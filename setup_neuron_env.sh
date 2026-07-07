#!/bin/bash
# Sets up the AWS Neuron SDK + PyTorch NeuronX environment from scratch on a
# vanilla Ubuntu trn2.3xlarge instance.
#
# The assignment's install.sh assumes a Stanford-provided custom AMI that
# already has a venv at /opt/aws_neuronx_venv_pytorch_2_8 with the Neuron
# driver/runtime/compiler pre-installed. If you're on a plain Ubuntu AMI
# instead, run this script FIRST to build that same venv, then run the
# original install.sh (unmodified) to append the bashrc activation line and
# set up InfluxDB for neuron-profile.
#
#   sudo ./setup_neuron_env.sh
#   source install.sh
#
# The default trn2.3xlarge root EBS volume is tiny (a few GB) -- nowhere near
# enough for the Neuron driver/toolchain + a torch venv. Rather than paying to
# grow the (billed separately) root EBS volume, this script uses the
# instance's local NVMe instance store (bundled free with the instance) for
# the large stuff: it's formatted/mounted at $SCRATCH_DIR, and the venv is
# created there with a symlink back to the path install.sh expects.
#
# Reference: https://awsdocs-neuron.readthedocs-hosted.com/en/latest/setup/pytorch/manual.html

set -euo pipefail

VENV_PATH="/opt/aws_neuronx_venv_pytorch_2_8"
SCRATCH_DIR="/mnt/scratch"
SCRATCH_DEVICE="/dev/nvme1n1"
VENV_DIR="$SCRATCH_DIR/aws_neuronx_venv_pytorch_2_8"
TORCH_NEURONX_VERSION="2.8.*"
NEURONX_CC_VERSION="2.*"
REAL_USER="${SUDO_USER:-$USER}"

if [ "$EUID" -ne 0 ]; then
    echo "Please run this script with sudo: sudo ./setup_neuron_env.sh"
    exit 1
fi

. /etc/os-release
echo "==> Detected ${PRETTY_NAME} (codename: ${VERSION_CODENAME})"

# The Neuron apt repo only publishes for focal/jammy/noble (20.04/22.04/24.04).
# On a newer/unsupported release, fall back to the noble (24.04) packages --
# usually fine since glibc/ABI is backward compatible, but this is a
# best-effort compatibility shim, not an AWS-tested combination.
NEURON_CODENAME="$VERSION_CODENAME"
case "$VERSION_CODENAME" in
    focal|jammy|noble) ;;
    *)
        echo "==> WARNING: ${PRETTY_NAME} (${VERSION_CODENAME}) is not a Neuron-supported release."
        echo "    Falling back to the 'noble' (Ubuntu 24.04) apt packages as a best-effort compat shim."
        NEURON_CODENAME="noble"
        ;;
esac

# 0. Format (if needed) and mount the local NVMe instance store, then point
# /opt/aws_neuronx_venv_pytorch_2_8 at it via symlink. Instance store data is
# lost if the instance is *stopped* (not on reboot), which is fine since
# capacity-block instances aren't meant to be stopped.
if [ -b "$SCRATCH_DEVICE" ]; then
    if ! blkid "$SCRATCH_DEVICE" > /dev/null 2>&1; then
        echo "==> Formatting unformatted instance store $SCRATCH_DEVICE as ext4"
        mkfs.ext4 -q -F "$SCRATCH_DEVICE"
    fi

    mkdir -p "$SCRATCH_DIR"
    if ! mountpoint -q "$SCRATCH_DIR"; then
        echo "==> Mounting $SCRATCH_DEVICE at $SCRATCH_DIR"
        mount "$SCRATCH_DEVICE" "$SCRATCH_DIR"
    fi

    SCRATCH_UUID=$(blkid -s UUID -o value "$SCRATCH_DEVICE")
    if ! grep -q "$SCRATCH_UUID" /etc/fstab; then
        echo "UUID=$SCRATCH_UUID $SCRATCH_DIR ext4 defaults,nofail 0 2" >> /etc/fstab
    fi

    chown "$REAL_USER":"$REAL_USER" "$SCRATCH_DIR"
else
    echo "==> No instance store found at $SCRATCH_DEVICE; falling back to $VENV_PATH on the root volume"
    SCRATCH_DIR="/opt"
    VENV_DIR="$VENV_PATH"
fi

# 1. Add the Neuron apt repository and its signing key (using the modern
# signed-by keyring approach -- apt-key is removed on newer Ubuntu releases)
if [ ! -f /etc/apt/sources.list.d/neuron.list ]; then
    echo "==> Adding Neuron apt repository"
    mkdir -p /etc/apt/keyrings
    wget -qO - https://apt.repos.neuron.amazonaws.com/GPG-PUB-KEY-AMAZON-AWS-NEURON.PUB | gpg --dearmor -o /etc/apt/keyrings/aws-neuron.gpg
    tee /etc/apt/sources.list.d/neuron.list > /dev/null <<EOF
deb [signed-by=/etc/apt/keyrings/aws-neuron.gpg] https://apt.repos.neuron.amazonaws.com ${NEURON_CODENAME} main
EOF
fi

# 2. Install the Neuron driver, runtime, and profiling/monitoring tools
echo "==> Installing Neuron driver/runtime/tools (apt)"
apt-get update -y
apt-get install -y linux-headers-"$(uname -r)" g++ git python3-venv
apt-get install -y aws-neuronx-dkms=2.* \
                    aws-neuronx-collectives=2.* \
                    aws-neuronx-runtime-lib=2.* \
                    aws-neuronx-tools=2.*

if ! grep -q '/opt/aws/neuron/bin' /etc/environment; then
    echo 'PATH="/opt/aws/neuron/bin:'"$PATH"'"' | tee -a /etc/environment > /dev/null
fi
export PATH=/opt/aws/neuron/bin:$PATH

# 3. Create the python venv on the big scratch disk, then symlink it to the
# exact path install.sh's bashrc line expects (/opt/... lives on the small
# root volume, so it can't hold the venv itself).
if [ ! -d "$VENV_DIR" ]; then
    echo "==> Creating venv at $VENV_DIR"
    sudo -u "$REAL_USER" python3 -m venv "$VENV_DIR"
fi

if [ "$VENV_DIR" != "$VENV_PATH" ] && [ ! -e "$VENV_PATH" ]; then
    ln -s "$VENV_DIR" "$VENV_PATH"
elif [ "$VENV_DIR" != "$VENV_PATH" ] && [ ! -L "$VENV_PATH" ]; then
    echo "WARNING: $VENV_PATH already exists and isn't a symlink to $VENV_DIR; leaving it alone."
fi

# 4. Install torch-neuronx, the NKI compiler (neuronx-cc), and other deps
# used directly by the assignment kernels (numpy, torchvision). Pip's
# download/build cache also goes on the scratch disk so it doesn't fill up
# the root volume.
echo "==> Installing torch-neuronx / neuronx-cc into the venv"
sudo -u "$REAL_USER" env PIP_CACHE_DIR="$SCRATCH_DIR/pip-cache" bash <<PYSETUP
source "$VENV_DIR/bin/activate"
python -m pip install -U pip
python -m pip config set global.extra-index-url https://pip.repos.neuron.amazonaws.com
python -m pip install "neuronx-cc==${NEURONX_CC_VERSION}" "torch-neuronx==${TORCH_NEURONX_VERSION}" torchvision numpy
PYSETUP

echo "==> Verifying installation"
sudo -u "$REAL_USER" bash -c "source '$VENV_PATH/bin/activate' && python -c 'import torch, torch_neuronx, neuronxcc; print(f\"torch {torch.__version__}, torch-neuronx {torch_neuronx.__version__}\")'"

cat <<EOF

Done. The Neuron kernel driver was just installed via dkms; if 'neuron-ls'
below doesn't show your device(s), reboot the instance and re-check.

  neuron-ls

Next, run the assignment's own install.sh to wire up ~/.bashrc and InfluxDB:

  source install.sh

EOF
