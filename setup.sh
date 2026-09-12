#!/usr/bin/env bash
# Complete, repeatable setup for a clean Ubuntu 22.04/24.04 installation.
#
# Usage:
#   ./setup.sh
#   RECREATE_VENV=1 ./setup.sh       # rebuild an existing .venv
#   INSTALL_NVIDIA_DRIVER=1 ./setup.sh # install Ubuntu's recommended driver
#
# The NVIDIA kernel driver must already be installed by the host/VM image. The
# script verifies it, but deliberately does not replace a running kernel driver.
set -Eeuo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT_DIR"

PYTHON_BIN="${PYTHON_BIN:-python3.11}"
VENV_DIR="${VENV_DIR:-$ROOT_DIR/.venv}"
SEEDVR2_DIR="$ROOT_DIR/third_party/seedvr2"
SEEDVR2_REPO="https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git"
SEEDVR2_COMMIT="4490bd1f482e026674543386bb2a4d176da245b9" # v2.5.24

log() { printf '\n==> %s\n' "$*"; }
die() { printf '\nERROR: %s\n' "$*" >&2; exit 1; }
trap 'printf "\nERROR: setup failed at line %s.\n" "$LINENO" >&2' ERR

if [[ ! -r /etc/os-release ]]; then
    die "This installer requires Ubuntu Linux (/etc/os-release was not found)."
fi
# shellcheck disable=SC1091
source /etc/os-release
[[ "${ID:-}" == "ubuntu" ]] || die "Unsupported distribution: ${PRETTY_NAME:-unknown}. Ubuntu is required."

if [[ $EUID -eq 0 ]]; then
    SUDO=()
elif command -v sudo >/dev/null 2>&1; then
    SUDO=(sudo)
else
    die "sudo is required when setup.sh is not run as root."
fi

log "Installing Ubuntu system dependencies"
export DEBIAN_FRONTEND=noninteractive
"${SUDO[@]}" apt-get update -y
"${SUDO[@]}" apt-get install -y --no-install-recommends \
    ca-certificates curl wget git git-lfs ffmpeg \
    build-essential pkg-config software-properties-common pciutils \
    libsndfile1 libgl1 libglib2.0-0 espeak-ng \
    python3-pip python3-dev

# Ubuntu 24.04 defaults to Python 3.12 and its standard repositories may expose
# some Python 3.11 metadata without providing python3.11-venv. Check the exact
# packages instead of checking only `python3.11`.
python311_has_venv() {
    command -v "$PYTHON_BIN" >/dev/null 2>&1 && \
        "$PYTHON_BIN" -c 'import ensurepip, venv' >/dev/null 2>&1
}

apt_has_candidate() {
    local package="$1"
    local candidate
    candidate="$(apt-cache policy "$package" 2>/dev/null | awk '/Candidate:/ {print $2; exit}')"
    [[ -n "$candidate" && "$candidate" != "(none)" ]]
}

if ! python311_has_venv; then
    log "Installing Python 3.11 with venv support"
    if ! apt_has_candidate python3.11 || \
       ! apt_has_candidate python3.11-venv || \
       ! apt_has_candidate python3.11-dev; then
        log "Enabling the deadsnakes PPA for Python 3.11"
        "${SUDO[@]}" add-apt-repository -y ppa:deadsnakes/ppa
        "${SUDO[@]}" apt-get update -y
    fi

    apt_has_candidate python3.11 || die "No python3.11 package is available after enabling the PPA."
    apt_has_candidate python3.11-venv || die "No python3.11-venv package is available after enabling the PPA."
    apt_has_candidate python3.11-dev || die "No python3.11-dev package is available after enabling the PPA."
    "${SUDO[@]}" apt-get install -y python3.11 python3.11-venv python3.11-dev
fi

command -v "$PYTHON_BIN" >/dev/null 2>&1 || die "Python 3.11 installation failed."
"$PYTHON_BIN" -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version'
python311_has_venv || die "Python 3.11 is installed, but its venv/ensurepip modules are unavailable."

log "Checking NVIDIA driver"
if command -v nvidia-smi >/dev/null 2>&1; then
    nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv,noheader
else
    cat >&2 <<'EOF'
WARNING: nvidia-smi was not found. The application can install, but GPU models
will not run efficiently. On a cloud VM, select an NVIDIA GPU image/extension.
EOF
        if [[ "${INSTALL_NVIDIA_DRIVER:-0}" == "1" ]]; then
                if ! lspci | grep -qi nvidia; then
                        die "INSTALL_NVIDIA_DRIVER=1 was set, but no NVIDIA PCI device was detected."
                fi
                log "Installing Ubuntu's recommended NVIDIA driver"
                "${SUDO[@]}" apt-get install -y ubuntu-drivers-common "linux-headers-$(uname -r)"
                "${SUDO[@]}" ubuntu-drivers install
                cat >&2 <<'EOF'
The NVIDIA driver was installed. Reboot after setup completes, then rerun
setup.sh to perform the CUDA smoke test:
    sudo reboot
EOF
        else
                cat >&2 <<'EOF'
On bare Ubuntu, install the recommended driver by rerunning this script with:
    INSTALL_NVIDIA_DRIVER=1 ./setup.sh
Then reboot before running GPU workloads.
EOF
        fi
fi

log "Creating Python 3.11 virtual environment"
if [[ "${RECREATE_VENV:-0}" == "1" && -d "$VENV_DIR" ]]; then
    rm -rf "$VENV_DIR"
fi
if [[ ! -x "$VENV_DIR/bin/python" ]]; then
    "$PYTHON_BIN" -m venv "$VENV_DIR"
fi
# shellcheck disable=SC1091
source "$VENV_DIR/bin/activate"
python -c 'import sys; assert sys.version_info[:2] == (3, 11), sys.version'

log "Installing Python dependencies (this can take several minutes)"
python -m pip install --upgrade pip wheel "setuptools<81"
python -m pip install -r requirements.txt

# facexlib's metadata permits NumPy 2 and basicsr; allowing pip to resolve it
# normally breaks AudioSR/current torchvision. Runtime dependencies are listed
# explicitly in requirements.txt, so install facexlib itself without deps.
python -m pip install --no-deps "facexlib==0.3.0"
# Do not run `pip check`: facexlib declares opencv-python, while this project
# intentionally uses the API-compatible opencv-python-headless package.

log "Fetching the pinned SeedVR2 standalone pipeline"
if [[ -e "$SEEDVR2_DIR" && ! -d "$SEEDVR2_DIR/.git" ]]; then
    die "$SEEDVR2_DIR exists but is not a Git checkout. Move/remove it and rerun setup.sh."
fi
if [[ ! -d "$SEEDVR2_DIR/.git" ]]; then
    mkdir -p "$(dirname "$SEEDVR2_DIR")"
    git clone --filter=blob:none "$SEEDVR2_REPO" "$SEEDVR2_DIR"
fi
git -C "$SEEDVR2_DIR" fetch --quiet origin "$SEEDVR2_COMMIT"
git -C "$SEEDVR2_DIR" checkout --quiet --detach "$SEEDVR2_COMMIT"
[[ "$(git -C "$SEEDVR2_DIR" rev-parse HEAD)" == "$SEEDVR2_COMMIT" ]] || \
    die "SeedVR2 checkout verification failed."

log "Creating application directories"
mkdir -p models/SEEDVR2 models/RIFE models/FACE output/batch

log "Verifying installation"
command -v ffmpeg >/dev/null 2>&1 || die "ffmpeg is unavailable."
ffmpeg -version | head -n 1
python - <<'PY'
import importlib
import subprocess
import sys

required = {
    "cv2": "opencv-python-headless",
    "diffusers": "diffusers",
    "facexlib": "facexlib",
    "ffmpeg": "ffmpeg-python",
    "gradio": "gradio",
    "librosa": "librosa",
    "omegaconf": "omegaconf",
    "safetensors": "safetensors",
    "spandrel": "spandrel",
    "torch": "torch",
    "torchaudio": "torchaudio",
    "torchvision": "torchvision",
}
failed = []
for module, package in required.items():
    try:
        importlib.import_module(module)
    except Exception as exc:
        failed.append(f"{package} ({exc})")
if failed:
    raise SystemExit("Import verification failed: " + ", ".join(failed))

import gradio, numpy, torch
print(f"Python {sys.version.split()[0]}")
print(f"Gradio {gradio.__version__} | NumPy {numpy.__version__}")
print(f"PyTorch {torch.__version__}")
print(f"CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"CUDA runtime: {torch.version.cuda}")
    print(f"GPU: {torch.cuda.get_device_name(0)}")
    x = torch.ones(1, device="cuda")
    print(f"CUDA smoke test: {x.item():.0f}")
elif subprocess.run(["sh", "-c", "command -v nvidia-smi"], check=False).returncode == 0:
    raise SystemExit(
        "NVIDIA driver is visible, but this PyTorch build cannot use CUDA. "
        "Check the driver version and reinstall requirements.txt."
    )
PY

cat <<EOF

Setup complete.

Model weights are downloaded automatically on first use:
  Real-ESRGAN: ~70 MB       SeedVR2: ~6.5 GB
  RIFE:       ~25 MB       Face models: ~900 MB
Internet access and approximately 10 GB of free disk space are required for
the first model runs.

Single-video app:
  source "$VENV_DIR/bin/activate"
  python app.py
  Open http://<server-ip>:7860

Batch app:
  source "$VENV_DIR/bin/activate"
  python batch-videos.py
  Open http://<server-ip>:7862

If the server is remote, allow TCP 7860/7862 in its firewall/security group,
or use an SSH tunnel. Do not expose Gradio directly to the public Internet
without authentication or a secured reverse proxy.
EOF
