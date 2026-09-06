#!/usr/bin/env bash
# Setup script for Azure VM (Ubuntu, NVIDIA H100, Python 3.11)
set -euo pipefail

cd "$(dirname "$0")"

echo "==> Installing system packages (ffmpeg)…"
sudo apt-get update -y
sudo apt-get install -y ffmpeg python3.11-venv

echo "==> Checking NVIDIA driver…"
if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "WARNING: nvidia-smi not found. Install the NVIDIA driver / CUDA toolkit first:"
    echo "  sudo apt-get install -y ubuntu-drivers-common && sudo ubuntu-drivers autoinstall"
    echo "Continuing anyway (app will fall back to CPU)…"
else
    nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
fi

echo "==> Creating Python 3.11 virtual environment…"
python3.11 -m venv .venv
source .venv/bin/activate

echo "==> Installing Python dependencies…"
pip install --upgrade pip wheel
pip install -r requirements.txt

echo "==> Fetching SeedVR2 (numz/ComfyUI-SeedVR2_VideoUpscaler, standalone pipeline)…"
SEEDVR2_DIR="third_party/seedvr2"
SEEDVR2_COMMIT="4490bd1f482e026674543386bb2a4d176da245b9"   # v2.5.24 (2025-12-24)
if [[ ! -d "$SEEDVR2_DIR/.git" ]]; then
    git clone --filter=blob:none https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler.git "$SEEDVR2_DIR"
fi
git -C "$SEEDVR2_DIR" fetch --quiet origin "$SEEDVR2_COMMIT" || true
git -C "$SEEDVR2_DIR" checkout --quiet "$SEEDVR2_COMMIT"
echo "    SeedVR2 weights (~6.5 GB) are downloaded to models/SEEDVR2 on first use."

echo "==> RIFE v4.26 (frame interpolation) — code is vendored in third_party/rife;"
echo "    weights (~25 MB) are downloaded to models/RIFE on first use."

echo "==> Face restoration (CodeFormer + GFPGAN) — installing facexlib without deps"
echo "    (its metadata would drag in numpy 2.x, which breaks audiosr)…"
pip install --no-deps "facexlib==0.3.0"
echo "    Weights (~900 MB: CodeFormer, GFPGANv1.4, RetinaFace, ParseNet) download to"
echo "    models/FACE on first use."

echo "==> Verifying CUDA availability in PyTorch…"
python - <<'EOF'
import torch
print(f"torch {torch.__version__} | CUDA available: {torch.cuda.is_available()}")
if torch.cuda.is_available():
    print(f"GPU: {torch.cuda.get_device_name(0)}")
EOF

echo ""
echo "Setup complete. Start the app with:"
echo "  source .venv/bin/activate && python app.py"
echo "The UI will be available on http://<vm-ip>:7860 (open port 7860 in the Azure NSG)."
