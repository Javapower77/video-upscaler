# Video Upscaler — Image + Faces + Frame Rate + Audio

Upscale the resolution, restore faces, raise the frame rate and enhance the
audio of a video using state-of-the-art open-source models. Designed to run on
an **Azure VM with an NVIDIA H100 GPU** (Ubuntu, Python 3.11).

Pipeline: `upscale (Real-ESRGAN | SeedVR2) → restore faces (CodeFormer | GFPGAN) → interpolate (RIFE) → enhance audio (AudioSR) → remux`

**Limits:** max 2 minutes, max 300 MB.

## Setup (Azure VM · Ubuntu · H100)

```bash
./setup.sh
```

This installs `ffmpeg`, creates a Python 3.11 virtual environment in `.venv`,
installs all dependencies (PyTorch with CUDA 12.x for Hopper/sm_90), and vendors
the SeedVR2 pipeline into `third_party/seedvr2`. SeedVR2 weights (~6.5 GB),
RIFE weights (~25 MB) and face-restoration weights (~900 MB) are downloaded to
`models/` on first use.

## Run

```bash
source .venv/bin/activate
python app.py
```

The Gradio UI listens on `0.0.0.0:7860`. Open port **7860** in the Azure
Network Security Group (or use an SSH tunnel: `ssh -L 7860:localhost:7860 <vm>`).

Environment overrides:

| Variable | Default | Description |
|----------|---------|-------------|
| `GRADIO_SERVER_NAME` | `0.0.0.0` | Bind address |
| `GRADIO_SERVER_PORT` | `7860` | Port |
| `SEEDVR2_DIT_MODEL` | `seedvr2_ema_3b_fp16.safetensors` | SeedVR2 DiT checkpoint (see below) |
| `SEEDVR2_BATCH_SIZE` | `33` | *Max* frames per SeedVR2 batch (4n+1); auto-reduced for large outputs |
| `SEEDVR2_GB_PER_FRAME_MP` | `0.8` | VRAM model used for batch planning (GB per frame·megapixel) |
| `SEEDVR2_VAE_TILE_MP` | `2.0` | Output megapixels above which tiled VAE encode/decode is enabled |
| `SEEDVR2_ATTENTION` | auto | `sdpa` / `flash_attn_2` / `flash_attn_3` |
| `SEEDVR2_MODEL_DIR` | `models/SEEDVR2` | Where SeedVR2 weights are stored |
| `SEEDVR2_DEBUG` | `0` | `1` = verbose SeedVR2 logging (VRAM, timings) |
| `RIFE_MODEL_DIR` | `models/RIFE` | Where RIFE weights are stored |
| `FACE_MODEL_DIR` | `models/FACE` | Where CodeFormer / GFPGAN / facexlib weights are stored |
| `FACE_TEMPORAL_EMA` | `0.35` | Blend weight of the previous frame's restored face (0 = off) |

## H100 optimisations

- fp16 inference for Real-ESRGAN (when the model supports half precision)
- TF32 matmuls + cuDNN autotuning enabled (Hopper)
- Large 1024-px tiles — most 1080p/4K frames processed with minimal tiling
- SeedVR2: DiT + VAE kept resident in VRAM between requests (only the first
  request pays the load cost); bf16 compute; 33-frame temporal batches.
  Optional: `pip install flash-attn` for Flash-Attention 3 (auto-detected).

## Models

| Task | Model | Output |
|------|-------|--------|
| Video (fast/animation) | Real-ESRGAN `realesr-animevideov3` | 2× or 4× upscale |
| Video (realistic/photo) | Real-ESRGAN `RealESRGAN_x4plus` | 2× or 4× upscale |
| Video (high quality / slow) | SeedVR2 `seedvr2_ema_3b_fp16` (one-step diffusion) | 2× or 4× upscale |
| Faces (low-quality) | CodeFormer (fidelity 0–1) | restored 512-px face crops, blended back |
| Faces (conservative) | GFPGAN v1.4 | restored 512-px face crops, blended back |
| Frame rate | RIFE v4.26 (Practical-RIFE) | 2×/4× source, 60 or 120 fps |
| Audio (general) | AudioSR `audiosr_basic` | 48 kHz |
| Audio (speech) | AudioSR `audiosr_speech` | 48 kHz |

### SeedVR2 notes

SeedVR2 ([ByteDance-Seed/SeedVR](https://github.com/ByteDance-Seed/SeedVR),
ICLR 2026) is a one-step diffusion transformer for real-world video restoration.
Compared with Real-ESRGAN it is temporally consistent (no per-frame flicker) and
produces much sharper, cleaner detail on compressed / degraded footage — at the
cost of speed. Measured on this H100 with the 3B fp16 model, 320×176 input,
250 frames:

| Scale | Output | Time (warm) | Peak VRAM |
|-------|--------|-------------|-----------|
| 2× | 640×352 | ≈ 65 s | ≈ 10 GB |
| 4× | 1280×704 | ≈ 225 s | ≈ 24 GB |
| 4× (624×832 in, 109 f) | 2496×3328 | ≈ 500 s | ≈ 26 GB (batch auto-reduced to 9, VAE tiling on) |

**Memory management.** Activation memory grows with output pixels × frames per
batch, so the batch size is planned per request from the free VRAM
(`_pick_batch_size`), VAE tiling is enabled automatically above
`SEEDVR2_VAE_TILE_MP`, and all per-run tensors are released in a `finally`
block so a failed request never leaves VRAM pinned. `PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True`
is set at startup to avoid allocator fragmentation.

Other checkpoints can be selected via `SEEDVR2_DIT_MODEL`
(`seedvr2_ema_7b_fp16.safetensors`, `seedvr2_ema_7b_sharp_fp16.safetensors`,
fp8 variants…). The 7B models are ~2.5× slower for a modest quality gain.

The pipeline code is vendored from
[numz/ComfyUI-SeedVR2_VideoUpscaler](https://github.com/numz/ComfyUI-SeedVR2_VideoUpscaler)
(Apache-2.0) and driven in-process by `seedvr2_upscaler.py`.

### Face restoration notes

Blind face restoration runs *after* spatial upscaling (faces are restored at the
final resolution) and *before* RIFE. Per frame: RetinaFace-ResNet50 detects
faces → each is aligned to the 512×512 FFHQ template → restored → pasted back
through a ParseNet soft mask, so only face pixels change.

| Model | Best for | Licence |
|-------|----------|---------|
| **CodeFormer** | heavily degraded / low-res faces; `fidelity` slider trades detail (0) against identity (1), default 0.7 | S-Lab **non-commercial** |
| **GFPGAN v1.4** | conservative restoration that keeps identity; faster | Apache-2.0 |

Both are per-image models, so two guards reduce video flicker: faces with an
inter-ocular distance < 5 px are ignored (they pop in/out), and each restored
crop is blended with the previous frame's restored crop when the face has
barely moved (`FACE_TEMPORAL_EMA`, default 0.35). Frames with no faces are
passed through untouched. Measured on the H100: 640×480, 100 frames, ~4 faces
per frame → CodeFormer 52 s, GFPGAN 45 s (dominated by detection + paste-back),
< 2 GB VRAM.

Implementation: CodeFormer's two architecture files are vendored in
`third_party/codeformer` with the `basicsr` imports removed; GFPGAN is loaded
through `spandrel`; detection/alignment/masking use `facexlib` (installed
`--no-deps` to keep `numpy < 2`). We deliberately avoid the PyPI `basicsr` and
`gfpgan` packages, which fail on current torchvision.

### RIFE (frame interpolation) notes

[Practical-RIFE](https://github.com/hzwer/Practical-RIFE) v4.26 (MIT) inserts
motion-compensated intermediate frames. It runs *after* spatial upscaling so it
interpolates sharp frames, and keeps the clip duration unchanged.

- **Multiplier planning** — the output rate is always an integer multiple of the
  source (`ceil(target/src)`), so every original frame is preserved. NTSC rates
  are tolerated: 29.97 → "60 fps" gives ×2 = 59.94 fps, not ×3.
- **Scene-cut guard** — if two consecutive frames have SSIM < 0.2 (a hard cut)
  the previous frame is held instead of blending two shots together.
- **fp16** inference; flow estimated at half scale for ≥ 4 MP frames (as the
  RIFE README recommends for 4K).
- Measured on the H100: 320×180, 300 frames → ×3 in **4.8 s**, ×5 in **8.2 s**
  (≈ 190 output fps), < 0.1 GB VRAM. Cost scales with output pixels.

`third_party/rife/IFNet_HDv3.py` and `warplayer.py` are the upstream files with
one local patch: the warp grid cache is keyed on dtype so fp16 works. The
`flownet.pkl` weights are the author's official release, fetched from
[`hzwer/RIFE`](https://huggingface.co/hzwer/RIFE) on Hugging Face.

## Batch processing

`batch-videos.py` applies Upscale / Frame Rate / Face Restoration to many videos in one run
(outputs `<name>_slp.mp4` in `output/batch/`, "Download All" as TAR). See
[BATCH_PROCESSING.md](BATCH_PROCESSING.md).

```bash
GRADIO_SERVER_PORT=7862 python batch-videos.py
```
