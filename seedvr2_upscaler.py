"""SeedVR2 one-step diffusion video upscaling (high quality / slow).

Wraps the standalone SeedVR2 pipeline from numz/ComfyUI-SeedVR2_VideoUpscaler
(vendored under ``third_party/seedvr2``) and runs it in-process on the local
H100. The DiT + VAE stay resident on the GPU between calls (≈95 GB VRAM is
plenty), so only the first request pays the model-load cost.
"""
import gc
import math
import os
import sys
import numpy as np
import ffmpeg
import torch

_HERE = os.path.dirname(os.path.abspath(__file__))
SEEDVR2_DIR = os.path.join(_HERE, "third_party", "seedvr2")
MODEL_DIR = os.environ.get("SEEDVR2_MODEL_DIR", os.path.join(_HERE, "models", "SEEDVR2"))

if SEEDVR2_DIR not in sys.path:
    sys.path.insert(0, SEEDVR2_DIR)

# ---------------------------------------------------------------------------
# Configuration (tuned for a single H100 80/95 GB)
# ---------------------------------------------------------------------------

# fp16 3B: best quality/speed trade-off on an H100. 7B is available too but is
# ~2.5x slower for a modest quality gain.
DIT_MODEL = os.environ.get("SEEDVR2_DIT_MODEL", "seedvr2_ema_3b_fp16.safetensors")

# Frames per DiT batch — MUST be 4n+1. 33 frames ≈ 1.3 s @ 25 fps is a good
# temporal window; larger batches are faster per-frame but need more VRAM at
# higher output resolutions. This is the *maximum*; the effective batch is
# scaled down automatically with output resolution (see _pick_batch_size).
BATCH_SIZE = int(os.environ.get("SEEDVR2_BATCH_SIZE", "33"))
TEMPORAL_OVERLAP = 3     # frames blended between batches (smooth transitions)
PREPEND_FRAMES = 4       # reversed frames prepended to reduce start artifacts
COLOR_CORRECTION = "lab" # perceptual colour match to source (recommended)

# VRAM budget. Activation memory of the DiT (window attention + VAE) scales
# roughly linearly with output pixels × frames per batch. Empirically on this
# H100: 33 frames @ 1280×704 (~0.9 MP) peaks around 24 GB → ~0.8 GB per
# (frame · megapixel). We budget against the *free* VRAM at call time.
_GB_PER_FRAME_MP = float(os.environ.get("SEEDVR2_GB_PER_FRAME_MP", "0.8"))
_VRAM_SAFETY = 0.75      # never plan to use more than 75 % of what is free

# VAE tiling kicks in above this many output megapixels (per frame). The VAE
# is the first thing to OOM at high resolution; tiles are 1024 px with 128 px
# overlap, which is seamless in practice.
VAE_TILE_THRESHOLD_MP = float(os.environ.get("SEEDVR2_VAE_TILE_MP", "2.0"))
VAE_TILE_SIZE = 1024
VAE_TILE_OVERLAP = 128

# Reduce allocator fragmentation ("23 GiB reserved but unallocated" in the OOM
# report). Must be set before the first CUDA allocation — harmless otherwise.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

# Enable Flash-Attention 3 automatically if the package is present (Hopper).
def _pick_attention_mode() -> str:
    forced = os.environ.get("SEEDVR2_ATTENTION")
    if forced:
        return forced
    try:
        import flash_attn_interface  # noqa: F401  (FA3 package)
        return "flash_attn_3"
    except ImportError:
        pass
    try:
        import flash_attn  # noqa: F401
        return "flash_attn_2"
    except ImportError:
        return "sdpa"

ATTENTION_MODE = _pick_attention_mode()

_state: dict = {}   # holds Debug, ctx, runner between calls


# ---------------------------------------------------------------------------
# Model preparation (cached)
# ---------------------------------------------------------------------------

def _get_runner():
    """Load (or reuse) SeedVR2 DiT + VAE on cuda:0 and return (ctx, runner, debug)."""
    if "runner" in _state:
        return _state["ctx"], _state["runner"], _state["debug"]

    from src.utils.debug import Debug
    from src.utils.downloads import download_weight
    from src.utils.model_registry import DEFAULT_VAE
    from src.core.generation_utils import setup_generation_context, prepare_runner

    debug = Debug(enabled=os.environ.get("SEEDVR2_DEBUG", "0") == "1")

    os.makedirs(MODEL_DIR, exist_ok=True)
    download_weight(DIT_MODEL, DEFAULT_VAE, model_dir=MODEL_DIR, debug=debug)

    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    ctx = setup_generation_context(
        dit_device=device,
        vae_device=device,
        # Offload target == inference device → cleanup between phases becomes a
        # no-op, so DiT and VAE stay resident in VRAM across requests.
        dit_offload_device=device,
        vae_offload_device=device,
        tensor_offload_device="cpu",  # intermediate latents → RAM (long clips)
        debug=debug,
    )

    runner, cache_context = prepare_runner(
        dit_model=DIT_MODEL,
        vae_model=DEFAULT_VAE,
        model_dir=MODEL_DIR,
        debug=debug,
        ctx=ctx,
        dit_cache=True,
        vae_cache=True,
        dit_id="app_dit",
        vae_id="app_vae",
        block_swap_config=None,
        encode_tiled=False,
        decode_tiled=False,
        attention_mode=ATTENTION_MODE,
    )
    ctx["cache_context"] = cache_context

    _state.update(ctx=ctx, runner=runner, debug=debug)
    return ctx, runner, debug


def _set_vae_tiling(runner, enabled: bool) -> None:
    """Toggle tiled VAE encode/decode on the cached runner for this request."""
    runner.encode_tiled = enabled
    runner.decode_tiled = enabled
    runner.encode_tile_size = (VAE_TILE_SIZE, VAE_TILE_SIZE)
    runner.decode_tile_size = (VAE_TILE_SIZE, VAE_TILE_SIZE)
    runner.encode_tile_overlap = (VAE_TILE_OVERLAP, VAE_TILE_OVERLAP)
    runner.decode_tile_overlap = (VAE_TILE_OVERLAP, VAE_TILE_OVERLAP)
    runner.tile_debug = "false"


def _free_vram_gb() -> float:
    if not torch.cuda.is_available():
        return float("inf")
    free, _total = torch.cuda.mem_get_info()
    return free / 1e9


def _pick_batch_size(n_frames: int, out_w: int, out_h: int) -> int:
    """Largest 4n+1 batch ≤ BATCH_SIZE that fits the VRAM budget for this output size."""
    out_mp = (out_w * out_h) / 1e6
    budget_gb = _free_vram_gb() * _VRAM_SAFETY
    max_frames_by_vram = int(budget_gb / max(_GB_PER_FRAME_MP * out_mp, 1e-6))

    batch = min(BATCH_SIZE, n_frames, max(max_frames_by_vram, 1))
    batch = max(1, ((batch - 1) // 4) * 4 + 1)  # snap down to 4n+1
    return batch


def _release_vram() -> None:
    """Drop cached allocator blocks between requests (models stay resident)."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()


# ---------------------------------------------------------------------------
# Frame I/O helpers (FFmpeg pipes, same approach as video_upscaler.py)
# ---------------------------------------------------------------------------

def _read_all_frames(input_path: str, w: int, h: int) -> torch.Tensor:
    """Decode the whole clip to a [T, H, W, 3] float32 tensor in [0, 1] (RGB)."""
    out, _ = (
        ffmpeg
        .input(input_path)
        .output("pipe:", format="rawvideo", pix_fmt="rgb24")
        .run(capture_stdout=True, quiet=True)
    )
    frames = np.frombuffer(out, dtype=np.uint8).reshape(-1, h, w, 3)
    return torch.from_numpy(frames.copy()).float().div_(255.0)


def _write_frames(frames: torch.Tensor, output_path: str, fps: float) -> None:
    """Encode a [T, H, W, 3] float tensor in [0, 1] (RGB) to H.264 MP4."""
    t, h, w, _ = frames.shape
    writer = (
        ffmpeg
        .input("pipe:", format="rawvideo", pix_fmt="rgb24", s=f"{w}x{h}", r=fps)
        .output(output_path, vcodec="libx264", pix_fmt="yuv420p", crf=18, preset="fast")
        .overwrite_output()
        .run_async(pipe_stdin=True, quiet=True)
    )
    frames_u8 = frames.clamp_(0, 1).mul_(255).round_().to(torch.uint8).numpy()
    for i in range(t):
        writer.stdin.write(frames_u8[i].tobytes())
    writer.stdin.close()
    writer.wait()


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def upscale_video_seedvr2(
    input_path: str,
    output_path: str,
    scale: int,
    info: dict,
    progress_cb=None,
) -> None:
    """Upscale ``input_path`` by ``scale`` (2 or 4) with SeedVR2 → ``output_path``.

    SeedVR2 targets a *short-side resolution* rather than a fixed factor, so we
    compute ``resolution = min(w, h) * scale``; aspect ratio is preserved.
    """
    from src.core.generation_utils import compute_generation_info, load_text_embeddings, script_directory
    from src.core.generation_phases import (
        encode_all_batches, upscale_all_batches, decode_all_batches, postprocess_all_batches,
    )

    ctx, runner, debug = _get_runner()

    w, h, fps = info["width"], info["height"], info["fps"]
    resolution = min(w, h) * scale
    target_h, target_w = h * scale, w * scale
    out_mp = (target_w * target_h) / 1e6

    # Start from a clean allocator so leftovers from a previous (possibly
    # failed) request don't eat into this run's budget.
    _release_vram()

    # Per-run context reset (keep device / dtype configuration, as the CLI does)
    keep = {"dit_device", "vae_device", "dit_offload_device", "vae_offload_device",
            "tensor_offload_device", "compute_dtype", "cache_context"}
    for k in list(ctx.keys()):
        if k not in keep:
            del ctx[k]

    ctx["text_embeds"] = load_text_embeddings(script_directory, ctx["dit_device"], ctx["compute_dtype"], debug)

    frames = _read_all_frames(input_path, w, h)
    n_frames = frames.shape[0]

    # Memory-aware settings for this particular clip
    batch = _pick_batch_size(n_frames, target_w, target_h)
    overlap = TEMPORAL_OVERLAP if batch > TEMPORAL_OVERLAP + 1 else 0
    use_tiling = out_mp > VAE_TILE_THRESHOLD_MP
    _set_vae_tiling(runner, use_tiling)
    print(f"[seedvr2] {w}x{h} → {target_w}x{target_h} ({out_mp:.2f} MP), "
          f"{n_frames} frames, batch={batch}, overlap={overlap}, "
          f"vae_tiling={'on' if use_tiling else 'off'}, "
          f"free VRAM={_free_vram_gb():.1f} GB", flush=True)

    # Map the 4 phases onto a single 0‥1 progress bar
    phase_span = {"Phase 1": (0.00, 0.15), "Phase 2": (0.15, 0.70),
                  "Phase 3": (0.70, 0.92), "Phase 4": (0.92, 1.00)}

    def _cb(step, total, _frames=None, msg=""):
        if not progress_cb or total <= 0:
            return
        for key, (lo, hi) in phase_span.items():
            if msg.startswith(key):
                progress_cb(lo + (hi - lo) * step / total)
                return

    seed = 42
    result = None
    try:
        frames, gen_info = compute_generation_info(
            ctx=ctx, images=frames, resolution=resolution, max_resolution=0,
            batch_size=batch, uniform_batch_size=True, seed=seed,
            prepend_frames=PREPEND_FRAMES, temporal_overlap=overlap, debug=debug,
        )

        ctx = encode_all_batches(
            runner, ctx=ctx, images=frames, debug=debug,
            batch_size=batch, uniform_batch_size=True, seed=seed,
            progress_callback=_cb, temporal_overlap=overlap,
            resolution=resolution, max_resolution=0,
            input_noise_scale=0.0, color_correction=COLOR_CORRECTION,
        )
        ctx = upscale_all_batches(
            runner, ctx=ctx, debug=debug, progress_callback=_cb,
            seed=seed, latent_noise_scale=0.0, cache_model=True,
        )
        ctx = decode_all_batches(
            runner, ctx=ctx, debug=debug, progress_callback=_cb, cache_model=True,
        )
        ctx = postprocess_all_batches(
            ctx=ctx, debug=debug, progress_callback=_cb,
            color_correction=COLOR_CORRECTION, prepend_frames=PREPEND_FRAMES,
            temporal_overlap=overlap, batch_size=batch,
        )

        result = ctx.pop("final_video")
        if result.is_cuda:
            result = result.cpu()
        result = result.to(torch.float32)

        # SeedVR2 pads to a multiple of 16 internally; crop to the exact target size
        result = result[:n_frames, :target_h, :target_w, :3]

        _write_frames(result, output_path, fps)

    except torch.OutOfMemoryError as e:
        raise RuntimeError(
            f"SeedVR2 ran out of GPU memory at {target_w}x{target_h} with batch={batch}. "
            f"Try a smaller scale factor, a shorter/lower-resolution clip, or lower "
            f"SEEDVR2_BATCH_SIZE / SEEDVR2_GB_PER_FRAME_MP."
        ) from e
    finally:
        # Always drop per-run tensors — even on failure — so the next request
        # starts from a clean slate. Models remain cached on the GPU.
        for k in ("all_latents", "all_upscaled_latents", "batch_samples",
                  "final_video", "text_embeds", "video_transform", "true_target_dims"):
            ctx.pop(k, None)
        del frames
        if result is not None:
            del result
        _release_vram()
