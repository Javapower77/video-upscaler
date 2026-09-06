"""RIFE v4.26 frame interpolation (frame-rate upscaling).

Uses the Practical-RIFE IFNet (hzwer/Practical-RIFE, MIT) vendored under
``third_party/rife``. Weights are the author's official v4.26 release,
fetched from the ``hzwer/RIFE`` Hugging Face repo on first use.

Pipeline position: runs *after* spatial upscaling so it interpolates the
sharp frames, and *before* audio remux (audio timing is unchanged — we only
add frames, the duration stays identical).
"""
import os
import sys
import zipfile
import numpy as np
import ffmpeg
import torch
import torch.nn.functional as F

_HERE = os.path.dirname(os.path.abspath(__file__))
RIFE_DIR = os.path.join(_HERE, "third_party", "rife")
MODEL_DIR = os.environ.get("RIFE_MODEL_DIR", os.path.join(_HERE, "models", "RIFE"))
WEIGHTS_PATH = os.path.join(MODEL_DIR, "rife_v4.26_flownet.pkl")

# Official release from the author's HF repo (same file as the Google Drive link
# in the Practical-RIFE README).
HF_REPO = "hzwer/RIFE"
HF_ZIP = "RIFEv4.26_0921.zip"
ZIP_MEMBER = "RIFEv4.26_0921/flownet.pkl"

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_FP16 = DEVICE.type == "cuda"

# RIFE processes the full frame at once; downscale the flow estimate for very
# large frames (the README recommends scale=0.5 for 4K). Output is unaffected.
FLOW_SCALE_THRESHOLD_MP = 4.0   # ≥ ~2880x1440 → scale 0.5

# Scene-cut guard: if two consecutive frames are this dissimilar, don't try to
# interpolate across them — duplicate the first frame instead (avoids ghosting).
SCENE_CUT_SSIM = 0.2

if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

_model = None


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def _ensure_weights() -> str:
    if os.path.exists(WEIGHTS_PATH):
        return WEIGHTS_PATH
    from huggingface_hub import hf_hub_download
    os.makedirs(MODEL_DIR, exist_ok=True)
    zip_path = hf_hub_download(HF_REPO, HF_ZIP, local_dir=os.path.join(MODEL_DIR, "_dl"))
    with zipfile.ZipFile(zip_path) as zf, zf.open(ZIP_MEMBER) as src, open(WEIGHTS_PATH, "wb") as dst:
        dst.write(src.read())
    return WEIGHTS_PATH


def _load_model():
    global _model
    if _model is not None:
        return _model

    from third_party.rife.IFNet_HDv3 import IFNet

    net = IFNet()
    state = torch.load(_ensure_weights(), map_location="cpu", weights_only=True)
    # Training checkpoints are saved from a DDP wrapper → strip "module."
    state = {k.replace("module.", ""): v for k, v in state.items()}
    net.load_state_dict(state, strict=False)
    net.eval().to(DEVICE)
    if USE_FP16:
        net.half()
    _model = net
    return net


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _ssim_small(a: torch.Tensor, b: torch.Tensor) -> float:
    """Cheap global SSIM on 32x32 thumbnails — enough to detect hard cuts."""
    a = F.interpolate(a.float(), (32, 32), mode="bilinear", align_corners=False)
    b = F.interpolate(b.float(), (32, 32), mode="bilinear", align_corners=False)
    mu_a, mu_b = a.mean(), b.mean()
    var_a, var_b = a.var(unbiased=False), b.var(unbiased=False)
    cov = ((a - mu_a) * (b - mu_b)).mean()
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return float(((2 * mu_a * mu_b + c1) * (2 * cov + c2)) /
                 ((mu_a ** 2 + mu_b ** 2 + c1) * (var_a + var_b + c2)))


@torch.no_grad()
def _interpolate(net, i0: torch.Tensor, i1: torch.Tensor, t: float, scale: float) -> torch.Tensor:
    """Frame at time t ∈ (0,1) between i0 and i1 (both [1,3,H,W] padded, model dtype)."""
    scale_list = [16 / scale, 8 / scale, 4 / scale, 2 / scale, 1 / scale]
    # IFNet does `timestep = (x[:, :1] * 0 + 1) * t` for a python float, which is
    # fine; with fp16 weights some builds still up-cast — pass an explicit tensor.
    ts = torch.full((1, 1, 1, 1), t, device=i0.device, dtype=i0.dtype)
    _flow, _mask, merged = net(torch.cat((i0, i1), 1), ts, scale_list)
    return merged[-1]


def plan_multiplier(src_fps: float, target_fps: float) -> int:
    """Integer frame multiplier to reach ≈ target_fps (1 = no interpolation).

    Tolerates NTSC-style rates: 29.97 → 60 gives ×2 (59.94 fps), not ×3.
    Otherwise rounds *up* so the result is never below the requested rate.
    """
    ratio = target_fps / src_fps
    if ratio <= 1.02:
        return 1
    nearest = round(ratio)
    if nearest >= 2 and abs(ratio - nearest) / nearest <= 0.02:
        return int(nearest)
    return int(np.ceil(ratio))


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def interpolate_video(
    input_path: str,
    output_path: str,
    target_fps: float,
    progress_cb=None,
) -> dict:
    """Raise the frame rate of ``input_path`` to ≥ ``target_fps`` → ``output_path``.

    The output fps is ``src_fps * multi`` where ``multi = ceil(target/src)``;
    e.g. 25 → 60 gives 75 fps (multi=3), 30 → 60 gives 60 fps (multi=2). This
    keeps every original frame and inserts ``multi-1`` evenly spaced frames
    between each pair — cleaner than resampling to a non-integer ratio.

    Returns ``{"src_fps", "out_fps", "multi", "frames_in", "frames_out"}``.
    """
    probe = ffmpeg.probe(input_path)
    vs = next(s for s in probe["streams"] if s["codec_type"] == "video")
    w, h = int(vs["width"]), int(vs["height"])
    num, den = vs.get("r_frame_rate", "30/1").split("/")
    src_fps = int(num) / max(int(den), 1)

    multi = plan_multiplier(src_fps, target_fps)
    out_fps = src_fps * multi
    if multi == 1:
        # Nothing to do — just copy through.
        ffmpeg.input(input_path).output(output_path, c="copy").overwrite_output().run(quiet=True)
        n = int(vs.get("nb_frames", 0) or 0)
        return {"src_fps": src_fps, "out_fps": src_fps, "multi": 1, "frames_in": n, "frames_out": n}

    net = _load_model()
    dtype = torch.float16 if USE_FP16 else torch.float32

    # Flow-estimation scale (README: 0.5 for 4K-class inputs)
    scale = 0.5 if (w * h) / 1e6 >= FLOW_SCALE_THRESHOLD_MP else 1.0
    tmp = max(128, int(128 / scale))
    ph = ((h - 1) // tmp + 1) * tmp
    pw = ((w - 1) // tmp + 1) * tmp
    pad = (0, pw - w, 0, ph - h)

    frame_bytes = w * h * 3
    reader = (
        ffmpeg.input(input_path)
        .output("pipe:", format="rawvideo", pix_fmt="rgb24")
        .run_async(pipe_stdout=True, quiet=True)
    )
    writer = (
        ffmpeg.input("pipe:", format="rawvideo", pix_fmt="rgb24", s=f"{w}x{h}", r=out_fps)
        .output(output_path, vcodec="libx264", pix_fmt="yuv420p", crf=18, preset="fast")
        .overwrite_output()
        .run_async(pipe_stdin=True, quiet=True)
    )

    def to_tensor(raw: bytes) -> torch.Tensor:
        arr = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
        t = torch.from_numpy(arr.copy()).to(DEVICE).permute(2, 0, 1).unsqueeze(0).float().div_(255.0)
        return F.pad(t, pad).to(dtype)

    def write(t: torch.Tensor) -> None:
        img = t[0, :, :h, :w].float().clamp_(0, 1).mul_(255).round_().to(torch.uint8)
        writer.stdin.write(img.permute(1, 2, 0).cpu().numpy().tobytes())

    total_in = int(vs.get("nb_frames", 0) or 0) or max(1, int(float(probe["format"]["duration"]) * src_fps))
    frames_in = frames_out = 0
    timesteps = [(i + 1) / multi for i in range(multi - 1)]

    raw = reader.stdout.read(frame_bytes)
    if not raw:
        reader.stdout.close(); reader.wait(); writer.stdin.close(); writer.wait()
        return {"src_fps": src_fps, "out_fps": out_fps, "multi": multi, "frames_in": 0, "frames_out": 0}

    prev = to_tensor(raw)
    frames_in += 1
    try:
        while True:
            raw = reader.stdout.read(frame_bytes)
            if not raw:
                break
            cur = to_tensor(raw)
            frames_in += 1

            write(prev); frames_out += 1
            if _ssim_small(prev, cur) < SCENE_CUT_SSIM:
                # Hard cut: hold the last frame instead of blending two shots
                for _ in timesteps:
                    write(prev); frames_out += 1
            else:
                for t in timesteps:
                    write(_interpolate(net, prev, cur, t, scale)); frames_out += 1
            prev = cur

            if progress_cb and total_in:
                progress_cb(min(frames_in / total_in, 1.0))

        # Last original frame, padded with copies to keep duration == input
        write(prev); frames_out += 1
        for _ in timesteps:
            write(prev); frames_out += 1
    finally:
        reader.stdout.close(); reader.wait()
        writer.stdin.close(); writer.wait()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    return {"src_fps": src_fps, "out_fps": out_fps, "multi": multi,
            "frames_in": frames_in, "frames_out": frames_out}
