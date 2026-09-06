"""Real-ESRGAN video upscaling via spandrel + FFmpeg frame pipe.

Optimised for a single NVIDIA H100 (80 GB): fp16 inference, TF32 matmuls,
and a large tile size so most frames are processed in a single pass.
"""
import os
import urllib.request
import numpy as np
import cv2
import ffmpeg
import torch

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
USE_FP16 = DEVICE.type == "cuda"

# Enable TF32 on Hopper for faster fp32 fallback paths
if DEVICE.type == "cuda":
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    torch.backends.cudnn.benchmark = True

# With 80 GB of VRAM, tiles can be large — most 1080p/4K frames fit in 1–4 tiles
TILE_SIZE = 1024
TILE_THRESHOLD = 2048  # only tile frames larger than this

WEIGHTS = {
    "animevideov3": (
        "realesr-animevideov3.pth",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.2.5.0/realesr-animevideov3.pth",
    ),
    "x4plus": (
        "RealESRGAN_x4plus.pth",
        "https://github.com/xinntao/Real-ESRGAN/releases/download/v0.1.0/RealESRGAN_x4plus.pth",
    ),
}

_model_cache: dict = {}


def _load_model(model_name: str):
    if model_name in _model_cache:
        return _model_cache[model_name]

    from spandrel import ModelLoader

    fname, url = WEIGHTS[model_name]
    if not os.path.exists(fname):
        urllib.request.urlretrieve(url, fname)

    descriptor = ModelLoader().load_from_file(fname)
    module = descriptor.model.eval().to(DEVICE)
    if USE_FP16 and descriptor.supports_half:
        module = module.half()

    _model_cache[model_name] = (module, descriptor.scale)
    return module, descriptor.scale


def _tile_upscale(module, tensor: torch.Tensor, model_scale: int, tile: int = TILE_SIZE, pad: int = 10) -> torch.Tensor:
    """Process large frame in overlapping tiles to stay within VRAM."""
    b, c, h, w = tensor.shape
    out_h, out_w = h * model_scale, w * model_scale
    output = torch.zeros(b, c, out_h, out_w, device=tensor.device, dtype=tensor.dtype)

    for y0 in range(0, h, tile - pad * 2):
        for x0 in range(0, w, tile - pad * 2):
            y1 = max(0, y0 - pad);  y2 = min(h, y0 + tile + pad)
            x1 = max(0, x0 - pad);  x2 = min(w, x0 + tile + pad)

            with torch.no_grad():
                t_out = module(tensor[:, :, y1:y2, x1:x2])

            oy = (y0 - y1) * model_scale
            ox = (x0 - x1) * model_scale
            oy2 = oy + min(tile, h - y0) * model_scale
            ox2 = ox + min(tile, w - x0) * model_scale
            dest_y = y0 * model_scale
            dest_x = x0 * model_scale
            output[:, :, dest_y:dest_y + (oy2 - oy), dest_x:dest_x + (ox2 - ox)] = t_out[:, :, oy:oy2, ox:ox2]

    return output


def _enhance_frame(module, model_scale: int, frame_bgr: np.ndarray, outscale: int) -> np.ndarray:
    h, w = frame_bgr.shape[:2]
    # spandrel expects RGB [0,1] NCHW
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    t = torch.from_numpy(rgb).permute(2, 0, 1).float().div(255.0).unsqueeze(0).to(DEVICE)
    model_dtype = next(module.parameters()).dtype
    t = t.to(model_dtype)

    if max(h, w) > TILE_THRESHOLD:
        out = _tile_upscale(module, t, model_scale)
    else:
        with torch.no_grad():
            out = module(t)

    # If model is 4x but user wants 2x, downsample output
    if outscale != model_scale:
        out_h = int(h * outscale)
        out_w = int(w * outscale)
        out = torch.nn.functional.interpolate(
            out.float(), size=(out_h, out_w), mode="bicubic", align_corners=False
        )

    out_np = out.squeeze(0).permute(1, 2, 0).float().clamp(0, 1).mul(255).byte().cpu().numpy()
    return cv2.cvtColor(out_np, cv2.COLOR_RGB2BGR)


def upscale_video(
    input_path: str,
    output_path: str,
    model_name: str,
    scale: int,
    info: dict,
    progress_cb=None,
) -> None:
    module, model_scale = _load_model(model_name)

    w, h = info["width"], info["height"]
    fps = info["fps"]
    out_w, out_h = w * scale, h * scale
    frame_bytes = w * h * 3

    reader = (
        ffmpeg
        .input(input_path)
        .output("pipe:", format="rawvideo", pix_fmt="bgr24")
        .run_async(pipe_stdout=True, quiet=True)
    )
    writer = (
        ffmpeg
        .input("pipe:", format="rawvideo", pix_fmt="bgr24", s=f"{out_w}x{out_h}", r=fps)
        .output(output_path, vcodec="libx264", pix_fmt="yuv420p", crf=18, preset="fast")
        .overwrite_output()
        .run_async(pipe_stdin=True, quiet=True)
    )

    total = max(1, int(info["duration"] * fps))
    done = 0

    while True:
        raw = reader.stdout.read(frame_bytes)
        if not raw:
            break
        frame = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3)
        out_frame = _enhance_frame(module, model_scale, frame, scale)
        writer.stdin.write(out_frame.tobytes())
        done += 1
        if progress_cb:
            progress_cb(done / total)

    reader.stdout.close();  reader.wait()
    writer.stdin.close();   writer.wait()
