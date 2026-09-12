"""Blind face restoration for video — CodeFormer and GFPGAN v1.4.

Pipeline position: runs *after* spatial upscaling (so faces are restored at the
final resolution) and *before* RIFE frame interpolation.

Per frame:
  1. Detect faces (RetinaFace-ResNet50 via facexlib) and align each to the
     512×512 FFHQ template.
  2. Restore each crop with CodeFormer (adjustable fidelity) or GFPGAN.
  3. Paste back using a ParseNet soft mask so only skin/hair/eyes/etc. change;
     background and non-face pixels are untouched.

Temporal stability: both models are per-image, so we reduce flicker by
(a) keeping detection stable via a small ``eye_dist_threshold`` (ignore tiny /
spurious faces), (b) blending each restored face crop with its restored
predecessor when the face barely moved (EMA on the aligned crop), and
(c) leaving frames without detected faces completely untouched.
"""
import os
import sys
import cv2
import numpy as np
import ffmpeg
import torch
from torchvision.transforms.functional import normalize

_HERE = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.environ.get("FACE_MODEL_DIR", os.path.join(_HERE, "models", "FACE"))
if _HERE not in sys.path:
    sys.path.insert(0, _HERE)

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

WEIGHTS = {
    "codeformer": ("codeformer.pth",
                   "https://github.com/sczhou/CodeFormer/releases/download/v0.1.0/codeformer.pth"),
    "gfpgan": ("GFPGANv1.4.pth",
               "https://github.com/TencentARC/GFPGAN/releases/download/v1.3.4/GFPGANv1.4.pth"),
    # facexlib helpers (detector + parser) — facexlib downloads these itself into
    # MODEL_DIR when missing, listed here for setup/documentation completeness.
    "_detector": ("detection_Resnet50_Final.pth",
                  "https://github.com/xinntao/facexlib/releases/download/v0.1.0/detection_Resnet50_Final.pth"),
    "_parser": ("parsing_parsenet.pth",
                "https://github.com/xinntao/facexlib/releases/download/v0.2.2/parsing_parsenet.pth"),
}

FACE_SIZE = 512
# Skip faces whose inter-ocular distance is below this (px in the source frame).
# Tiny faces restore badly and pop in/out between frames.
MIN_EYE_DIST = 5
# Temporal EMA on aligned crops: weight of the *previous* restored crop when the
# face has barely moved. 0 disables. Keeps skin texture from shimmering.
TEMPORAL_EMA = float(os.environ.get("FACE_TEMPORAL_EMA", "0.35"))
# A face is considered "the same, barely moved" if its aligned landmarks moved
# less than this many pixels (in the 512 template space).
EMA_MAX_SHIFT = 6.0

_models: dict = {}
_helper = None


# ---------------------------------------------------------------------------
# Loading
# ---------------------------------------------------------------------------

def _ensure_weight(key: str) -> str:
    fname, url = WEIGHTS[key]
    path = os.path.join(MODEL_DIR, fname)
    if not os.path.exists(path):
        os.makedirs(MODEL_DIR, exist_ok=True)
        import urllib.request
        urllib.request.urlretrieve(url, path)
    return path


def _get_helper():
    """facexlib FaceRestoreHelper: detection + alignment + parse-mask paste-back."""
    global _helper
    if _helper is None:
        from facexlib.utils.face_restoration_helper import FaceRestoreHelper
        _ensure_weight("_detector")
        _ensure_weight("_parser")
        _helper = FaceRestoreHelper(
            upscale_factor=1,          # we restore at the already-upscaled resolution
            face_size=FACE_SIZE,
            crop_ratio=(1, 1),
            det_model="retinaface_resnet50",
            use_parse=True,
            device=DEVICE,
            model_rootpath=MODEL_DIR,
        )
    return _helper


def _get_model(name: str):
    if name in _models:
        return _models[name]

    if name == "codeformer":
        from third_party.codeformer.codeformer_arch import CodeFormer
        net = CodeFormer(dim_embd=512, codebook_size=1024, n_head=8, n_layers=9,
                         connect_list=["32", "64", "128", "256"])
        sd = torch.load(_ensure_weight("codeformer"), map_location="cpu", weights_only=True)["params_ema"]
        net.load_state_dict(sd, strict=True)
        net = net.eval().to(DEVICE)
    elif name == "gfpgan":
        from spandrel import ModelLoader
        desc = ModelLoader().load_from_file(_ensure_weight("gfpgan"))
        net = desc.model.eval().to(DEVICE)
    else:
        raise ValueError(f"Unknown face model: {name}")

    _models[name] = net
    return net


# ---------------------------------------------------------------------------
# Per-crop restoration
# ---------------------------------------------------------------------------

@torch.no_grad()
def _restore_crop(net, name: str, face_bgr: np.ndarray, fidelity: float) -> np.ndarray:
    """512×512 BGR uint8 → restored 512×512 BGR uint8."""
    t = torch.from_numpy(cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)).permute(2, 0, 1).float().div_(255.0)
    normalize(t, (0.5, 0.5, 0.5), (0.5, 0.5, 0.5), inplace=True)
    t = t.unsqueeze(0).to(DEVICE)

    try:
        if name == "codeformer":
            out = net(t, w=fidelity, adain=True)[0]
        else:  # gfpgan — returns (image, rgb_list)
            out = net(t)[0]
    except Exception:
        # Model failure on a single crop must not kill the whole video; keep original
        return face_bgr

    out = out.squeeze(0).float().clamp_(-1, 1).add_(1).div_(2).mul_(255).round_().byte()
    out = out.permute(1, 2, 0).cpu().numpy()
    return cv2.cvtColor(out, cv2.COLOR_RGB2BGR)


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def restore_faces_video(
    input_path: str,
    output_path: str,
    model_name: str = "codeformer",     # "codeformer" | "gfpgan"
    fidelity: float = 0.7,              # CodeFormer only: 0 = max quality, 1 = max fidelity
    progress_cb=None,
) -> dict:
    """Restore faces in every frame of ``input_path`` → ``output_path`` (same size / fps).

    Returns ``{"frames", "frames_with_faces", "faces_restored"}``.
    """
    net = _get_model(model_name)
    helper = _get_helper()

    probe = ffmpeg.probe(input_path)
    vs = next(s for s in probe["streams"] if s["codec_type"] == "video")
    w, h = int(vs["width"]), int(vs["height"])
    fps_str = vs.get("r_frame_rate", "30/1")
    total = int(vs.get("nb_frames", 0) or 0) or max(1, int(float(probe["format"]["duration"]) * eval(fps_str)))
    frame_bytes = w * h * 3

    reader = (
        ffmpeg.input(input_path)
        .output("pipe:", format="rawvideo", pix_fmt="bgr24")
        .run_async(pipe_stdout=True, quiet=True)
    )
    writer = (
        ffmpeg.input("pipe:", format="rawvideo", pix_fmt="bgr24", s=f"{w}x{h}", r=fps_str)
        .output(output_path, vcodec="libx264", pix_fmt="yuv420p", crf=18, preset="fast")
        .overwrite_output()
        .run_async(pipe_stdin=True, quiet=True)
    )

    # Temporal state: per-face (landmarks, restored crop) from the previous frame
    prev_faces: list[tuple[np.ndarray, np.ndarray]] = []

    n = n_with_faces = n_faces = 0
    completed = False
    try:
        while True:
            raw = reader.stdout.read(frame_bytes)
            if not raw:
                break
            frame = np.frombuffer(raw, dtype=np.uint8).reshape(h, w, 3).copy()
            n += 1

            helper.clean_all()
            helper.read_image(frame)
            helper.get_face_landmarks_5(only_center_face=False, resize=640, eye_dist_threshold=MIN_EYE_DIST)

            if helper.all_landmarks_5:
                helper.align_warp_face()
                cur_faces = []
                for lm, crop in zip(helper.all_landmarks_5, helper.cropped_faces):
                    restored = _restore_crop(net, model_name, crop, fidelity)

                    # Temporal EMA with the closest previous face (if it barely moved)
                    if TEMPORAL_EMA > 0 and prev_faces:
                        dists = [np.abs(lm - plm).max() for plm, _ in prev_faces]
                        j = int(np.argmin(dists))
                        if dists[j] <= EMA_MAX_SHIFT:
                            restored = cv2.addWeighted(
                                restored, 1.0 - TEMPORAL_EMA, prev_faces[j][1], TEMPORAL_EMA, 0)

                    helper.add_restored_face(restored)
                    cur_faces.append((lm.copy(), restored))
                    n_faces += 1

                helper.get_inverse_affine()
                frame = helper.paste_faces_to_input_image(upsample_img=frame)
                prev_faces = cur_faces
                n_with_faces += 1
            else:
                prev_faces = []

            writer.stdin.write(np.ascontiguousarray(frame[:, :, :3]).tobytes())
            if progress_cb:
                progress_cb(min(n / total, 1.0))
        completed = True
    finally:
        reader.stdout.close()
        writer.stdin.close()
        if not completed:
            reader.terminate()
            writer.terminate()
        reader.wait()
        writer.wait()
        helper.clean_all()
        if DEVICE.type == "cuda":
            torch.cuda.empty_cache()

    return {"frames": n, "frames_with_faces": n_with_faces, "faces_restored": n_faces}
