"""Blind face restoration for video — CodeFormer and GFPGAN v1.4.

Pipeline position: runs *after* spatial upscaling (so faces are restored at the
final resolution) and *before* RIFE frame interpolation.

Per frame:
  1. Detect faces (RetinaFace-ResNet50 via facexlib) and align each to the
     512×512 FFHQ template.
  2. Restore each crop with CodeFormer (adjustable fidelity) or GFPGAN.
  3. Paste back using a ParseNet soft mask so only skin/hair/eyes/etc. change;
     background and non-face pixels are untouched.

Temporal stability (both models are per-image):
  (a) skip tiny / spurious faces via ``eye_dist_threshold``;
  (b) greedy identity tracking + EMA of 5-point landmarks *before* warp, so the
      affine does not swim;
  (c) motion-adaptive EMA of restored 512 crops (always on, weight decays with
      landmark motion — not a hard 6 px cutoff);
  (d) mix a fraction of the original aligned crop back in (GFPGAN has no
      fidelity slider; CodeFormer already has ``w``);
  (e) hold the last restored face for a few missed detections instead of
      snapping back to the unrestored frame.
"""
from __future__ import annotations

import math
import os
import sys
from dataclasses import dataclass

import cv2
import ffmpeg
import numpy as np
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
MIN_EYE_DIST = float(os.environ.get("FACE_MIN_EYE_DIST", "8"))
# Temporal EMA on aligned crops: *maximum* weight of the previous restored crop
# when the face is still. Weight decays with motion; 0 disables crop EMA.
TEMPORAL_EMA = float(os.environ.get("FACE_TEMPORAL_EMA", "0.50"))
# Landmark EMA *before* affine warp. Higher = more stable crop, more lag.
LANDMARK_EMA = float(os.environ.get("FACE_LANDMARK_EMA", "0.50"))
# Characteristic motion (as a fraction of inter-ocular distance) at which crop
# EMA falls to ~37% of TEMPORAL_EMA. Larger = keep blending through speech.
EMA_MOTION_TAU = float(os.environ.get("FACE_EMA_MOTION_TAU", "0.40"))
# Keep pasting the last restored face for this many consecutive missed detections.
HOLD_FRAMES = int(os.environ.get("FACE_HOLD_FRAMES", "3"))
# Mix original aligned crop back into the restored crop (GFPGAN has no ``w``).
# CodeFormer already has a fidelity slider, so its default is lower.
BLEND_ORIGINAL = os.environ.get("FACE_BLEND_ORIGINAL")  # None → model default
# Match gate as a multiple of inter-ocular distance (mean landmark L2).
MATCH_GATE = float(os.environ.get("FACE_MATCH_GATE", "1.6"))
# Looser gate used only when a leftover 1-to-1 pair remains after the first pass.
MATCH_GATE_LOOSE = float(os.environ.get("FACE_MATCH_GATE_LOOSE", "3.0"))


@dataclass
class _Track:
    landmarks: np.ndarray   # (5, 2) smoothed, source-frame pixels
    restored: np.ndarray    # 512×512 BGR uint8
    affine: np.ndarray      # 2×3 warp from crop → aligned 512
    missed: int = 0


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


def _blend_u8(a: np.ndarray, b: np.ndarray, w_b: float) -> np.ndarray:
    """``out = (1-w_b)*a + w_b*b``, both BGR uint8."""
    w_b = float(np.clip(w_b, 0.0, 1.0))
    if w_b <= 1e-4:
        return a
    if w_b >= 1.0 - 1e-4:
        return b
    return cv2.addWeighted(a, 1.0 - w_b, b, w_b, 0)


def _eye_dist(lm: np.ndarray) -> float:
    return float(np.linalg.norm(lm[0] - lm[1]))


def _lm_cost(a: np.ndarray, b: np.ndarray) -> float:
    return float(np.linalg.norm(a - b, axis=1).mean())


def _match_tracks(detections: list[np.ndarray], tracks: list[_Track]) -> tuple[dict[int, int], set[int], set[int]]:
    """Greedy unique matching by mean 5-point landmark L2, gated by eye distance.

    Returns ``(det_idx → track_idx, unmatched_dets, unmatched_tracks)``.
    """
    n_d, n_t = len(detections), len(tracks)
    if n_d == 0 or n_t == 0:
        return {}, set(range(n_d)), set(range(n_t))

    costs = np.empty((n_d, n_t), dtype=np.float64)
    for i, lm in enumerate(detections):
        for j, tr in enumerate(tracks):
            costs[i, j] = _lm_cost(lm, tr.landmarks)

    used_d: set[int] = set()
    used_t: set[int] = set()
    matches: dict[int, int] = {}

    order = np.argsort(costs, axis=None)
    for k in order:
        i, j = divmod(int(k), n_t)
        if i in used_d or j in used_t:
            continue
        eye = max(_eye_dist(detections[i]), _eye_dist(tracks[j].landmarks), 1.0)
        if costs[i, j] <= MATCH_GATE * eye:
            matches[i] = j
            used_d.add(i)
            used_t.add(j)

    leftover_d = [i for i in range(n_d) if i not in used_d]
    leftover_t = [j for j in range(n_t) if j not in used_t]
    # Talking-head / camera cut: a single leftover pair is almost always the same face.
    if len(leftover_d) == 1 and len(leftover_t) == 1:
        i, j = leftover_d[0], leftover_t[0]
        eye = max(_eye_dist(detections[i]), _eye_dist(tracks[j].landmarks), 1.0)
        if costs[i, j] <= MATCH_GATE_LOOSE * eye:
            matches[i] = j
            used_d.add(i)
            used_t.add(j)

    return matches, set(range(n_d)) - used_d, set(range(n_t)) - used_t


def _smooth_landmarks(detected: np.ndarray, previous: np.ndarray) -> np.ndarray:
    if LANDMARK_EMA <= 0:
        return detected.astype(np.float64, copy=True)
    a = float(np.clip(LANDMARK_EMA, 0.0, 0.95))
    return a * previous + (1.0 - a) * detected


def _crop_ema_weight(detected: np.ndarray, previous: np.ndarray) -> float:
    """Weight of the *previous* restored crop. Decays with landmark motion."""
    if TEMPORAL_EMA <= 0:
        return 0.0
    eye = max(_eye_dist(detected), _eye_dist(previous), 1.0)
    shift_norm = _lm_cost(detected, previous) / eye
    tau = max(EMA_MOTION_TAU, 1e-3)
    return float(TEMPORAL_EMA) * math.exp(-shift_norm / tau)


def _original_blend_weight(model_name: str) -> float:
    if BLEND_ORIGINAL is not None:
        return float(np.clip(float(BLEND_ORIGINAL), 0.0, 1.0))
    # GFPGAN has no fidelity mix; lock a bit of the source crop so pores/teeth
    # don't shimmer independently every frame. CodeFormer already has ``w``.
    return 0.22 if model_name == "gfpgan" else 0.08


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
    orig_w = _original_blend_weight(model_name)

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

    tracks: list[_Track] = []

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
            helper.get_face_landmarks_5(
                only_center_face=False, resize=640, eye_dist_threshold=MIN_EYE_DIST)

            detected = [np.asarray(lm, dtype=np.float64) for lm in helper.all_landmarks_5]
            matches, _unmatched_d, unmatched_t = _match_tracks(detected, tracks)

            # Smooth landmarks of matched faces *before* the affine warp so the
            # 512 crop does not swim independently of the restorer.
            if detected:
                helper.all_landmarks_5 = [
                    _smooth_landmarks(lm, tracks[matches[i]].landmarks) if i in matches else lm
                    for i, lm in enumerate(detected)
                ]
                helper.align_warp_face()

            next_tracks: list[_Track] = []

            for i, (lm_raw, crop) in enumerate(zip(detected, helper.cropped_faces)):
                restored = _restore_crop(net, model_name, crop, fidelity)
                if orig_w > 0:
                    restored = _blend_u8(restored, crop, orig_w)

                if i in matches:
                    tr = tracks[matches[i]]
                    w_prev = _crop_ema_weight(lm_raw, tr.landmarks)
                    if w_prev > 0.02:
                        restored = _blend_u8(restored, tr.restored, w_prev)

                next_tracks.append(_Track(
                    landmarks=np.asarray(helper.all_landmarks_5[i], dtype=np.float64),
                    restored=restored,
                    affine=helper.affine_matrices[i].copy(),
                    missed=0,
                ))
                helper.add_restored_face(restored)
                n_faces += 1

            # Detection hysteresis: keep pasting a face that flickered off for a
            # couple of frames instead of exposing the unrestored crop.
            for j in unmatched_t:
                tr = tracks[j]
                tr.missed += 1
                if tr.missed <= HOLD_FRAMES:
                    helper.affine_matrices.append(tr.affine.copy())
                    helper.add_restored_face(tr.restored)
                    next_tracks.append(tr)

            tracks = next_tracks

            if helper.restored_faces:
                helper.get_inverse_affine()
                frame = helper.paste_faces_to_input_image(upsample_img=frame)
                n_with_faces += 1

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
