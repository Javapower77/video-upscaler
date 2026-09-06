import gc
import os
import shutil
import tempfile

# Must be set before torch initialises CUDA — reduces allocator fragmentation.
os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gradio as gr
import torch

from utils import validate_video, extract_audio, remux, make_tmp_dir
from video_upscaler import upscale_video as _upscale_video
from seedvr2_upscaler import upscale_video_seedvr2 as _upscale_video_seedvr2
from frame_interpolator import interpolate_video as _interpolate_video, plan_multiplier
from face_restorer import restore_faces_video as _restore_faces_video
from audio_enhancer import enhance_audio as _enhance_audio

# UI label → internal model key
VIDEO_MODELS = {
    "animevideov3 (fast / animation)": "animevideov3",
    "x4plus (realistic / photo)": "x4plus",
    "SeedVR2 (high quality / slow)": "seedvr2",
}

# Frame-rate options (RIFE v4.26). "Off" keeps the source frame rate.
FPS_OPTIONS = {
    "Off (keep source)": None,
    "2× source": "2x",
    "4× source": "4x",
    "60 fps": 60.0,
    "120 fps": 120.0,
}

# Face restoration options. "Off" leaves faces as the upscaler produced them.
FACE_OPTIONS = {
    "Off": None,
    "CodeFormer (best for low-quality faces)": "codeformer",
    "GFPGAN v1.4 (conservative / fast)": "gfpgan",
}


# ---------------------------------------------------------------------------
# GPU functions (run directly on the local H100 — no ZeroGPU allocation)
# ---------------------------------------------------------------------------

def _free_gpu():
    """Return cached allocator blocks to the driver (models stay loaded)."""
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def gpu_upscale_video(input_path, output_path, model_name, scale, info, progress_cb=None):
    if model_name == "seedvr2":
        _upscale_video_seedvr2(input_path, output_path, scale, info, progress_cb=progress_cb)
    else:
        _upscale_video(input_path, output_path, model_name, scale, info, progress_cb=progress_cb)


def gpu_interpolate_video(input_path, output_path, target_fps, progress_cb=None):
    return _interpolate_video(input_path, output_path, target_fps, progress_cb=progress_cb)


def gpu_restore_faces(input_path, output_path, model_name, fidelity, progress_cb=None):
    return _restore_faces_video(input_path, output_path, model_name=model_name,
                                fidelity=fidelity, progress_cb=progress_cb)


def gpu_enhance_audio(input_path, output_path, model_name):
    _enhance_audio(input_path, output_path, model_name)


def _resolve_target_fps(fps_choice: str, src_fps: float):
    """Turn a UI choice into an absolute target fps (or None for off)."""
    v = FPS_OPTIONS.get(fps_choice)
    if v is None:
        return None
    if isinstance(v, str) and v.endswith("x"):
        return src_fps * int(v[:-1])
    return float(v)


# ---------------------------------------------------------------------------
# Orchestrator — upscale → restore faces → interpolate → audio → remux
# ---------------------------------------------------------------------------

def _allocate_spans(stages: list[str], start: float = 0.05, end: float = 0.95) -> dict:
    """Split the progress bar between enabled stages, weighted by typical cost."""
    weights = {"upscale": 5.0, "faces": 3.0, "fps": 1.5, "audio": 3.0}
    total = sum(weights[s] for s in stages) or 1.0
    spans, cur = {}, start
    for s in stages:
        nxt = cur + (end - start) * weights[s] / total
        spans[s] = (cur, nxt)
        cur = nxt
    return spans


def process(
    video_file,
    do_video: bool,
    video_model: str,
    scale_str: str,
    face_choice: str,
    face_fidelity: float,
    fps_choice: str,
    do_audio: bool,
    audio_model: str,
    progress=gr.Progress(track_tqdm=True),
):
    if video_file is None:
        raise gr.Error("Please upload a video.")
    do_fps = FPS_OPTIONS.get(fps_choice) is not None
    face_key = FACE_OPTIONS.get(face_choice)
    do_faces = face_key is not None
    if not (do_video or do_audio or do_fps or do_faces):
        raise gr.Error("Enable at least one of: video upscaling, face restoration, "
                       "frame-rate upscaling or audio enhancement.")

    tmp = make_tmp_dir()
    try:
        progress(0, desc="Validating…")
        try:
            info = validate_video(video_file)
        except ValueError as e:
            raise gr.Error(str(e))

        scale = int(scale_str.replace("x", ""))
        model_key = VIDEO_MODELS.get(video_model, "x4plus")

        upscaled_video = os.path.join(tmp, "upscaled.mp4")
        faces_video = os.path.join(tmp, "faces.mp4")
        interpolated_video = os.path.join(tmp, "interpolated.mp4")
        enhanced_audio = os.path.join(tmp, "enhanced.wav")
        original_audio = os.path.join(tmp, "original.wav")
        output_path = os.path.join(tmp, "output.mp4")

        # Progress-bar spans for the enabled stages
        stages = [s for s, on in (("upscale", do_video), ("faces", do_faces),
                                  ("fps", do_fps), ("audio", do_audio)) if on]
        spans = _allocate_spans(stages)
        audio_start = spans["audio"][0] if do_audio else 0.95

        # --- Video upscaling ---
        if do_video:
            desc = ("Upscaling video with SeedVR2 (diffusion — this takes a few minutes)…"
                    if model_key == "seedvr2" else "Upscaling video…")
            lo, hi = spans["upscale"]
            progress(lo, desc=desc)
            try:
                gpu_upscale_video(
                    video_file, upscaled_video, model_key, scale, info,
                    progress_cb=lambda p: progress(lo + (hi - lo) * p, desc=desc),
                )
            except torch.OutOfMemoryError:
                _free_gpu()
                raise gr.Error(
                    f"Out of GPU memory upscaling {info['width']}x{info['height']} at {scale}x. "
                    "Try a smaller scale factor or a shorter / lower-resolution clip."
                )
            except RuntimeError as e:
                _free_gpu()
                raise gr.Error(str(e))
            progress(hi, desc="Video upscaled.")
        else:
            upscaled_video = video_file  # pass through

        # --- Face restoration (CodeFormer / GFPGAN) ---
        video_for_remux = upscaled_video
        if do_faces:
            lo, hi = spans["faces"]
            label = "CodeFormer" if face_key == "codeformer" else "GFPGAN"
            desc = f"Restoring faces with {label}…"
            progress(lo, desc=desc)
            try:
                stats = gpu_restore_faces(
                    upscaled_video, faces_video, face_key, float(face_fidelity),
                    progress_cb=lambda p: progress(lo + (hi - lo) * p, desc=desc),
                )
            except torch.OutOfMemoryError:
                _free_gpu()
                raise gr.Error("Out of GPU memory during face restoration.")
            if stats["faces_restored"] > 0:
                video_for_remux = faces_video
                progress(hi, desc=f"Restored {stats['faces_restored']} faces in "
                                  f"{stats['frames_with_faces']}/{stats['frames']} frames.")
            else:
                progress(hi, desc="No faces detected — skipping face restoration.")

        # --- Frame-rate upscaling (RIFE) ---
        if do_fps:
            target_fps = _resolve_target_fps(fps_choice, info["fps"])
            multi = plan_multiplier(info["fps"], target_fps)
            lo, hi = spans["fps"]
            if multi > 1:
                desc = f"Interpolating frames with RIFE ({info['fps']:.2f} → {info['fps'] * multi:.2f} fps)…"
                progress(lo, desc=desc)
                try:
                    gpu_interpolate_video(
                        video_for_remux, interpolated_video, target_fps,
                        progress_cb=lambda p: progress(lo + (hi - lo) * p, desc=desc),
                    )
                except torch.OutOfMemoryError:
                    _free_gpu()
                    raise gr.Error("Out of GPU memory during frame interpolation. Try a lower scale factor.")
                video_for_remux = interpolated_video
                progress(hi, desc="Frame rate upscaled.")
            else:
                progress(hi, desc=f"Source is already ≥ {target_fps:.0f} fps — skipping interpolation.")

        # --- Audio enhancement ---
        audio_for_remux = None
        if do_audio:
            progress(audio_start, desc="Extracting audio…")
            has_audio = extract_audio(video_file, original_audio)
            if has_audio:
                progress(audio_start + 0.03, desc="Enhancing audio…")
                gpu_enhance_audio(original_audio, enhanced_audio, audio_model)
                audio_for_remux = enhanced_audio
                progress(0.95, desc="Audio enhanced.")
            else:
                progress(0.95, desc="No audio track found — skipping audio enhancement.")
        else:
            # Keep original audio
            has_audio = extract_audio(video_file, original_audio)
            if has_audio:
                audio_for_remux = original_audio

        # --- Remux ---
        progress(0.97, desc="Muxing output…")
        remux(video_for_remux, audio_for_remux, output_path)
        progress(1.0, desc="Done!")

        # Copy to a stable temp path Gradio can serve
        final = tempfile.NamedTemporaryFile(suffix=".mp4", delete=False)
        shutil.copy2(output_path, final.name)
        return final.name

    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        _free_gpu()


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Video Upscaler") as demo:
    gr.Markdown(
        "## Video Upscaler — Image + Faces + Frame Rate + Audio\n"
        "Upscale video resolution (Real-ESRGAN / SeedVR2), restore faces (CodeFormer / GFPGAN), "
        "raise the frame rate (RIFE) and enhance audio (AudioSR).  \n"
        "**Limits:** max 2 minutes · max 300 MB"
    )

    with gr.Row():
        video_in = gr.Video(label="Input video", sources=["upload"])
        video_out = gr.Video(label="Enhanced output", interactive=False)

    with gr.Row():
        with gr.Column():
            do_video = gr.Checkbox(label="Upscale video", value=True)
            video_model = gr.Radio(
                list(VIDEO_MODELS.keys()),
                value="animevideov3 (fast / animation)",
                label="Video model",
            )
            scale = gr.Radio(["2x", "4x"], value="2x", label="Scale factor")
            gr.Markdown(
                "_animevideov3 / x4plus_ — Real-ESRGAN, per-frame, near real-time  \n"
                "_SeedVR2_ — one-step diffusion video restoration (ByteDance). Temporally "
                "consistent and far sharper on real-world / compressed footage, but "
                "~1–4 fps on the H100 (≈1–4 min per 10 s clip). First run downloads ~6.5 GB "
                "of weights.  \n"
                "_Tip: 4x is best for clips under 30 s._"
            )

        with gr.Column():
            face_choice = gr.Radio(
                list(FACE_OPTIONS.keys()),
                value="Off",
                label="Face restoration",
            )
            face_fidelity = gr.Slider(
                0.0, 1.0, value=0.7, step=0.05,
                label="CodeFormer fidelity",
                info="0 = maximum detail (may alter identity) · 1 = maximum fidelity to input. "
                     "Ignored for GFPGAN.",
            )
            gr.Markdown(
                "Detects faces (RetinaFace), restores each 512-px crop and blends it back "
                "with a parsing mask — background pixels are untouched. Runs after "
                "upscaling. Skin texture is temporally smoothed to reduce flicker.  \n"
                "_CodeFormer_ — best on heavily degraded / low-res faces (non-commercial licence).  \n"
                "_GFPGAN v1.4_ — more conservative, keeps identity, Apache-2.0."
            )

        with gr.Column():
            fps_choice = gr.Radio(
                list(FPS_OPTIONS.keys()),
                value="Off (keep source)",
                label="Frame rate (RIFE v4.26)",
            )
            gr.Markdown(
                "Inserts motion-interpolated frames between originals — smoother motion, "
                "same duration. Runs after upscaling. NTSC rates are handled "
                "(29.97 → 60 gives 59.94 fps). Hard scene cuts are detected and not "
                "blended. Near real-time on the H100."
            )

        with gr.Column():
            do_audio = gr.Checkbox(label="Enhance audio", value=True)
            audio_model = gr.Radio(
                ["audiosr_basic", "audiosr_speech"],
                value="audiosr_basic",
                label="Audio model",
            )
            gr.Markdown(
                "_audiosr\\_basic_ — all content (speech + music)  \n"
                "_audiosr\\_speech_ — optimised for voice"
            )

    run_btn = gr.Button("Upscale", variant="primary")

    _OFF, _CF, _GFP = list(FACE_OPTIONS.keys())
    gr.Examples(
        examples=[
            ["examples/animation.mp4", True,  "animevideov3 (fast / animation)", "4x", _OFF, 0.7, "Off (keep source)", True,  "audiosr_basic"],
            ["examples/realistic.mp4", True,  "x4plus (realistic / photo)",      "2x", _OFF, 0.7, "60 fps",            True,  "audiosr_basic"],
            ["examples/faces.mp4",     True,  "x4plus (realistic / photo)",      "2x", _CF,  0.7, "Off (keep source)", False, "audiosr_basic"],
            ["examples/faces.mp4",     False, "x4plus (realistic / photo)",      "2x", _GFP, 0.7, "Off (keep source)", False, "audiosr_basic"],
            ["examples/realistic.mp4", True,  "SeedVR2 (high quality / slow)",   "2x", _OFF, 0.7, "Off (keep source)", False, "audiosr_basic"],
            ["examples/animation.mp4", False, "animevideov3 (fast / animation)", "2x", _OFF, 0.7, "2× source",         False, "audiosr_basic"],
            ["examples/animation.mp4", False, "animevideov3 (fast / animation)", "2x", _OFF, 0.7, "Off (keep source)", True,  "audiosr_speech"],
        ],
        inputs=[video_in, do_video, video_model, scale, face_choice, face_fidelity, fps_choice, do_audio, audio_model],
        label="Examples",
    )
    run_btn.click(
        process,
        inputs=[video_in, do_video, video_model, scale, face_choice, face_fidelity, fps_choice, do_audio, audio_model],
        outputs=video_out,
    )

if __name__ == "__main__":
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7860")),
        ssr_mode=False,
    )
