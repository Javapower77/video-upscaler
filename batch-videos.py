"""Video Batch Processing — Gradio UI for upscaling / face restoration / frame-rate
upscaling of N videos in one run.  Reuses the single-video pipeline in app.py.

Run:  GRADIO_SERVER_PORT=7862 python batch-videos.py
"""
import gc
import os
import shutil
import tarfile
import tempfile
import threading
import time
import traceback
from dataclasses import dataclass, field

os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

import gradio as gr
import torch

from utils import validate_video, extract_audio, remux, make_tmp_dir
from frame_interpolator import plan_multiplier
from app import (
    VIDEO_MODELS, FPS_OPTIONS, FACE_OPTIONS,
    gpu_upscale_video, gpu_restore_faces, gpu_interpolate_video,
    _resolve_target_fps, _free_gpu,
)

OUTPUT_SUFFIX = "_slp"
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output", "batch")
OPERATIONS = ["Upscale", "Frame Rate", "Face Restoration"]
STATUS_ICON = {"pending": "⏳", "running": "🔄", "done": "✅", "error": "❌", "stopped": "⏹️"}


class StopRequested(Exception):
    """Raised from inside a progress callback when the user presses Stop."""


@dataclass
class Job:
    src: str
    name: str
    status: str = "pending"
    stage: str = ""
    pct: float = 0.0
    log: list = field(default_factory=list)
    output: str | None = None
    elapsed: float = 0.0


class BatchState:
    """Shared between the worker thread and the UI poller."""

    def __init__(self):
        self.lock = threading.Lock()
        self.jobs: list[Job] = []
        self.running = False
        self.stop_event = threading.Event()
        self.thread: threading.Thread | None = None
        self.settings: dict = {}
        self.tar_path: str | None = None

    def reset(self, files, settings):
        self.jobs = [Job(src=f, name=os.path.basename(f)) for f in files]
        self.settings = settings
        self.stop_event.clear()
        self.tar_path = None

    def finished_all(self) -> bool:
        return bool(self.jobs) and not self.running and all(j.status == "done" for j in self.jobs)


STATE = BatchState()


def _out_name(name: str) -> str:
    stem, _ = os.path.splitext(name)
    return f"{stem}{OUTPUT_SUFFIX}.mp4"


def _log(job: Job, msg: str):
    with STATE.lock:
        job.log.append(f"[{time.strftime('%H:%M:%S')}] {msg}")


def _set(job: Job, **kw):
    with STATE.lock:
        for k, v in kw.items():
            setattr(job, k, v)


def _check_stop():
    if STATE.stop_event.is_set():
        raise StopRequested()


def _progress_cb(job: Job, stage: str):
    def cb(p: float):
        _check_stop()
        _set(job, stage=stage, pct=float(p))
    return cb


def _process_one(job: Job, s: dict):
    """Same stage order as app.process(): upscale → faces → RIFE → remux."""
    ops = s["ops"]
    do_video = "Upscale" in ops
    do_faces = "Face Restoration" in ops
    do_fps = "Frame Rate" in ops
    tmp = make_tmp_dir()
    t0 = time.time()
    try:
        _set(job, status="running", stage="Validating", pct=0.0)
        _log(job, "Validating input…")
        info = validate_video(job.src)
        _log(job, f"Input: {info['width']}x{info['height']} @ {info['fps']:.2f} fps, "
                  f"{info['duration']:.1f}s")
        _check_stop()

        upscaled = os.path.join(tmp, "upscaled.mp4")
        faces = os.path.join(tmp, "faces.mp4")
        interp = os.path.join(tmp, "interpolated.mp4")
        orig_audio = os.path.join(tmp, "original.wav")
        final_tmp = os.path.join(tmp, "output.mp4")
        current = job.src

        if do_video:
            model_key = VIDEO_MODELS.get(s["video_model"], "x4plus")
            scale = int(s["scale"].replace("x", ""))
            stage = f"Upscaling {scale}x ({model_key})"
            _set(job, stage=stage, pct=0.0)
            _log(job, f"Upscale: model={model_key}, scale={scale}x → "
                      f"{info['width']*scale}x{info['height']*scale}")
            try:
                gpu_upscale_video(current, upscaled, model_key, scale, info,
                                  progress_cb=_progress_cb(job, stage))
            except torch.OutOfMemoryError:
                _free_gpu()
                raise RuntimeError(f"Out of GPU memory upscaling at {scale}x.")
            current = upscaled
            _log(job, "Upscale done.")

        if do_faces:
            face_key = FACE_OPTIONS.get(s["face_choice"])
            if face_key is None:
                _log(job, "Face Restoration selected but model is 'Off' — skipped.")
            else:
                label = "CodeFormer" if face_key == "codeformer" else "GFPGAN"
                stage = f"Restoring faces ({label})"
                _set(job, stage=stage, pct=0.0)
                _log(job, f"Face restoration: model={label}, fidelity={s['fidelity']:.2f}")
                try:
                    stats = gpu_restore_faces(current, faces, face_key, float(s["fidelity"]),
                                              progress_cb=_progress_cb(job, stage))
                except torch.OutOfMemoryError:
                    _free_gpu()
                    raise RuntimeError("Out of GPU memory during face restoration.")
                if stats["faces_restored"] > 0:
                    current = faces
                    _log(job, f"Restored {stats['faces_restored']} faces in "
                              f"{stats['frames_with_faces']}/{stats['frames']} frames.")
                else:
                    _log(job, "No faces detected — face restoration skipped.")

        if do_fps:
            target = _resolve_target_fps(s["fps_choice"], info["fps"])
            if target is None:
                _log(job, "Frame Rate selected but option is 'Off' — skipped.")
            else:
                multi = plan_multiplier(info["fps"], target)
                if multi > 1:
                    stage = f"RIFE {info['fps']:.2f} → {info['fps']*multi:.2f} fps"
                    _set(job, stage=stage, pct=0.0)
                    _log(job, f"Frame interpolation: x{multi} ({info['fps']:.2f} → "
                              f"{info['fps']*multi:.2f} fps)")
                    try:
                        gpu_interpolate_video(current, interp, target,
                                              progress_cb=_progress_cb(job, stage))
                    except torch.OutOfMemoryError:
                        _free_gpu()
                        raise RuntimeError("Out of GPU memory during frame interpolation.")
                    current = interp
                    _log(job, "Frame interpolation done.")
                else:
                    _log(job, f"Source already ≥ {target:.0f} fps — interpolation skipped.")

        _check_stop()
        _set(job, stage="Muxing audio", pct=0.99)
        audio = orig_audio if extract_audio(job.src, orig_audio) else None
        _log(job, "Audio track kept." if audio else "No audio track.")
        remux(current, audio, final_tmp)

        os.makedirs(OUTPUT_DIR, exist_ok=True)
        dest = os.path.join(OUTPUT_DIR, _out_name(job.name))
        shutil.copy2(final_tmp, dest)
        _set(job, status="done", stage="Done", pct=1.0, output=dest,
             elapsed=time.time() - t0)
        _log(job, f"Saved → {os.path.basename(dest)} ({time.time()-t0:.1f}s)")

    except StopRequested:
        _set(job, status="stopped", stage="Stopped by user", elapsed=time.time() - t0)
        _log(job, "Stopped by user.")
        raise
    except Exception as e:  # noqa: BLE001
        _set(job, status="error", stage=f"Error: {e}", elapsed=time.time() - t0)
        _log(job, f"ERROR: {e}")
        traceback.print_exc()
    finally:
        shutil.rmtree(tmp, ignore_errors=True)
        _free_gpu()


def _worker():
    try:
        for job in STATE.jobs:
            if STATE.stop_event.is_set():
                _set(job, status="stopped", stage="Stopped by user")
                continue
            _process_one(job, STATE.settings)
    except StopRequested:
        for j in STATE.jobs:
            if j.status in ("pending", "running"):
                _set(j, status="stopped", stage="Stopped by user")
    finally:
        with STATE.lock:
            STATE.running = False


# ---------------------------------------------------------------------------
# UI helpers
# ---------------------------------------------------------------------------

def _render_status() -> str:
    with STATE.lock:
        jobs = list(STATE.jobs)
    if not jobs:
        return "_No videos queued. Upload videos and press **Start Process**._"
    done = sum(j.status == "done" for j in jobs)
    lines = [f"### Batch progress: {done}/{len(jobs)} videos completed\n"]
    for i, j in enumerate(jobs, 1):
        bar = "█" * int(j.pct * 20) + "░" * (20 - int(j.pct * 20))
        lines.append(f"**{i}. {j.name}** {STATUS_ICON[j.status]} `{bar}` {j.pct*100:5.1f}%  "
                     f"— {j.stage or 'Waiting…'}")
        for entry in j.log[-6:]:
            lines.append(f"  - {entry}")
        lines.append("")
    return "\n".join(lines)


def _outputs() -> list[str]:
    with STATE.lock:
        return [j.output for j in STATE.jobs if j.output]


def _ui_snapshot():
    """Common return for start/stop/tick: status md, files, start-btn, download-btn, timer."""
    running = STATE.running
    finished = STATE.finished_all()
    return (
        _render_status(),
        _outputs(),
        gr.update(value="Stop Process" if running else "Start Process",
                  variant="stop" if running else "primary"),
        gr.update(interactive=finished),
        gr.Timer(active=running),
    )


def on_start_stop(files, ops, video_model, scale, face_choice, fidelity, fps_choice):
    if STATE.running:                          # acting as "Stop Process"
        STATE.stop_event.set()
        if STATE.thread:
            STATE.thread.join(timeout=120)
        return _ui_snapshot()

    if not files:
        raise gr.Error("Please upload at least one video before starting.")
    if not ops:
        raise gr.Error("Select at least one operation (Upscale, Frame Rate, Face Restoration).")
    if "Face Restoration" in ops and FACE_OPTIONS.get(face_choice) is None:
        raise gr.Error("Face Restoration is selected — choose CodeFormer or GFPGAN.")
    if "Frame Rate" in ops and FPS_OPTIONS.get(fps_choice) is None:
        raise gr.Error("Frame Rate is selected — choose a target frame rate.")

    paths = [f if isinstance(f, str) else f.name for f in files]
    bad = [os.path.basename(p) for p in paths if not os.path.isfile(p)]
    if bad:
        raise gr.Error(f"Missing files: {', '.join(bad)}")

    STATE.reset(paths, dict(ops=ops, video_model=video_model, scale=scale,
                            face_choice=face_choice, fidelity=fidelity, fps_choice=fps_choice))
    STATE.running = True
    STATE.thread = threading.Thread(target=_worker, daemon=True)
    STATE.thread.start()
    return _ui_snapshot()


def on_tick():
    return _ui_snapshot()


def on_download_all():
    outs = _outputs()
    if not outs:
        raise gr.Error("No processed videos to download.")
    path = os.path.join(tempfile.gettempdir(), f"batch_videos_{int(time.time())}.tar")
    with tarfile.open(path, "w") as tar:
        for p in outs:
            tar.add(p, arcname=os.path.basename(p))
    STATE.tar_path = path
    return gr.update(value=path)


# ---------------------------------------------------------------------------
# Gradio UI
# ---------------------------------------------------------------------------

with gr.Blocks(title="Video Batch Processing") as demo:
    gr.Markdown("## Video Batch Processing — Upscale · Frame Rate · Face Restoration\n"
                "Process many videos with one configuration. Output = `<name>_slp.mp4`.  \n"
                "**Limits per video:** max 2 minutes · max 300 MB")

    gr.Markdown("### 1. Upload videos")
    files_in = gr.File(label="Videos", file_count="multiple", file_types=["video"], type="filepath")

    gr.Markdown("### 2. Operations (applied to every video, in this order)")
    ops = gr.CheckboxGroup(OPERATIONS, value=["Upscale"], label="Operations")
    with gr.Row():
        with gr.Column():
            video_model = gr.Radio(list(VIDEO_MODELS.keys()),
                                   value="animevideov3 (fast / animation)", label="Upscale model")
            scale = gr.Radio(["2x", "4x"], value="2x", label="Scale factor")
        with gr.Column():
            fps_choice = gr.Radio(list(FPS_OPTIONS.keys()), value="2× source",
                                  label="Frame rate (RIFE v4.26)")
        with gr.Column():
            face_choice = gr.Radio(list(FACE_OPTIONS.keys()),
                                   value="CodeFormer (best for low-quality faces)",
                                   label="Face restoration model")
            fidelity = gr.Slider(0.0, 1.0, value=0.7, step=0.05, label="CodeFormer fidelity")

    with gr.Row():
        start_btn = gr.Button("Start Process", variant="primary")
        dl_btn = gr.DownloadButton("Download All (.tar)", interactive=False)

    gr.Markdown("### 3. Processing details")
    status_md = gr.Markdown(_render_status())

    gr.Markdown("### 4. Processed videos (click to download)")
    files_out = gr.File(label="Outputs", file_count="multiple", interactive=False)

    timer = gr.Timer(1.0, active=False)
    outs = [status_md, files_out, start_btn, dl_btn, timer]
    start_btn.click(on_start_stop,
                    inputs=[files_in, ops, video_model, scale, face_choice, fidelity, fps_choice],
                    outputs=outs)
    timer.tick(on_tick, outputs=outs)
    dl_btn.click(on_download_all, outputs=dl_btn)

if __name__ == "__main__":
    demo.launch(
        server_name=os.environ.get("GRADIO_SERVER_NAME", "0.0.0.0"),
        server_port=int(os.environ.get("GRADIO_SERVER_PORT", "7862")),
        ssr_mode=False,
    )
