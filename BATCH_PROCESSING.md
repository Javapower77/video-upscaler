# Video Batch Processing — Technical Documentation

`batch-videos.py` is a Gradio application that applies the same enhancement pipeline as
`app.py` to **N videos in one run**. It reuses the existing single-video code without
duplicating it.

## Run

```bash
source .venv/bin/activate
GRADIO_SERVER_PORT=7862 python batch-videos.py      # default port 7862 (app.py uses 7860/7861)
```

## UI layout

| Section | Component | Purpose |
|---|---|---|
| 1. Upload videos | `gr.File(file_count="multiple")` | Queue any number of videos (≤ 2 min, ≤ 300 MB each — enforced by `utils.validate_video`). |
| 2. Operations | `gr.CheckboxGroup` + per-operation settings | **Upscale** (Real-ESRGAN animevideov3 / x4plus / SeedVR2, 2x/4x), **Frame Rate** (RIFE v4.26: 2×, 4×, 60, 120 fps), **Face Restoration** (CodeFormer / GFPGAN + fidelity). One configuration is applied to every video. |
| 3. Processing details | `gr.Markdown`, refreshed every 1 s by `gr.Timer` | Per-video status icon (⏳ pending · 🔄 running · ✅ done · ❌ error · ⏹️ stopped), text progress bar, current stage, and the last 6 timestamped log lines (input info, model/scale, faces restored, fps change, output name, elapsed time). |
| 4. Processed videos | `gr.File(file_count="multiple")` | Each `<name>_slp.mp4` appears **as soon as it finishes**; click to download individually. |
| Buttons | `Start Process` / `Stop Process`, `Download All (.tar)` | See below. |

## Buttons

* **Start Process** — validates: ≥ 1 video uploaded, ≥ 1 operation selected, and that the
  chosen sub-option is not "Off" for Face Restoration / Frame Rate. Starts a background
  worker thread and relabels itself **Stop Process** (red).
* **Stop Process** — sets a `threading.Event`. The worker checks it inside every
  `progress_cb` call (i.e. every frame) and between stages, raising `StopRequested`;
  the current video is marked ⏹️ and remaining videos are skipped. Button returns to
  **Start Process**.
* **Download All (.tar)** — disabled until **every** video finished with ✅. Builds an
  uncompressed TAR of all `_slp.mp4` files and serves it via `gr.DownloadButton`.

## Processing pipeline (per video, identical order to `app.py`)

```
validate → [Upscale] → [Face Restoration] → [Frame Rate (RIFE)] → remux original audio → save
```

Functions reused from `app.py`: `gpu_upscale_video`, `gpu_restore_faces`,
`gpu_interpolate_video`, `_resolve_target_fps`, `_free_gpu`, and the option maps
`VIDEO_MODELS`, `FPS_OPTIONS`, `FACE_OPTIONS`. Helpers reused from `utils.py`:
`validate_video`, `extract_audio`, `remux`, `make_tmp_dir`; `plan_multiplier` from
`frame_interpolator.py`. Audio enhancement (AudioSR) is not part of the batch scope; the
original audio track is remuxed unchanged.

Behaviour notes (same as `app.py`):
* If no faces are detected the face stage is skipped and the upscaled video is used.
* If the source fps already ≥ target, RIFE is skipped.
* `torch.OutOfMemoryError` is caught per stage, VRAM is freed, and the video is marked ❌
  with the message — the batch continues with the next video.
* Any other exception marks that video ❌ (full traceback on the console) and the batch
  continues.

## Output

* Directory: `output/batch/` (git-ignored).
* Naming: original stem + `_slp` + `.mp4`, e.g. `holiday.mov` → `holiday_slp.mp4`.
* Intermediate files live in a per-video temp dir removed in `finally`.

## Architecture

```
Gradio UI thread                       Worker thread (daemon)
────────────────                       ──────────────────────
Start → on_start_stop()  ──spawn──▶    _worker() → for job: _process_one(job)
gr.Timer(1 s) → on_tick() ◀─reads──    BatchState (lock-protected Job list)
Stop  → stop_event.set() ──────────▶   progress_cb → _check_stop() → StopRequested
```

`BatchState` (module-level singleton) holds the `Job` list (`status`, `stage`, `pct`,
`log`, `output`, `elapsed`), the run flag, the stop event and the last TAR path. Because
GPU models are loaded once per process and kept resident (see `README.md`), the batch
avoids reload cost between videos. Only one batch may run at a time.
