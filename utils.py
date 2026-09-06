import subprocess
import tempfile
import os
import ffmpeg


MAX_DURATION_S = 120   # 2 minutes
MAX_FILE_BYTES = 300 * 1024 * 1024  # 300 MB


def validate_video(path: str) -> dict:
    """Return {duration, fps, width, height} or raise ValueError."""
    if os.path.getsize(path) > MAX_FILE_BYTES:
        raise ValueError(f"File exceeds 300 MB limit.")

    probe = ffmpeg.probe(path)
    vs = next((s for s in probe["streams"] if s["codec_type"] == "video"), None)
    if vs is None:
        raise ValueError("No video stream found.")

    duration = float(probe["format"]["duration"])
    if duration > MAX_DURATION_S:
        raise ValueError(f"Video is {duration:.0f}s — max allowed is {MAX_DURATION_S}s (2 min).")

    fps_parts = vs.get("r_frame_rate", "30/1").split("/")
    fps = int(fps_parts[0]) / max(int(fps_parts[1]), 1)

    return {
        "duration": duration,
        "fps": fps,
        "width": int(vs["width"]),
        "height": int(vs["height"]),
    }


def extract_audio(video_path: str, out_wav: str) -> bool:
    """Extract audio track to WAV. Returns False if video has no audio."""
    probe = ffmpeg.probe(video_path)
    has_audio = any(s["codec_type"] == "audio" for s in probe["streams"])
    if not has_audio:
        return False
    (
        ffmpeg
        .input(video_path)
        .output(out_wav, acodec="pcm_s16le", ar=44100, ac=2)
        .overwrite_output()
        .run(quiet=True)
    )
    return True


def remux(video_path: str, audio_path: str | None, output_path: str) -> None:
    """Combine video-only file with optional audio into final MP4."""
    v = ffmpeg.input(video_path)
    if audio_path:
        a = ffmpeg.input(audio_path)
        (
            ffmpeg
            .output(v, a, output_path, vcodec="copy", acodec="aac", audio_bitrate="192k")
            .overwrite_output()
            .run(quiet=True)
        )
    else:
        (
            ffmpeg
            .output(v, output_path, vcodec="copy")
            .overwrite_output()
            .run(quiet=True)
        )


def make_tmp_dir() -> str:
    d = tempfile.mkdtemp(prefix="upscale_")
    return d
