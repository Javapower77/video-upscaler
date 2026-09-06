"""Chunked audio super-resolution using AudioSR."""
import os
import tempfile
import numpy as np
import torch
import soundfile as sf
import librosa

CHUNK_DUR = 5.12        # AudioSR internal window (seconds)
OVERLAP = 0.5           # crossfade overlap (seconds)
AUDIOSR_OUT_SR = 48000  # AudioSR always outputs at 48 kHz

_audiosr_cache: dict = {}


def _load_audiosr(model_name: str):
    if model_name in _audiosr_cache:
        return _audiosr_cache[model_name]
    from audiosr import build_model
    m = build_model(model_name=model_name, device="cuda" if torch.cuda.is_available() else "cpu")
    _audiosr_cache[model_name] = m
    return m


def _process_audiosr(model, audio_path: str, out_path: str, progress_cb=None) -> None:
    from audiosr import super_resolution

    waveform, sr = librosa.load(audio_path, sr=None, mono=False)
    if waveform.ndim == 1:
        waveform = waveform[np.newaxis, :]

    chunk_samples = int(CHUNK_DUR * sr)
    overlap_samples = int(OVERLAP * sr)
    stride = max(1, chunk_samples - overlap_samples)

    # Overlap for crossfade is in output-SR space
    out_overlap = int(OVERLAP * AUDIOSR_OUT_SR)

    positions = list(range(0, waveform.shape[1], stride))
    n = len(positions)
    chunks_out = []

    for i, start in enumerate(positions):
        end = min(start + chunk_samples, waveform.shape[1])
        chunk = waveform[:, start:end]

        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as f:
            tmp_in = f.name
        try:
            sf.write(tmp_in, chunk.T, sr)
            # super_resolution returns a numpy array (channels, samples) at AUDIOSR_OUT_SR
            out_wav = super_resolution(
                model, tmp_in,
                guidance_scale=3.5,
                ddim_steps=50,
                latent_t_per_second=12.8,
                seed=42,
            )
        finally:
            os.unlink(tmp_in)

        out_wav = np.squeeze(out_wav)        # collapse any size-1 dims: (1,1,N) → (N,)
        if out_wav.ndim == 1:
            out_wav = out_wav[np.newaxis, :]  # ensure (channels, samples)
        chunks_out.append(out_wav)

        if progress_cb:
            progress_cb((i + 1) / n)

    _concat_chunks(chunks_out, out_path, out_overlap)


def _concat_chunks(chunks: list, out_path: str, overlap_samples: int) -> None:
    """Concatenate chunks with linear crossfade on overlapping regions."""
    if not chunks:
        return
    result = chunks[0]

    for nxt in chunks[1:]:
        ov = min(overlap_samples, result.shape[1], nxt.shape[1])
        if ov > 0:
            fade_out = np.linspace(1, 0, ov)
            fade_in  = np.linspace(0, 1, ov)
            result[:, -ov:] = result[:, -ov:] * fade_out + nxt[:, :ov] * fade_in
            result = np.concatenate([result, nxt[:, ov:]], axis=1)
        else:
            result = np.concatenate([result, nxt], axis=1)

    sf.write(out_path, result.T, AUDIOSR_OUT_SR)


def enhance_audio(
    input_path: str,
    output_path: str,
    model_name: str,   # "audiosr_basic" | "audiosr_speech"
    progress_cb=None,
) -> None:
    audiosr_key = "basic" if model_name == "audiosr_basic" else "speech"
    model = _load_audiosr(audiosr_key)
    _process_audiosr(model, input_path, output_path, progress_cb)
