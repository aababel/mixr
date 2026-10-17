"""Shared audio loading for preview.py and mix.py (owner: Aarav).

Everything is rendered at one sample rate and channel count so previews
and the final mix go through identical processing.
"""
from functools import lru_cache
from pathlib import Path

import librosa
import numpy as np

RENDER_SR = 44100
RENDER_CHANNELS = 2


@lru_cache(maxsize=4)
def _load_cached(path: str, mtime_ns: int) -> np.ndarray:
    y, _ = librosa.load(path, sr=RENDER_SR, mono=False)
    frames = y[:, None] if y.ndim == 1 else y.T  # -> (samples, ch)
    if frames.shape[1] == 1:
        frames = np.repeat(frames, RENDER_CHANNELS, axis=1)
    elif frames.shape[1] > RENDER_CHANNELS:
        frames = frames[:, :RENDER_CHANNELS]
    frames = np.ascontiguousarray(frames, dtype=np.float32)
    frames.setflags(write=False)  # cached and shared: nobody may mutate it
    return frames


def load_render_audio(path: Path) -> np.ndarray:
    """Decode to float32 (samples, 2) at RENDER_SR. Cached for repeat A/B calls."""
    p = Path(path)
    return _load_cached(str(p.resolve()), p.stat().st_mtime_ns)
