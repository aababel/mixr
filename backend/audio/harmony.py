"""Harmonic match between parts of two songs (owner: Aarav).

A song's single key estimate says little about the stretch that is actually
playing: a pre-chorus can sit on a different chord than the drop. For cue
picking, each song gets a chroma series (12 pitch-class energies per frame,
median-smoothed over time so drum hits don't count), and choose_windows
compares the part of A that plays under the blend with the part of B that
plays over it.

compatibility() is the cosine between the two parts' mean chroma after each
is centered on its own mean: 1 = the same pitch classes stand out (in key
together), around 0 = unrelated, negative = they emphasize different notes
(the blend rubs). Mapped to 0-1.

Computed from the audio at mix time and cached per file, so it needs no
schema change.
"""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path

import librosa
import numpy as np
from scipy.ndimage import median_filter

CHROMA_SR = 22050
CHROMA_HOP = 4096  # ~0.19 s per frame: parts are averaged over seconds anyway
CHROMA_N_FFT = 8192  # long window: pitch resolution down to the bass, drum hits smeared out
MEDIAN_FRAMES = 5  # ~1 s median over time: held notes stay, hits and fills drop out


@dataclass(frozen=True)
class Chroma:
    frames: np.ndarray  # (n, 12), each frame normalized to max 1
    hop: float  # seconds per frame

    def mean(self, t0: float, t1: float) -> np.ndarray | None:
        i0, i1 = max(0, int(t0 / self.hop)), int(np.ceil(t1 / self.hop))
        seg = self.frames[i0:i1]
        return seg.mean(axis=0) if seg.shape[0] else None


@lru_cache(maxsize=16)
def _chroma_cached(path: str, mtime: float) -> Chroma | None:
    try:
        y, sr = librosa.load(path, sr=CHROMA_SR, mono=True)
    except Exception:
        return None
    if y.size < CHROMA_N_FFT:
        return None
    frames = librosa.feature.chroma_stft(y=y, sr=sr, n_fft=CHROMA_N_FFT, hop_length=CHROMA_HOP)
    frames = median_filter(frames, size=(1, MEDIAN_FRAMES), mode="nearest")
    return Chroma(np.ascontiguousarray(frames.T, dtype=np.float32), CHROMA_HOP / sr)


def chroma_for(path: Path) -> Chroma | None:
    """Chroma series of an audio file, None if it can't be read."""
    p = Path(path)
    try:
        return _chroma_cached(str(p.resolve()), p.stat().st_mtime)
    except OSError:
        return None


def compatibility(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    """0-1 harmonic match of two mean chroma vectors, None without data."""
    if a is None or b is None:
        return None
    a, b = a - a.mean(), b - b.mean()
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < 1e-9 or nb < 1e-9:
        return None
    return float((np.dot(a, b) / (na * nb) + 1.0) / 2.0)
