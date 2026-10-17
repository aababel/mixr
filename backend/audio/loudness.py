"""Per-track loudness matching (owner: Aarav).

Each track gets ONE gain, computed from that track alone:

    gain_db = target_lufs - measured_lufs

then limited to a safe range and to the track's peak headroom (no limiter is
applied to whole tracks, so a boost may not push any sample above the
ceiling). Because the gain depends only on the track's own audio, an A/B
preview and the final mix always apply the identical gain to a given track,
whatever the playlist around it.

Loudness is ITU-R BS.1770-4 integrated loudness (K-weighting, 400 ms blocks
with 75% overlap, -70 LUFS absolute gate, -10 LU relative gate). Filter
coefficients follow libebur128, so they are exact at any sample rate.
"""
from dataclasses import dataclass
from functools import lru_cache
from pathlib import Path
from typing import Optional

import numpy as np
from scipy.signal import sosfilt

from backend.audio.loading import RENDER_SR, load_render_audio


@dataclass(frozen=True)
class LoudnessSettings:
    target_lufs: float = -14.0  # common streaming reference level
    max_boost_db: float = 12.0
    max_cut_db: float = 24.0
    peak_ceiling: float = 0.98  # linear sample peak a boost may reach (~ -0.2 dBFS)


DEFAULT_LOUDNESS = LoudnessSettings()


@dataclass(frozen=True)
class TrackGain:
    """The gain actually applied to one track, and why."""

    gain_db: float
    measured_lufs: Optional[float]  # None when the track is silent or too short to measure
    target_lufs: float
    peak_limited: bool  # True when peak headroom stopped the boost short of the target
    note: Optional[str] = None  # human-readable reason when the target was not reached


# ---------------------------------------------------------------- measurement


def _k_weighting(sr: int) -> list[tuple[np.ndarray, np.ndarray]]:
    """BS.1770 pre-filter (high shelf) and RLB high-pass as (b, a) pairs."""
    # Stage 1: high shelf, +4 dB above ~1.7 kHz (head acoustics).
    f0, gain_db, q = 1681.974450955533, 3.999843853973347, 0.7071752369554196
    k = np.tan(np.pi * f0 / sr)
    vh = 10.0 ** (gain_db / 20.0)
    vb = vh ** 0.4996667741545416
    a0 = 1.0 + k / q + k * k
    b1 = np.array([(vh + vb * k / q + k * k) / a0, 2.0 * (k * k - vh) / a0, (vh - vb * k / q + k * k) / a0])
    a1 = np.array([1.0, 2.0 * (k * k - 1.0) / a0, (1.0 - k / q + k * k) / a0])
    # Stage 2: RLB high-pass at ~38 Hz.
    f0, q = 38.13547087602444, 0.5003270373238773
    k = np.tan(np.pi * f0 / sr)
    d = 1.0 + k / q + k * k
    b2 = np.array([1.0, -2.0, 1.0])
    a2 = np.array([1.0, 2.0 * (k * k - 1.0) / d, (1.0 - k / q + k * k) / d])
    return [(b1, a1), (b2, a2)]


def integrated_loudness(audio: np.ndarray, sr: int) -> Optional[float]:
    """BS.1770-4 integrated loudness in LUFS of (samples, ch) or (samples,) audio.

    Returns None for audio shorter than one 400 ms block or entirely below
    the -70 LUFS absolute gate (silence).
    """
    x = np.asarray(audio, dtype=np.float64)
    if x.ndim == 1:
        x = x[:, None]
    block, hop = int(round(0.4 * sr)), int(round(0.1 * sr))
    if x.shape[0] < block:
        return None

    sos = np.vstack([np.concatenate([b, a]) for b, a in _k_weighting(sr)])
    x = sosfilt(sos, x, axis=0)

    # Mean square per 400 ms block via a cumulative sum (fast on long tracks).
    # L/R/mono channel weights are all 1.0, so sum over channels.
    power = np.einsum("ij,ij->i", x, x)
    csum = np.concatenate([[0.0], np.cumsum(power)])
    starts = np.arange(0, x.shape[0] - block + 1, hop)
    z = (csum[starts + block] - csum[starts]) / block

    with np.errstate(divide="ignore"):
        lk = -0.691 + 10.0 * np.log10(z)
    z = z[lk > -70.0]  # absolute gate
    if z.size == 0:
        return None
    rel_gate = -0.691 + 10.0 * np.log10(z.mean()) - 10.0
    with np.errstate(divide="ignore"):
        z = z[-0.691 + 10.0 * np.log10(z) > rel_gate]  # relative gate
    return float(-0.691 + 10.0 * np.log10(z.mean()))


# ---------------------------------------------------------------- gain


def compute_track_gain(
    audio: np.ndarray, sr: int, settings: LoudnessSettings = DEFAULT_LOUDNESS
) -> TrackGain:
    """Gain that brings this track to the target loudness, within safe limits."""
    measured = integrated_loudness(audio, sr)
    if measured is None:
        return TrackGain(0.0, None, settings.target_lufs, False,
                         "loudness not measurable (silent or shorter than 0.4 s); gain 0 dB")

    wanted = settings.target_lufs - measured
    gain = min(max(wanted, -settings.max_cut_db), settings.max_boost_db)
    notes = []
    if gain != wanted:
        notes.append(f"gain range limited to {-settings.max_cut_db:+.0f}..{settings.max_boost_db:+.0f} dB")

    peak_limited = False
    peak = float(np.max(np.abs(audio)))
    if gain > 0 and peak > 0:
        headroom_db = 20.0 * np.log10(settings.peak_ceiling / peak)
        if gain > headroom_db:
            gain = max(0.0, headroom_db)
            peak_limited = True
            notes.append(f"boost capped by peak headroom (sample peak {peak:.3f})")

    note = None
    if notes:
        reached = measured + gain
        note = (f"measured {measured:.1f} LUFS, gain {gain:+.1f} dB reaches {reached:.1f} LUFS "
                f"instead of the {settings.target_lufs:.1f} LUFS target: " + "; ".join(notes))
    return TrackGain(float(gain), measured, settings.target_lufs, peak_limited, note)


@lru_cache(maxsize=16)
def _gain_cached(path: str, mtime_ns: int) -> TrackGain:
    return compute_track_gain(load_render_audio(Path(path)), RENDER_SR)


def track_gain_for_path(path: Path) -> TrackGain:
    """Gain for an audio file, measured on the same decoded audio the renderers use.

    Cached by path and modification time, so the A/B previews and the final mix
    reuse one measurement and always agree.
    """
    p = Path(path)
    return _gain_cached(str(p.resolve()), p.stat().st_mtime_ns)


def gain_notes(side: str, track_id: str, gain: TrackGain) -> list[str]:
    """Manifest notes for a track whose loudness gain did not reach the target."""
    return [f"{side} {track_id}: {gain.note}"] if gain.note else []
