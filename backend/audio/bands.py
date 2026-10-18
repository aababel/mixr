"""Split-frequency swap for the overlap (owner: Aarav).

Each side of the seam is split into three bands with zero-phase filters:
    low  = Butterworth low-pass  at LOW_HZ,  run forward and backward (sosfiltfilt)
    high = Butterworth high-pass at HIGH_HZ, run forward and backward
    mid  = signal - low - high
so low + mid + high reproduces the input exactly, and a band at gain 1 on
both bands it shares with the others leaves the audio untouched.

Gains across the seam:
    lows   swap sharply at the midpoint (one basslines at a time)
    highs  swap early, at ~30% (one set of hi-hats at a time)
    mids   the request's fade_style crossfade (equal-power by default)
Swap points move to the nearest matched beat when one is close, so the swap
falls on a kick and between off-beat hats.
"""
from functools import lru_cache

import numpy as np
from scipy.signal import butter, sosfiltfilt

LOW_HZ = 200.0
HIGH_HZ = 5000.0
FILTER_ORDER = 4  # per pass; sosfiltfilt doubles the effective slope
BASS_SWAP_AT = 0.5
HIGH_SWAP_AT = 0.30
BASS_SNAP_WINDOW = (0.42, 0.58)  # snap the swap to a beat only inside this share of the seam
HIGH_SNAP_WINDOW = (0.25, 0.35)
BASS_RAMP_SECONDS = 0.040  # short enough to sound like a hard swap, long enough not to click
HIGH_RAMP_SECONDS = 0.030


@lru_cache(maxsize=8)
def _filters(sr: int):
    nyq = sr / 2.0
    low = butter(FILTER_ORDER, LOW_HZ / nyq, btype="low", output="sos")
    high = butter(FILTER_ORDER, min(HIGH_HZ, 0.45 * sr) / nyq, btype="high", output="sos")
    return low, high


def split_bands(x: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(low, mid, high) of (samples, ch) audio; low + mid + high == x."""
    x64 = np.asarray(x, dtype=np.float64)
    low_sos, high_sos = _filters(sr)
    if x64.shape[0] <= 3 * (2 * low_sos.shape[0] + 1):  # too short for filtfilt padding
        z = np.zeros_like(x64)
        return z, x64, z.copy()
    low = sosfiltfilt(low_sos, x64, axis=0)
    high = sosfiltfilt(high_sos, x64, axis=0)
    return low, x64 - low - high, high


def swap_point(n: int, at: float, window: tuple[float, float], beats: list[int]) -> int:
    """Sample where a band swaps: the beat nearest at*n if one lies in the window."""
    target = at * n
    inside = [b for b in beats if window[0] * n <= b <= window[1] * n]
    if inside:
        return int(min(inside, key=lambda b: abs(b - target)))
    return int(round(target))


def swap_gains(n: int, at: int, ramp: int) -> tuple[np.ndarray, np.ndarray]:
    """Equal-power swap finishing at sample `at`: outgoing 1 -> 0, incoming 0 -> 1."""
    at = min(max(at, 1), n)
    ramp = max(1, min(ramp, at))
    g_in = np.zeros(n)
    g_in[at:] = 1.0
    t = (np.arange(ramp) + 0.5) / ramp
    g_in[at - ramp:at] = np.sin(t * np.pi / 2)
    g_out = np.sqrt(np.clip(1.0 - g_in ** 2, 0.0, 1.0))
    return g_out[:, None], g_in[:, None]


def band_swap(
    out_seg: np.ndarray, in_seg: np.ndarray, sr: int, beats: list[int],
    mid_fade_out: np.ndarray, mid_fade_in: np.ndarray,
) -> tuple[np.ndarray, str]:
    """Blend two equal-length (samples, ch) segments band by band.

    Returns (blend float64, description for AppliedTransition.applied_eq).
    """
    n = out_seg.shape[0]
    o_low, o_mid, o_high = split_bands(out_seg, sr)
    i_low, i_mid, i_high = split_bands(in_seg, sr)
    bass_at = swap_point(n, BASS_SWAP_AT, BASS_SNAP_WINDOW, beats)
    high_at = swap_point(n, HIGH_SWAP_AT, HIGH_SNAP_WINDOW, beats)
    bo, bi = swap_gains(n, bass_at, int(round(BASS_RAMP_SECONDS * sr)))
    ho, hi = swap_gains(n, high_at, int(round(HIGH_RAMP_SECONDS * sr)))
    blend = (o_low * bo + i_low * bi + o_high * ho + i_high * hi
             + o_mid * mid_fade_out + i_mid * mid_fade_in)
    desc = (f"3-band swap: lows <{LOW_HZ:.0f} Hz swapped at {bass_at / sr:.2f}s "
            f"({100 * bass_at / n:.0f}%), highs >{HIGH_HZ:.0f} Hz swapped at {high_at / sr:.2f}s "
            f"({100 * high_at / n:.0f}%), mids crossfaded")
    return blend, desc
