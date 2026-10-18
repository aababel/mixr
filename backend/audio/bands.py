"""Split-frequency swap for the overlap (owner: Aarav).

Each side of the seam is split into three bands with zero-phase filters:
    low  = Butterworth low-pass  at LOW_HZ,  run forward and backward (sosfiltfilt)
    high = Butterworth high-pass at HIGH_HZ, run forward and backward
    mid  = signal - low - high
so low + mid + high reproduces the input exactly, and a band at gain 1 on
both bands it shares with the others leaves the audio untouched.

Gains across the seam. The blend lands on a phrase boundary: the outgoing
window ends on a phrase start, and with a whole-phrase blend the incoming
song is a whole phrase in there too. Everything resolves on that downbeat,
so the incoming song's next phrase (its chorus or drop) hits as the
outgoing one leaves, instead of the handover falling mid-phrase:
    lows   swap sharply on the landing beat (bass_swap_sample: the last
           matched beat of the region; one bassline at a time)
    highs  swap halfway, on a beat (one set of hi-hats at a time)
    mids   transition.py crosses them over the second half (mid_window),
           with fade_style; the incoming mids sit at MID_UNDER before that

All of this happens inside the blend region [r0, r1) of the seam (the whole
seam unless transition.py moved the crossfade off a fade-in head or a
fade-out tail): before r0 only the outgoing track sounds, after r1 only the
incoming one. Swap fractions are fractions of the region.
"""
from functools import lru_cache

import numpy as np
from scipy.signal import butter, sosfiltfilt

LOW_HZ = 200.0
HIGH_HZ = 5000.0
FILTER_ORDER = 4  # per pass; sosfiltfilt doubles the effective slope
LAND_MIN_SHARE = 0.75  # the landing beat must be in the last quarter of the region
LAND_TOLERANCE_SECONDS = 0.005
HIGH_SWAP_AT = 0.5
HIGH_SNAP_WINDOW = (0.42, 0.58)  # snap the swap to a beat only inside this share of the region
MID_FROM = 0.5  # mids cross over from this share of the way to the landing beat
BASS_RAMP_SECONDS = 0.040  # short enough to sound like a hard swap, long enough not to click
HIGH_RAMP_SECONDS = 0.030
# Incoming mids before they cross over: a hint of the next song under the
# outgoing one. With a vocal swap (both songs carry melody at the seam) the
# mids then trade over one beat ending on the landing beat, so two vocals
# are never loud together.
MID_UNDER = 0.2  # -14 dB
MID_SWAP_BEATS = 1.0
DEFAULT_BEAT_SECONDS = 0.5


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


def swap_point(n: int, at: float, window: tuple[float, float], beats: list[int],
               r0: int = 0) -> int:
    """Sample where a band swaps: the beat nearest r0 + at*n if one lies in the
    window. n is the region length, r0 its start in the seam."""
    target = r0 + at * n
    inside = [b for b in beats if r0 + window[0] * n <= b <= r0 + window[1] * n]
    if inside:
        return int(min(inside, key=lambda b: abs(b - target)))
    return int(round(target))


def bass_swap_sample(n: int, beats: list[int], sr: int, region: tuple[int, int] | None = None) -> int:
    """The landing beat: the last matched beat in the region's last quarter
    (the region's end without one)."""
    r0, r1 = region if region else (0, n)
    tol = int(LAND_TOLERANCE_SECONDS * sr)
    inside = [b for b in beats if r0 + LAND_MIN_SHARE * (r1 - r0) <= b <= r1 + tol]
    return int(min(max(inside), n)) if inside else r1


def mid_window(n: int, beats: list[int], sr: int, region: tuple[int, int] | None = None) -> tuple[int, int]:
    """(start, landing beat) of the mid crossover: from the matched beat
    nearest MID_FROM of the way to the landing beat."""
    r0 = region[0] if region else 0
    land = bass_swap_sample(n, beats, sr, region)
    start = swap_point(land - r0, MID_FROM, (MID_FROM - 0.08, MID_FROM + 0.08), beats, r0)
    return min(start, land - 1), land


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
    region: tuple[int, int] | None = None,
    mid_swap: bool = False,
) -> tuple[np.ndarray, str]:
    """Blend two equal-length (samples, ch) segments band by band.

    region: (r0, r1) blend region in samples, default the whole seam.
    mid_fade_out / mid_fade_in: the mid crossover, 1/0 before mid_window's
    start and 0/1 after its landing beat.
    mid_swap: both sides carry melody, so the mids swap over one beat into
    the landing beat instead of following the mid fades.
    Returns (blend float64, description for AppliedTransition.applied_eq).
    """
    n = out_seg.shape[0]
    r0, r1 = region if region else (0, n)
    o_low, o_mid, o_high = split_bands(out_seg, sr)
    i_low, i_mid, i_high = split_bands(in_seg, sr)
    bass_at = bass_swap_sample(n, beats, sr, region)
    high_at = swap_point(r1 - r0, HIGH_SWAP_AT, HIGH_SNAP_WINDOW, beats, r0)
    bo, bi = swap_gains(n, bass_at, int(round(BASS_RAMP_SECONDS * sr)))
    ho, hi = swap_gains(n, high_at, int(round(HIGH_RAMP_SECONDS * sr)))
    if mid_swap:
        beat = float(np.median(np.diff(beats))) if len(beats) >= 2 else DEFAULT_BEAT_SECONDS * sr
        mid_fade_out, mid_fade_in = swap_gains(n, bass_at, int(round(MID_SWAP_BEATS * beat)))
    under = np.zeros(n)
    under[r0:bass_at] = MID_UNDER
    mid_fade_in = np.maximum(mid_fade_in, under[:, None])
    blend = (o_low * bo + i_low * bi + o_high * ho + i_high * hi
             + o_mid * mid_fade_out + i_mid * mid_fade_in)
    mids = (f"mids (vocals) swapped over one beat into {bass_at / sr:.2f}s" if mid_swap
            else "mids crossed over in the second half")
    mids += f", incoming mids {20 * np.log10(MID_UNDER):.0f} dB before"
    desc = (f"3-band swap: lows <{LOW_HZ:.0f} Hz swapped at {bass_at / sr:.2f}s "
            f"({100 * (bass_at - r0) / max(1, r1 - r0):.0f}%), highs >{HIGH_HZ:.0f} Hz swapped at "
            f"{high_at / sr:.2f}s ({100 * (high_at - r0) / max(1, r1 - r0):.0f}%), {mids}")
    if (r0, r1) != (0, n):
        desc += f" over {r0 / sr:.2f}-{r1 / sr:.2f}s of the {n / sr:.2f}s seam"
    return blend, desc
