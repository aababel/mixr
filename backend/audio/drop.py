"""Drop transition: build tension on the outgoing song, land the incoming drop.

choose_windows picks it when the incoming song has a real drop (a phrase
where its energy jumps). The incoming window then starts a few bars before
that drop (the build), and TrackWindow.drop_seconds marks it. The seam:

  * The outgoing song gets a resonant high-pass sweep (SWEEP_FROM_HZ up to
    SWEEP_TO_HZ, slow at first, fast at the end), so it thins out and rises.
  * Its last bar before the drop rolls: beat 1 plays, then half-beat
    repeats, then quarter and eighth repeats of it, accelerating.
  * The beat before the drop is silence, apart from a short echo of the
    roll's last slice.
  * The incoming drop then lands on the downbeat with full bass.

layered (tempos close enough to beat-match): the incoming build plays under
the sweep without its lows, rising to BUILD_LEVEL, and drops out for the
silent beat. The seam is planned by seam_plan.py like a blend, so every beat
lines up and the drop falls on a matched beat.

cut (tempos too far apart): only the outgoing song plays in the seam, on its
own beat grid, and the incoming song starts cold on its drop. Its build is
skipped. This is the usual way to switch tempos.
"""
from dataclasses import dataclass

import numpy as np
from scipy.signal import sosfilt

from backend.audio.bands import split_bands
from backend.audio.seam_plan import SeamPlan

SWEEP_FROM_HZ = 40.0
SWEEP_TO_HZ = 1800.0
SWEEP_CURVE = 1.8  # cutoff fraction = progress ** this: stays low, then rises fast
SWEEP_Q = 2.2  # resonance on the second section: the rising "wooo"
SWEEP_BLOCK = 256  # samples per coefficient update
SWEEP_MAKEUP = 2.0  # gain at the top of the sweep, so the build rises instead of fading
BUILD_LEVEL = 0.85  # incoming build mids/highs reach this before the silent beat
BUILD_CURVE = 1.3
CUT_BARS = 2  # cut variant: outgoing bars processed before the drop
BAR_BEATS = 4
SLICE_FADE_SECONDS = 0.002
EDGE_FADE_SECONDS = 0.004
ECHO_GAINS = (0.45, 0.2)  # the roll's last slice repeated into the silent beat


@dataclass
class DropPlan(SeamPlan):
    drop_at: int = 0  # seam sample where the incoming drop lands
    period: float = 0.5  # beat length before the drop, seconds
    layered: bool = True


def _highpass_sos(fc: float, q: float, sr: int) -> np.ndarray:
    w0 = 2 * np.pi * min(fc, 0.45 * sr) / sr
    alpha = np.sin(w0) / (2 * q)
    c = np.cos(w0)
    b = np.array([(1 + c) / 2, -(1 + c), (1 + c) / 2])
    a = np.array([1 + alpha, -2 * c, 1 - alpha])
    return np.concatenate([b / a[0], a / a[0]])


def highpass_sweep(x: np.ndarray, sr: int, n_sweep: int) -> np.ndarray:
    """Resonant 4th-order high-pass whose cutoff rises over the first n_sweep
    samples (held at the top after). Coefficients change every SWEEP_BLOCK
    samples; the filter state carries over, so the sweep is continuous."""
    y = np.empty_like(x, dtype=np.float64)
    zi = np.zeros((2, 2, x.shape[1]))
    n_sweep = max(1, n_sweep)
    for i in range(0, x.shape[0], SWEEP_BLOCK):
        prog = min(1.0, (i + SWEEP_BLOCK / 2) / n_sweep)
        fc = SWEEP_FROM_HZ * (SWEEP_TO_HZ / SWEEP_FROM_HZ) ** (prog ** SWEEP_CURVE)
        sos = np.stack([_highpass_sos(fc, 0.707, sr), _highpass_sos(fc, SWEEP_Q, sr)])
        y[i:i + SWEEP_BLOCK], zi = sosfilt(sos, x[i:i + SWEEP_BLOCK], axis=0, zi=zi)
    return y


def _faded(seg: np.ndarray, k: int) -> np.ndarray:
    seg = seg.copy()
    k = min(k, seg.shape[0] // 4)
    if k > 1:
        ramp = np.linspace(0.0, 1.0, k)[:, None]
        seg[:k] *= ramp
        seg[-k:] *= ramp[::-1]
    return seg


def roll(a: np.ndarray, start: int, p: int, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Three beats starting at `start`: beat 1 as is, then 2 x half-beat,
    2 x quarter-beat and 4 x eighth-beat repeats of it. Returns (3 beats of
    audio, the last slice for the echo)."""
    src = a[start:start + p]
    k = int(SLICE_FADE_SECONDS * sr)
    parts = [src]
    for size, count in ((p // 2, 2), (p // 4, 2), (p // 8, 4)):
        piece = _faded(src[:size], k)
        parts.extend([piece] * count)
    out = np.concatenate(parts, axis=0)
    if out.shape[0] < 3 * p:
        out = np.concatenate([out, np.zeros((3 * p - out.shape[0], a.shape[1]))], axis=0)
    return out[:3 * p], parts[-1]


def _ramp_edge(g: np.ndarray, at: int, k: int, up: bool) -> None:
    """Soften a 0/1 step in gain curve g at sample `at` over k samples."""
    lo, hi = max(0, at - k), min(g.size, at + k)
    if hi > lo:
        g[lo:hi] = np.linspace(0.0, 1.0, hi - lo) if up else np.linspace(1.0, 0.0, hi - lo)


def render_drop(out_seg: np.ndarray, in_seg: np.ndarray, sr: int, plan: DropPlan,
                g_out: float, g_in: float) -> tuple[np.ndarray, str]:
    """Seam audio for a drop plan, gains applied. out_seg / in_seg are already
    aligned to the seam (warped when the plan glides). For a cut, in_seg is
    silent before drop_at. Returns ((seam_len, ch) float32, description)."""
    n, d = plan.seam_len, plan.drop_at
    p = max(1, int(round(plan.period * sr)))
    ch = out_seg.shape[1]
    edge = int(EDGE_FADE_SECONDS * sr)
    gap = max(0, d - p)

    a = np.zeros((n, ch))
    m = min(n, out_seg.shape[0])
    a[:m] = out_seg[:m].astype(np.float64) * g_out
    rolled = gap >= 3 * p
    echo = None
    if rolled:
        a[gap - 3 * p:gap], echo = roll(a, gap - 3 * p, p, sr)
    a = highpass_sweep(a, sr, gap)
    prog = np.clip(np.arange(n) / max(1, gap), 0.0, 1.0) ** SWEEP_CURVE
    a *= (1.0 + (SWEEP_MAKEUP - 1.0) * prog)[:, None]
    keep = np.ones(n)
    keep[gap:] = 0.0
    _ramp_edge(keep, gap, edge, up=False)
    seam = a * keep[:, None]
    if echo is not None:
        echo = highpass_sweep(echo, sr, 1) * SWEEP_MAKEUP  # at the top of the sweep, like the roll
        for r, gain in enumerate(ECHO_GAINS):
            at = gap + r * echo.shape[0]
            k = min(echo.shape[0], max(0, d - at))
            seam[at:at + k] += echo[:k] * gain

    b = np.zeros((n, ch))
    m = min(n, in_seg.shape[0])
    b[:m] = in_seg[:m].astype(np.float64) * g_in
    low, mid, high = split_bands(b, sr)
    t = np.arange(n, dtype=np.float64)
    build = BUILD_LEVEL * np.clip(t / max(1, gap), 0.0, 1.0) ** BUILD_CURVE
    build[gap:] = 0.0
    if not plan.layered:
        build[:] = 0.0
    full = np.zeros(n)
    full[d:] = 1.0
    _ramp_edge(full, d, edge, up=True)
    seam += low * full[:, None] + (mid + high) * np.maximum(build, full)[:, None]

    desc = (f"drop: outgoing high-pass sweep {SWEEP_FROM_HZ:.0f} Hz -> {SWEEP_TO_HZ / 1000:.1f} kHz "
            f"over {gap / sr:.2f}s"
            + (", last bar rolled (1, 1/2, 1/4, 1/8 beat)" if rolled else "")
            + f", one beat of silence, incoming drop at {d / sr:.2f}s with full bass"
            + ("; incoming build under the sweep without its bass" if plan.layered
               else "; incoming starts cold on its drop (tempos too far apart to layer)"))
    return seam.astype(np.float32), desc


def plan_drop_cut(out_len: int, in_len: int, sr: int, end_t: float, period: float,
                  drop_rel: float) -> DropPlan:
    """Cut variant: the outgoing song's last CUT_BARS bars ending at its bar
    line end_t, then the incoming song from drop_rel (its build is skipped)."""
    end = min(out_len, int(round(end_t * sr)))
    o_start = max(0, end - int(round(CUT_BARS * BAR_BEATS * period * sr)))
    drop_at = end - o_start
    lead = int(EDGE_FADE_SECONDS * sr)  # the incoming drop's first samples, faded in
    first = min(int(round(drop_rel * sr)), in_len - lead - 1)
    plan = DropPlan(kind="drop_cut", o_start=o_start, in_consumed=first + lead,
                    seam_len=drop_at + lead, beats=[drop_at], target_seconds=drop_rel,
                    drop_at=drop_at, period=period, layered=False)
    plan.notes.append(f"drop (cut): outgoing's last {drop_at / sr:.2f}s swept and rolled into a "
                      f"beat of silence; incoming starts on its drop {first / sr:.2f}s into its window")
    return plan
