"""Seam planning: where a transition starts, how long it is, and how each
side is time-warped (owner: Aarav).

Pure arithmetic on beat lists, no audio. transition.py renders the plan.

Seam model (unchanged): the overlap ends at the outgoing track's last sample
and the incoming track starts at its sample 0. With a tempo glide the seam
(output) length differs from the number of samples each side consumes, so a
plan has three lengths:
    out_consumed = len(outgoing) - o_start
    in_consumed  = incoming samples played inside the seam
    seam_len     = output samples
mix.py stitches outgoing[:o_start] + seam + incoming[in_consumed:].

Kinds of plan:
  glide         beats matched beat-for-beat; both sides warped so the output
                tempo ramps linearly from the outgoing tempo to the incoming
                tempo across the blend (needs Rubber Band)
  beat_matched  tempos already close enough that the grids drift < 5 ms over
                the blend: beats matched, no warp
  short         tempos differ > MAX_GLIDE_DEVIATION after half/double folding
                (or Rubber Band missing): ~2-beat blend ending on an incoming
                beat, no warp
  snapped       no usable beat grid: the previous behaviour (first incoming
                beat snapped onto an outgoing beat, no warp)
  plain         beat_align=False: requested length as is
"""
import math
from dataclasses import dataclass, field

import numpy as np
from scipy.signal import butter, sosfiltfilt

MAX_GLIDE_DEVIATION = 0.12  # max |tempo ratio - 1| (after folding) for a glide
MAX_DRIFT_SECONDS = 0.005  # below this total drift, beats are matched without warping
SHORT_BLEND_BEATS = 2.0
GRID_MAX_RMS_SECONDS = 0.02  # inlier jitter above this: grid rejected
GRID_INLIER_SECONDS = 0.035  # detected beats further than this from the grid are ignored
GRID_MIN_BEATS = 6
KICK_HZ = 150.0  # kick-band low-pass for phase locking
KICK_WIN_SECONDS, KICK_HOP_SECONDS = 0.0116, 0.0029  # energy frames for kick onsets
KICK_BIN_SECONDS = 0.004  # phase histogram resolution
KICK_MIN_SHARPNESS = 1.8  # fold peak / mean below this = no clear kick pulse (breakdowns ~1.5)
KICK_BPM_RANGE = (90.0, 180.0)
GRID_MIN_INLIER_SHARE = 0.6  # at least this share of the window's beats must agree
GRID_WINDOW_MARGIN_SECONDS = 10.0
MIN_PERIOD, MAX_PERIOD = 60.0 / 220.0, 60.0 / 50.0


@dataclass(frozen=True)
class Grid:
    """Constant-tempo beat grid fitted to detected beats: beat k at first + k * period."""

    first: float  # seconds, the fitted time of the first detected beat in the window
    period: float  # seconds per beat

    @property
    def bpm(self) -> float:
        return 60.0 / self.period


def fit_grid(beats: list[float], t_lo: float, t_hi: float) -> Grid | None:
    """Constant-tempo grid through the detected beats in [t_lo, t_hi].

    librosa beat times sit on an 11-23 ms frame grid and the tracker
    hiccups in breakdowns and intros (a few beats too early or late, a
    missed or doubled beat). So:
      1. period = mean of the regular gaps (within 8% of the median), which
         averages out frame quantization;
      2. phase  = circular mean of the beat times modulo that period, which
         a few stray beats cannot drag far;
      3. beats more than GRID_INLIER_SECONDS off the grid are dropped and the
         grid refitted by least squares on the rest (a few rounds).
    Returns None when too few beats agree, or they agree only loosely.
    """
    b = np.array(sorted(x for x in beats if t_lo <= x <= t_hi), dtype=np.float64)
    if b.size < GRID_MIN_BEATS:
        return None
    gaps = np.diff(b)
    med = float(np.median(gaps))
    if med <= 0:
        return None
    regular = gaps[np.abs(gaps - med) <= 0.08 * med]
    if regular.size < GRID_MIN_BEATS - 1:
        return None
    period = float(np.mean(regular))
    phase = float(np.angle(np.mean(np.exp(2j * np.pi * b / period)))) * period / (2 * np.pi)

    inl = np.ones(b.size, dtype=bool)
    for _ in range(4):
        k = np.round((b - phase) / period)
        res = b - (phase + k * period)
        inl = np.abs(res) <= GRID_INLIER_SECONDS
        _, first_idx = np.unique(k[inl], return_index=True)  # one beat per grid index
        kk, bb = k[inl][first_idx], b[inl][first_idx]
        if kk.size < GRID_MIN_BEATS:
            return None
        A = np.stack([np.ones_like(kk), kk], axis=1)
        (phase, period), *_ = np.linalg.lstsq(A, bb, rcond=None)
        phase, period = float(phase), float(period)
        if not (MIN_PERIOD <= period <= MAX_PERIOD):
            return None

    k = np.round((b - phase) / period)
    res = b - (phase + k * period)
    inl = np.abs(res) <= GRID_INLIER_SECONDS
    rms = float(np.sqrt(np.mean(res[inl] ** 2))) if inl.any() else np.inf
    if inl.sum() < max(GRID_MIN_BEATS, GRID_MIN_INLIER_SHARE * b.size) or rms > GRID_MAX_RMS_SECONDS:
        return None
    first_k = k[inl][0]  # grid index of the first beat that agrees with the grid
    return Grid(phase + first_k * period, period)


def beat_period(beats: list[float], t_lo: float, t_hi: float) -> float | None:
    """Tempo seed: mean of the regular beat gaps in [t_lo, t_hi] (or the whole
    list if the window holds too few beats)."""
    for lo, hi in ((t_lo, t_hi), (-np.inf, np.inf)):
        b = np.array(sorted(x for x in beats if lo <= x <= hi), dtype=np.float64)
        if b.size >= GRID_MIN_BEATS:
            gaps = np.diff(b)
            med = float(np.median(gaps))
            regular = gaps[np.abs(gaps - med) <= 0.08 * med]
            if med > 0 and regular.size >= GRID_MIN_BEATS - 1:
                return float(np.mean(regular))
    return None


_KICK_SOS = {}


def _low_onsets(mono: np.ndarray, sr: int) -> tuple[np.ndarray, np.ndarray]:
    """Onset strength of the kick band: positive log-energy flux below
    KICK_HZ. Returns (flux, onset time in seconds of each value).

    The low-pass is zero-phase, and a jump in frame i is timed where a sharp
    onset first enters that frame (its last hop), so times are not delayed.
    """
    if sr not in _KICK_SOS:
        _KICK_SOS[sr] = butter(4, KICK_HZ, btype="low", fs=sr, output="sos")
    win = max(8, int(round(KICK_WIN_SECONDS * sr)))
    hop = max(1, int(round(KICK_HOP_SECONDS * sr)))
    x = np.asarray(mono, dtype=np.float64)
    if x.size < max(win * 4, 64):
        return np.zeros(0), np.zeros(0)
    x = sosfiltfilt(_KICK_SOS[sr], x)
    n = (x.size - win) // hop
    c = np.concatenate([[0.0], np.cumsum(x * x)])
    starts = np.arange(n) * hop
    energy = (c[starts + win] - c[starts]) / win
    loud = np.log(energy + 1e-7)
    flux = np.maximum(0.0, np.diff(loud, prepend=loud[0]))
    times = (starts + win - hop / 2) / sr
    return flux, times


def kick_pulse(mono: np.ndarray, sr: int, seed_period: float) -> tuple[float, float, float] | None:
    """(sharpness, phase_seconds, period) of the clearest kick pulse near
    seed_period, folded into the 90-180 BPM range the analysis uses."""
    flux, times = _low_onsets(mono, sr)
    if flux.size < 8 or flux.sum() <= 0:
        return None

    def fold(period: float) -> tuple[float, float]:
        bins = max(16, int(round(period / KICK_BIN_SECONDS)))
        idx = np.floor((times % period) / period * bins).astype(int) % bins
        hist = np.bincount(idx, weights=flux, minlength=bins)
        hist = hist + 0.5 * (np.roll(hist, 1) + np.roll(hist, -1))  # tolerate bin edges
        peak = int(np.argmax(hist))
        sharp = float(hist[peak] / (np.mean(hist) + 1e-12))
        return sharp, (peak + 0.5) / bins * period

    base = seed_period
    while base > 60.0 / KICK_BPM_RANGE[0]:
        base /= 2
    while base <= 60.0 / KICK_BPM_RANGE[1]:
        base *= 2
    best = max((fold(p) + (p,) for p in base * (1 + np.linspace(-0.015, 0.015, 61))),
               key=lambda r: r[0])
    p0 = best[2]
    for p in p0 * (1 + np.linspace(-0.0006, 0.0006, 25)):  # fine period search
        r = fold(p) + (p,)
        if r[0] > best[0]:
            best = r
    return best


def kick_grid(mono: np.ndarray, sr: int, t_offset: float, seed_period: float) -> Grid | None:
    """Grid locked to the kicks in a stretch of audio starting at t_offset s.

    Folds the kick-band onset strength at candidate periods near the tracker's
    tempo; the sharpest fold gives period and phase. The beat tracker's own
    phase is not used: on real songs it can jump between the kick and
    syncopated parts by a quarter beat or more within one section. Returns
    None when no clear pulse shows (no steady kick, e.g. an ambient outro).
    """
    res = kick_pulse(mono, sr, seed_period)
    if res is None or res[0] < KICK_MIN_SHARPNESS:
        return None
    _, phase, period = res
    return Grid(float(t_offset + phase), float(period))


def fold_ratio(out_period: float, in_period: float) -> tuple[float, float]:
    """(ratio, factor): ratio = incoming tempo / outgoing tempo after folding
    into [1/sqrt2, sqrt2]; factor scales the incoming beat period (2 = pair
    every other incoming beat, 0.5 = pair half-beats)."""
    r, factor = out_period / in_period, 1.0
    while r > math.sqrt(2):
        r, factor = r / 2, factor * 2
    while r < 1 / math.sqrt(2):
        r, factor = r * 2, factor / 2
    return r, factor


def deviation(ratio: float) -> float:
    return max(ratio, 1.0 / ratio) - 1.0


@dataclass
class SeamPlan:
    kind: str
    o_start: int
    in_consumed: int
    seam_len: int
    out_map: list[tuple[int, int]] | None = None  # None = no warp
    in_map: list[tuple[int, int]] | None = None
    beats: list[int] = field(default_factory=list)  # seam-relative output samples of matched beats
    tempo_ratio: float | None = None  # incoming/outgoing tempo, set when a warp is applied
    target_seconds: float = 0.0  # length the planner aimed for
    notes: list[str] = field(default_factory=list)

    @property
    def warped(self) -> bool:
        return self.out_map is not None or self.in_map is not None


def _clean_map(points: list[tuple[float, float]], sr: int) -> list[tuple[int, int]]:
    out: list[tuple[int, int]] = []
    for s, t in points:
        p = (int(round(s * sr)), int(round(t * sr)))
        if out and (p[0] <= out[-1][0] or p[1] <= out[-1][1]):
            continue  # zero-length segment (e.g. no pre-roll); keep it strictly increasing
        out.append(p)
    return out


def plan_matched(
    out_len: int, in_len: int, sr: int, g_out: Grid, g_in: Grid,
    target: float, lo: float, hi: float, can_warp: bool,
) -> SeamPlan | str | None:
    """Beat-for-beat blend of N beats (a multiple of 4 near `target`).

    Returns a SeamPlan, "needs_warp" if the tempos drift audibly but Rubber
    Band is unavailable, or None if no N fits the safe range.
    """
    out_dur, in_dur = out_len / sr, in_len / sr
    ratio, factor = fold_ratio(g_out.period, g_in.period)
    p_o, p_i = g_out.period, g_in.period * factor
    in_pre = g_in.first - p_i * math.floor(g_in.first / p_i)  # first grid beat at or after 0
    k_last = math.floor((out_dur - 1e-3 - g_out.first) / p_o)
    last_out_beat = g_out.first + k_last * p_o
    tail = out_dur - last_out_beat  # outgoing audio after its last grid beat, < one beat

    p_avg = 0.5 * (p_o + p_i)
    n0 = max(4, 4 * int(round(target / (4 * p_avg))))
    for n in list(range(n0, 3, -4)) + [2]:
        drift = n * abs(p_o - p_i)
        warp = drift > MAX_DRIFT_SECONDS
        if warp and not can_warp:
            return "needs_warp"
        if warp:
            f = (np.arange(n) + 0.5) / n
            d = 1.0 / ((1 - f) / p_o + f / p_i)  # linear tempo ramp, one rate per beat
            u0 = in_pre * p_o / p_i  # incoming pre-roll plays at the outgoing tempo
            tail_out = tail * p_i / p_o  # outgoing tail plays at the incoming tempo
        else:
            d = np.full(n, p_o)
            u0, tail_out = in_pre, tail
        u = u0 + np.concatenate([[0.0], np.cumsum(d)])  # output time of matched beat m
        seam = u[-1] + tail_out
        out_cons = u0 + n * p_o + tail
        in_cons = (in_pre + n * p_i + tail_out) if warp else out_cons
        if out_cons > out_dur or out_cons > hi + 1e-9 or in_cons > hi + 1e-9 or in_cons > in_dur:
            continue
        if seam < lo - 1e-9:
            continue

        o_start = out_len - int(round(out_cons * sr))
        plan = SeamPlan(
            kind="glide" if warp else "beat_matched",
            o_start=o_start,
            in_consumed=int(round(in_cons * sr)) if warp else out_len - o_start,
            seam_len=int(round(seam * sr)) if warp else out_len - o_start,
            beats=[int(round(x * sr)) for x in u],
            target_seconds=target,
        )
        bpm_o, bpm_i = 60.0 / p_o, 60.0 / p_i
        fold = "" if factor == 1.0 else f" (incoming {g_in.bpm:.1f} BPM counted at {bpm_i:.1f} after half/double folding)"
        if warp:
            plan.out_map = _clean_map([(0, 0)] + [(u0 + m * p_o, u[m]) for m in range(n + 1)]
                                      + [(out_cons, seam)], sr)
            plan.in_map = _clean_map([(0, 0)] + [(in_pre + m * p_i, u[m]) for m in range(n + 1)]
                                     + [(in_cons, seam)], sr)
            # rounding can shift the end points by a sample; pin them to the integer lengths
            plan.out_map[-1] = (out_len - o_start, plan.seam_len)
            plan.in_map[-1] = (plan.in_consumed, plan.seam_len)
            plan.tempo_ratio = ratio
            plan.notes.append(
                f"tempo glide over {n} beats{fold}: output tempo ramps {bpm_o:.1f} -> {bpm_i:.1f} BPM; "
                f"outgoing warped from its own tempo to the incoming tempo, incoming from the "
                f"outgoing tempo to its own; blend {seam:.2f}s heard, uses {out_cons:.2f}s of "
                f"outgoing and {in_cons:.2f}s of incoming"
            )
        else:
            plan.notes.append(
                f"beats matched over {n} beats without warping: {bpm_o:.1f} vs {bpm_i:.1f} BPM{fold} "
                f"drifts {drift * 1000:.1f} ms over the blend"
            )
        return plan
    return None


def plan_short(out_len: int, in_len: int, sr: int, g_in: Grid, lo: float, hi: float) -> SeamPlan | None:
    """~2-beat blend that ends exactly on an incoming beat, preferring a bar
    line (every 4th beat counted from the first detected beat) within one beat
    of the 2-beat target. No warp, so there is no time for the beats to drift."""
    p = g_in.period
    n_before = math.floor(g_in.first / p)
    in_pre = g_in.first - n_before * p
    target = SHORT_BLEND_BEATS * p
    lo = min(lo, 1.5 * p)  # a 2-beat cut may be shorter than the general 1 s minimum
    limit = min(hi, out_len / sr, in_len / sr)
    cands = []
    j = 0
    while in_pre + j * p <= limit + 1e-9:
        t = in_pre + j * p
        if t >= lo - 1e-9:
            cands.append((t, (j - n_before) % 4 == 0))
        j += 1
    if not cands:
        return None
    bars = [c for c in cands if c[1] and abs(c[0] - target) <= p + 1e-9]
    t, on_bar = min(bars or cands, key=lambda c: abs(c[0] - target))
    ov = int(round(t * sr))
    beats = [int(round((in_pre + j * p) * sr)) for j in range(int(t / p) + 2)]
    plan = SeamPlan(
        kind="short", o_start=out_len - ov, in_consumed=ov, seam_len=ov,
        beats=[b for b in beats if 0 <= b <= ov], target_seconds=target,
    )
    where = "an incoming bar line" if on_bar else "an incoming beat"
    plan.notes.append(f"short blend of {t:.2f}s ({t / p:.1f} beats at {g_in.bpm:.1f} BPM) ending on {where}")
    return plan
