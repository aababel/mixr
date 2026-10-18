"""Shared transition renderer (owner: Aarav).

ONE function powers both A/B previews and the final mix, so a preview is an
exact excerpt of what the final mix will contain at that seam.

Seam model (all times in seconds in the public contract):
  * The overlap always ends exactly at the outgoing track's last sample.
  * The incoming track always starts at its sample 0.
  * The seam blends outgoing[o_start:] with incoming[:consumed]; with a
    tempo glide both are time-warped to the seam length, which can differ
    from either.
  * mix.py stitches: outgoing[:o_start] + seam + incoming[consumed:]
    which keeps every sample of both tracks (no truncation, no skipping).

Processing actually performed (and recorded in AppliedTransition):
  1. Clamp the requested overlap to a safe range (SafetyLimits).
  2. Find a fade-out tail on the outgoing track and a fade-in head on the
     incoming one (edges.py). The seam is lengthened by both so the
     crossfade itself happens on full-level music: it starts after the
     incoming head and finishes where the outgoing tail begins. The edges
     still play inside the seam, under the other track.
     Plan the seam from both beat lists (seam_plan.py):
       - tempos within 15% (after half/double folding): match beats
         beat-for-beat. A request that lands on a whole bar (choose_windows
         aims it at the incoming song's next event) plays exactly that many
         bars; any other is rounded to a whole phrase (32 or 16 beats), else
         bars. If the grids would drift, glide the tempo (step 3)
       - tempos further apart, one side with no steady kick at the seam
         (pulse_curve): beatless blend of the requested length (up to that
         beatless stretch), no warp, nothing to clash
       - tempos further apart with a beat on both sides: echo-out (echo.py)
       - no steady beat grid: snap the first incoming beat onto an
         outgoing beat (no warp)
  3. Tempo glide: Rubber Band time map warps the outgoing side from its
     tempo to the incoming tempo, and the incoming side from the outgoing
     tempo to its own, so every matched beat lands together.
     tempo_change = incoming/outgoing tempo ratio when a warp is applied.
  4. Per-track gain (outgoing_gain_db / incoming_gain_db, from loudness.py),
     the same gain mix.py applies to the rest of each track.
  5. Split-frequency swap (bands.py) over the blend region, landing on the
     phrase boundary at its last matched beat: lows swap on that beat, highs
     halfway, mids cross over in the second half with fade_style. Recorded
     in applied_eq.
  6. Look-ahead gain limiter on the seam only if its peak exceeds the
     threshold: turned down around peaks, never clipped.

Vocal clash: when melody data is passed and BOTH sides carry lead melody at
the seam (outgoing's last 8 s, incoming's first 8 s), the blend's mids swap
on a beat at the bass swap instead of crossfading (bands.py, mid_swap). If
the beats can't be matched (even with a beatless side: two melodies don't
blend without a beat to swap on), steps 2-5 are replaced by an echo-out (echo.py): A plays its last bars with the bass
filtered out, its last beat echoes, and B drops on a downbeat at A's bar
line, under the echo tail.

Drop (drop.py): request.style == "drop" (choose_windows found a drop in the
incoming song; requested_overlap_seconds is its build). The outgoing song is
swept up to the incoming drop, which lands on a matched beat at full level;
with tempos too far apart it is a cut with a roll and a beat of silence.
Takes precedence over the vocal swap.
"""
from dataclasses import dataclass

import numpy as np
from scipy.ndimage import minimum_filter1d, uniform_filter1d

from backend.api.schemas import AppliedTransition, RenderMode, TransitionRequest
from backend.audio.bands import band_swap, mid_window
from backend.audio.drop import DropPlan, plan_drop_cut, render_drop
from backend.audio.echo import (
    MELODY_CLASH, last_bar_end, melody_clash, plan_echo_out, render_echo_out,
)
from backend.audio.edges import MAX_EDGE_SECONDS, fade_in_seconds, fade_out_seconds
from backend.audio.seam_plan import (
    GRID_WINDOW_MARGIN_SECONDS, MAX_GLIDE_DEVIATION, PULSE_HOP_SECONDS, SeamPlan, beat_period,
    beatless_run, deviation, fit_grid, fold_ratio, kick_grid, plan_beatless, plan_matched, plan_short,
    pulse_curve,
)
from backend.audio.stretch import engine_name, rubberband_available, timemap_stretch

MIN_BLEND_SECONDS = 1.0  # blend region shorter than this: edges are ignored
WARP_CONTEXT_SECONDS = 0.25  # extra audio fed to Rubber Band beyond each warped edge
EDGE_BLEND_SECONDS = 0.010  # crossfade between untouched track and warped seam edge
# A beat grid that can't be found over the blend's own span is retried over
# this much audio: a breakdown can hide the kick for half a minute.
GRID_RETRY_SECONDS = 64.0
# Tempos too far apart to match: a side with no steady kick for at least this
# long at the seam gets a beatless blend instead of an echo-out.
BEATLESS_MIN_SECONDS = 4.0
LIMIT_HOLD_SECONDS = 0.030  # limiter look-ahead and hold either side of a peak
LIMIT_EDGE_SECONDS = 0.010
PULSE_WINDOW_MARGIN_SECONDS = 12.0  # extra audio for measuring the pulse when analysis stored none


@dataclass(frozen=True)
class SafetyLimits:
    min_overlap_seconds: float = 1.0
    max_overlap_seconds: float = 32.0
    max_fraction_of_track: float = 0.5  # overlap may not exceed this share of either track
    limiter_threshold: float = 0.9  # seam peak above this triggers the soft limiter


DEFAULT_SAFETY = SafetyLimits()


# ---------------------------------------------------------------- helpers


def _to_frames(audio: np.ndarray, name: str) -> np.ndarray:
    """Return float32 audio shaped (samples, channels)."""
    a = np.asarray(audio, dtype=np.float32)
    if a.ndim == 1:
        a = a[:, None]
    elif a.ndim == 2:
        # librosa uses channels-first (ch, samples); detect and transpose.
        if a.shape[0] <= 8 and a.shape[1] > a.shape[0]:
            a = a.T
    else:
        raise ValueError(f"{name} audio must be 1-D or 2-D, got shape {a.shape}")
    if a.shape[0] == 0:
        raise ValueError(f"{name} audio is empty")
    return np.ascontiguousarray(a)


def _match_channels(out: np.ndarray, inc: np.ndarray, notes: list[str]):
    if out.shape[1] == inc.shape[1]:
        return out, inc
    if out.shape[1] == 1:
        notes.append(f"outgoing upmixed from mono to {inc.shape[1]} channels")
        return np.repeat(out, inc.shape[1], axis=1), inc
    if inc.shape[1] == 1:
        notes.append(f"incoming upmixed from mono to {out.shape[1]} channels")
        return out, np.repeat(inc, out.shape[1], axis=1)
    raise ValueError(
        f"channel mismatch: outgoing {out.shape[1]}, incoming {inc.shape[1]}"
    )


def _seam_cap(out_dur: float, in_dur: float, extra: float, safety: SafetyLimits) -> float:
    """Longest whole seam: the blend limit plus the fade edges, and never more
    than max_fraction_of_track of either track."""
    return min(
        safety.max_overlap_seconds + extra,
        safety.max_fraction_of_track * out_dur,
        safety.max_fraction_of_track * in_dur,
    )


def _blend_region(plan: SeamPlan, sr: int, out_len: int, head: float, tail: float) -> tuple[int, int]:
    """(r0, r1) in seam samples: the crossfade starts where the incoming head
    has faded up and ends where the outgoing tail begins, each moved onto the
    nearest matched beat inside that range."""
    out_cons = out_len - plan.o_start
    h, t = int(round(head * sr)), int(round(tail * sr))

    def to_seam(points, x):
        if not points:
            return float(x)
        src, dst = zip(*points)
        return float(np.interp(x, src, dst))

    r0 = to_seam(plan.in_map, h) if h else 0.0
    r1 = to_seam(plan.out_map, out_cons - t) if t else float(plan.seam_len)
    inside = [b for b in plan.beats if r0 - 1 <= b <= r1 + 1]
    if h and inside:
        r0 = min(inside)
    if t and inside:
        r1 = max(inside)
    return int(round(r0)), int(round(min(r1, plan.seam_len)))


def _safe_range(out_dur: float, in_dur: float, safety: SafetyLimits) -> tuple[float, float]:
    hi = min(
        safety.max_overlap_seconds,
        safety.max_fraction_of_track * out_dur,
        safety.max_fraction_of_track * in_dur,
    )
    lo = min(safety.min_overlap_seconds, hi)  # very short tracks: lo collapses to hi
    return lo, hi


def _beat_snap(
    target: float,
    out_dur: float,
    out_beats: list[float],
    in_beats: list[float],
    lo: float,
    hi: float,
) -> float | None:
    """Overlap length that puts incoming's first beat on an outgoing beat.

    Incoming sample 0 sits at outgoing time (out_dur - ov), so its first beat
    lands at out_dur - ov + in_beats[0]. Setting that equal to an outgoing beat
    b gives ov = out_dur - b + in_beats[0]. Pick the candidate in [lo, hi]
    closest to the target. Returns None if no candidate fits.
    """
    if not out_beats or not in_beats:
        return None
    first_in = in_beats[0]
    cands = [out_dur - b + first_in for b in out_beats]
    cands = [c for c in cands if lo - 1e-9 <= c <= hi + 1e-9]
    if not cands:
        return None
    return min(cands, key=lambda c: abs(c - target))


def _fades(n: int, style: str, region: tuple[int, int] | None = None) -> tuple[np.ndarray, np.ndarray]:
    """Crossfade curves over n samples. With region (r0, r1) the fade runs
    over [r0, r1) only: outgoing 1 / incoming 0 before, 0 / 1 after."""
    r0, r1 = region if region else (0, n)
    m = max(1, r1 - r0)
    t = (np.clip(np.arange(n, dtype=np.float64) - r0, -0.5, m - 0.5) + 0.5) / m
    if style == "equal_power":
        fade_out, fade_in = np.cos(t * np.pi / 2), np.sin(t * np.pi / 2)
    elif style == "linear":
        fade_out, fade_in = 1.0 - t, t
    else:
        raise ValueError(f"unknown fade_style {style!r}")
    return fade_out.astype(np.float32)[:, None], fade_in.astype(np.float32)[:, None]


def limit_peaks(x: np.ndarray, threshold: float, sr: int,
                edge_seconds: float = LIMIT_EDGE_SECONDS) -> np.ndarray:
    """Look-ahead gain limiter: the audio is turned down around each peak
    instead of clipped, so loud passages keep their shape. With edge_seconds
    the gain eases back to 1 over the first and last edge_seconds, so a seam
    meets the unlimited track bodies without a step."""
    mag = np.max(np.abs(x), axis=1)
    need = np.minimum(1.0, threshold / np.maximum(mag, 1e-12))
    hold = max(2, int(round(LIMIT_HOLD_SECONDS * sr)))
    gain = minimum_filter1d(need, size=2 * hold + 1, mode="nearest")
    # two box passes reach at most `hold` samples, inside every min window, so gain(t) <= need(t)
    box = 2 * (hold // 2) + 1
    gain = uniform_filter1d(uniform_filter1d(gain, box, mode="nearest"), box, mode="nearest")
    edge = min(int(round(edge_seconds * sr)), x.shape[0] // 2)
    if edge:
        ramp = np.linspace(0.0, 1.0, edge)
        gain[:edge] = 1.0 - (1.0 - gain[:edge]) * ramp
        gain[-edge:] = 1.0 - (1.0 - gain[-edge:]) * ramp[::-1]
    return x * gain.astype(np.float32)[:, None]


# ---------------------------------------------------------------- public API


def gain_factor(gain_db: float) -> np.float32:
    """Linear float32 factor for a dB gain. mix.py uses this too, so the
    track body and its seams are scaled by the bit-identical factor."""
    return np.float32(10.0 ** (float(gain_db) / 20.0))


def _grid(audio: np.ndarray, sr: int, beats: list[float], t_lo: float, t_hi: float):
    """Beat grid near a seam: locked to the kicks in the audio when there is a
    clear kick pulse (tracker beats only seed the tempo), else fitted to the
    tracker's beats. Returns (Grid or None, description of the source)."""
    seed = beat_period(beats, t_lo, t_hi)
    if seed is not None:
        mono = audio[int(t_lo * sr):int(t_hi * sr)].mean(axis=1)
        g = kick_grid(mono, sr, t_lo, seed)
        if g is not None:
            return g, "kick onsets"
    return fit_grid(beats, t_lo, t_hi), "tracked beats"


def _plan_seam(
    out: np.ndarray, inc: np.ndarray, sr: int,
    out_beats: list[float], in_beats: list[float],
    request: TransitionRequest, clamped: float, lo: float, hi: float, notes: list[str],
    head: float = 0.0, tail: float = 0.0, hi_all: float | None = None,
    beatless: tuple[float, float] = (0.0, 0.0),
) -> SeamPlan | None:
    """Choose glide / beat_matched / beatless / short / snapped / plain (see
    seam_plan.py). None: the tempos are too far apart and both sides have a
    beat, so the seam is an echo-out.

    clamped, lo, hi describe the blend itself; head / tail (seconds of fade on
    the incoming start / outgoing end) are added on top for the whole seam.
    beatless: (outgoing, incoming) seconds with no steady kick at the seam."""
    out_len, in_len = out.shape[0], inc.shape[0]
    out_dur, in_dur = out_len / sr, in_len / sr

    def identity(kind: str, seconds: float, beats: list[int] | None = None) -> SeamPlan:
        ov = max(1, min(int(round(seconds * sr)), out_len, in_len))
        return SeamPlan(kind=kind, o_start=out_len - ov, in_consumed=ov, seam_len=ov,
                        beats=[b for b in (beats or []) if 0 <= b <= ov], target_seconds=seconds)

    if not request.beat_align:
        return identity("plain", clamped)
    extra = head + tail
    target_all = clamped + extra
    hi_all = min(hi + extra, out_dur, in_dur) if hi_all is None else hi_all
    lo_all = min(lo + extra, hi_all)

    span = hi_all + GRID_WINDOW_MARGIN_SECONDS
    g_out, src_out = _grid(out, sr, out_beats, max(0.0, out_dur - span), out_dur)
    g_in, src_in = _grid(inc, sr, in_beats, 0.0, min(in_dur, span))
    wide = max(span, GRID_RETRY_SECONDS)
    if g_out is None and out_dur > span:
        g_out, src_out = _grid(out, sr, out_beats, max(0.0, out_dur - wide), out_dur)
    if g_in is None and in_dur > span:
        g_in, src_in = _grid(inc, sr, in_beats, 0.0, min(in_dur, wide))
    if g_out and g_in:
        notes.append(f"beat grids: outgoing {g_out.bpm:.2f} BPM from {src_out}, "
                     f"incoming {g_in.bpm:.2f} BPM from {src_in}")
    if g_out and g_in:
        ratio, _ = fold_ratio(g_out.period, g_in.period)
        if deviation(ratio) <= MAX_GLIDE_DEVIATION:
            plan = plan_matched(out_len, in_len, sr, g_out, g_in, target_all, lo_all, hi_all,
                                rubberband_available(), after_land=tail)
            if isinstance(plan, SeamPlan):
                return plan
            if plan == "needs_warp":
                short = plan_short(out_len, in_len, sr, g_in, lo, hi_all, head, tail)
                if short:
                    short.notes.insert(0, f"tempo glide skipped: Rubber Band is not installed "
                                          f"({g_out.bpm:.1f} -> {g_in.bpm:.1f} BPM would drift); "
                                          f"requested overlap not used")
                    return short
            else:
                notes.append("beat-matched blend does not fit the safe overlap range")
        else:
            why = (f"tempos {g_out.bpm:.1f} and {g_in.bpm:.1f} BPM differ by {deviation(ratio):.0%} "
                   f"after half/double folding (glide limit {MAX_GLIDE_DEVIATION:.0%})")
            run_out, run_in = beatless
            if max(run_out, run_in) >= BEATLESS_MIN_SECONDS:
                side = "outgoing" if run_out >= run_in else "incoming"
                seconds = min(clamped, max(run_out, run_in))
                ov = min(seconds + head + tail, hi_all, out_dur, in_dur)
                beats = (list(in_beats) if side == "outgoing"
                         else [b - (out_dur - ov) for b in out_beats])
                plan = plan_beatless(out_len, in_len, sr, seconds, head, tail, beats, hi_all, side)
                plan.notes.insert(0, f"{why}: no warp")
                return plan
            notes.append(f"{why}, and both sides have a beat: echo-out instead of a blend")
            return None
    elif out_beats and in_beats:
        notes.append("beat grid too irregular to match beat-for-beat; only the first beat is aligned")

    # Fallback: previous behaviour, first incoming beat snapped onto an outgoing beat.
    snapped = _beat_snap(target_all, out_dur, list(out_beats), list(in_beats), lo_all, hi_all)
    if snapped is None:
        notes.append("beat alignment skipped: no beats or no aligned overlap in safe range")
        return identity("plain", clamped)
    return identity("snapped", snapped, [int(round(b * sr)) for b in in_beats])


def _edge_blend(warped: np.ndarray, original: np.ndarray, at_start: bool, sr: int) -> None:
    """Crossfade a few ms between the untouched track and its warped seam
    segment, where the warp rate is 1, so the seam joins the body seamlessly."""
    n = min(int(round(EDGE_BLEND_SECONDS * sr)), warped.shape[0] // 4, original.shape[0])
    if n < 2:
        return
    w = ((np.arange(n) + 0.5) / n).astype(np.float32)[:, None]
    if at_start:
        warped[:n] = original[:n] * (1 - w) + warped[:n] * w
    else:
        warped[-n:] = warped[-n:] * (1 - w) + original[-n:] * w


def _warp_outgoing(out: np.ndarray, plan: SeamPlan, sr: int) -> np.ndarray:
    ctx = min(int(round(WARP_CONTEXT_SECONDS * sr)), plan.o_start)  # warm-up audio for Rubber Band
    seg = out[plan.o_start - ctx:]
    tmap = [(0, 0)] + [(s + ctx, t + ctx) for s, t in plan.out_map if (s, t) != (0, 0) or ctx]
    tmap = [p for i, p in enumerate(tmap) if i == 0 or p != tmap[i - 1]]
    warped = timemap_stretch(seg, sr, tmap, plan.seam_len + ctx)[ctx:].copy()
    _edge_blend(warped, out[plan.o_start:], True, sr)
    return warped


def _warp_incoming(inc: np.ndarray, plan: SeamPlan, sr: int) -> np.ndarray:
    ctx = min(int(round(WARP_CONTEXT_SECONDS * sr)), inc.shape[0] - plan.in_consumed)
    seg = inc[:plan.in_consumed + ctx]
    tmap = list(plan.in_map)
    if ctx:
        tmap.append((plan.in_consumed + ctx, plan.seam_len + ctx))
    warped = timemap_stretch(seg, sr, tmap, plan.seam_len + ctx)[:plan.seam_len].copy()
    _edge_blend(warped, inc[:plan.in_consumed], False, sr)
    return warped


def render_transition(
    outgoing: np.ndarray,
    incoming: np.ndarray,
    sr: int,
    outgoing_beats_seconds: list[float],
    incoming_beats_seconds: list[float],
    request: TransitionRequest,
    mode: RenderMode,
    preview_context_seconds: float = 6.0,
    *,
    safety: SafetyLimits = DEFAULT_SAFETY,
    outgoing_gain_db: float = 0.0,
    incoming_gain_db: float = 0.0,
    outgoing_melody: list[float] | None = None,
    incoming_melody: list[float] | None = None,
    melody_hop_seconds: float | None = None,
    outgoing_fade_out_seconds: float | None = None,
    incoming_fade_in_seconds: float | None = None,
    outgoing_pulse: list[float] | None = None,
    incoming_pulse: list[float] | None = None,
    pulse_hop_seconds: float | None = None,
) -> tuple[np.ndarray, AppliedTransition]:
    """Render the seam between two tracks.

    preview: pre-roll from outgoing + the seam + post-roll from incoming.
             The seam samples are bit-identical to final mode.
    final:   only the overlapped seam. mix.py stitches full tracks around it
             using seam_sample_bounds().

    Audio may be (samples,), (samples, ch) or channels-first (ch, samples).
    Output is float32 shaped (samples, ch).

    outgoing_gain_db / incoming_gain_db: whole-track gain for each side,
    recorded as applied_gain_db. Pass the same value for a track at both of
    its seams and in its preview.

    outgoing_melody / incoming_melody: Track.melody_activity of each side
    (already sliced to the played window), on a melody_hop_seconds grid.
    When both edges are melodic the seam is an echo-out (echo.py). Leave
    them None to never echo out.

    outgoing_fade_out_seconds / incoming_fade_in_seconds: fade lengths stored
    by analysis (track.metadata) for a side that plays to its real end / from
    its real start, so the renderer and choose_windows use the identical
    value. None = measure from the audio here (edges.py), as before.

    outgoing_pulse / incoming_pulse: the kick pulse analysis stored
    (track.metadata["pulse"], sliced to the window) on pulse_hop_seconds.
    With tempos too far apart to match, a side with no steady kick at the
    seam gets a beatless blend, else the seam echoes out. None = measure from
    the audio here (seam_plan.pulse_curve).

    In the result, actual_overlap_seconds is how much of the INCOMING track
    plays inside the seam (what mix.py skips afterwards). With a tempo glide
    the seam itself is a different length: in final mode that is
    output_duration_seconds.
    """
    if sr <= 0:
        raise ValueError("sr must be positive")
    if mode not in ("preview", "final"):
        raise ValueError(f"unknown mode {mode!r}")

    notes: list[str] = []
    out = _to_frames(outgoing, "outgoing")
    inc = _to_frames(incoming, "incoming")
    out, inc = _match_channels(out, inc, notes)

    out_dur = out.shape[0] / sr
    in_dur = inc.shape[0] / sr
    requested = float(request.requested_overlap_seconds)

    # 1. clamp to safe range
    lo, hi = _safe_range(out_dur, in_dur, safety)
    clamped = min(max(requested, lo), hi)
    changed_for_safety = abs(clamped - requested) > 1e-9
    if changed_for_safety:
        notes.append(
            f"overlap clamped from {requested:.3f}s to {clamped:.3f}s "
            f"(safe range {lo:.3f}-{hi:.3f}s)"
        )

    # 2. fade edges, then plan: beat matching, tempo glide or short blend
    tail = head = 0.0
    if request.beat_align:
        tail = (_stored_edge(outgoing_fade_out_seconds, out_dur)
                if outgoing_fade_out_seconds is not None else fade_out_seconds(out, sr))
        head = (_stored_edge(incoming_fade_in_seconds, in_dur)
                if incoming_fade_in_seconds is not None else fade_in_seconds(inc, sr))
    hi_all = _seam_cap(out_dur, in_dur, head + tail, safety)
    if head + tail > 0 and hi_all - (head + tail) < lo:
        notes.append(f"fade edges ignored (outgoing tail {tail:.2f}s, incoming head {head:.2f}s): "
                     f"tracks too short to fit them")
        head = tail = 0.0
        hi_all = hi
    g_out, g_in = gain_factor(outgoing_gain_db), gain_factor(incoming_gain_db)
    clash, mel_a, mel_b = (melody_clash(outgoing_melody, incoming_melody, melody_hop_seconds,
                                        out_dur, tail, head)
                           if request.beat_align else (False, None, None))
    if mel_a is not None and mel_b is not None:
        notes.append(f"melody at the seam: outgoing {mel_a:.2f}, incoming {mel_b:.2f} "
                     f"(echo-out when both >= {MELODY_CLASH})")
    if request.style == "drop" and request.beat_align:
        # The build runs on full-level music: the outgoing fade-out is cut, not swept.
        cut = int(round(tail * sr))
        if cut and out.shape[0] - cut > int(hi * sr):
            notes.append(f"drop: outgoing's {tail:.2f}s fade-out cut so the build is on full-level music")
            out = out[:out.shape[0] - cut]
        seam, plan, eq_desc = _drop(out, inc, sr, list(outgoing_beats_seconds),
                                    list(incoming_beats_seconds), request, lo, hi, notes, g_out, g_in)
        notes.extend(plan.notes)
        changed_for_safety = False
        region = None
    else:
        beatless = (_beatless_runs(out, inc, sr, outgoing_beats_seconds, incoming_beats_seconds,
                                   outgoing_pulse, incoming_pulse, pulse_hop_seconds, head, tail, hi)
                    if request.beat_align else (0.0, 0.0))
        blended = _blend(out, inc, sr, outgoing_beats_seconds, incoming_beats_seconds, request,
                         clamped, lo, hi, notes, head, tail, hi_all, g_out, g_in, mid_swap=clash,
                         beatless=beatless)
        if blended is not None:
            seam, plan, eq_desc, region = blended
            if clash:
                notes.append("both sides have lead melody at the seam: the vocals swap on a beat "
                             "instead of crossfading, so two melodies are never loud together")
        else:
            cut = int(round(tail * sr))
            if cut and out.shape[0] - cut > int(hi * sr):
                notes.append(f"echo-out: outgoing's {tail:.2f}s fade-out cut so it echoes out of "
                             f"full-level music")
                out = out[:out.shape[0] - cut]
            seam, plan, eq_desc = _echo_out(out, inc, sr, list(outgoing_beats_seconds),
                                            list(incoming_beats_seconds), safety, g_out, g_in, clamped, head)
            notes.extend(plan.notes)
            why = ("both sides have lead melody at the seam and the beats can't be matched for a "
                   "vocal swap" if clash else "the beats can't be matched for a blend")
            notes.append(f"requested overlap sets the echo-out's length instead of a blend: {why}")
            changed_for_safety = clash  # far-apart tempos: windows.py planned this echo-out
            region = None
    o_start, in_cons, seam_len = plan.o_start, plan.in_consumed, plan.seam_len

    # 6. soft limiter, only when needed
    peak = float(np.max(np.abs(seam)))
    if peak > safety.limiter_threshold:
        seam = limit_peaks(seam, safety.limiter_threshold, sr)
        new_peak = float(np.max(np.abs(seam)))
        notes.append(
            f"soft limiter on seam: peak {peak:.3f} -> {new_peak:.3f} "
            f"(threshold {safety.limiter_threshold})"
        )
    seam = seam.astype(np.float32)

    if mode == "final":
        audio = seam
    else:
        ctx = int(round(max(0.0, preview_context_seconds) * sr))
        pre = out[max(0, o_start - ctx):o_start] * g_out
        post = inc[in_cons:in_cons + ctx] * g_in
        audio = np.concatenate([pre, seam, post], axis=0).astype(np.float32)

    actual = in_cons / sr
    adjust = 0.0
    if plan.kind not in ("plain", "echo_out", "drop", "drop_cut"):
        adjust = actual - clamped - (head + tail if region else 0.0)
    applied = AppliedTransition(
        from_id=request.from_id,
        to_id=request.to_id,
        mode=mode,
        fade_style=request.fade_style,
        requested_overlap_seconds=requested,
        actual_overlap_seconds=actual,
        overlap_changed_for_safety=changed_for_safety,
        outgoing_start_seconds=o_start / sr,
        incoming_start_seconds=0.0,
        beat_alignment_adjustments_seconds=adjust,
        applied_gain_db={"outgoing": float(outgoing_gain_db), "incoming": float(incoming_gain_db)},
        applied_eq=eq_desc,
        tempo_change=plan.tempo_ratio if plan.warped else None,
        safety_adjustments=notes,
        output_duration_seconds=audio.shape[0] / sr,
    )
    return audio, applied


def _stored_edge(seconds: float, dur: float) -> float:
    """A fade length from analysis, kept inside what edges.py could return."""
    return float(min(max(0.0, seconds), MAX_EDGE_SECONDS, dur))


def _beatless_runs(out, inc, sr, out_beats, in_beats, out_pulse, in_pulse, hop, head, tail,
                   limit) -> tuple[float, float]:
    """(seconds before the outgoing fade tail, seconds after the incoming fade
    head) with no steady kick, up to limit. From the stored pulse curves, else
    measured from the audio near the seam."""
    out_dur, in_dur = out.shape[0] / sr, inc.shape[0] / sr
    runs = []
    for curve, audio, beats, at_end in ((out_pulse, out, out_beats, True), (in_pulse, inc, in_beats, False)):
        dur = out_dur if at_end else in_dur
        t = dur - tail if at_end else head
        step = hop
        if not curve:
            seed = beat_period(list(beats), 0.0, dur)
            if seed is None:
                runs.append(0.0)
                continue
            span = min(dur, limit + PULSE_WINDOW_MARGIN_SECONDS)
            lo = dur - span if at_end else 0.0
            mono = audio[int(lo * sr):int((lo + span) * sr)].mean(axis=1)
            curve, step = pulse_curve(mono, sr, seed, PULSE_HOP_SECONDS), PULSE_HOP_SECONDS
            t -= lo
        runs.append(beatless_run(curve, step, t, forward=not at_end, limit=limit))
    return runs[0], runs[1]


def _echo_out(out, inc, sr, out_beats, in_beats, safety, g_out, g_in, requested, head=0.0):
    """Echo-out seam (echo.py): (seam, plan, eq description). requested: the
    seconds of the outgoing song to filter out before its last bar line;
    head: the incoming song's quiet lead-in."""
    out_dur, in_dur = out.shape[0] / sr, inc.shape[0] / sr
    grid_out, _ = _grid(out, sr, out_beats, max(0.0, out_dur - 20.0), out_dur)
    grid_in, _ = _grid(inc, sr, in_beats, 0.0, min(in_dur, 20.0))
    first = grid_in.first if grid_in is not None else (in_beats[0] if in_beats else 0.0)
    plan = plan_echo_out(out.shape[0], inc.shape[0], sr, grid_out, first,
                         safety.max_fraction_of_track, requested, head)
    seam = render_echo_out(out, inc, sr, plan, g_out, g_in)
    bars = f"last {plan.bars} bars" if plan.bars > 1 else "last bar"
    eq = (f"echo-out: outgoing lows (<200 Hz) ramped out over its {bars}; last beat echoed "
          f"{plan.repeats}x on a 1-beat delay, high-passed at 300 Hz, each repeat "
          f"{-20 * np.log10(plan.feedback):.0f} dB quieter")
    return seam, plan, eq


def _drop(out, inc, sr, out_beats, in_beats, request, lo, hi, notes, g_out, g_in):
    """Drop seam (drop.py): (seam, plan, eq description). Layered on a
    beat-matched plan when the incoming drop falls on one of its matched
    beats, else a cut on the outgoing song's own grid."""
    drop_rel = float(request.requested_overlap_seconds)  # the build: incoming start to its drop
    out_len, in_len = out.shape[0], inc.shape[0]
    if drop_rel <= hi:
        plan = _plan_seam(out, inc, sr, out_beats, in_beats, request, drop_rel, lo, hi, [])
        if plan is not None and plan.kind in ("glide", "beat_matched") and len(plan.beats) >= 2:
            x = drop_rel * sr
            if plan.in_map:
                src, dst = zip(*plan.in_map)
                x = float(np.interp(x, src, dst))
            beats = np.asarray(plan.beats)
            period = float(np.median(np.diff(beats)))
            near = int(beats[np.argmin(np.abs(beats - x))])
            # The drop comes from the tracker's phrase grid; the seam's beats are
            # locked to the kicks, which can sit up to half a beat away from it.
            if abs(near - x) < period / 2 and 2 * period <= near <= plan.seam_len:
                out_seg = _warp_outgoing(out, plan, sr) if plan.out_map else out[plan.o_start:]
                in_seg = _warp_incoming(inc, plan, sr) if plan.in_map else inc[:plan.in_consumed]
                dp = DropPlan(kind="drop", o_start=plan.o_start, in_consumed=plan.in_consumed,
                              seam_len=plan.seam_len, out_map=plan.out_map, in_map=plan.in_map,
                              beats=plan.beats, tempo_ratio=plan.tempo_ratio,
                              target_seconds=drop_rel, notes=list(plan.notes),
                              drop_at=near, period=period / sr, layered=True)
                seam, desc = render_drop(out_seg, in_seg, sr, dp, float(g_out), float(g_in))
                return seam, dp, desc
            notes.append("drop: the incoming drop isn't on a matched beat of the blend, so it's a cut")
    out_dur = out_len / sr
    grid_out, _ = _grid(out, sr, out_beats, max(0.0, out_dur - 20.0), out_dur)
    end_t, period = last_bar_end(out_dur, grid_out)
    dp = plan_drop_cut(out_len, in_len, sr, end_t, period, drop_rel)
    in_seg = np.zeros((dp.seam_len, inc.shape[1]), dtype=np.float32)
    in_seg[dp.drop_at:] = inc[dp.in_consumed - (dp.seam_len - dp.drop_at):dp.in_consumed]
    seam, desc = render_drop(out[dp.o_start:], in_seg, sr, dp, float(g_out), float(g_in))
    return seam, dp, desc


def _blend(out, inc, sr, outgoing_beats_seconds, incoming_beats_seconds, request,
           clamped, lo, hi, notes, head, tail, hi_all, g_out, g_in, mid_swap=False,
           beatless=(0.0, 0.0)):
    """Normal blend (steps 2-5): (seam, plan, eq description, region). None
    when the seam should echo out instead: both sides have beats that can't be
    matched, or (mid_swap: both sides carry melody) only a short or snapped
    plan fits."""
    plan_notes: list[str] = []
    plan = _plan_seam(out, inc, sr, list(outgoing_beats_seconds),
                      list(incoming_beats_seconds), request, clamped, lo, hi, plan_notes,
                      head, tail, hi_all, beatless)
    if plan is None:
        notes.extend(plan_notes)
        return None
    if mid_swap and plan.kind not in ("glide", "beat_matched", "plain"):
        return None  # beats that don't line up: a long blend would trainwreck
    notes.extend(plan_notes)
    notes.extend(plan.notes)
    o_start, in_cons, seam_len = plan.o_start, plan.in_consumed, plan.seam_len

    region = None
    if plan.kind != "plain" and head + tail > 0:
        r0, r1 = _blend_region(plan, sr, out.shape[0], head, tail)
        if r1 - r0 >= MIN_BLEND_SECONDS * sr or r1 - r0 >= 0.8 * (plan.seam_len - (head + tail) * sr):
            region = (r0, r1)
            edge = []
            if tail:
                edge.append(f"outgoing fades out over its last {tail:.2f}s")
            if head:
                edge.append(f"incoming fades in over its first {head:.2f}s")
            notes.append(f"{' and '.join(edge)}: seam lengthened to cover the fade, crossfade "
                         f"runs {r0 / sr:.2f}-{r1 / sr:.2f}s of the {seam_len / sr:.2f}s seam so "
                         f"the fade plays under the other track instead of inside the blend")
        else:
            notes.append(f"fade edges ignored: blend region would be {(r1 - r0) / sr:.2f}s")

    # 3. time-warp each side onto the shared output beat grid
    out_seg = _warp_outgoing(out, plan, sr) if plan.out_map else out[o_start:]
    in_seg = _warp_incoming(inc, plan, sr) if plan.in_map else inc[:in_cons]
    if plan.warped:
        notes.append(f"time-warped with {engine_name()} time map")

    # 4. per-track gain, 5. split-frequency swap with the mids crossfaded
    fade_out, fade_in = _fades(seam_len, request.fade_style, mid_window(seam_len, plan.beats, sr, region))
    blend, eq_desc = band_swap(out_seg * g_out, in_seg * g_in, sr, plan.beats, fade_out, fade_in,
                               region, mid_swap=mid_swap)
    eq_desc = eq_desc.replace("mids crossed over", f"mids crossed over ({request.fade_style})")
    return blend.astype(np.float32), plan, eq_desc, region


def seam_sample_bounds(applied: AppliedTransition, sr: int) -> tuple[int, int]:
    """(outgoing_start_sample, incoming_samples_consumed) for stitching in mix.py.

    Stitch as: outgoing[:o_start] + seam + incoming[consumed:]. The second
    value is how much of the incoming track the seam used, NOT the seam
    length; they differ when the seam is tempo-glided. actual_overlap_seconds
    holds whole samples (consumed / sr), so rounding recovers them exactly.
    """
    o_start = int(round(applied.outgoing_start_seconds * sr))
    ov = int(round(applied.actual_overlap_seconds * sr))
    return o_start, ov
