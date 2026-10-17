"""Shared transition renderer (owner: Aarav).

ONE function powers both A/B previews and the final mix, so a preview is an
exact excerpt of what the final mix will contain at that seam.

Seam model (all times in seconds in the public contract):
  * The overlap always ends exactly at the outgoing track's last sample.
  * The incoming track always starts at its sample 0.
  * So the overlap region is outgoing[o_start:] blended with incoming[:ov].
  * mix.py stitches: outgoing[:o_start] + seam + incoming[ov:]
    which keeps every sample of both tracks (no truncation, no skipping).

Processing actually performed (and recorded in AppliedTransition):
  1. Clamp the requested overlap to a safe range (SafetyLimits).
  2. Optionally beat-snap: shift the overlap so the incoming track's first
     beat lands on an outgoing beat, staying inside the safe range.
  3. Per-track gain (outgoing_gain_db / incoming_gain_db, default 0 dB),
     applied to the seam and to preview context. The caller supplies one
     gain per TRACK (see loudness.py), and mix.py applies the same gain to
     the rest of that track, so level never jumps at a seam boundary.
  4. Fade out / fade in (equal-power or linear).
  5. Soft limiter on the seam only if its peak exceeds the threshold.
No EQ and no tempo change are applied.
"""
from dataclasses import dataclass

import numpy as np

from backend.api.schemas import AppliedTransition, RenderMode, TransitionRequest


@dataclass(frozen=True)
class SafetyLimits:
    min_overlap_seconds: float = 1.0
    max_overlap_seconds: float = 16.0
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


def _fades(n: int, style: str) -> tuple[np.ndarray, np.ndarray]:
    t = (np.arange(n, dtype=np.float64) + 0.5) / n  # midpoints, never exactly 0 or 1
    if style == "equal_power":
        fade_out, fade_in = np.cos(t * np.pi / 2), np.sin(t * np.pi / 2)
    elif style == "linear":
        fade_out, fade_in = 1.0 - t, t
    else:
        raise ValueError(f"unknown fade_style {style!r}")
    return fade_out.astype(np.float32)[:, None], fade_in.astype(np.float32)[:, None]


def _soft_limit(x: np.ndarray, threshold: float) -> np.ndarray:
    """Leave samples below threshold untouched; squash the rest toward 1.0."""
    headroom = 1.0 - threshold
    mag = np.abs(x)
    over = mag > threshold
    y = x.copy()
    y[over] = np.sign(x[over]) * (
        threshold + headroom * np.tanh((mag[over] - threshold) / headroom)
    )
    return y


# ---------------------------------------------------------------- public API


def gain_factor(gain_db: float) -> np.float32:
    """Linear float32 factor for a dB gain. mix.py uses this too, so the
    track body and its seams are scaled by the bit-identical factor."""
    return np.float32(10.0 ** (float(gain_db) / 20.0))


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
) -> tuple[np.ndarray, AppliedTransition]:
    """Render the seam between two tracks.

    preview: pre-roll from outgoing + the seam + post-roll from incoming.
             The seam samples are bit-identical to final mode.
    final:   only the overlapped seam (length = actual overlap). mix.py
             stitches full tracks around it using seam_sample_bounds().

    Audio may be (samples,), (samples, ch) or channels-first (ch, samples).
    Output is float32 shaped (samples, ch).

    outgoing_gain_db / incoming_gain_db: whole-track gain for each side,
    recorded as applied_gain_db. Pass the same value for a track at both of
    its seams and in its preview.
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

    # 2. beat snap
    chosen = clamped
    if request.beat_align:
        snapped = _beat_snap(
            clamped, out_dur, list(outgoing_beats_seconds),
            list(incoming_beats_seconds), lo, hi,
        )
        if snapped is None:
            notes.append("beat alignment skipped: no beats or no aligned overlap in safe range")
        else:
            chosen = snapped

    ov = int(round(chosen * sr))
    ov = max(1, min(ov, out.shape[0], inc.shape[0]))
    o_start = out.shape[0] - ov
    actual = ov / sr
    beat_adjust = actual - clamped if request.beat_align else 0.0

    # 3. per-track gain, 4. fades
    g_out, g_in = gain_factor(outgoing_gain_db), gain_factor(incoming_gain_db)
    fade_out, fade_in = _fades(ov, request.fade_style)
    seam = out[o_start:] * (fade_out * g_out) + inc[:ov] * (fade_in * g_in)

    # 5. soft limiter, only when needed
    peak = float(np.max(np.abs(seam)))
    if peak > safety.limiter_threshold:
        seam = _soft_limit(seam, safety.limiter_threshold)
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
        post = inc[ov:ov + ctx] * g_in
        audio = np.concatenate([pre, seam, post], axis=0).astype(np.float32)

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
        beat_alignment_adjustments_seconds=beat_adjust,
        applied_gain_db={"outgoing": float(outgoing_gain_db), "incoming": float(incoming_gain_db)},
        applied_eq=None,
        tempo_change=None,
        safety_adjustments=notes,
        output_duration_seconds=audio.shape[0] / sr,
    )
    return audio, applied


def seam_sample_bounds(applied: AppliedTransition, sr: int) -> tuple[int, int]:
    """(outgoing_start_sample, overlap_samples) for stitching in mix.py.

    Stitch as: outgoing[:o_start] + seam + incoming[overlap:].
    """
    o_start = int(round(applied.outgoing_start_seconds * sr))
    ov = int(round(applied.actual_overlap_seconds * sr))
    return o_start, ov
