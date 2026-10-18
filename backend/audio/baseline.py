"""Plain crossfade baseline for the before/after demo (owner: Aarav).

This is what an ordinary music player does at a song change: the last N
seconds of the outgoing track fade out while the first N seconds of the
incoming track fade in. No beat alignment, tempo glide, EQ swap, cue points,
echo-out or drop. It exists only so judges can hear the same song pair
handled two ways; it is never used in a Mixr mix.

To keep the comparison fair it gets the same things Mixr's preview gets that
are not "intelligence": the same decoded audio (loading.load_render_audio),
the same per-track loudness gain (loudness.track_gain_for_path), the same
equal-power curve, the same context around the seam
(preview.PREVIEW_CONTEXT_SECONDS) and the same output leveling
(preview.level_preview), so neither version wins by being louder.
"""
from pathlib import Path

import numpy as np
import soundfile as sf

from backend.audio.loading import RENDER_SR, load_render_audio
from backend.audio.loudness import track_gain_for_path
from backend.audio.preview import PREVIEW_CONTEXT_SECONDS, level_preview
from backend.audio.transition import _fades

CLIP_CEILING = 0.98


def plain_crossfade(
    outgoing: np.ndarray,
    incoming: np.ndarray,
    sr: int,
    seconds: float,
    context_seconds: float = PREVIEW_CONTEXT_SECONDS,
    outgoing_gain_db: float = 0.0,
    incoming_gain_db: float = 0.0,
) -> tuple[np.ndarray, dict]:
    """Excerpt around an end-to-start equal-power crossfade.

    outgoing / incoming: (frames, channels) float32 at sr.
    Returns (audio, applied) where applied says what was actually done, in seconds.
    The overlap is shortened, and that is recorded, if either track is shorter than it.
    """
    out = np.asarray(outgoing, dtype=np.float32)
    inc = np.asarray(incoming, dtype=np.float32)
    if out.ndim == 1:
        out = out[:, None]
    if inc.ndim == 1:
        inc = inc[:, None]
    if out.shape[1] != inc.shape[1]:
        ch = max(out.shape[1], inc.shape[1])
        out = np.repeat(out, ch // out.shape[1], axis=1)
        inc = np.repeat(inc, ch // inc.shape[1], axis=1)

    notes = []
    n = int(round(seconds * sr))
    n_max = min(len(out), len(inc))
    if n > n_max:
        n = n_max
        notes.append(f"crossfade shortened to {n / sr:.2f} s (track shorter than requested)")
    n = max(n, 1)
    ctx = int(round(context_seconds * sr))

    g_out = np.float32(10.0 ** (outgoing_gain_db / 20.0))
    g_in = np.float32(10.0 ** (incoming_gain_db / 20.0))
    head_start = max(0, len(out) - n - ctx)
    head = out[head_start:len(out) - n] * g_out
    fade_out, fade_in = _fades(n, "equal_power")
    seam = out[len(out) - n:] * g_out * fade_out + inc[:n] * g_in * fade_in
    tail = inc[n:n + ctx] * g_in
    audio = np.concatenate([head, seam, tail]).astype(np.float32)

    peak = float(np.max(np.abs(audio))) if audio.size else 0.0
    if peak > CLIP_CEILING:
        audio *= np.float32(CLIP_CEILING / peak)
        notes.append(f"scaled by {20 * np.log10(CLIP_CEILING / peak):+.2f} dB to avoid clipping")

    applied = {
        "style": "plain end-to-start crossfade",
        "fade_style": "equal_power",
        "requested_overlap_seconds": float(seconds),
        "actual_overlap_seconds": n / sr,
        "outgoing_start_seconds": (len(out) - n) / sr,
        "incoming_start_seconds": 0.0,
        "beat_aligned": False,
        "tempo_change": None,
        "applied_eq": None,
        "applied_gain_db": {"outgoing": outgoing_gain_db, "incoming": incoming_gain_db},
        "safety_adjustments": notes,
        "output_duration_seconds": len(audio) / sr,
    }
    return audio, applied


def render_plain_crossfade(
    outgoing_path: Path,
    incoming_path: Path,
    seconds: float,
    out_path: Path,
    *,
    loudness_match: bool = True,
    level_match: bool = True,
) -> dict:
    """Write the plain-crossfade excerpt for two files to out_path (WAV)."""
    out = load_render_audio(Path(outgoing_path))
    inc = load_render_audio(Path(incoming_path))
    g_out = g_in = 0.0
    if loudness_match:
        g_out = track_gain_for_path(Path(outgoing_path)).gain_db
        g_in = track_gain_for_path(Path(incoming_path)).gain_db
    audio, applied = plain_crossfade(out, inc, RENDER_SR, seconds,
                                     outgoing_gain_db=g_out, incoming_gain_db=g_in)
    if level_match:
        audio, level_db, note = level_preview(audio, RENDER_SR)
        applied["applied_gain_db"]["preview_output"] = level_db
        if note:
            applied["safety_adjustments"].append(note)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), audio, RENDER_SR, subtype="PCM_16")
    return applied
