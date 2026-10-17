"""A/B calibration previews (owner: Aarav).

Uses render_transition(mode="preview"), the same code path as the final
mix, so the seam a listener votes on is exactly what the mix will contain.
Each track gets the same loudness-matching gain here as in render_mix, so
A and B (and the final mix) differ only in overlap.
"""
from pathlib import Path

import soundfile as sf

from backend.api.schemas import AppliedTransition, Track, TransitionRequest
from backend.audio.loading import RENDER_SR, load_render_audio
from backend.audio.loudness import gain_notes, track_gain_for_path
from backend.audio.transition import render_transition

PREVIEW_CONTEXT_SECONDS = 6.0


def render_preview(
    outgoing_path: Path,
    incoming_path: Path,
    outgoing: Track,
    incoming: Track,
    request: TransitionRequest,
    out_path: Path,
    *,
    loudness_match: bool = True,
) -> AppliedTransition:
    """Write a short preview excerpt around the seam to out_path (WAV).

    loudness_match: apply each track's loudness gain (same as render_mix).
    applied_gain_db in the result records the gains actually used.
    """
    out_audio = load_render_audio(outgoing_path)
    in_audio = load_render_audio(incoming_path)
    g_out = g_in = None
    if loudness_match:
        g_out, g_in = track_gain_for_path(outgoing_path), track_gain_for_path(incoming_path)
    audio, applied = render_transition(
        out_audio,
        in_audio,
        RENDER_SR,
        outgoing.beats_seconds,
        incoming.beats_seconds,
        request,
        "preview",
        PREVIEW_CONTEXT_SECONDS,
        outgoing_gain_db=g_out.gain_db if g_out else 0.0,
        incoming_gain_db=g_in.gain_db if g_in else 0.0,
    )
    if loudness_match:
        notes = gain_notes("outgoing", outgoing.id, g_out) + gain_notes("incoming", incoming.id, g_in)
        if notes:
            applied = applied.model_copy(
                update={"safety_adjustments": applied.safety_adjustments + notes}
            )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), audio, RENDER_SR, subtype="PCM_16")
    return applied
