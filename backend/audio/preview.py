"""A/B calibration previews (owner: Aarav).

Uses render_transition(mode="preview"), the same code path as the final
mix, so the seam a listener votes on is exactly what the mix will contain.
Each track gets the same loudness-matching gain here as in render_mix, so
A and B (and the final mix) differ only in overlap.
"""
from pathlib import Path

import soundfile as sf

from backend.api.schemas import AppliedTransition, Track, TrackWindow, TransitionRequest
from backend.audio.loading import RENDER_SR, load_render_audio
from backend.audio.loudness import gain_notes, track_gain_for_path
from backend.audio.transition import render_transition
from backend.audio.windows import edge_fades_for_seam, play, with_entry

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
    outgoing_window: TrackWindow | None = None,
    incoming_window: TrackWindow | None = None,
) -> AppliedTransition:
    """Write a short preview excerpt around the seam to out_path (WAV).

    loudness_match: apply each track's loudness gain (same as render_mix).
    applied_gain_db in the result records the gains actually used.
    outgoing_window / incoming_window: the windows the mix will play (None =
    whole track), so the preview is the seam the mix will contain. Times in
    the result are measured from the start of each window.
    """
    out_p = play(load_render_audio(outgoing_path), outgoing, outgoing_window, RENDER_SR)
    in_p = play(load_render_audio(incoming_path), incoming, incoming_window, RENDER_SR)
    g_out = g_in = None
    if loudness_match:
        g_out, g_in = track_gain_for_path(outgoing_path), track_gain_for_path(incoming_path)
    fades = edge_fades_for_seam(out_p, outgoing, in_p, incoming)
    audio, applied = render_transition(
        out_p.audio,
        in_p.audio,
        RENDER_SR,
        out_p.beats,
        in_p.beats,
        with_entry(request, in_p.window if incoming_window else None),
        "preview",
        PREVIEW_CONTEXT_SECONDS,
        outgoing_gain_db=g_out.gain_db if g_out else 0.0,
        incoming_gain_db=g_in.gain_db if g_in else 0.0,
        outgoing_melody=out_p.melody,
        incoming_melody=in_p.melody,
        melody_hop_seconds=outgoing.energy_hop_seconds,
        outgoing_fade_out_seconds=fades[0],
        incoming_fade_in_seconds=fades[1],
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
