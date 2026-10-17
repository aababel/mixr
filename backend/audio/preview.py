"""A/B calibration previews (owner: Aarav).

Uses render_transition(mode="preview"), the same code path as the final
mix, so the seam a listener votes on is exactly what the mix will contain.
"""
from pathlib import Path

import soundfile as sf

from backend.api.schemas import AppliedTransition, Track, TransitionRequest
from backend.audio.loading import RENDER_SR, load_render_audio
from backend.audio.transition import render_transition

PREVIEW_CONTEXT_SECONDS = 6.0


def render_preview(
    outgoing_path: Path,
    incoming_path: Path,
    outgoing: Track,
    incoming: Track,
    request: TransitionRequest,
    out_path: Path,
) -> AppliedTransition:
    """Write a short preview excerpt around the seam to out_path (WAV)."""
    out_audio = load_render_audio(outgoing_path)
    in_audio = load_render_audio(incoming_path)
    audio, applied = render_transition(
        out_audio,
        in_audio,
        RENDER_SR,
        outgoing.beats_seconds,
        incoming.beats_seconds,
        request,
        "preview",
        PREVIEW_CONTEXT_SECONDS,
    )
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), audio, RENDER_SR, subtype="PCM_16")
    return applied
