"""Shared transition renderer (owner: Aarav).

ONE function powers both A/B previews and the final mix.
Status: contract only, not implemented yet.
"""
import numpy as np

from backend.api.schemas import AppliedTransition, RenderMode, TransitionRequest


def render_transition(
    outgoing: np.ndarray,
    incoming: np.ndarray,
    sr: int,
    outgoing_beats_seconds: list[float],
    incoming_beats_seconds: list[float],
    request: TransitionRequest,
    mode: RenderMode,
    preview_context_seconds: float = 6.0,
) -> tuple[np.ndarray, AppliedTransition]:
    """Render the seam between two tracks.

    preview: return a short excerpt around the seam.
    final:   return the overlapped seam region; mix.py stitches full tracks
             around it so no musical content is dropped.
    """
    raise NotImplementedError("render_transition is not implemented yet")
