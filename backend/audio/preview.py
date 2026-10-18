"""A/B calibration previews (owner: Aarav).

Uses render_transition(mode="preview"), the same code path as the final
mix, so the seam a listener votes on is exactly what the mix will contain.
Each track gets the same loudness-matching gain here as in render_mix, so
A and B (and the final mix) differ only in overlap.

Preview leveling: a longer blend sums two songs for longer, so the long
preview tends to come out louder than the short one, and louder tends to win
an A/B vote. After rendering, each preview excerpt is scaled by one uniform
gain so its integrated loudness is PREVIEW_TARGET_LUFS. This is a single
scalar on the finished excerpt (the seam itself is untouched, so the preview
is still exactly the mix's seam, only at a fixed playback level). The gain is
recorded as applied_gain_db["preview_output"].
"""
from pathlib import Path
from typing import Optional

import numpy as np
import soundfile as sf

from backend.api.schemas import AppliedTransition, Track, TrackWindow, TransitionRequest
from backend.audio.loading import RENDER_SR, load_render_audio
from backend.audio.loudness import DEFAULT_LOUDNESS, gain_notes, integrated_loudness, track_gain_for_path
from backend.audio.transition import DEFAULT_SAFETY, SafetyLimits, render_transition
from backend.audio.windows import edge_fades_for_seam, play, with_entry

PREVIEW_CONTEXT_SECONDS = 6.0
# Below the -14 LUFS track target, so leveling is nearly always a cut and the
# peak ceiling never stops one preview short of the level the other reached.
PREVIEW_TARGET_LUFS = -16.0


def level_preview(
    audio: np.ndarray,
    sr: int,
    target_lufs: float = PREVIEW_TARGET_LUFS,
    peak_ceiling: float = DEFAULT_LOUDNESS.peak_ceiling,
) -> tuple[np.ndarray, float, Optional[str]]:
    """Scale a whole preview excerpt to target_lufs integrated loudness.

    Returns (leveled audio, gain_db applied, note or None). A boost never
    pushes a sample past peak_ceiling; when that stops it short of the
    target, or the excerpt can't be measured, the note says so.
    """
    measured = integrated_loudness(audio, sr)
    if measured is None:
        return audio, 0.0, "preview level: loudness not measurable; gain 0 dB"
    gain = target_lufs - measured
    note = None
    peak = float(np.max(np.abs(audio)))
    if gain > 0 and peak > 0:
        headroom = 20.0 * np.log10(peak_ceiling / peak)
        if gain > headroom:
            gain = max(0.0, np.floor(headroom * 100) / 100)  # round down: never past the ceiling
            note = (f"preview level: measured {measured:.1f} LUFS, boost capped by peak "
                    f"headroom at {gain:+.1f} dB, reaches {measured + gain:.1f} LUFS "
                    f"instead of {target_lufs:.1f}")
    gain = round(float(gain), 2)  # the recorded value is exactly the applied one
    leveled = (audio * np.float32(10.0 ** (gain / 20.0))).astype(audio.dtype, copy=False)
    return leveled, gain, note


def render_preview(
    outgoing_path: Path,
    incoming_path: Path,
    outgoing: Track,
    incoming: Track,
    request: TransitionRequest,
    out_path: Path,
    *,
    loudness_match: bool = True,
    level_match: bool = True,
    outgoing_window: TrackWindow | None = None,
    incoming_window: TrackWindow | None = None,
    safety: SafetyLimits = DEFAULT_SAFETY,
) -> AppliedTransition:
    """Write a short preview excerpt around the seam to out_path (WAV).

    loudness_match: apply each track's loudness gain (same as render_mix).
    applied_gain_db in the result records the gains actually used.
    level_match: scale the finished excerpt to PREVIEW_TARGET_LUFS so A and
    B play at the same loudness; the gain is applied_gain_db["preview_output"]
    (key absent when level_match=False).
    outgoing_window / incoming_window: the windows the mix will play (None =
    whole track), so the preview is the seam the mix will contain. Times in
    the result are measured from the start of each window.
    safety: the same limits the mix renders with.
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
        safety=safety,
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
    if level_match:
        audio, level_db, level_note = level_preview(audio, RENDER_SR)
        update = {"applied_gain_db": {**applied.applied_gain_db, "preview_output": level_db}}
        if level_note:
            update["safety_adjustments"] = applied.safety_adjustments + [level_note]
        applied = applied.model_copy(update=update)
    out_path = Path(out_path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(out_path), audio, RENDER_SR, subtype="PCM_16")
    return applied
