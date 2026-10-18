"""Final mix renderer (owner: Aarav).

Every seam goes through render_transition(mode="final"), the same code the
A/B previews use. Every track is written in full; only the seams overlap:

    t0[:start0] + seam0 + t1[ov0:start1] + seam1 + t2[ov1:] ...

where start_i is where seam i begins in track i and ov_i is how many
incoming samples seam i consumed (seam_sample_bounds). With a tempo glide
the seam's own length differs from ov_i; that is fine, because each seam is
written whole and the next track resumes at ov_i. The renderer caps both
sides of a seam at 50% of their track, so the head and tail seams of a
middle track never collide.
The mix is streamed to disk, holding at most two decoded tracks at once.

Loudness matching: each track gets one gain (loudness.py) applied to its
whole body AND passed to render_transition for both of its seams, using the
same float32 factor, so level is continuous across seam boundaries and the
seams match render_preview exactly.
"""
from pathlib import Path

import numpy as np
import soundfile as sf

from backend.api.schemas import AppliedTransition, MixManifest, Track, TransitionRequest
from backend.audio.loading import RENDER_SR, load_render_audio
from backend.audio.loudness import DEFAULT_LOUDNESS, TrackGain, gain_notes, track_gain_for_path
from backend.audio.stretch import rubberband_available
from backend.audio.transition import gain_factor, render_transition, seam_sample_bounds

_CHUNK = RENDER_SR * 10  # scale and write long track bodies 10 s at a time


def _write(f: sf.SoundFile, block: np.ndarray, stats: dict, gain_db: float = 0.0) -> None:
    g = gain_factor(gain_db)
    for i in range(0, block.shape[0], _CHUNK):
        part = block[i:i + _CHUNK]
        if gain_db != 0.0:
            part = part * g
        clipped = int(np.count_nonzero(np.abs(part) > 1.0))
        if clipped:
            stats["clipped"] += clipped
            part = np.clip(part, -1.0, 1.0)
        f.write(part)
        stats["frames"] += part.shape[0]


def render_mix(
    mix_id: str,
    tracks: list[Track],
    paths: list[Path],
    overlap_seconds: list[float],
    out_path: Path,
    *,
    calibrated: bool = False,
    overlap_preference: float = 0.0,
    loudness_match: bool = True,
) -> MixManifest:
    """Write the full mix to out_path (WAV). The manifest reports actual values.

    loudness_match: apply each track's loudness gain (same as render_preview).
    Each transition's applied_gain_db records the gains actually used.
    """
    if len(tracks) < 1 or len(tracks) != len(paths):
        raise ValueError("tracks and paths must be non-empty and the same length")
    if len(overlap_seconds) != len(tracks) - 1:
        raise ValueError(f"need {len(tracks) - 1} overlaps, got {len(overlap_seconds)}")
    if any(o <= 0 for o in overlap_seconds):
        raise ValueError("every requested overlap must be > 0 seconds")

    out_path = Path(out_path)
    if out_path.resolve() in {Path(p).resolve() for p in paths}:
        raise ValueError("out_path must not be one of the input tracks")
    out_path.parent.mkdir(parents=True, exist_ok=True)

    transitions: list[AppliedTransition] = []
    durations: list[float] = []
    stats = {"frames": 0, "clipped": 0}
    gains: list[TrackGain | None] = []

    def gain_for(i: int) -> TrackGain | None:
        while len(gains) <= i:
            gains.append(track_gain_for_path(paths[len(gains)]) if loudness_match else None)
        return gains[i]

    def db(i: int) -> float:
        g = gain_for(i)
        return g.gain_db if g else 0.0

    with sf.SoundFile(str(out_path), "w", RENDER_SR, 2, subtype="PCM_16") as f:
        current = load_render_audio(paths[0])
        head = 0  # samples of `current` already consumed by the previous seam
        for i, track in enumerate(tracks):
            durations.append(current.shape[0] / RENDER_SR)
            if i == len(tracks) - 1:
                _write(f, current[head:], stats, db(i))
                break

            nxt_track = tracks[i + 1]
            nxt = load_render_audio(paths[i + 1])
            request = TransitionRequest(
                from_id=track.id,
                to_id=nxt_track.id,
                requested_overlap_seconds=float(overlap_seconds[i]),
            )
            seam, applied = render_transition(
                current, nxt, RENDER_SR,
                track.beats_seconds, nxt_track.beats_seconds,
                request, "final",
                outgoing_gain_db=db(i), incoming_gain_db=db(i + 1),
            )
            if loudness_match:
                notes = (gain_notes("outgoing", track.id, gain_for(i))
                         + gain_notes("incoming", nxt_track.id, gain_for(i + 1)))
                if notes:
                    applied = applied.model_copy(
                        update={"safety_adjustments": applied.safety_adjustments + notes}
                    )
            o_start, ov = seam_sample_bounds(applied, RENDER_SR)
            if o_start < head:  # cannot happen with the 50% cap; guard anyway
                raise RuntimeError(f"seam {i} overlaps the previous seam in track {track.id}")

            _write(f, current[head:o_start], stats, db(i))
            _write(f, seam, stats)  # gain already applied inside the seam
            transitions.append(applied)
            current, head = nxt, ov

    if loudness_match:
        limitations = [
            f"loudness matched per track toward {DEFAULT_LOUDNESS.target_lufs:.1f} LUFS "
            "(BS.1770 integrated, whole track); no limiter on track bodies, "
            "so a boost stops at the track's peak headroom",
        ]
        short = [t.id for t, g in zip(tracks, gains) if g and g.note]
        if short:
            limitations.append(f"loudness target not reached for {', '.join(short)}")
        if len(tracks) == 1 and gains[0] and gains[0].note:
            limitations.append(f"{tracks[0].id}: {gains[0].note}")
    else:
        limitations = ["no loudness normalization between tracks (0 dB gain)"]
    short = [f"{t.from_id}->{t.to_id}" for t in transitions
             if any(n.startswith("short blend") for n in t.safety_adjustments)]
    if short:
        limitations.append(
            f"short 2-beat blend instead of the requested overlap at {', '.join(short)} "
            "(tempos too far apart to glide, or Rubber Band not installed)"
        )
    unmatched = [f"{t.from_id}->{t.to_id}" for t in transitions
                 if any("beat alignment skipped" in n or "only the first beat" in n
                        for n in t.safety_adjustments)]
    if unmatched:
        limitations.append(f"beats not matched beat-for-beat at {', '.join(unmatched)}")
    if transitions and not rubberband_available():
        limitations.append("Rubber Band not installed: no tempo glide on any seam")
    if stats["clipped"]:
        limitations.append(f"{stats['clipped']} samples clipped to [-1, 1] when writing 16-bit WAV")
    missing_beats = [t.id for t in tracks if not t.beats_seconds]
    if missing_beats:
        limitations.append(f"no beat grid for {', '.join(missing_beats)}; those seams are not beat-aligned")

    return MixManifest(
        mix_id=mix_id,
        ordered_track_ids=[t.id for t in tracks],
        transitions=transitions,
        source_durations_seconds=durations,
        output_duration_seconds=stats["frames"] / RENDER_SR,
        calibrated=calibrated,
        overlap_preference=overlap_preference,
        limitations=limitations,
    )
