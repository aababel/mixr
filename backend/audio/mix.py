"""Final mix renderer (owner: Aarav).

Every seam goes through render_transition(mode="final"), the same code the
A/B previews use. Every track is written in full; only the seams overlap:

    t0[:end0] + seam0 + t1[ov0:end1] + seam1 + t2[ov1:] ...

where end_i = len(t_i) - ov_i. The renderer caps each overlap at 50% of
either track, so the head and tail seams of a middle track never collide.
The mix is streamed to disk, holding at most two decoded tracks at once.
"""
from pathlib import Path

import numpy as np
import soundfile as sf

from backend.api.schemas import AppliedTransition, MixManifest, Track, TransitionRequest
from backend.audio.loading import RENDER_SR, load_render_audio
from backend.audio.transition import render_transition, seam_sample_bounds


def _write(f: sf.SoundFile, block: np.ndarray, stats: dict) -> None:
    clipped = int(np.count_nonzero(np.abs(block) > 1.0))
    if clipped:
        stats["clipped"] += clipped
        block = np.clip(block, -1.0, 1.0)
    f.write(block)
    stats["frames"] += block.shape[0]


def render_mix(
    mix_id: str,
    tracks: list[Track],
    paths: list[Path],
    overlap_seconds: list[float],
    out_path: Path,
    *,
    calibrated: bool = False,
    overlap_preference: float = 0.0,
) -> MixManifest:
    """Write the full mix to out_path (WAV). The manifest reports actual values."""
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

    with sf.SoundFile(str(out_path), "w", RENDER_SR, 2, subtype="PCM_16") as f:
        current = load_render_audio(paths[0])
        head = 0  # samples of `current` already consumed by the previous seam
        for i, track in enumerate(tracks):
            durations.append(current.shape[0] / RENDER_SR)
            if i == len(tracks) - 1:
                _write(f, current[head:], stats)
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
            )
            o_start, ov = seam_sample_bounds(applied, RENDER_SR)
            if o_start < head:  # cannot happen with the 50% cap; guard anyway
                raise RuntimeError(f"seam {i} overlaps the previous seam in track {track.id}")

            _write(f, current[head:o_start], stats)
            _write(f, seam, stats)
            transitions.append(applied)
            current, head = nxt, ov

    limitations = [
        "no loudness normalization between tracks (0 dB gain)",
        "no EQ or tempo change applied",
    ]
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
