"""Stand-in for backend.audio.mix.render_mix, used only by tests and local UI checks.

Follows the signature in docs/integration.md with a plain linear crossfade.
"""
from pathlib import Path
from types import ModuleType

import numpy as np
import soundfile as sf

from backend.api.schemas import AppliedTransition, MixManifest, Track


def render_mix(
    mix_id: str,
    tracks: list[Track],
    paths: list[Path],
    overlap_seconds: list[float],
    out_path: Path,
) -> MixManifest:
    sr = tracks[0].sample_rate
    audio = [sf.read(str(p), always_2d=True)[0].mean(axis=1) for p in paths]
    out = audio[0]
    transitions = []
    for i, (requested, nxt) in enumerate(zip(overlap_seconds, audio[1:])):
        limit = min(len(audio[i]), len(nxt)) / sr / 2
        actual = min(requested, limit)
        n = int(actual * sr)
        fade = np.linspace(0.0, 1.0, n)
        seam = out[len(out) - n :] * (1 - fade) + nxt[:n] * fade
        outgoing_start = (len(audio[i]) - n) / sr
        out = np.concatenate([out[: len(out) - n], seam, nxt[n:]])
        transitions.append(
            AppliedTransition(
                from_id=tracks[i].id,
                to_id=tracks[i + 1].id,
                mode="final",
                fade_style="linear",
                requested_overlap_seconds=requested,
                actual_overlap_seconds=n / sr,
                overlap_changed_for_safety=actual < requested,
                outgoing_start_seconds=outgoing_start,
                incoming_start_seconds=0.0,
                safety_adjustments=(
                    [f"Overlap limited to half the shorter track ({limit:.1f}s)"]
                    if actual < requested
                    else []
                ),
                output_duration_seconds=n / sr,
            )
        )
    sf.write(str(out_path), out, sr)
    return MixManifest(
        mix_id=mix_id,
        ordered_track_ids=[t.id for t in tracks],
        transitions=transitions,
        source_durations_seconds=[t.duration_seconds for t in tracks],
        output_duration_seconds=len(out) / sr,
        calibrated=False,
        overlap_preference=0.0,
        limitations=["Test renderer: linear crossfade, no beat alignment"],
    )


def as_module() -> ModuleType:
    module = ModuleType("backend.audio.mix")
    module.render_mix = render_mix
    return module
