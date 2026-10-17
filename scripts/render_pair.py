"""Listen test for the shared renderer on two real files.

Usage:
    python scripts/render_pair.py OUTGOING.mp3 INCOMING.mp3 [--short 2] [--long 5] [--out out/]

Writes to the output folder:
    preview_A_short.wav, preview_B_long.wav   A/B previews (same renderer as the final mix)
    full_mix_short.wav, full_mix_long.wav     both complete tracks stitched at each overlap
    applied.json                              what the renderer actually did

Beats come from librosa.beat.beat_track as a stopgap until backend/audio/analysis.py exists.
"""
import argparse
import json
import sys
from pathlib import Path

import librosa
import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.schemas import TransitionRequest  # noqa: E402
from backend.audio.transition import render_transition, seam_sample_bounds  # noqa: E402

SR = 44100


def load(path):
    y, _ = librosa.load(path, sr=SR, mono=False)  # channels-first or 1-D
    mono = librosa.to_mono(y) if y.ndim == 2 else y
    _, frames = librosa.beat.beat_track(y=mono, sr=SR)
    beats = librosa.frames_to_time(frames, sr=SR).tolist()
    frames_audio = y.T if y.ndim == 2 else y[:, None]
    return np.ascontiguousarray(frames_audio, dtype=np.float32), beats


def main():
    p = argparse.ArgumentParser()
    p.add_argument("outgoing")
    p.add_argument("incoming")
    p.add_argument("--short", type=float, default=2.0)
    p.add_argument("--long", type=float, default=5.0)
    p.add_argument("--fade", choices=["equal_power", "linear"], default="equal_power")
    p.add_argument("--out", default="out")
    args = p.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    out, ob = load(args.outgoing)
    inc, ib = load(args.incoming)
    if out.shape[1] != inc.shape[1]:  # keep stitching simple: match channel counts up front
        ch = max(out.shape[1], inc.shape[1])
        out = np.repeat(out, ch // out.shape[1], axis=1)
        inc = np.repeat(inc, ch // inc.shape[1], axis=1)

    report = {}
    for label, secs, name in [("A", args.short, "short"), ("B", args.long, "long")]:
        r = TransitionRequest(
            from_id=Path(args.outgoing).stem, to_id=Path(args.incoming).stem,
            requested_overlap_seconds=secs, fade_style=args.fade,
        )
        prev, ap_prev = render_transition(out, inc, SR, ob, ib, r, "preview")
        seam, ap_final = render_transition(out, inc, SR, ob, ib, r, "final")
        o_start, ov = seam_sample_bounds(ap_final, SR)
        mix = np.concatenate([out[:o_start], seam, inc[ov:]])

        sf.write(out_dir / f"preview_{label}_{name}.wav", prev, SR)
        sf.write(out_dir / f"full_mix_{name}.wav", mix, SR)
        report[label] = {"preview": ap_prev.model_dump(), "final": ap_final.model_dump()}
        print(
            f"{label}: requested {secs:.2f}s -> actual {ap_final.actual_overlap_seconds:.3f}s "
            f"(beat adj {ap_final.beat_alignment_adjustments_seconds:+.3f}s) "
            f"notes={ap_final.safety_adjustments}"
        )

    gap = abs(report["A"]["final"]["actual_overlap_seconds"] - report["B"]["final"]["actual_overlap_seconds"])
    if gap < 1.0:
        print(f"WARNING: A and B differ by only {gap:.2f}s after clamping/snapping")
    (out_dir / "applied.json").write_text(json.dumps(report, indent=2))
    print(f"wrote {out_dir}/")


if __name__ == "__main__":
    main()
