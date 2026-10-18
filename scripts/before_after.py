"""Before/after demo: the same song pair as a plain crossfade and as a Mixr seam.

Usage:
    python scripts/before_after.py OUTGOING.mp3 INCOMING.mp3 [--overlap 16] [--before-overlap 6] [--out out/before_after/]

Writes to the output folder:
    before_plain_crossfade.wav   what a normal player does: last N s of A over first N s of B
    after_mixr.wav               Mixr's real seam (render_preview, same path as the app's Tune step)
    before_then_after.wav        both back to back with a short gap, for playing during the pitch
    applied.json                 what each version actually did

"After" is what the app does in Full songs mode with no tuning: real analysis,
the cue points choose_windows picks, and the neutral recommended overlap.
Both versions use the same decoded audio, track loudness gains, equal-power
curve, seam context and output leveling, so the difference you hear is the
seam itself, not volume.
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.schemas import Track, TransitionRequest  # noqa: E402
from backend.audio.analysis import analyze_track  # noqa: E402
from backend.audio.baseline import render_plain_crossfade  # noqa: E402
from backend.audio.harmony import chroma_for  # noqa: E402
from backend.audio.loading import RENDER_SR  # noqa: E402
from backend.audio.preview import render_preview  # noqa: E402
from backend.logic.calibration import recommended_overlap_seconds  # noqa: E402
from backend.logic.windows import choose_windows  # noqa: E402

GAP_SECONDS = 1.5


def analyze(path: Path, track_id: str) -> Track:
    info = sf.info(str(path))
    tr = Track(id=track_id, filename=path.name,
               duration_seconds=info.frames / info.samplerate, sample_rate=info.samplerate)
    tr = analyze_track(path, tr)
    if tr.analysis_status != "ok":
        sys.exit(f"analysis failed for {path.name}: {tr.analysis_error}")
    print(f"{path.name}: bpm {tr.bpm}, key {tr.key_camelot or 'unknown'}")
    return tr


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("outgoing")
    ap.add_argument("incoming")
    ap.add_argument("--overlap", type=float, default=None,
                    help="requested overlap for both versions, seconds (default: Mixr's neutral recommendation)")
    ap.add_argument("--before-overlap", type=float, default=None,
                    help="plain crossfade length, seconds (default: same as --overlap)")
    ap.add_argument("--out", default="out/before_after")
    args = ap.parse_args()

    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)
    pa, pb = Path(args.outgoing), Path(args.incoming)
    ta, tb = analyze(pa, "a"), analyze(pb, "b")
    overlap = args.overlap if args.overlap is not None else recommended_overlap_seconds(0.0)

    before_overlap = args.before_overlap if args.before_overlap is not None else overlap
    before = render_plain_crossfade(pa, pb, before_overlap, out_dir / "before_plain_crossfade.wav")

    windows = choose_windows([ta, tb], [overlap], None, [chroma_for(pa), chroma_for(pb)])
    req = TransitionRequest(from_id=ta.id, to_id=tb.id, requested_overlap_seconds=overlap)
    after = render_preview(pa, pb, ta, tb, req, out_dir / "after_mixr.wav",
                           outgoing_window=windows[0], incoming_window=windows[1])

    b, _ = sf.read(out_dir / "before_plain_crossfade.wav", dtype="float32")
    a, _ = sf.read(out_dir / "after_mixr.wav", dtype="float32")
    gap = np.zeros((int(GAP_SECONDS * RENDER_SR), b.shape[1]), dtype=np.float32)
    sf.write(out_dir / "before_then_after.wav", np.concatenate([b, gap, a]), RENDER_SR, subtype="PCM_16")

    report = {
        "before": before,
        "after": after.model_dump(),
        "windows": [w.model_dump() for w in windows],
    }
    (out_dir / "applied.json").write_text(json.dumps(report, indent=2))

    print(f"\nbefore: plain crossfade {before['actual_overlap_seconds']:.2f}s at the very end of "
          f"{pa.name}, level {before['applied_gain_db']['preview_output']:+.2f} dB")
    print(f"after:  {after.actual_overlap_seconds:.2f}s (requested {overlap:.1f}s), "
          f"starts {after.outgoing_start_seconds:.1f}s into the played part of {pa.name}, "
          f"beat adj {after.beat_alignment_adjustments_seconds:+.3f}s, "
          f"tempo {after.tempo_change or 'not changed'}, eq {after.applied_eq or 'none'}, "
          f"level {after.applied_gain_db.get('preview_output', 0.0):+.2f} dB")
    for w in windows:
        print(f"  plays {w.track_id}: {w.start_seconds:.1f}-{w.end_seconds:.1f}s  {w.reason or ''}")
    for note in after.safety_adjustments:
        print(f"  note: {note}")
    print(f"wrote {out_dir}/ (play before_then_after.wav)")


if __name__ == "__main__":
    main()
