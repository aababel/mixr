"""Build a full Mixr mix from real files, in the order given.

Usage:
    python scripts/make_mix.py song1.mp3 song2.mp3 song3.mp3 [--overlap 8] [--out out/mix.wav] [--no-loudness]

Runs analysis, renders every seam with the shared renderer, writes the mix
and manifest.json next to it. Ordering and scoring are Sahil's; this script
keeps your order so you can test the audio on its own.
"""
import argparse
import json
import sys
import time
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.schemas import Track  # noqa: E402
from backend.audio.analysis import analyze_track  # noqa: E402
from backend.audio.mix import render_mix  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--overlap", type=float, default=8.0, help="requested overlap per seam, seconds")
    ap.add_argument("--out", default="out/mix.wav")
    ap.add_argument("--no-loudness", action="store_true", help="skip loudness matching (0 dB gain)")
    args = ap.parse_args()
    if len(args.files) < 2:
        sys.exit("need at least two files")

    paths = [Path(f) for f in args.files]
    tracks = []
    t0 = time.time()
    for i, p in enumerate(paths):
        info = sf.info(str(p))
        tr = Track(id=f"t{i}", filename=p.name,
                   duration_seconds=info.frames / info.samplerate, sample_rate=info.samplerate)
        tr = analyze_track(p, tr)
        print(f"t{i} {p.name}: {tr.analysis_status}, bpm {tr.bpm}, key {tr.key_camelot}")
        tracks.append(tr)
    t1 = time.time()

    out = Path(args.out)
    manifest = render_mix("local", tracks, paths, [args.overlap] * (len(paths) - 1), out,
                          loudness_match=not args.no_loudness)
    t2 = time.time()

    out.with_name("manifest.json").write_text(json.dumps(manifest.model_dump(), indent=2))
    print(f"\nanalysis {t1 - t0:.1f}s, mix {t2 - t1:.1f}s")
    for t in manifest.transitions:
        print(f"{t.from_id} -> {t.to_id}: requested {t.requested_overlap_seconds:.1f}s, "
              f"actual {t.actual_overlap_seconds:.2f}s, starts at {t.outgoing_start_seconds:.1f}s "
              f"of {t.from_id}, gain {t.applied_gain_db['outgoing']:+.1f}/{t.applied_gain_db['incoming']:+.1f} dB"
              f"  {t.safety_adjustments or ''}")
    print(f"sources {sum(manifest.source_durations_seconds):.1f}s -> mix {manifest.output_duration_seconds:.1f}s")
    print(f"wrote {out} and {out.with_name('manifest.json')}")


if __name__ == "__main__":
    main()
