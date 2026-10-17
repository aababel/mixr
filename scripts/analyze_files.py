"""Print Mixr's analysis for real audio files.

Usage:
    python scripts/analyze_files.py song1.mp3 song2.mp3 ...
"""
import sys
import time
from pathlib import Path

import soundfile as sf

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from backend.api.schemas import Track  # noqa: E402
from backend.audio.analysis import analyze_track  # noqa: E402


def main():
    if len(sys.argv) < 2:
        sys.exit(__doc__)
    for arg in sys.argv[1:]:
        path = Path(arg)
        info = sf.info(str(path))
        track = Track(
            id=path.stem, filename=path.name,
            duration_seconds=info.frames / info.samplerate, sample_rate=info.samplerate,
        )
        t0 = time.time()
        out = analyze_track(path, track)
        m = out.metadata
        print(f"\n{path.name}  ({time.time() - t0:.1f}s to analyze)")
        if out.analysis_status != "ok":
            print(f"  ERROR: {out.analysis_error}")
            continue
        print(f"  duration   {out.duration_seconds:.1f}s")
        print(f"  bpm        {out.bpm}  ({len(out.beats_seconds)} beats, first at {out.beats_seconds[0] if out.beats_seconds else '-'}s)")
        print(f"  key        {out.key_camelot or 'unknown'}  ({m['key_name']}, confidence {m['key_confidence']})")
        print(f"  energy     intro {m['intro_energy']:.3f}  outro {m['outro_energy']:.3f}  peak {max(out.energy_rms):.3f}")
        for note in m["analysis_notes"]:
            print(f"  note       {note}")


if __name__ == "__main__":
    main()
