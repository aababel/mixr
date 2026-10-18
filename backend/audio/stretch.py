"""Time-map warping with the Rubber Band CLI (owner: Aarav).

Same mechanism as pyrubberband.timemap_stretch (the `rubberband` binary with
--timemap), called directly so the temp files are 32-bit float. pyrubberband
writes 16-bit temp WAVs, which would clip decoded MP3 samples above 1.0 and
add a second round of quantization before the final 16-bit write. The
audio is also passed 12 dB down (see below), because the CLI otherwise
lowers the whole output's gain when stretched transients pass full scale.

Install the binary: `brew install rubberband` (macOS) or
`apt-get install rubberband-cli` (Debian/Ubuntu). Without it,
rubberband_available() is False and transition.py does not warp.
"""
import os
import shutil
import subprocess
import tempfile
from functools import lru_cache

import numpy as np
import soundfile as sf

_HEADROOM = np.float32(0.25)  # -12 dB while inside Rubber Band
TimeMap = list[tuple[int, int]]  # (source_sample, target_sample), both increasing


@lru_cache(maxsize=1)
def _binary() -> tuple[str | None, bool]:
    """(path to rubberband, supports the R3 --fine engine)."""
    path = shutil.which("rubberband")
    if path is None:
        return None, False
    try:
        out = subprocess.run([path, "--version"], capture_output=True, text=True, timeout=10)
        text = (out.stdout or out.stderr).strip().splitlines()
        major = int(text[-1].split(".")[0]) if text else 0
    except (OSError, ValueError, subprocess.SubprocessError):
        major = 0
    return path, major >= 3


def rubberband_available() -> bool:
    return _binary()[0] is not None


def engine_name() -> str:
    path, fine = _binary()
    if path is None:
        return "unavailable"
    return "Rubber Band R3 (--fine)" if fine else "Rubber Band R2"


def is_identity(time_map: TimeMap) -> bool:
    return all(s == t for s, t in time_map)


def timemap_stretch(y: np.ndarray, sr: int, time_map: TimeMap, out_len: int) -> np.ndarray:
    """Warp (samples, ch) float32 audio so source sample s lands at target t
    for every (s, t) in time_map. time_map must start at (0, 0) and end at
    (len(y), out_len). Returns float32 (out_len, ch)."""
    path, fine = _binary()
    if path is None:
        raise RuntimeError("rubberband is not installed")
    if time_map[0] != (0, 0) or time_map[-1] != (y.shape[0], out_len):
        raise ValueError("time_map must run from (0, 0) to (len(y), out_len)")
    for (s0, t0), (s1, t1) in zip(time_map, time_map[1:]):
        if s1 <= s0 or t1 <= t0:
            raise ValueError("time_map must be strictly increasing")

    tmp = tempfile.mkdtemp(prefix="mixr_rb_")
    src, dst, tm = (os.path.join(tmp, n) for n in ("in.wav", "out.wav", "map.txt"))
    try:
        # The CLI restarts at reduced gain (or clamps) whenever its output would
        # pass full scale, and smeared transients often do. Stretching is linear,
        # so run 12 dB down and scale back up: level is preserved exactly.
        sf.write(src, y * _HEADROOM, sr, subtype="FLOAT")
        with open(tm, "w") as f:
            # The final point is implied by --time; rubberband wants interior anchors.
            for s, t in time_map[1:-1]:
                f.write(f"{s} {t}\n")
        args = [path, "-q", "--ignore-clipping", "--time", repr(out_len / y.shape[0]), "--timemap", tm]
        if fine:
            args.append("--fine")
        subprocess.run(args + [src, dst], check=True, capture_output=True, timeout=120)
        out, _ = sf.read(dst, dtype="float32", always_2d=True)
        out = out / _HEADROOM
    finally:
        shutil.rmtree(tmp, ignore_errors=True)

    if out.shape[0] >= out_len:
        out = out[:out_len]
    else:
        out = np.concatenate([out, np.zeros((out_len - out.shape[0], out.shape[1]), np.float32)])
    return np.ascontiguousarray(out, dtype=np.float32)
