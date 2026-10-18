"""Fade-out tails and fade-in heads at the edges of a track (owner: Aarav).

Many songs end with a fade to silence after their last bar (the JJTBEATZ
remixes: 3-8 s), and some start with a fade-in. If the transition's
crossfade sits on top of that, the outgoing side fades twice (its own fade
plus ours) and the blend dips. transition.py uses these lengths to place
the crossfade before the tail and after the head; the edges still play
inside the seam, mixed under the other track.

Detection works on the level envelope only (RMS in short windows, dB):
  tail: the last moment the level is still within EDGE_TOLERANCE_DB of the
        median of the few seconds before it; after that it only falls.
  head: the mirror image at the start.
An edge counts only if the far end is at least MIN_DROP_DB below where the
fade begins, so a song that simply ends on a loud hit, or a quiet but steady
intro section, has no edge. Edges shorter than MIN_EDGE_SECONDS are ignored
(a gap between the last hits, not a fade) and longer ones are capped at
MAX_EDGE_SECONDS.
"""
import numpy as np

WINDOW_SECONDS = 0.4  # RMS window, about one beat at 150 BPM, smooths kick pumping
HOP_SECONDS = 0.1
REFERENCE_SECONDS = 4.0  # level an edge is compared against: median of this much audio next to it
EDGE_TOLERANCE_DB = 3.0
MIN_DROP_DB = 12.0
MIN_EDGE_SECONDS = 1.0
MAX_EDGE_SECONDS = 12.0
SEARCH_SECONDS = MAX_EDGE_SECONDS + REFERENCE_SECONDS + 1.0


def _levels_db(mono: np.ndarray, sr: int) -> np.ndarray:
    """RMS level in dB of WINDOW_SECONDS windows every HOP_SECONDS."""
    win, hop = max(1, int(WINDOW_SECONDS * sr)), max(1, int(HOP_SECONDS * sr))
    if mono.shape[0] < win:
        return np.zeros(0)
    power = np.concatenate([[0.0], np.cumsum(np.asarray(mono, dtype=np.float64) ** 2)])
    starts = np.arange(0, mono.shape[0] - win + 1, hop)
    mean_sq = (power[starts + win] - power[starts]) / win
    return 10.0 * np.log10(mean_sq + 1e-10)


def _head_frames(db: np.ndarray) -> int:
    """Frames at the start of `db` that are a fade up from much lower."""
    ref_n = int(round(REFERENCE_SECONDS / HOP_SECONDS))
    max_n = int(round(MAX_EDGE_SECONDS / HOP_SECONDS))
    if db.size < ref_n + 2:
        return 0
    for i in range(0, min(max_n, db.size - ref_n) + 1):
        if db[i] >= np.median(db[i:i + ref_n]) - EDGE_TOLERANCE_DB:
            break
    else:
        i = min(max_n, db.size - ref_n)  # fade longer than MAX_EDGE_SECONDS: capped
    if i * HOP_SECONDS < MIN_EDGE_SECONDS - 1e-9 or db[0] > db[i] - MIN_DROP_DB:
        return 0
    return i


def _mono(audio: np.ndarray) -> np.ndarray:
    a = np.asarray(audio, dtype=np.float32)
    return a.mean(axis=1) if a.ndim == 2 else a


def fade_out_seconds(audio: np.ndarray, sr: int) -> float:
    """Length of the fade to silence at the end of `audio` ((samples, ch) or
    (samples,)), 0.0 if there is none. Measured to the last sample."""
    mono = _mono(audio)
    n = min(mono.shape[0], int(SEARCH_SECONDS * sr))
    tail = mono[mono.shape[0] - n:]
    db = _levels_db(tail[::-1], sr)  # reversed: a fade-out reads as a fade-in
    frames = _head_frames(db)
    if frames == 0:
        return 0.0
    # frame k (reversed) covers the window ending k * hop before the end; the
    # fade begins where the last full-level window ends.
    return float(min(frames * HOP_SECONDS, MAX_EDGE_SECONDS))


def fade_in_seconds(audio: np.ndarray, sr: int) -> float:
    """Length of the fade up from silence at the start of `audio`, 0.0 if none."""
    mono = _mono(audio)
    head = mono[:min(mono.shape[0], int(SEARCH_SECONDS * sr))]
    frames = _head_frames(_levels_db(head, sr))
    return float(min(frames * HOP_SECONDS, MAX_EDGE_SECONDS))
