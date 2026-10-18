"""Track windows: play only [start, end] of a track (owner: Aarav).

Shared by mix.py and preview.py so the A/B previews blend exactly the
windows the mix will contain. A window is cut out of the decoded audio and
everything time-based moves with it (beats shifted by -start, melody
sliced), so render_transition sees the window as if it were the whole
track. Times in the resulting AppliedTransition are therefore measured
from the start of the window.
"""
from dataclasses import dataclass

import numpy as np

from backend.api.schemas import Track, TrackWindow, TransitionRequest

MIN_WINDOW_SECONDS = 4.0
FADE_IN_BEATS = 1  # a mix that starts mid-song fades in over about a beat
FADE_OUT_BEATS = 8  # and one that ends mid-song fades out over about 2 bars
DEFAULT_BEAT_SECONDS = 0.5
EDGE_TOLERANCE_SECONDS = 0.05


@dataclass
class Played:
    audio: np.ndarray  # (samples, ch), read-only when untrimmed
    beats: list[float]  # window-relative
    melody: list[float] | None  # window-relative, on track.energy_hop_seconds
    window: TrackWindow  # the window actually used (clamped to the audio)
    trimmed_start: bool
    trimmed_end: bool


FADE_OUT_KEY = "fade_out_seconds"  # track.metadata keys written by analysis.py
FADE_IN_KEY = "fade_in_seconds"


def stored_fade(track: Track, key: str) -> float | None:
    """Fade length analysis stored in track.metadata, None if absent or unusable."""
    v = track.metadata.get(key)
    if isinstance(v, bool) or not isinstance(v, (int, float)) or not np.isfinite(v) or v < 0:
        return None
    return float(v)


def edge_fades_for_seam(out_p: "Played", out_t: Track, in_p: "Played", in_t: Track):
    """(outgoing fade-out, incoming fade-in) to hand render_transition: the
    stored value when that side plays to its real end / from its real start,
    None (measure from the audio) when a window cut it."""
    tail = None if out_p.trimmed_end else stored_fade(out_t, FADE_OUT_KEY)
    head = None if in_p.trimmed_start else stored_fade(in_t, FADE_IN_KEY)
    return tail, head


def beat_seconds(track: Track) -> float:
    return 60.0 / track.bpm if track.bpm else DEFAULT_BEAT_SECONDS


def with_entry(request: TransitionRequest, incoming: TrackWindow | None) -> TransitionRequest:
    """The request for a seam into `incoming` (the window actually played):
    a drop transition when the window marks a drop, its build as the overlap."""
    if incoming is None or incoming.drop_seconds is None:
        return request
    build = incoming.drop_seconds - incoming.start_seconds
    if build <= 0:
        return request
    return request.model_copy(update={"style": "drop", "requested_overlap_seconds": build})


def check_windows(tracks: list[Track], windows: list[TrackWindow]) -> None:
    if len(windows) != len(tracks):
        raise ValueError(f"need one window per track ({len(tracks)}), got {len(windows)}")
    for t, w in zip(tracks, windows):
        if w.track_id != t.id:
            raise ValueError(f"window for {w.track_id!r} is in the slot of track {t.id!r}")
        if not (0.0 <= w.start_seconds < w.end_seconds):
            raise ValueError(f"window for {t.id!r} must have 0 <= start < end")
        if w.end_seconds - w.start_seconds < MIN_WINDOW_SECONDS:
            raise ValueError(f"window for {t.id!r} is shorter than {MIN_WINDOW_SECONDS}s")


def play(audio: np.ndarray, track: Track, window: TrackWindow | None, sr: int) -> Played:
    """Cut `audio` (the whole decoded track) to `window` (None = all of it)."""
    n = audio.shape[0]
    melody = list(track.melody_activity) or None
    if window is None:
        w = TrackWindow(track_id=track.id, start_seconds=0.0, end_seconds=n / sr)
        return Played(audio, list(track.beats_seconds), melody, w, False, False)

    s = min(max(0, int(round(window.start_seconds * sr))), n - 1)
    e = min(n, int(round(window.end_seconds * sr)))
    if e <= s:
        raise ValueError(f"window for {track.id!r} starts after the end of the audio")
    start, end = s / sr, e / sr
    beats = [b - start for b in track.beats_seconds if start <= b < end]
    hop = track.energy_hop_seconds
    if melody and hop:
        melody = melody[int(start / hop):int(np.ceil(end / hop))] or None
    used = window.model_copy(update={"start_seconds": start, "end_seconds": end})
    # a window within EDGE_TOLERANCE_SECONDS of the track's edge plays to that edge
    # (the track's listed duration and its decoded length can differ by a few ms)
    tol = int(EDGE_TOLERANCE_SECONDS * sr)
    return Played(audio[s:e], beats, melody, used, s > tol, e < n - tol)


def edge_fades(p: Played, track: Track, sr: int, first: bool, last: bool) -> tuple[np.ndarray, list[str]]:
    """Fade-in for the first window and fade-out for the last one when they
    cut into the song (other cuts are inside a seam). Returns new audio."""
    audio, notes = p.audio, []
    beat = beat_seconds(track)
    if first and p.trimmed_start:
        k = min(audio.shape[0] // 4, int(FADE_IN_BEATS * beat * sr))
        audio = audio.copy()
        audio[:k] *= np.linspace(0.0, 1.0, k, dtype=np.float32)[:, None]
        notes.append(f"{track.id}: mix starts at {p.window.start_seconds:.2f}s into the song, "
                     f"faded in over {k / sr:.2f}s")
    if last and p.trimmed_end:
        k = min(audio.shape[0] // 4, int(FADE_OUT_BEATS * beat * sr))
        audio = audio.copy() if audio is p.audio else audio
        audio[-k:] *= np.linspace(1.0, 0.0, k, dtype=np.float32)[:, None]
        notes.append(f"{track.id}: mix ends at {p.window.end_seconds:.2f}s into the song, "
                     f"faded out over {k / sr:.2f}s")
    return audio, notes
