"""Where each song is entered and left: cue points (owner: Sahil).

choose_windows picks one window [start, end] per track so every blend hands
the melody over cleanly: song A is left as its vocal or melody line ends,
song B comes in where it's instrumental, and only one melody plays at a time.

Candidate cues are phrase starts (8 bars, from cues.py), else bar lines on
the beat grid, else a plain time grid, plus each song's own start and end.
Every seam scores each (A end, B start) pair on

  handoff    A's melody_activity low or falling over the blend, B's low in
             its second half (the renderer holds B's mids down until then
             and lands the blend on its last beat; bands.py). Measured over
             the overlap, or over the stretch
             echo.py checks if that's longer, so a clean cue here also means
             the renderer won't fall back to echo-out.
  groove     the blend itself is full: both songs are near their normal
             loudness while they overlap. Quiet melody alone also matches an
             outro running into a sparse intro, which is dead air.
  flow       no long stretch without melody across the seam: from A's last
             melodic moment to B's first. Picking quiet-melody cues alone
             can leave half a minute of instrumental intro after A's vocal.
  payoff     B arrives: its energy and melody are up right after the blend,
             and best of all its energy jumps there (a chorus or drop
             starting on the landing beat), so the hook lands as A leaves.
  energy     A's loudness at its end close to B's at its start.
  structure  leaving at a section boundary or the song's own ending,
             entering at the song's start, an intro or a break.
  harmony    the part of A under the blend and the part of B over it
             stress the same pitch classes (harmony.py chroma, passed in
             from the audio; neutral without it).

Any part of a song can meet any part of the next one: set mode picks both
cues freely, so the two parts that match best meet wherever they are.

How long each blend runs depends on the two songs (plan_windows): every
seam is scored at whole phrases of 16, 32 and 64 beats of the outgoing
song's tempo (up to the renderer's 32 s), and each (A end, B start) takes
the length that scores best there. Longer blends earn a bonus when the
notes line up and lose it when the two melodies would sing over each other;
tempos too far apart to beat-match get one short blend. The A/B
preference leans every seam longer or shorter.

and each window pays a length cost: full mode trims as little as possible
(sparse stretches are cheap to lose), set mode sizes windows to hit
target_seconds and keeps each song's best (loudest) section. A dynamic program picks the windows that maximize the
summed seam quality minus the length costs over the whole running order, so
a middle song's entry and exit are chosen together and never collide.
"""
import math
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Literal, Optional, Sequence

import numpy as np

from backend.api.schemas import Section, Track, TrackWindow
from backend.audio.cues import BEATS_PER_BAR
from backend.audio.echo import CLASH_SECONDS, MELODY_CLASH, melody_clash
from backend.audio.edges import MAX_EDGE_SECONDS, MIN_DROP_DB
from backend.audio.harmony import Chroma, compatibility
from backend.audio.seam_plan import MAX_GLIDE_DEVIATION, deviation, fold_ratio
from backend.audio.transition import DEFAULT_SAFETY, SafetyLimits
from backend.audio.windows import MIN_WINDOW_SECONDS

# Seam quality. Melody handoff matters most; keep these summing to 1.0.
WEIGHTS = {"handoff": 0.25, "groove": 0.15, "flow": 0.15, "payoff": 0.15, "energy": 0.05,
           "structure": 0.10, "harmony": 0.15}
# harmony at or above this is called out as in key, below the low mark as rubbing.
HARMONY_GOOD = 0.75
HARMONY_POOR = 0.4

# Energy (relative to the song's 95th percentile) that counts as fully playing.
FULL_ENERGY = 0.7
# How far past the blend B's payoff is measured.
PAYOFF_SECONDS = 16.0
# Below this groove the reason says the blend is sparse.
SPARSE_GROOVE = 0.7
# flow: seconds across a seam with no melody from either song. Up to
# GAP_OK is breathing room; GAP_DEAD past that scores 0.
GAP_OK_SECONDS = 8.0
GAP_DEAD_SECONDS = 24.0
GAP_CAP_SECONDS = 48.0
MELODY_SMOOTH_SECONDS = 4.0  # melody "on" = this moving average >= QUIET_MELODY

# A seam scoring below this, or with a handoff below POOR_HANDOFF, is called
# out in the reason as the least-bad cue.
POOR_SEAM = 0.5
POOR_HANDOFF = 0.3

# Stand-in for a sub-score whose inputs are missing.
NEUTRAL = 0.5

# Taken off the handoff when both sides reach echo.py's clash level: the
# renderer won't blend that seam at all, it echoes A out instead.
CLASH_PENALTY = 0.25

# melody_activity wording for reasons.
QUIET_MELODY = 0.35
BUSY_MELODY = 0.55

# Full mode: each trimmed second costs this much seam quality, and no edge
# loses more than this fraction of its song. A sparse second (energy under
# FULL_ENERGY) costs proportionally less, down to SPARSE_TRIM_WEIGHT.
FULL_TRIM_COST_PER_SECOND = 0.003
FULL_MAX_TRIM_FRACTION = 0.3
SPARSE_TRIM_WEIGHT = 0.2

# Set mode.
SET_MIN_WINDOW_SECONDS = 45.0
DEFAULT_SET_SECONDS_PER_TRACK = 90.0
SET_TOLERANCE = 0.10
SET_LENGTH_COST = 1.0  # per unit of relative length error
BEST_SECTION_COST = 0.15  # for leaving out the whole best section: a better-matched seam can win
SET_LENGTH_PASSES = 4
BEST_SECONDS = 30.0  # length of the "best part" every set window keeps

# track.metadata keys for fade lengths measured by edges.py during analysis.
# When present they're the same numbers the renderer works from; without
# them the edges are estimated from energy_rms, which reads fades short.
FADE_OUT_KEY = "fade_out_seconds"
FADE_IN_KEY = "fade_in_seconds"

# Fallback cue grid when a track has no beats.
TIME_GRID_SECONDS = 4.0
# Phrase starts closer than this to a song's start or end merge into it.
EDGE_SNAP_SECONDS = 1.0
# A cue within this of a section boundary counts as on it.
BOUNDARY_SECONDS = 0.5

GRID_NAMES = {"phrase": "phrase", "bar": "bar", "time": ""}

# Drop transitions (backend/audio/drop.py). A drop is a phrase start where
# the energy over the next DROP_LOOK_SECONDS jumps by DROP_JUMP (full
# strength) over the DROP_LOOK_SECONDS before, up to at least DROP_LEVEL.
DROP_LOOK_SECONDS = 8.0
DROP_JUMP = 0.3
DROP_LEVEL = 0.85
DROP_MIN_STRENGTH = 0.4
# The incoming build before its drop: the overlap rounded to whole bars.
BUILD_MIN_BARS = 2
BUILD_MAX_SECONDS = 24.0
# Drop seam quality = DROP_BASE + DROP_STRENGTH_WEIGHT * strength
# + 0.1 x A's energy as it leaves + 0.1 x how natural leaving A there is.
# Capped near 0.65: a long beat-matched blend is what makes a set flow, so a
# drop only replaces a blend that would be weak (clean ones score 0.75-0.9).
DROP_BASE = 0.2
DROP_STRENGTH_WEIGHT = 0.25
# Tempos further apart than the renderer can glide get a ~2-beat blend: the
# blend's quality is scaled by this, so a drop (cut) can take over.
UNMATCHED_BLEND = 0.6
# The reason says B takes over on an energy jump at or above this rise.
LANDING_RISE = 0.5

# Blend length (plan_windows): each seam is scored at whole phrases of the
# outgoing song's beat and takes the length that suits those two songs.
# Under 16 beats a beat-matched blend reads as a cut, not a mix.
BLEND_FALLBACK_SECONDS = (4.0, 8.0, 16.0, 24.0)  # no tempo to count beats in
MIN_BLEND_SECONDS = 4.0
MAX_BLEND_SECONDS = DEFAULT_SAFETY.max_overlap_seconds
# Tempos the renderer can't glide between get a short blend (it plays ~2 beats).
UNMATCHED_BLEND_SECONDS = 4.0
# A longer blend earns LENGTH_WEIGHT x reach x (LENGTH_BASE + 2 x (harmony - 0.5)
# - stack + PREFERENCE_LEAN x overlap_preference), reach 0 at MIN_BLEND_SECONDS
# to 1 at MAX_BLEND_SECONDS. Songs in key ride together for long; songs that
# rub, or whose melodies would sing over each other (stack: the geometric
# mean of the two melody levels over the blend, plus CLASH_STACK where the
# renderer has to swap the vocals), hand over quickly. The A/B votes lean
# every seam longer or shorter without overriding the songs.
LENGTH_WEIGHT = 0.1
LENGTH_BASE = 0.5
PREFERENCE_LEAN = 1.0
CLASH_STACK = 0.5

TransitionLength = Literal["auto", "longer", "extended"]


@dataclass(frozen=True)
class BlendSize:
    """The user's transition length setting, as blend lengths to try and how
    hard to lean toward the long end."""
    beats: tuple[int, ...]  # phrase lengths to try, in beats of the outgoing song
    fallback: tuple[float, ...]  # seconds to try without a tempo
    cap: float  # longest blend the renderer may play (SafetyLimits.max_overlap_seconds)
    lean: float  # added to overlap_preference
    stack_weight: float  # how strongly two melodies singing at once shorten a blend
    size_for_longest: bool  # set mode sizes windows for the longest length, not the middle one


# auto: the songs decide. longer: every seam leans long and may reach 64
# beats at 96+ BPM. extended: up to 128 beats and 60 s, vocal clashes still
# shorten a blend but half as much, and set-mode windows grow to fit.
BLEND_SIZES: dict[str, BlendSize] = {
    "auto": BlendSize((16, 32, 64), BLEND_FALLBACK_SECONDS, MAX_BLEND_SECONDS, 0.0, 1.0, False),
    "longer": BlendSize((16, 32, 64), (8.0, 16.0, 24.0, 32.0), 40.0, 1.0, 1.0, False),
    "extended": BlendSize((16, 32, 64, 128), (8.0, 16.0, 32.0, 48.0, 60.0), 60.0, 1.5, 0.5, True),
}


def overlap_limits(transition_length: str = "auto") -> SafetyLimits:
    """The renderer's safety limits for a transition length setting."""
    return replace(DEFAULT_SAFETY, max_overlap_seconds=BLEND_SIZES[transition_length].cap)


@dataclass
class _Cues:
    track: Track
    grid: np.ndarray  # candidate cue times, first 0, last the duration
    kind: str  # "phrase" | "bar" | "time"
    melody: Optional[np.ndarray]
    energy: Optional[np.ndarray]  # energy_rms relative to the track's 95th percentile
    hop: Optional[float]
    head: float  # quiet lead-in at the song's start, seconds
    tail: float  # fade-out at the song's end, seconds
    drop_only: frozenset = frozenset()  # grid times added only as drop build starts
    chroma: Optional[Chroma] = None

    @property
    def duration(self) -> float:
        return self.track.duration_seconds

    def content(self, t: np.ndarray) -> np.ndarray:
        """Seconds of music from 0 to t, a sparse second counting as less."""
        if self.energy is None or not self.hop:
            return np.asarray(t, dtype=np.float64)
        weight = np.clip(self.energy / FULL_ENERGY, SPARSE_TRIM_WEIGHT, 1.0)
        cum = np.concatenate([[0.0], np.cumsum(weight) * self.hop])
        times = np.arange(cum.size) * self.hop
        return np.interp(t, times, cum)

    @property
    def name(self) -> str:
        stem = Path(self.track.filename).stem
        return stem if len(stem) <= 40 else stem[:39] + "…"


@dataclass
class _Seam:
    quality: np.ndarray  # (len(A ends), len(B starts))
    handoff: np.ndarray
    groove: np.ndarray
    energy: np.ndarray
    in_payoff: np.ndarray  # B's melody right after the blend
    gap: np.ndarray  # seconds with no melody from either song across the seam
    in_quiet: np.ndarray  # seconds from B's start until its melody is on
    out_level: np.ndarray
    out_first: np.ndarray
    out_last: np.ndarray
    out_full: np.ndarray
    in_level: np.ndarray
    in_full: np.ndarray
    span: float
    is_drop: np.ndarray  # (A ends, B starts): the drop transition beats the blend here
    drop_at: dict  # B start time -> (its drop time, strength)
    in_rise: Optional[np.ndarray] = None  # B's energy jump on the landing beat, 0-1
    harmony: Optional[np.ndarray] = None  # (A ends, B starts), None without chroma
    overlap: float = 0.0  # the blend length this seam was scored at
    stack: Optional[np.ndarray] = None  # (A ends, B starts): melody both carry over the blend


@dataclass
class _SeamChoice:
    """One seam scored at one or more blend lengths; every (A end, B start)
    takes the length that scores best there."""
    options: list[_Seam]  # shortest first
    score: np.ndarray  # (A ends, B starts): the chosen length's quality plus its length bonus
    pick: np.ndarray  # (A ends, B starts): index into options

    def at(self, i: int, j: int) -> _Seam:
        return self.options[int(self.pick[i, j])]


def _fixed(seam: _Seam) -> _SeamChoice:
    return _SeamChoice([seam], seam.quality, np.zeros(seam.quality.shape, dtype=int))


def _length_bonus(seam: _Seam, preference: float, stack_weight: float = 1.0) -> np.ndarray:
    """preference: overlap_preference plus the setting's lean. Reach keeps
    growing past MAX_BLEND_SECONDS for extended blends."""
    reach = (math.log2(max(seam.overlap, MIN_BLEND_SECONDS) / MIN_BLEND_SECONDS)
             / math.log2(MAX_BLEND_SECONDS / MIN_BLEND_SECONDS))
    harmony = seam.harmony if seam.harmony is not None else NEUTRAL
    stack = seam.stack if seam.stack is not None else 0.0
    bonus = LENGTH_WEIGHT * reach * (LENGTH_BASE + 2.0 * (harmony - 0.5) - stack_weight * stack
                                     + PREFERENCE_LEAN * preference)
    return np.where(seam.is_drop, 0.0, bonus)  # a drop's length is its build, not a blend


def _choose_length(options: list[_Seam], preference: float, stack_weight: float) -> _SeamChoice:
    scores = np.stack([o.quality + _length_bonus(o, preference, stack_weight) for o in options])
    pick = np.argmax(scores, axis=0)
    return _SeamChoice(options, np.take_along_axis(scores, pick[None], 0)[0], pick)


def blend_lengths(a: Track, b: Track, transition_length: str = "auto") -> list[float]:
    """Blend lengths to try between A and B, shortest first: whole phrases
    (16-64 beats, up to 128 for "extended") of A's beat within the setting's
    cap, one short blend when the tempos can't be beat-matched, a plain time
    ladder without tempo."""
    size = BLEND_SIZES[transition_length]
    if not tempos_blend(a, b):
        return [UNMATCHED_BLEND_SECONDS]
    if not a.bpm or not math.isfinite(a.bpm) or a.bpm <= 0:
        return list(size.fallback)
    beat = 60.0 / a.bpm
    fits = [n * beat for n in size.beats if MIN_BLEND_SECONDS <= n * beat <= size.cap]
    return fits or [float(np.clip(16 * beat, MIN_BLEND_SECONDS, size.cap))]


def tempos_blend(a: Track, b: Track) -> bool:
    """Whether the renderer can beat-match a blend between the two songs
    (half/double time folded). True when either tempo is unknown."""
    if not a.bpm or not b.bpm:
        return True
    ratio, _ = fold_ratio(60.0 / a.bpm, 60.0 / b.bpm)
    return deviation(ratio) <= MAX_GLIDE_DEVIATION


def find_drops(track: Track) -> list[tuple[float, float]]:
    """(time, strength 0-1) of the song's drops: phrase starts where the
    energy jumps up to near full. [] without phrases or energy."""
    hop = track.energy_hop_seconds
    if not track.phrase_starts_seconds or not track.energy_rms or not hop:
        return []
    energy = np.asarray(track.energy_rms, dtype=np.float64)
    ref = float(np.percentile(energy, 95))
    if ref <= 0:
        return []
    energy = np.clip(energy / ref, 0.0, 1.0)
    out = []
    for t in track.phrase_starts_seconds:
        if t < DROP_LOOK_SECONDS or t > track.duration_seconds - DROP_LOOK_SECONDS:
            continue
        before = _mean(energy, hop, t - DROP_LOOK_SECONDS, t)
        after = _mean(energy, hop, t, t + DROP_LOOK_SECONDS)
        strength = (np.clip((after - before) / DROP_JUMP, 0.0, 1.0)
                    * np.clip(after / DROP_LEVEL, 0.0, 1.0))
        if strength >= DROP_MIN_STRENGTH:
            out.append((float(t), round(float(strength), 3)))
    return out


def _build_starts(track: Track, overlap: float) -> dict:
    """Build start -> (drop time, strength) for every drop of the song: the
    beat a whole number of bars (about `overlap`) before it."""
    beats = np.asarray(track.beats_seconds, dtype=np.float64)
    if beats.size < 2 or not track.bpm:
        return {}
    bar = BEATS_PER_BAR * 60.0 / track.bpm
    bars = int(np.clip(round(overlap / bar), BUILD_MIN_BARS, max(BUILD_MIN_BARS, BUILD_MAX_SECONDS // bar)))
    starts = {}
    for t, strength in find_drops(track):
        i = int(np.argmin(np.abs(beats - t)))
        j = i - bars * BEATS_PER_BAR
        if j >= 0 and beats[j] > EDGE_SNAP_SECONDS:
            starts[round(float(beats[j]), 3)] = (round(float(beats[i]), 3), strength)
    return starts


def _series(values: Sequence[float], hop: Optional[float]) -> Optional[np.ndarray]:
    if not values or not hop or hop <= 0:
        return None
    arr = np.asarray(values, dtype=np.float64)
    arr = np.where(np.isfinite(arr), arr, 0.0)
    return arr


def _cues(track: Track) -> _Cues:
    duration = track.duration_seconds
    if track.phrase_starts_seconds:
        points, kind = track.phrase_starts_seconds, "phrase"
    elif len(track.beats_seconds) >= 2:
        points, kind = track.beats_seconds[::BEATS_PER_BAR], "bar"
    else:
        points, kind = list(np.arange(TIME_GRID_SECONDS, duration, TIME_GRID_SECONDS)), "time"
    inner = sorted({round(float(p), 3) for p in points
                    if EDGE_SNAP_SECONDS < p < duration - EDGE_SNAP_SECONDS})
    hop = track.energy_hop_seconds
    energy = _series(track.energy_rms, hop)
    if energy is not None:
        ref = float(np.percentile(energy, 95))
        energy = np.clip(energy / ref, 0.0, 1.0) if ref > 0 else None
    melody = _series(track.melody_activity, hop)
    if melody is not None:
        melody = np.clip(melody, 0.0, 1.0)
    head = _measured_edge(track, FADE_IN_KEY)
    if head is None:
        head = _edge_seconds(energy, hop, from_end=False)
    tail = _measured_edge(track, FADE_OUT_KEY)
    if tail is None:
        tail = _edge_seconds(energy, hop, from_end=True)
    return _Cues(track, np.array([0.0, *inner, duration]), kind, melody, energy, hop, head, tail)


def _measured_edge(track: Track, key: str) -> Optional[float]:
    """Fade length analysis stored in track.metadata, None if absent or unusable."""
    value = track.metadata.get(key)
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        return None
    if not math.isfinite(value) or value < 0:
        return None
    return float(min(value, MAX_EDGE_SECONDS))


def _edge_seconds(energy: Optional[np.ndarray], hop: Optional[float], from_end: bool) -> float:
    """Quiet lead-in or fade-out at a song's edge, from energy_rms: the stretch
    more than MIN_DROP_DB below full level. transition.py places a blend that
    touches the song's own start or end past these edges (edges.py measures
    them exactly from the audio), so cues at the edges are judged there too."""
    if energy is None or not hop:
        return 0.0
    loud = np.flatnonzero(energy >= 10 ** (-MIN_DROP_DB / 20))
    if loud.size == 0:
        return 0.0
    frames = energy.size - 1 - loud[-1] if from_end else loud[0]
    return float(min(frames * hop, MAX_EDGE_SECONDS))


def _mean(series: Optional[np.ndarray], hop: Optional[float], t0: float, t1: float) -> float:
    """Mean of series over [t0, t1] seconds, NaN when there's no data there."""
    if series is None or not hop:
        return math.nan
    i0, i1 = max(0, int(t0 / hop)), int(math.ceil(t1 / hop))
    seg = series[i0:i1]
    return float(seg.mean()) if seg.size else math.nan


def _section_at(sections: list[Section], t: float) -> Optional[Section]:
    for s in sections:
        if s.start_seconds - BOUNDARY_SECONDS <= t < s.end_seconds - BOUNDARY_SECONDS:
            return s
    return None


def _on_boundary(sections: list[Section], t: float) -> bool:
    return any(abs(s.start_seconds - t) <= BOUNDARY_SECONDS for s in sections[1:])


def _structure_out(c: _Cues, e: float) -> float:
    """How natural it is to leave the song at e."""
    if e >= c.duration:
        return 1.0
    sections = c.track.sections
    if not sections:
        return NEUTRAL
    here = _section_at(sections, e)
    label = here.label if here else None
    if _on_boundary(sections, e):
        before = _section_at(sections, e - 2 * BOUNDARY_SECONDS)
        if before is not None and before.label == "intro":
            return 0.3
        return 1.0 if label in ("outro", "break") else 0.8
    return 0.8 if label == "outro" else NEUTRAL


def _structure_in(c: _Cues, s: float) -> float:
    """How natural it is to enter the song at s."""
    if s <= 0:
        return 1.0
    sections = c.track.sections
    if not sections:
        return NEUTRAL
    here = _section_at(sections, s)
    label = here.label if here else None
    if label == "outro":
        return 0.2
    if _on_boundary(sections, s):
        return 1.0 if label in ("intro", "break") else 0.8
    return NEUTRAL


def _nan_to(arr: np.ndarray, value: float) -> np.ndarray:
    return np.where(np.isnan(arr), value, arr)


def _quiet_run(c: _Cues, t: float, forward: bool) -> float:
    """Seconds from t until the song's melody is on (forward), or since it
    was last on (backward), capped at GAP_CAP_SECONDS. 0 without data."""
    if c.melody is None or not c.hop:
        return 0.0
    n = max(1, int(round(MELODY_SMOOTH_SECONDS / c.hop)))
    i = int(t / c.hop)
    cap = int(GAP_CAP_SECONDS / c.hop)
    # Only the side of t that plays: B from its start on, A up to its end.
    seg = c.melody[i:i + cap] if forward else c.melody[max(0, i - cap):i][::-1]
    if not seg.size:
        return 0.0
    hits = np.flatnonzero(np.convolve(seg, np.ones(n) / n, mode="same") >= QUIET_MELODY)
    return float((hits[0] if hits.size else seg.size) * c.hop)


def _score_seam(a: _Cues, b: _Cues, overlap: float, drop_at: Optional[dict] = None) -> _Seam:
    drop_at = drop_at or {}
    span = max(overlap, CLASH_SECONDS)
    # Where the blend sits: before A's fade-out tail at its own end, after
    # B's lead-in at its own start.
    ends = [e - a.tail if e >= a.duration else e for e in a.grid]
    starts = [s + b.head if s <= 0 else s for s in b.grid]
    out_full = np.array([_mean(a.melody, a.hop, e - span, e) for e in ends])
    out_first = np.array([_mean(a.melody, a.hop, e - span, e - span / 2) for e in ends])
    out_last = np.array([_mean(a.melody, a.hop, e - span / 2, e) for e in ends])
    in_full = np.array([_mean(b.melody, b.hop, s, s + span) for s in starts])
    in_last = np.array([_mean(b.melody, b.hop, s + span / 2, s + span) for s in starts])

    out_level = _nan_to(0.35 * out_full + 0.65 * out_last, NEUTRAL)
    in_level = _nan_to(0.65 * in_last + 0.35 * in_full, NEUTRAL)
    falling = _nan_to(np.maximum(0.0, out_first - out_last), 0.0)
    x, y = out_level[:, None], in_level[None, :]
    # Two melodies at once is the main thing to avoid (the product term);
    # beyond that, A still singing as it leaves is worse than B starting busy.
    handoff = 1.0 - (0.5 * np.sqrt(x * y) + 0.3 * x + 0.2 * y) + 0.3 * falling[:, None]
    # Echo-out is judged on echo.py's fixed CLASH_SECONDS at each edge, not on
    # span: with a long overlap the two can disagree.
    out_edge = np.array([_mean(a.melody, a.hop, e - CLASH_SECONDS, e) for e in ends])
    in_edge = np.array([_mean(b.melody, b.hop, s, s + CLASH_SECONDS) for s in starts])
    clash = (_nan_to(out_edge, 0.0) >= MELODY_CLASH)[:, None] & (_nan_to(in_edge, 0.0) >= MELODY_CLASH)[None, :]
    handoff = np.clip(handoff - CLASH_PENALTY * clash, 0.0, 1.0)
    # How much melody the two stack over the blend (unknown counts as none):
    # the longer the blend, the longer that lasts.
    out_heard = _nan_to(0.35 * out_full + 0.65 * out_last, 0.0)
    in_heard = _nan_to(0.65 * in_last + 0.35 * in_full, 0.0)
    stack = np.sqrt(out_heard[:, None] * in_heard[None, :]) + CLASH_STACK * clash

    e_out = np.array([_mean(a.energy, a.hop, e - span, e) for e in ends])
    e_in = np.array([_mean(b.energy, b.hop, s, s + span) for s in starts])
    energy = _nan_to(1.0 - np.abs(e_out[:, None] - e_in[None, :]), NEUTRAL)

    x, y = e_out[:, None], e_in[None, :]
    groove = _nan_to(np.clip((0.6 * (x + y) / 2 + 0.4 * np.minimum(x, y)) / FULL_ENERGY, 0.0, 1.0),
                     NEUTRAL)

    after = [(s + span, s + span + PAYOFF_SECONDS) for s in starts]
    in_payoff = np.array([_mean(b.melody, b.hop, t0, t1) for t0, t1 in after])
    e_after = np.array([_mean(b.energy, b.hop, t0, t1) for t0, t1 in after])
    # The landing beat: B's energy over the next DROP_LOOK_SECONDS against
    # the blend before it, scored like a drop.
    e_land = np.array([_mean(b.energy, b.hop, s + span, s + span + DROP_LOOK_SECONDS) for s in starts])
    rise = _nan_to(np.clip((e_land - e_in) / DROP_JUMP, 0.0, 1.0) * np.clip(e_land / DROP_LEVEL, 0.0, 1.0),
                   0.0)
    payoff = _nan_to(0.35 * np.clip(e_after / FULL_ENERGY, 0.0, 1.0) + 0.35 * in_payoff + 0.3 * rise,
                     NEUTRAL)

    # Seam timeline with A's blend end at 0: A's melody last sounds at
    # -out_quiet, B's first sounds at in_quiet - span.
    out_quiet = np.array([_quiet_run(a, e, forward=False) for e in ends])
    in_quiet = np.array([_quiet_run(b, s, forward=True) for s in starts])
    gap = np.maximum(0.0, out_quiet[:, None] + in_quiet[None, :] - span)
    flow = np.clip(1.0 - (gap - GAP_OK_SECONDS) / GAP_DEAD_SECONDS, 0.0, 1.0)
    if a.melody is None or b.melody is None:
        flow = np.full_like(flow, NEUTRAL)

    st_out = np.array([_structure_out(a, e) for e in ends])
    st_in = np.array([_structure_in(b, s) for s in starts])
    structure = (st_out[:, None] + st_in[None, :]) / 2

    harmony = None
    if a.chroma is not None and b.chroma is not None:
        ca = [a.chroma.mean(e - span, e) for e in ends]
        cb = [b.chroma.mean(s, s + span) for s in starts]
        harmony = np.array([[compatibility(x, y) for y in cb] for x in ca], dtype=float)
        harmony = _nan_to(harmony, NEUTRAL)

    # Quiet melody over a quiet blend is dead air, not a clean handoff.
    quality = (WEIGHTS["handoff"] * handoff * groove + WEIGHTS["groove"] * groove
               + WEIGHTS["flow"] * flow + WEIGHTS["payoff"] * payoff[None, :]
               + WEIGHTS["energy"] * energy + WEIGHTS["structure"] * structure
               + WEIGHTS["harmony"] * (harmony if harmony is not None else NEUTRAL))
    if not tempos_blend(a.track, b.track):
        quality = quality * UNMATCHED_BLEND

    is_drop = np.zeros_like(quality, dtype=bool)
    drop_cols = [(j, drop_at[round(float(s), 3)][1]) for j, s in enumerate(b.grid)
                 if round(float(s), 3) in drop_at]
    if drop_cols:
        leaving = _nan_to(np.clip(e_out / FULL_ENERGY, 0.0, 1.0), NEUTRAL)
        for j, strength in drop_cols:
            q = DROP_BASE + DROP_STRENGTH_WEIGHT * strength + 0.1 * leaving + 0.1 * st_out
            is_drop[:, j] = q > quality[:, j]
            quality[:, j] = np.maximum(quality[:, j], q)
            if round(float(b.grid[j]), 3) in b.drop_only:
                # Off the phrase grid: entering there only works as a drop.
                quality[~is_drop[:, j], j] = 0.0
    return _Seam(quality, handoff, groove, energy, in_payoff, gap, in_quiet, out_level, out_first, out_last,
                 out_full, in_level, in_full, span, is_drop, drop_at, in_rise=rise, harmony=harmony,
                 overlap=overlap, stack=stack)


def _window_melody(c: _Cues, start: float, end: float) -> list[float]:
    """melody_activity cut to [start, end] the way audio/windows.play cuts it."""
    if not c.track.melody_activity or not c.hop:
        return []
    return list(c.track.melody_activity[int(start / c.hop):int(math.ceil(end / c.hop))])


def _renderer_echoes(a: _Cues, a_start: float, a_end: float,
                     b: _Cues, b_start: float, b_end: float) -> bool:
    """Whether transition.py will echo this seam out, asked of echo.melody_clash
    with the same window-relative melody and length the renderer passes it.
    Fade edges only exist at a song's own ending or start; the renderer
    measures them from audio, this uses the energy estimate."""
    tail = a.tail if a_end >= a.duration else 0.0
    head = b.head if b_start <= 0 else 0.0
    clash, _, _ = melody_clash(_window_melody(a, a_start, a_end), _window_melody(b, b_start, b_end),
                               a.hop, a_end - a_start, tail, head)
    return clash


def best_section(track: Track) -> Optional[tuple[float, float]]:
    """(start, end) of the song's best part, None without data.

    A section labeled "chorus" or "drop" if analysis ever labels one (the
    loudest of them). Otherwise the loudest BEST_SECONDS of energy_rms,
    snapped out to the cue grid and kept out of a labeled intro or outro.
    cues.py sections alone are too coarse for this: one can span half a song.
    """
    hits = [s for s in track.sections if s.label in ("chorus", "drop")]
    if hits:
        best = max(hits, key=lambda s: s.energy)
        return best.start_seconds, best.end_seconds
    hop = track.energy_hop_seconds
    if not track.energy_rms or not hop:
        return None
    energy = np.asarray(track.energy_rms, dtype=np.float64)
    energy = np.where(np.isfinite(energy), energy, 0.0)  # as _series does
    for s in track.sections:
        if s.label in ("intro", "outro"):
            energy[int(s.start_seconds / hop):int(math.ceil(s.end_seconds / hop))] = 0.0
    n = max(1, min(len(energy), int(round(BEST_SECONDS / hop))))
    sums = np.convolve(energy, np.ones(n), mode="valid")
    i = int(np.argmax(sums))
    t0, t1 = i * hop, min(track.duration_seconds, (i + n) * hop)
    grid = _cues(track).grid
    return float(grid[grid <= t0 + BOUNDARY_SECONDS].max()), float(grid[grid >= t1 - BOUNDARY_SECONDS].min())


def _min_lengths(tracks: list[Track], overlaps: list[float]) -> list[float]:
    """Each window must be at least twice each adjoining overlap (the renderer
    caps an overlap at half a window), which also keeps a middle song's head
    and tail seams apart."""
    out = []
    for i in range(len(tracks)):
        head = overlaps[i - 1] if i > 0 else 0.0
        tail = overlaps[i] if i < len(overlaps) else 0.0
        out.append(max(MIN_WINDOW_SECONDS, 2 * head, 2 * tail))
    return out


def _unary(c: _Cues, i: int, n: int, min_len: float, target: Optional[float],
           best: Optional[tuple[float, float]], lead: float = 0.0) -> np.ndarray:
    """Cost-adjusted score of every (start, end) for one track; -inf = invalid.
    lead: the incoming blend's length. The song is only heard in full from
    where that blend lands, so its best section counts as kept from there."""
    g, dur = c.grid, c.duration
    s, e = g[:, None], g[None, :]
    length = e - s
    whole = (s == 0) & (e == dur)
    if dur < min_len:
        return np.where(whole, 0.0, -np.inf)

    if target is None:
        max_trim = FULL_MAX_TRIM_FRACTION * dur
        valid = (length >= min_len) & (s <= max_trim) & (dur - e <= max_trim)
        if i == 0:
            valid &= s == 0
        if i == n - 1:
            valid &= e == dur
        content = c.content(g)
        trimmed = content[:, None] + (content[-1] - content[None, :])
        score = -FULL_TRIM_COST_PER_SECOND * trimmed
    else:
        floor = max(min_len, min(SET_MIN_WINDOW_SECONDS, dur))
        valid = length >= floor
        score = -SET_LENGTH_COST * np.abs(length - target) / max(target, 1.0)
        if best is not None:
            b0, b1 = best
            heard = s + lead
            covered = np.clip(np.minimum(e, b1) - np.maximum(heard, b0), 0.0, None)
            need = np.maximum(np.minimum(b1 - b0, e - heard), 0.0)
            score = score - BEST_SECTION_COST * (1.0 - covered / np.maximum(need, 1e-6))
        # Tie-break: a set opens near its first song's start and closes near
        # its last song's end.
        if i == 0:
            score = score - 1e-4 * s
        if i == n - 1:
            score = score - 1e-4 * (dur - e)
    return np.where(valid | whole, score, -np.inf)


def _solve(cues: list[_Cues], seams: list[_SeamChoice], unaries: list[np.ndarray]) -> tuple[list[tuple[int, int]], float]:
    """Best (start index, end index) per track: max of seam quality + unary."""
    n = len(cues)
    # f[e] = best total with track i ending at grid index e; back[i][e] = start index.
    best_s = np.argmax(unaries[0], axis=0)
    f = unaries[0][best_s, np.arange(unaries[0].shape[1])]
    backs_s, backs_prev = [best_s], []
    for i in range(1, n):
        q = seams[i - 1].score
        total = f[:, None] + q  # (prev ends, starts)
        prev_e = np.argmax(total, axis=0)
        g = total[prev_e, np.arange(total.shape[1])]
        cand = unaries[i] + g[:, None]  # (starts, ends)
        s_idx = np.argmax(cand, axis=0)
        f = cand[s_idx, np.arange(cand.shape[1])]
        backs_s.append(s_idx)
        backs_prev.append(prev_e)
    e = int(np.argmax(f))
    score = float(f[e])
    picks: list[tuple[int, int]] = []
    for i in range(n - 1, -1, -1):
        s = int(backs_s[i][e])
        picks.append((s, e))
        if i > 0:
            e = int(backs_prev[i - 1][s])
    return picks[::-1], score


def _length_targets(cues: list[_Cues], floors: list[float], total: float) -> list[float]:
    """Split total window seconds across tracks as evenly as their lengths allow."""
    lo = np.array([min(f, c.duration) for f, c in zip(floors, cues)])
    hi = np.array([c.duration for c in cues])
    if total <= lo.sum():
        return lo.tolist()
    if total >= hi.sum():
        return hi.tolist()
    a, b = 0.0, float(hi.max())
    for _ in range(60):
        mid = (a + b) / 2
        if np.clip(mid, lo, hi).sum() < total:
            a = mid
        else:
            b = mid
    return np.clip(b, lo, hi).tolist()


def _clock(t: float) -> str:
    t = max(0.0, t)
    return f"{int(t // 60)}:{int(t % 60):02d}"


def _melody_phrase(level: float) -> str:
    if level < QUIET_MELODY:
        return f"its melody is quiet ({level:.2f})"
    if level < BUSY_MELODY:
        return f"its melody is moderate ({level:.2f})"
    return f"its melody is still busy ({level:.2f})"


def _start_text(c: _Cues, s: float, seam: Optional[_Seam], j: int) -> str:
    cue = GRID_NAMES[c.kind]
    if seam is None:
        return "Starts at the top" if s <= 0 else f"Starts at {_clock(s)}" + (f" on a {cue} start" if cue else "")
    text = "Comes in from its start" if s <= 0 else f"Comes in at {_clock(s)}" + (f" on a {cue} start" if cue else "")
    here = _section_at(c.track.sections, s) if s > 0 else None
    if here is not None and here.label in ("intro", "break") and _on_boundary(c.track.sections, s):
        text += f" (its {here.label})"
    full = seam.in_full[j]
    if seam.in_rise is not None and seam.in_rise[j] >= LANDING_RISE:
        land = s + (0.0 if s > 0 else c.head) + seam.span
        near = c.grid[np.argmin(np.abs(c.grid - land))]
        if abs(near - land) <= 0.3 * seam.span:
            land = float(near)  # the renderer lands whole-phrase blends on the cue grid
        return text + (f", building under the outgoing song, and takes over at "
                       f"{_clock(land)} as its energy jumps")
    if not math.isnan(full):
        text += f", where {_melody_phrase(float(seam.in_level[j]))}"
        arrives = float(seam.in_quiet[j]) - seam.span
        if seam.in_level[j] < QUIET_MELODY and -seam.span / 2 < arrives <= GAP_OK_SECONDS:
            text += (", then its melody kicks in right after the blend" if arrives <= 2.0
                     else f", then its melody comes in {arrives:.0f} s after the blend")
    return text


def _end_text(c: _Cues, e: float, seam: Optional[_Seam], i: int) -> str:
    cue = GRID_NAMES[c.kind]
    if seam is None:
        if e >= c.duration:
            return "plays to the end"
        return f"fades out at {_clock(e)}" + (f" on a {cue} start" if cue else "")
    text = "plays to its own ending" if e >= c.duration else f"leaves at {_clock(e)}" + (f" on a {cue} start" if cue else "")
    first, last = seam.out_first[i], seam.out_last[i]
    if not math.isnan(seam.out_full[i]):
        if first - last >= 0.1:
            text += f" as its melody drops ({first:.2f} to {last:.2f})"
        else:
            text += f" where {_melody_phrase(float(seam.out_level[i]))}"
    else:
        text += " (no melody data, so the cue follows energy and phrasing)"
    return text


def _length_text(c: _Cues, nxt: _Cues, seam: _Seam, options: list[_Seam], i: int, j: int,
                 capped: bool) -> str:
    """Why the blend into the next song is as long as it is ("" if there was no choice)."""
    beats = f"{round(seam.overlap * c.track.bpm / 60.0)} beats, " if c.track.bpm else ""
    size = f"{beats}{seam.overlap:.0f} s"
    if not tempos_blend(c.track, nxt.track):
        return (f"Short blend into {nxt.name} ({size}): their tempos are too far apart "
                f"to beat-match for long.")
    if len(options) < 2:
        return ""
    # Past the auto cap only the longer settings reach.
    extended = seam.overlap > MAX_BLEND_SECONDS + 1e-6
    if capped:
        if extended:
            return f"Extended blend into {nxt.name} ({size}), the longest that fits these windows."
        return f"Blends into {nxt.name} over {size}, the longest that fits these windows."
    longest = options[-1]
    if seam is longest:
        if seam.harmony is not None and seam.harmony[i, j] >= HARMONY_GOOD:
            why = "their notes line up"
        elif seam.quality[i, j] >= POOR_SEAM and seam.handoff[i, j] >= POOR_HANDOFF:
            why = "the melody hands over cleanly the whole way"
        else:
            why = "no shorter one scored better here"
        return f"{'Extended' if extended else 'Long'} blend into {nxt.name} ({size}): {why}."
    if extended:
        return f"Extended blend into {nxt.name} ({size}), the length that suits this pair best."
    if seam is options[0]:
        if longest.harmony is not None and longest.harmony[i, j] < HARMONY_POOR:
            why = "their notes rub over a longer stretch"
        elif longest.handoff[i, j] < seam.handoff[i, j] - 0.05:
            why = "a longer blend would stack the two melodies"
        else:
            why = "a longer one scored worse here"
        return f"Short blend into {nxt.name} ({size}): {why}."
    return f"Blends into {nxt.name} over {size}, the length that suits this pair best."


def choose_windows(
    tracks: list[Track],
    overlap_seconds: list[float],
    target_seconds: float | None = None,
    chroma: Optional[Sequence[Optional[Chroma]]] = None,
) -> list[TrackWindow]:
    """One window per track, same order, every seam blending for the given
    overlap_seconds. reason says why each cue was picked.

    chroma: per track, same order (harmony.chroma_for); None = no harmony term.

    target_seconds None = full mode: keep songs as long as possible, trimming
    an edge only to reach a cleaner handoff. Otherwise set mode: windows of at
    least 45 s (or the whole song if shorter) that keep each song's best
    section, with sum(window lengths) - sum(overlaps) close to target_seconds.
    """
    if not tracks:
        return []
    if len(overlap_seconds) != len(tracks) - 1:
        raise ValueError(f"need {len(tracks) - 1} overlaps for {len(tracks)} tracks, got {len(overlap_seconds)}")
    overlaps = [max(0.0, float(o)) for o in overlap_seconds]
    return _plan(tracks, [[o] for o in overlaps], target_seconds, chroma, None, BLEND_SIZES["auto"])[0]


def plan_windows(
    tracks: list[Track],
    overlap_preference: float = 0.0,
    target_seconds: float | None = None,
    chroma: Optional[Sequence[Optional[Chroma]]] = None,
    transition_length: TransitionLength = "auto",
) -> tuple[list[TrackWindow], list[float]]:
    """Windows as choose_windows picks them, plus the blend length of every
    seam, chosen for that pair of songs from blend_lengths: long when their
    notes line up and neither melody gets in the way, short when they rub or
    a vocal would overlap. overlap_preference (-1 short to +1 long, from the
    A/B votes) leans every seam's choice; transition_length ("auto",
    "longer", "extended") is the user's setting from BLEND_SIZES. Render with
    overlap_limits(transition_length) so the renderer allows those lengths.
    """
    if not tracks:
        return [], []
    size = BLEND_SIZES[transition_length]
    p = overlap_preference if math.isfinite(overlap_preference) else 0.0
    p = min(1.0, max(-1.0, p))
    ladders = [blend_lengths(a, b, transition_length) for a, b in zip(tracks, tracks[1:])]
    return _plan(tracks, ladders, target_seconds, chroma, p + size.lean, size)


def _plan(
    tracks: list[Track],
    ladders: list[list[float]],
    target_seconds: float | None,
    chroma: Optional[Sequence[Optional[Chroma]]],
    preference: Optional[float],
    size: BlendSize,
) -> tuple[list[TrackWindow], list[float]]:
    """ladders: blend lengths to try per seam, shortest first. preference
    (overlap_preference plus the setting's lean) None means each ladder is a
    single fixed length."""
    n = len(tracks)
    cues = [_cues(t) for t in tracks]
    if chroma is not None:
        if len(chroma) != n:
            raise ValueError(f"need {n} chroma entries, got {len(chroma)}")
        for c, ch in zip(cues, chroma):
            c.chroma = ch
    # Middle of each ladder (the top for "extended", so set-mode windows have
    # room for it): sizes drop builds and set-mode lengths before the blend
    # lengths are known.
    typical = [ladder[-1] if size.size_for_longest else ladder[len(ladder) // 2] for ladder in ladders]
    builds = [_build_starts(tracks[i + 1], typical[i]) for i in range(n - 1)]
    for i, starts in enumerate(builds):
        c = cues[i + 1]
        extra = [s for s in starts if 0.0 < s < c.duration]
        if extra:
            have = {round(float(g), 3) for g in c.grid}
            c.drop_only = frozenset(round(s, 3) for s in extra) - have
            c.grid = np.unique(np.concatenate([c.grid, np.round(extra, 3)]))
    options = [[_score_seam(cues[i], cues[i + 1], o, builds[i]) for o in ladders[i]] for i in range(n - 1)]
    seams = [_fixed(opts[0]) if preference is None else _choose_length(opts, preference, size.stack_weight)
             for opts in options]
    mins = _min_lengths(tracks, [ladder[0] for ladder in ladders])
    bests = [best_section(t) for t in tracks]

    def fitted(picks: list[tuple[int, int]], i: int) -> tuple[_Seam, bool]:
        """Seam i's chosen length, or the best one that fits if it won't: the
        renderer caps a blend at half of either window. Also whether it was cut."""
        (s0, e0), (s1, e1) = picks[i], picks[i + 1]
        fit = min(cues[i].grid[e0] - cues[i].grid[s0], cues[i + 1].grid[e1] - cues[i + 1].grid[s1]) / 2
        seam = seams[i].at(e0, s1)
        if seam.overlap <= fit or len(seams[i].options) < 2:
            return seam, False
        fitting = [o for o in seams[i].options if o.overlap <= fit]
        if not fitting:
            return seams[i].options[0], True
        return max(fitting, key=lambda o: o.quality[e0, s1]
                   + float(_length_bonus(o, preference, size.stack_weight)[e0, s1])), True

    def blends(picks: list[tuple[int, int]]) -> list[float]:
        return [fitted(picks, i)[0].overlap for i in range(n - 1)]

    if target_seconds is None:
        unaries = [_unary(c, i, n, mins[i], None, None) for i, c in enumerate(cues)]
        picks, _ = _solve(cues, seams, unaries)
    else:
        floors = [max(m, SET_MIN_WINDOW_SECONDS) for m in mins]
        total = target_seconds + sum(typical)
        candidates = []
        for _ in range(SET_LENGTH_PASSES):
            lengths = _length_targets(cues, floors, total)
            unaries = [_unary(c, i, n, mins[i], lengths[i], bests[i], typical[i - 1] if i else 0.0)
                       for i, c in enumerate(cues)]
            picks, score = _solve(cues, seams, unaries)
            want = target_seconds + sum(blends(picks))
            got = sum(cues[i].grid[e] - cues[i].grid[s] for i, (s, e) in enumerate(picks))
            candidates.append((abs(got - want) > SET_TOLERANCE * target_seconds, abs(got - want), -score, picks))
            if abs(got - want) <= SET_TOLERANCE * target_seconds * 0.5:
                break
            total += want - got
        picks = min(candidates, key=lambda c: c[:3])[3]

    chosen, capped = map(list, zip(*[fitted(picks, i) for i in range(n - 1)])) if n > 1 else ([], [])
    overlaps = [s.overlap for s in chosen]

    windows = []
    for i, (si, ei) in enumerate(picks):
        c = cues[i]
        s, e = float(c.grid[si]), float(c.grid[ei])
        head = chosen[i - 1] if i > 0 else None
        tail = chosen[i] if i < n - 1 else None
        next_name = cues[i + 1].name if i < n - 1 else ""
        drop_in = head is not None and bool(head.is_drop[picks[i - 1][1], si])
        drop_out = tail is not None and bool(tail.is_drop[ei, picks[i + 1][0]])
        drop_seconds = None
        if drop_in:
            drop_seconds, strength = head.drop_at[round(s, 3)]
            start = (f"Comes in at {_clock(s)} for a {drop_seconds - s:.0f} s build; its drop at "
                     f"{_clock(drop_seconds)} (energy jump {strength:.2f}) hits as "
                     f"{cues[i - 1].name} leaves")
        else:
            start = _start_text(c, s, head, si)
        if drop_out:
            how = (f"filtered up over {next_name}'s build into its drop"
                   if tempos_blend(c.track, cues[i + 1].track)
                   else f"filtered and rolled into a beat of silence before {next_name}'s drop")
            end = ("plays to its own ending" if e >= c.duration else f"leaves at {_clock(e)}") + f", {how}"
        else:
            end = _end_text(c, e, tail, ei)
        parts = [f"{start}; {end}."]
        if target_seconds is not None and bests[i] is not None:
            b0, b1 = bests[i]
            if s <= b0 + BOUNDARY_SECONDS and e >= b1 - BOUNDARY_SECONDS:
                parts.append(f"Keeps its loudest part ({_clock(b0)}-{_clock(b1)}).")
            elif min(e, b1) > max(s, b0):
                parts.append(f"Plays inside its loudest part ({_clock(b0)}-{_clock(b1)}).")
        if c.duration < mins[i]:
            parts.append("Too short to trim, so it plays in full.")
        if tail is not None and not drop_out:
            ej, next_end = picks[i + 1]
            nxt = cues[i + 1]
            clash = _renderer_echoes(c, s, e, nxt, float(nxt.grid[ej]), float(nxt.grid[next_end]))
            fix = ("the vocals swap on a beat" if tempos_blend(c.track, nxt.track)
                   else "the renderer echoes it out")
            if tail.quality[ei, ej] < POOR_SEAM or tail.handoff[ei, ej] < POOR_HANDOFF:
                why = ("both songs carry melody at every cue" if tail.handoff[ei, ej] < NEUTRAL
                       else "their energy never lines up")
                parts.append(f"No clean handoff into {next_name} ({why}), so this is the least-bad cue"
                             + (f" and {fix}." if clash else "."))
            elif clash:
                parts.append(f"Both it and {next_name} carry melody at this cue, so {fix} "
                             f"instead of crossfading.")
            if tail.harmony is not None:
                h = float(tail.harmony[ei, ej])
                if h >= HARMONY_GOOD:
                    parts.append(f"Its notes line up with {next_name}'s over the blend "
                                 f"(harmonic match {h:.2f}).")
                elif h < HARMONY_POOR:
                    parts.append(f"Its notes rub a little against {next_name}'s over the blend "
                                 f"(harmonic match {h:.2f}); no closer match scored better overall.")
            if tail.gap[ei, ej] > GAP_OK_SECONDS:
                parts.append(f"About {tail.gap[ei, ej]:.0f} s pass without melody from either song "
                             f"around the blend into {next_name}.")
            if tail.groove[ei, ej] < SPARSE_GROOVE:
                parts.append(f"The blend into {next_name} is on the sparse side; no fuller cue "
                             f"handed the melody over as cleanly.")
            if preference is not None:
                text = _length_text(c, nxt, tail, seams[i].options, ei, ej, capped[i])
                if text:
                    parts.append(text)
        windows.append(TrackWindow(track_id=c.track.id, start_seconds=round(s, 3),
                                   end_seconds=round(e, 3), reason=" ".join(parts),
                                   drop_seconds=drop_seconds))

    if target_seconds is not None:
        played = sum(w.end_seconds - w.start_seconds for w in windows) - sum(overlaps)
        if abs(played - target_seconds) > SET_TOLERANCE * target_seconds:
            why = ("the songs are too short to fill it" if played < target_seconds
                   else "each song needs room for the extended blends" if size.size_for_longest
                   else f"every song needs at least {SET_MIN_WINDOW_SECONDS:.0f} s")
            first = windows[0]
            windows[0] = first.model_copy(update={"reason": (
                f"{first.reason} The set runs about {_clock(played)} instead of the "
                f"{_clock(target_seconds)} target: {why}.")})
    return windows, overlaps
