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

How long each blend runs depends on the two songs (plan_windows): from each
B start, the blend can land on any of B's next events (a vocal coming in,
a chorus or other section, an energy lift, a drop, a phrase), counted in
whole bars of matched beats, as long as A can carry it (it may not start
before A's last vocal line or its last lift) and the two melodies don't
sing over each other for long. Each (A end, B start) takes the landing that
scores best there; longer blends earn a bonus when the notes line up.
Tempos too far apart to beat-match get a beatless blend where one side has
no steady kick, else an echo-out. The A/B preference and the transition
length setting lean every seam longer or shorter.

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
from backend.audio.echo import ECHO_BARS
from backend.audio.seam_plan import MAX_GLIDE_DEVIATION, beatless_run, deviation, fold_ratio
from backend.audio.transition import BEATLESS_MIN_SECONDS, DEFAULT_SAFETY, SafetyLimits
from backend.audio.windows import MIN_WINDOW_SECONDS, stored_pulse

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
# Tempos too far apart to beat-match: an echo-out keeps this share of the
# seam's quality, a beatless blend (one side has no kick) nearly all of it,
# and a drop cut (with its silent beat) this share, so the cut only wins
# where B's drop is strong and the cue is poor for anything else.
ECHO_BLEND = 0.8
BEATLESS_BLEND = 0.95
UNMATCHED_DROP = 0.8
# The reason says B takes over on an energy jump at or above this rise.
LANDING_RISE = 0.5

# Blend length (plan_windows). A blend plays B's opening under A's ending
# and lands (bass swap, bands.py) on B's next musical event: its vocal
# coming in, a section boundary, an energy lift, or at least a phrase or
# half-phrase line. Each (A end, B start) is offered every landing of B
# within the setting's cap, counted in whole bars of B so the renderer
# lands exactly there, and limited by how long A can carry a blend: back to
# its last event (a vocal line starting after a pause, or its energy
# lifting into a chorus or drop), so B never comes in under A's hook.
MIN_BLEND_SECONDS = 4.0  # reach is 0 here (and for anything shorter)
MAX_BLEND_SECONDS = DEFAULT_SAFETY.max_overlap_seconds
MIN_LANDING_BEATS = 8  # nearer landings read as a cut: the next one is used
LANDING_SECONDS_NO_TEMPO = 4.0  # the same without a tempo to count beats in
# Event strength by kind, 0-1: how much a landing there is worth. Events
# within a beat of each other merge, each extra kind adding EVENT_STACK.
EVENT_STRENGTH = {"chorus": 1.0, "drop": 1.0, "vocal": 0.9, "section": 0.7, "phrase": 0.5,
                  "break": 0.5, "half": 0.25, "outro": 0.1}
EVENT_NAMES = ("chorus", "drop", "vocal", "rise", "section", "break", "phrase", "half", "outro")
EVENT_STACK = 0.15
CLASH_BLEND_SECONDS = 8.0  # longest blend where both songs carry melody at the seam
STACK_LIMIT = 0.55  # _Seam.stack from which the two melodies sing over each other
LANDING_WEIGHT = 0.1  # seam quality per unit of landing strength
RISE_STEP = 0.15  # energy step (8 s after vs before a bar line) that counts as a lift
ONSET_QUIET_SECONDS = 4.0  # a melody coming in after this long without one is a new vocal line
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
    """The user's transition length setting. The lengths themselves come
    from the songs (where B's events fall); the setting caps them and leans
    the choice."""
    cap: float  # longest blend the renderer may play (SafetyLimits.max_overlap_seconds)
    lean: float  # added to overlap_preference
    stack_weight: float  # how strongly two melodies singing at once shorten a blend
    typical: float  # seconds per seam budgeted before the lengths are known (drop builds, set mode)
    echo_bars: int  # longest echo-out (tempos too far apart), in outgoing bars


# auto: the songs decide, up to 32 s. longer: every seam leans long and may
# land up to 40 s in, extended up to 60 s, with vocal clashes shortening a
# blend half as much and set-mode windows sized for long blends. Echo-outs
# filter out up to 2 bars (auto) or 4.
BLEND_SIZES: dict[str, BlendSize] = {
    "auto": BlendSize(MAX_BLEND_SECONDS, 0.0, 1.0, 16.0, 2),
    "longer": BlendSize(40.0, 1.0, 1.0, 24.0, 4),
    "extended": BlendSize(60.0, 1.5, 0.5, 40.0, 4),
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
    pulse: Optional[list] = None  # kick pulse (seam_plan.pulse_curve) from analysis
    pulse_hop: Optional[float] = None

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
    kind: str = "fixed"  # "blend" | "beatless" | "echo"; "fixed" = choose_windows' given length
    requests: Optional[np.ndarray] = None  # (B starts,): overlap to ask the renderer for there
    beats: Optional[int] = None  # blend: matched beats from B's first beat to the landing
    land: Optional[np.ndarray] = None  # (B starts,): strength of B's event at the landing, 0 = none
    land_at: Optional[np.ndarray] = None  # (B starts,): the landing, in B's seconds
    land_what: Optional[list] = None  # (B starts,): EVENT_NAMES entry, "" = a bar line
    bars: int = 0  # echo: outgoing bars filtered out

    def request(self, j: int) -> float:
        return float(self.requests[j]) if self.requests is not None else self.overlap


@dataclass
class _SeamChoice:
    """One seam scored at its candidate lengths; every (A end, B start) takes
    the best one it may use."""
    options: list[_Seam]
    score: np.ndarray  # (A ends, B starts): the chosen option's quality plus its bonuses
    pick: np.ndarray  # (A ends, B starts): index into options
    valid: list[np.ndarray]  # per option, (A ends, B starts): where it may be used
    sustain: Optional[list] = None  # per A end: (seconds A can carry a blend, what stops it, when)
    preference: float = 0.0
    stack_weight: float = 1.0
    clash: Optional[np.ndarray] = None  # (A ends, B starts): both carry melody at the seam

    def at(self, i: int, j: int) -> _Seam:
        return self.options[int(self.pick[i, j])]

    def value(self, o: _Seam, i: int, j: int) -> float:
        return float(_option_score(o, self.preference, self.stack_weight)[i, j])


def _fixed(seam: _Seam) -> _SeamChoice:
    shape = seam.quality.shape
    return _SeamChoice([seam], seam.quality, np.zeros(shape, dtype=int), [np.ones(shape, dtype=bool)])


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


def _option_score(o: _Seam, preference: float, stack_weight: float) -> np.ndarray:
    score = o.quality + _length_bonus(o, preference, stack_weight)
    if o.land is not None:
        score = score + np.where(o.is_drop, 0.0, LANDING_WEIGHT * o.land[None, :])
    return score


def _choose(options: list[_Seam], valid: list[np.ndarray], preference: float, stack_weight: float,
            sustain: Optional[list] = None) -> _SeamChoice:
    """Every cell takes its best valid option (a drop is valid everywhere:
    its length is its build)."""
    scores = np.stack([np.where(ok | o.is_drop, _option_score(o, preference, stack_weight), -np.inf)
                       for o, ok in zip(options, valid)])
    pick = np.argmax(scores, axis=0)
    return _SeamChoice(options, np.take_along_axis(scores, pick[None], 0)[0], pick, valid, sustain,
                       preference, stack_weight)


def tempos_blend(a: Track, b: Track) -> bool:
    """Whether the renderer can beat-match a blend between the two songs
    (half/double time folded). True when either tempo is unknown."""
    if not a.bpm or not b.bpm:
        return True
    ratio, _ = fold_ratio(60.0 / a.bpm, 60.0 / b.bpm)
    return deviation(ratio) <= MAX_GLIDE_DEVIATION


def _bar_lines(track: Track) -> np.ndarray:
    """Bar lines on the song's beat grid, phased to its phrase starts."""
    beats = np.asarray(track.beats_seconds, dtype=np.float64)
    if beats.size < BEATS_PER_BAR:
        return beats
    k0 = 0
    if track.phrase_starts_seconds:
        k0 = int(np.argmin(np.abs(beats - track.phrase_starts_seconds[0]))) % BEATS_PER_BAR
    return beats[k0::BEATS_PER_BAR]


def _melody_onsets(c: _Cues) -> np.ndarray:
    """Where the melody comes in after ONSET_QUIET_SECONDS or more without
    one (smoothed as _quiet_run): a new vocal line."""
    if c.melody is None or not c.hop:
        return np.zeros(0)
    n = max(1, int(round(MELODY_SMOOTH_SECONDS / c.hop)))
    on = np.convolve(c.melody, np.ones(n) / n, mode="same") >= QUIET_MELODY
    need = int(round(ONSET_QUIET_SECONDS / c.hop))
    starts = np.flatnonzero(on[1:] & ~on[:-1]) + 1
    quiet_from = np.concatenate([[0], np.flatnonzero(~on[1:] & on[:-1]) + 1])
    out = []
    for k in starts:
        last_off = quiet_from[quiet_from < k]
        if last_off.size and k - last_off[-1] >= need:
            out.append(k * c.hop)
    return np.array(out)


def _rise(c: _Cues, t: float) -> float:
    """Energy over the DROP_LOOK_SECONDS after t minus the DROP_LOOK_SECONDS before."""
    step = _mean(c.energy, c.hop, t, t + DROP_LOOK_SECONDS) - _mean(c.energy, c.hop, t - DROP_LOOK_SECONDS, t)
    return 0.0 if math.isnan(step) else step


def _events(c: _Cues) -> list[tuple[float, float, str]]:
    """The song's musical events, (time, strength, what), where a blend into
    it can land: its vocal or melody coming in, section boundaries (named by
    label), energy lifts at bar lines, and its phrase and half-phrase lines.
    Events within a beat of each other merge into the strongest, named by
    the most telling kind."""
    t = c.track
    beat = 60.0 / t.bpm if t.bpm else None
    raw: list[tuple[float, float, str]] = []
    for p in t.phrase_starts_seconds:
        raw.append((p, EVENT_STRENGTH["phrase"], "phrase"))
        if beat:
            raw.append((p + 4 * BEATS_PER_BAR * beat, EVENT_STRENGTH["half"], "half"))
    for prev, sec in zip(t.sections, t.sections[1:]):
        what = sec.label if sec.label in EVENT_STRENGTH else "section"
        strength = EVENT_STRENGTH[what]
        if what == "section":
            strength += 0.3 * float(np.clip((sec.energy - prev.energy) / 0.3, 0.0, 1.0))
        raw.append((sec.start_seconds, strength, what))
    bars = _bar_lines(t)
    for on in _melody_onsets(c):
        raw.append((float(bars[np.argmin(np.abs(bars - on))]) if bars.size else on,
                    EVENT_STRENGTH["vocal"], "vocal"))
    if bars.size and c.energy is not None:
        rise = np.array([_rise(c, b) for b in bars])
        for k, r in enumerate(rise):
            if r >= RISE_STEP and r >= rise[max(0, k - 2):k + 3].max():
                raw.append((float(bars[k]), 0.5 + 0.5 * min(1.0, (r - RISE_STEP) / RISE_STEP), "rise"))
    raw = sorted(e for e in raw if 0.0 < e[0] < c.duration)
    tol = 0.75 * beat if beat else 0.6
    events, group = [], []
    for e in raw + [(math.inf, 0.0, "")]:
        if group and e[0] - group[0][0] > tol:
            top = max(group, key=lambda g: g[1])
            kinds = {g[2] for g in group}
            what = next(k for k in EVENT_NAMES if k in kinds)
            events.append((top[0], min(1.2, top[1] + EVENT_STACK * (len(kinds) - 1)), what))
            group = []
        group.append(e)
    return events


def _sustain(c: _Cues, end: float, cap: float, onsets: np.ndarray) -> tuple[float, str, float]:
    """How long A can carry a blend that ends at `end`: back to its last
    event, so the blend never starts before A's last vocal line or under its
    lift into a chorus or drop. (seconds, what, when); what "" = nothing
    within cap."""
    best = (cap, "", end - cap)
    for on in onsets:
        if end - cap < on < end and end - on < best[0]:
            best = (end - on, "vocal", float(on))
    lifts = {round(s.start_seconds, 3): s.label for s in c.track.sections[1:] if s.label in ("chorus", "drop")}
    for p in c.track.phrase_starts_seconds:
        if end - cap < p < end and end - p < best[0]:
            label = lifts.get(round(p, 3))
            if label or _rise(c, p) >= RISE_STEP:
                best = (end - p, label or "rise", float(p))
    return best


def _blend_points(a: _Cues, b: _Cues) -> tuple[np.ndarray, np.ndarray]:
    """Where the blend sits: before A's fade-out tail at its own end, after
    B's lead-in at its own start."""
    return (np.array([e - a.tail if e >= a.duration else e for e in a.grid]),
            np.array([s + b.head if s <= 0 else s for s in b.grid]))


def _fill_fallback(options: list[_Seam], valid: list[np.ndarray], rooms: list[np.ndarray], tol: float) -> None:
    """Cells where no event landing fits its option's room (A ends, B
    starts): any length that fits, else the shortest."""
    none = ~np.any(valid, axis=0)
    if none.any():
        for o, ok, room in zip(options, valid, rooms):
            ok |= none & (o.requests[None, :] <= room + tol)
        none = ~np.any(valid, axis=0)
        shortest = int(np.argmin([o.requests.min() for o in options]))
        valid[shortest] |= none


def _edge_clash(a: _Cues, b: _Cues, ends, starts) -> np.ndarray:
    """(A ends, B starts): both edges carry melody over echo.py's
    CLASH_SECONDS, so the renderer swaps the vocals or echoes out."""
    out_edge = np.array([_mean(a.melody, a.hop, e - CLASH_SECONDS, e) for e in ends])
    in_edge = np.array([_mean(b.melody, b.hop, s, s + CLASH_SECONDS) for s in starts])
    return (_nan_to(out_edge, 0.0) >= MELODY_CLASH)[:, None] & (_nan_to(in_edge, 0.0) >= MELODY_CLASH)[None, :]


def _landing_choice(a: _Cues, b: _Cues, size: BlendSize, preference: float, drop_at: dict) -> _SeamChoice:
    """Tempos that can be beat-matched: every landing on one of B's events
    within the cap, counted in whole bars from B's first beat."""
    ends, starts = _blend_points(a, b)
    heads = starts - b.grid
    onsets = _melody_onsets(a)
    sustain = [_sustain(a, e, size.cap, onsets) for e in ends]
    limit = np.array([s[0] for s in sustain])
    # Two melodies at the seam: however the songs line up, keep it short.
    clash = _edge_clash(a, b, ends, starts)
    cell = np.where(clash, np.minimum(limit[:, None], CLASH_BLEND_SECONDS), limit[:, None])
    periods = None
    if a.track.bpm and b.track.bpm:
        _, factor = fold_ratio(60.0 / a.track.bpm, 60.0 / b.track.bpm)
        p_in = 60.0 / b.track.bpm * factor  # the renderer counts B's beats against A's
        periods = (p_in, 0.5 * (60.0 / a.track.bpm + p_in))
    first = b.track.beats_seconds[0] if b.track.beats_seconds else 0.0
    found: dict[float, dict[int, tuple[float, str, float]]] = {}
    events = _events(b)
    for j, s in enumerate(b.grid):
        if round(float(s), 3) in b.drop_only:
            continue
        for t, strength, what in events:
            if periods:
                p_in, p_avg = periods
                n = BEATS_PER_BAR * round((t - (s if s > 0 else first)) / p_in / BEATS_PER_BAR)
                key, length = float(n), n * p_avg - heads[j]
                if n < MIN_LANDING_BEATS:
                    continue
            else:
                key = length = round(t - starts[j], 1)
                if length < LANDING_SECONDS_NO_TEMPO:
                    continue
            if length <= size.cap + 1e-6 and strength > found.setdefault(key, {}).get(j, (0.0,))[0]:
                found[key][j] = (strength, what, t)
    unit = periods[1] * BEATS_PER_BAR if periods else 1.0
    for lim in set(np.minimum(cell, size.cap).ravel().tolist()):  # the longest bar count A can carry
        found.setdefault(float(max(1, math.floor(lim / unit)) * (BEATS_PER_BAR if periods else 1)), {})
    options, valid, rooms = [], [], []
    for key in sorted(found):
        span = key * periods[1] if periods else key
        seam = _score_seam(a, b, span, drop_at)
        req = np.maximum(span - heads, DEFAULT_SAFETY.min_overlap_seconds) if periods else np.full(len(starts), span)
        land, at, what = np.zeros(len(starts)), starts + req, [""] * len(starts)
        for j, (strength, w, t) in found[key].items():
            land[j], at[j], what[j] = strength, t, w
        seam.kind, seam.requests, seam.land, seam.land_at, seam.land_what = "blend", req, land, at, what
        seam.beats = int(key) if periods else None
        options.append(seam)
        # A blend that would sing both melodies across its length stays short too.
        room = np.where(seam.stack >= STACK_LIMIT, np.minimum(cell, CLASH_BLEND_SECONDS), cell)
        rooms.append(room)
        valid.append((land[None, :] > 0) & (req[None, :] <= room + unit / BEATS_PER_BAR))
    _fill_fallback(options, valid, rooms, unit / BEATS_PER_BAR)
    choice = _choose(options, valid, preference, size.stack_weight, sustain)
    choice.clash = clash
    return choice


def _unmatched_choice(a: _Cues, b: _Cues, size: BlendSize, preference: float, drop_at: dict) -> _SeamChoice:
    """Tempos too far apart to beat-match. Where A's ending or B's opening
    has no steady kick (as the renderer will measure it) and the two don't
    both carry melody there, a beatless blend as long as that stretch;
    elsewhere an echo-out of 1, 2 or 4 bars of A (up to the setting's
    echo_bars and what A can carry)."""
    ends, starts = _blend_points(a, b)
    limit = np.array([_sustain(a, e, size.cap, _melody_onsets(a))[0] for e in ends])
    run_out = np.array([beatless_run(a.pulse, a.pulse_hop, e, False, size.cap) for e in ends])
    run_in = np.array([beatless_run(b.pulse, b.pulse_hop, s, True, size.cap) for s in starts])
    run = np.maximum(run_out[:, None], run_in[None, :])
    beatless = (run >= BEATLESS_MIN_SECONDS) & ~_edge_clash(a, b, ends, starts)  # two melodies: echo out
    seconds = np.floor(run)
    options, valid = [], []
    for length in sorted(set(seconds[beatless].tolist())):
        seam = _score_seam(a, b, length, drop_at, factor=BEATLESS_BLEND)
        seam.kind, seam.requests = "beatless", np.full(len(starts), length)
        options.append(seam)
        valid.append(beatless & (seconds == length))
    beat = 60.0 / a.track.bpm
    for bars in sorted((n for n in ECHO_BARS if n <= size.echo_bars), reverse=True):  # ties: the longest
        length = bars * BEATS_PER_BAR * beat
        seam = _score_seam(a, b, length, drop_at, factor=ECHO_BLEND)
        seam.kind, seam.bars, seam.requests = "echo", bars, np.full(len(starts), length)
        options.append(seam)
        valid.append(~beatless & ((length <= limit[:, None] + beat) | (bars == 1)))
    return _choose(options, valid, preference, size.stack_weight)


def _seam_choice(a: _Cues, b: _Cues, size: BlendSize, preference: float, drop_at: dict) -> _SeamChoice:
    if tempos_blend(a.track, b.track):
        return _landing_choice(a, b, size, preference, drop_at)
    return _unmatched_choice(a, b, size, preference, drop_at)


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
    pulse, pulse_hop = stored_pulse(track)
    return _Cues(track, np.array([0.0, *inner, duration]), kind, melody, energy, hop, head, tail,
                 pulse=pulse, pulse_hop=pulse_hop)


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


def _score_seam(a: _Cues, b: _Cues, overlap: float, drop_at: Optional[dict] = None,
                factor: float = 1.0) -> _Seam:
    """The seam scored at one blend length. factor scales the blend's quality
    (an echo-out or beatless blend in place of a beat-matched one)."""
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
    clash = _edge_clash(a, b, ends, starts)
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
    quality = quality * factor
    drop_factor = 1.0 if tempos_blend(a.track, b.track) else UNMATCHED_DROP

    is_drop = np.zeros_like(quality, dtype=bool)
    drop_cols = [(j, drop_at[round(float(s), 3)][1]) for j, s in enumerate(b.grid)
                 if round(float(s), 3) in drop_at]
    if drop_cols:
        leaving = _nan_to(np.clip(e_out / FULL_ENERGY, 0.0, 1.0), NEUTRAL)
        for j, strength in drop_cols:
            q = drop_factor * (DROP_BASE + DROP_STRENGTH_WEIGHT * strength + 0.1 * leaving + 0.1 * st_out)
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


def _renderer_beatless(a: _Cues, i: int, b: _Cues, j: int, limit: float = MAX_BLEND_SECONDS) -> bool:
    """Whether transition.py will play a seam between tempos it can't match
    as a beatless blend: one side has no steady kick at the seam for
    BEATLESS_MIN_SECONDS (the same pulse curve and test the renderer uses)."""
    e, s = _blend_points(a, b)
    return max(beatless_run(a.pulse, a.pulse_hop, e[i], False, limit),
               beatless_run(b.pulse, b.pulse_hop, s[j], True, limit)) >= BEATLESS_MIN_SECONDS


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
    if seam.kind == "echo":
        text += ", right on the downbeat under the outgoing song's echo"
        return text + (f", where {_melody_phrase(float(seam.in_level[j]))}" if not math.isnan(full) else "")
    if seam.in_rise is not None and seam.in_rise[j] >= LANDING_RISE:
        if seam.land_at is not None:
            land = float(seam.land_at[j])
        elif seam.requests is not None:
            land = s + (0.0 if s > 0 else c.head) + float(seam.request(j))
        else:
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


LANDING_TEXT = {
    "chorus": "lands on its chorus", "drop": "lands on its drop", "vocal": "lands as its vocals come in",
    "rise": "lands as its energy lifts", "section": "lands on its next section", "break": "lands on its break",
    "phrase": "lands on its next phrase", "half": "lands halfway through a phrase",
    "outro": "lands on its outro", "": "lands on a bar line",
}
SUSTAIN_TEXT = {"vocal": "last vocal line", "chorus": "chorus", "drop": "drop", "rise": "energy lift"}


def _tempos(c: _Cues, nxt: _Cues) -> str:
    return f"{c.track.bpm:.0f} and {nxt.track.bpm:.0f} BPM"


def _length_text(c: _Cues, nxt: _Cues, seam: _Seam, choice: _SeamChoice, i: int, j: int,
                 capped: bool, cap: float) -> str:
    """How the seam into the next song plays and why it is as long as it is
    ("" for choose_windows' given length)."""
    seconds = seam.request(j)
    if seam.kind == "echo":
        bars = "bar" if seam.bars == 1 else f"{seam.bars} bars"
        return (f"Their tempos ({_tempos(c, nxt)}) are too far apart to beat-match and both have a beat, "
                f"so it echoes out over its last {bars} as its phrase ends and {nxt.name} comes in on "
                f"its downbeat under the echo.")
    if seam.kind == "beatless":
        e, s = _blend_points(c, nxt)
        run_out = beatless_run(c.pulse, c.pulse_hop, e[i], False, c.duration)
        run_in = beatless_run(nxt.pulse, nxt.pulse_hop, s[j], True, nxt.duration)
        if run_out >= run_in:
            return (f"Their tempos ({_tempos(c, nxt)}) are too far apart to beat-match, but its last "
                    f"{run_out:.0f} s have no beat, so {nxt.name} blends in under them over "
                    f"{seconds:.0f} s with no beats to clash.")
        beat_at = _clock(s[j] + run_in)
        lands = (f"lands as its beat comes in at {beat_at}" if run_in - seconds <= 1.5
                 else f"lands ahead of its beat, which comes in at {beat_at}")
        return (f"Their tempos ({_tempos(c, nxt)}) are too far apart to beat-match, but {nxt.name} opens "
                f"without a beat, so it blends in under this song over {seconds:.0f} s and {lands}.")
    if seam.kind != "blend":
        return ""
    beats = f" ({seam.beats} beats)" if seam.beats else ""
    what, at = seam.land_what[j], float(seam.land_at[j])
    lead = "Extended blend into" if seconds > MAX_BLEND_SECONDS + 1e-6 else "Blends into"
    lands = LANDING_TEXT[what] if what or seam.beats else "lands"  # no tempo: no bar lines
    text = f"{lead} {nxt.name} over {seconds:.0f} s{beats} and {lands} at {_clock(at)}"
    if capped:
        return text + ", the longest landing that fits these windows."
    if choice.clash is not None and choice.clash[i, j] and any(
            o.land is not None and o.land[j] > 0 and o.request(j) > CLASH_BLEND_SECONDS + 1.0
            for o in choice.options):
        return text + "; both songs carry melody at the seam, so it stays short instead of landing later."
    # The next stronger landing it passed up, and what ruled it out.
    later = [(o.request(j), k) for k, o in enumerate(choice.options)
             if o.land is not None and o.land[j] > seam.land[j] and o.request(j) > seconds
             and not choice.valid[k][i, j]]
    if not later:
        return text + "."
    o = choice.options[min(later)[1]]
    beat = 60.0 / c.track.bpm if c.track.bpm else 1.0
    sustain = choice.sustain[i] if choice.sustain else None
    if sustain and sustain[1] and o.request(j) > sustain[0] + beat:
        return (text + f"; a later landing would start the blend before its {SUSTAIN_TEXT[sustain[1]]} "
                f"at {_clock(sustain[2])}.")
    if o.stack is not None and o.stack[i, j] >= STACK_LIMIT:
        where = LANDING_TEXT[o.land_what[j]].removeprefix("lands ")
        return (text + f"; landing {where} at {_clock(float(o.land_at[j]))} would sing both songs' "
                f"melodies over each other for {o.request(j):.0f} s.")
    return text + "."


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
    return _plan(tracks, target_seconds, chroma, BLEND_SIZES["auto"], fixed=overlaps)[0]


def plan_windows(
    tracks: list[Track],
    overlap_preference: float = 0.0,
    target_seconds: float | None = None,
    chroma: Optional[Sequence[Optional[Chroma]]] = None,
    transition_length: TransitionLength = "auto",
) -> tuple[list[TrackWindow], list[float]]:
    """Windows as choose_windows picks them, plus the overlap to request at
    every seam, chosen for that pair of songs: the blend lands on B's next
    musical event (vocal entry, section, energy lift, phrase) that A can
    carry a blend into, long when their notes line up and neither melody
    gets in the way, short when they rub or a vocal would overlap. Tempos
    too far apart to beat-match get a beatless blend where one side has no
    beat, else an echo-out. overlap_preference (-1 short to +1 long, from
    the A/B votes) leans every seam's choice; transition_length ("auto",
    "longer", "extended") is the user's setting from BLEND_SIZES. Render with
    overlap_limits(transition_length) so the renderer allows those lengths.
    """
    if not tracks:
        return [], []
    size = BLEND_SIZES[transition_length]
    p = overlap_preference if math.isfinite(overlap_preference) else 0.0
    p = min(1.0, max(-1.0, p))
    return _plan(tracks, target_seconds, chroma, size, preference=p + size.lean)


def _plan(
    tracks: list[Track],
    target_seconds: float | None,
    chroma: Optional[Sequence[Optional[Chroma]]],
    size: BlendSize,
    preference: float = 0.0,
    fixed: Optional[list[float]] = None,
) -> tuple[list[TrackWindow], list[float]]:
    """fixed: one given overlap per seam (choose_windows); None = every seam
    picks its own (plan_windows), leaned by preference (overlap_preference
    plus the setting's lean)."""
    n = len(tracks)
    cues = [_cues(t) for t in tracks]
    if chroma is not None:
        if len(chroma) != n:
            raise ValueError(f"need {n} chroma entries, got {len(chroma)}")
        for c, ch in zip(cues, chroma):
            c.chroma = ch
    # Drop builds and set-mode lengths are sized before the blend lengths are known.
    typical = fixed if fixed is not None else [size.typical] * (n - 1)
    builds = [_build_starts(tracks[i + 1], typical[i]) for i in range(n - 1)]
    for i, starts in enumerate(builds):
        c = cues[i + 1]
        extra = [s for s in starts if 0.0 < s < c.duration]
        if extra:
            have = {round(float(g), 3) for g in c.grid}
            c.drop_only = frozenset(round(s, 3) for s in extra) - have
            c.grid = np.unique(np.concatenate([c.grid, np.round(extra, 3)]))
    if fixed is not None:
        seams = [_fixed(_score_seam(cues[i], cues[i + 1], fixed[i], builds[i],
                                    1.0 if tempos_blend(tracks[i], tracks[i + 1]) else ECHO_BLEND))
                 for i in range(n - 1)]
    else:
        seams = [_seam_choice(cues[i], cues[i + 1], size, preference, builds[i]) for i in range(n - 1)]
    mins = _min_lengths(tracks, [min(o.request(j) for o in s.options for j in range(len(cues[i + 1].grid)))
                                 for i, s in enumerate(seams)])
    bests = [best_section(t) for t in tracks]

    def fitted(picks: list[tuple[int, int]], i: int) -> tuple[_Seam, bool]:
        """Seam i's chosen option, or the best one that fits if it won't: the
        renderer caps a blend at half of either window. Also whether it was cut."""
        (s0, e0), (s1, e1) = picks[i], picks[i + 1]
        fit = min(cues[i].grid[e0] - cues[i].grid[s0], cues[i + 1].grid[e1] - cues[i + 1].grid[s1]) / 2
        choice = seams[i]
        seam = choice.at(e0, s1)
        if seam.request(s1) <= fit or len(choice.options) < 2:
            return seam, False
        fitting = [k for k, o in enumerate(choice.options) if o.request(s1) <= fit]
        if not fitting:
            return min(choice.options, key=lambda o: o.request(s1)), True
        usable = [k for k in fitting if choice.valid[k][e0, s1]] or fitting
        return choice.options[max(usable, key=lambda k: choice.value(choice.options[k], e0, s1))], True

    def blends(picks: list[tuple[int, int]]) -> list[float]:
        return [fitted(picks, i)[0].request(picks[i + 1][0]) for i in range(n - 1)]

    if target_seconds is None:
        unaries = [_unary(c, i, n, mins[i], None, None) for i, c in enumerate(cues)]
        picks, _ = _solve(cues, seams, unaries)
    else:
        floors = [max(m, SET_MIN_WINDOW_SECONDS) for m in mins]
        total = target_seconds + sum(typical)
        candidates, runs = [], []

        def attempt(lengths: list[float]) -> float:
            unaries = [_unary(c, i, n, mins[i], lengths[i], bests[i], typical[i - 1] if i else 0.0)
                       for i, c in enumerate(cues)]
            picks, score = _solve(cues, seams, unaries)
            want = target_seconds + sum(blends(picks))
            got = sum(cues[i].grid[e] - cues[i].grid[s] for i, (s, e) in enumerate(picks))
            candidates.append((abs(got - want) > SET_TOLERANCE * target_seconds, abs(got - want), -score, picks))
            runs.append((got - want, picks))
            return want - got

        for _ in range(SET_LENGTH_PASSES):
            miss = attempt(_length_targets(cues, floors, total))
            if abs(miss) <= SET_TOLERANCE * target_seconds * 0.5:
                break
            total += miss
        else:
            # Every window moving by a whole phrase at once can step over the
            # target: mix the closest passes either side of it, track by track.
            over = min((r for r in runs if r[0] > 0), key=lambda r: r[0], default=None)
            under = max((r for r in runs if r[0] < 0), key=lambda r: r[0], default=None)
            if over is not None and under is not None:
                short = [cues[i].grid[e] - cues[i].grid[s] for i, (s, e) in enumerate(under[1])]
                long = [cues[i].grid[e] - cues[i].grid[s] for i, (s, e) in enumerate(over[1])]
                lengths, need = list(short), -under[0]
                for i in sorted(range(n), key=lambda i: long[i] - short[i]):
                    if 0 < long[i] - short[i] <= 2 * need:
                        lengths[i], need = long[i], need - (long[i] - short[i])
                attempt(lengths)
        picks = min(candidates, key=lambda c: c[:3])[3]

    chosen, capped = map(list, zip(*[fitted(picks, i) for i in range(n - 1)])) if n > 1 else ([], [])
    overlaps = [chosen[i].request(picks[i + 1][0]) for i in range(n - 1)]

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
        if drop_out and fixed is None and not tempos_blend(c.track, cues[i + 1].track):
            parts.append(f"Its tempo is too far from {next_name}'s to beat-match, and {next_name}'s drop "
                         f"beat an echo-out or a beatless blend here, so it cuts into the drop.")
        if tail is not None and not drop_out:
            ej, next_end = picks[i + 1]
            nxt = cues[i + 1]
            clash = _renderer_echoes(c, s, e, nxt, float(nxt.grid[ej]), float(nxt.grid[next_end]))
            echoes = tail.kind == "echo" or (
                tail.kind == "fixed" and not tempos_blend(c.track, nxt.track)
                and not _renderer_beatless(c, ei, nxt, ej, tail.request(ej)))
            fix = "the renderer echoes it out" if echoes else "the vocals swap on a beat"
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
            if tail.groove[ei, ej] < SPARSE_GROOVE and tail.kind != "echo":
                parts.append(f"The blend into {next_name} is on the sparse side; no fuller cue "
                             f"handed the melody over as cleanly.")
            text = _length_text(c, nxt, tail, seams[i], ei, ej, capped[i], size.cap)
            if text:
                parts.append(text)
        windows.append(TrackWindow(track_id=c.track.id, start_seconds=round(s, 3),
                                   end_seconds=round(e, 3), reason=" ".join(parts),
                                   drop_seconds=drop_seconds))

    if target_seconds is not None:
        played = sum(w.end_seconds - w.start_seconds for w in windows) - sum(overlaps)
        if abs(played - target_seconds) > SET_TOLERANCE * target_seconds:
            why = ("the songs are too short to fill it" if played < target_seconds
                   else "each song needs room for the longer blends" if size.typical > MAX_BLEND_SECONDS / 2
                   else f"every song needs at least {SET_MIN_WINDOW_SECONDS:.0f} s")
            first = windows[0]
            windows[0] = first.model_copy(update={"reason": (
                f"{first.reason} The set runs about {_clock(played)} instead of the "
                f"{_clock(target_seconds)} target: {why}.")})
    return windows, overlaps
