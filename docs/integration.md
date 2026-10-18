# Integration contract

How `backend/api/routes.py` (Shaan) calls into `backend/audio` (Aarav) and
`backend/logic` (Sahil). Names and signatures here are a proposal: agree on
them, then each owner implements their side and the routes swap their stubs
for real calls.

All types come from `backend/api/schemas.py`. All time values are seconds.

## Aarav: `backend/audio`

### `analysis.py`

```python
def analyze_track(path: Path, track: Track) -> Track:
    """Return a copy of `track` with bpm, beats_seconds, key_camelot,
    energy_rms, energy_hop_seconds and analysis_status filled in.

    Never raises for bad audio: set analysis_status="error" and
    analysis_error to a human-readable reason instead.
    key_camelot stays None when key detection is unreliable."""
```

Used by `POST /api/analyze`. Upload already fills `id`, `filename`,
`duration_seconds` and `sample_rate`.

### `transition.py` (exists as a stub)

`render_transition(...)` as already defined. Routes never call it directly;
they go through the two functions below so preview and final share it.

### `preview.py`

```python
def render_preview(
    outgoing_path: Path, incoming_path: Path,
    outgoing: Track, incoming: Track,
    request: TransitionRequest, out_path: Path,
    *, loudness_match: bool = True,
) -> AppliedTransition:
    """Write a short preview excerpt around the seam to out_path (WAV)
    using render_transition(mode="preview")."""
```

Used by `POST /api/calibration/preview`, called twice per trial with
requests that differ only in `requested_overlap_seconds`.

### `mix.py`

```python
def render_mix(
    mix_id: str,
    tracks: list[Track],          # already in final order
    paths: list[Path],            # same order as tracks
    overlap_seconds: list[float], # requested overlap per seam, len(tracks) - 1
    out_path: Path,
    *, calibrated: bool = False, overlap_preference: float = 0.0,
    loudness_match: bool = True,
) -> MixManifest:
    """Write the full mix to out_path (WAV) using render_transition(mode="final")
    for every seam. Every track stays complete; only seams overlap.
    The manifest reports actual, not requested, values."""
```

`outgoing_start_seconds` is measured from the start of the outgoing track,
not from the start of the mix. The UI adds these up to show where each song
starts in the mix.

Used by `POST /api/mix`.

### Transition quality (`transition.py`, `seam_plan.py`, `bands.py`, `stretch.py`)

Routes need no change. Every seam (A/B previews and the final mix) now:

- **Matches beats beat-for-beat** over a whole number of bars (multiple of 4
  beats) near the requested overlap. Each side's grid is locked to the kicks
  in the audio near the seam (tempo seeded from `beats_seconds`), because on
  real songs the beat tracker's phase can jump by a quarter beat or more
  within a section. Without a clear kick it falls back to a robust fit of
  `beats_seconds`. The source is noted in `safety_adjustments`.
- **Glides the tempo** when the two tempos differ: inside the blend, Rubber
  Band warps the outgoing song from its tempo to the incoming tempo, and the
  incoming song from the outgoing tempo to its own, so every beat lands
  together. `tempo_change` = incoming/outgoing tempo ratio (e.g. 0.93 for
  138 -> 128 BPM), `null` when nothing was warped.
- **Folds half/double time per pair**: two songs are compared at whichever of
  1x, 2x or 0.5x is closest (100 and 200 BPM are 0% apart and match on every
  other beat), so the 90-180 BPM fold in analysis only changes the number
  shown, never which seams match.
- **Tempos too far apart to match** (more than 15% after half/double
  folding, e.g. 90 BPM rap into 128 BPM house): if A's ending or B's opening
  has no steady kick for at least 4 s (`seam_plan.pulse_curve`: how sharply
  the onset flux folds onto the beat period over 12 s windows, per second;
  below 1.8 is beatless), it is a **beatless blend** of the requested length,
  up to that beatless stretch, with no warp and the same band swaps
  (`plan_beatless`, note `"beatless blend of ..."`). Otherwise both sides
  have a beat and the seam is an **echo-out** (below). `analyze_track`
  stores the curve as `metadata["pulse"]` / `metadata["pulse_hop_seconds"]`
  so the planner and the renderer judge the same thing; without it the
  renderer measures the audio near the seam. The ~2-beat short blend is now
  only the fallback when Rubber Band is not installed.
- **Swaps frequency bands, landing on the phrase**: the blend resolves on its
  last matched beat, the phrase boundary in both songs. Lows (<200 Hz) swap
  on that beat, highs (>5 kHz) halfway, and mids cross over in the second
  half with `fade_style`; before that the incoming mids sit at -14 dB. So the
  incoming song builds under the outgoing one and its next phrase hits as
  the outgoing one leaves. Described in `applied_eq`.
- **Picks entries for the landing** (`choose_windows`): B's handoff is judged
  on the second half of the blend (what is heard), payoff adds B's energy
  jump on the landing beat, and set mode counts B's best section as kept
  from the landing, so B comes in one phrase before its chorus or drop
  instead of burying it under A.
- **Blends around fades** (`edges.py`): when the outgoing song ends with a
  fade to silence (most JJTBEATZ remixes: 2-8 s) or the incoming one starts
  with a fade-in, the seam is lengthened by the fade and the crossfade and
  band swaps happen on full-level music, finishing where the outgoing fade
  begins (starting after the incoming fade-in). The fade still plays inside
  the seam, under the other track. Noted in `safety_adjustments` ("outgoing
  fades out over its last ...") and `applied_eq` ("over a-b s of the n s
  seam"). Fades under 1 s are ignored, fades over 12 s are capped. The 32 s
  overlap limit applies to the blend; the whole seam may be up to 32 s plus
  the fades, and never more than half of either track.
- **Lands where it was aimed**: a requested length within half a beat of
  a whole number of bars (`plan_windows` aims every blend at an event in B,
  counted in bars) renders exactly that many bars, past any fade the seam
  is lengthened for, so the bass swap hits the event. Other requests (the
  Tune trials, 8 s and 24 s) round as before: 32 beats when within
  0.8-1.3x, else 16 beats (0.7-1.35x), else the nearest whole bar count.

Field meanings with a glide (no schema change):

| Field | Meaning |
| --- | --- |
| `actual_overlap_seconds` | how much of the incoming track plays inside the blend (where the next track resumes) |
| `output_duration_seconds` (final mode) | how long the blend lasts in the mix |
| `outgoing_start_seconds` | where the blend starts in the outgoing track (unchanged) |

With a fade extension, `actual_overlap_seconds` includes the fade, so it is
longer than the audible crossfade (e.g. 13.4 s for an 8 s blend over a 5.2 s
fade-out); `beat_alignment_adjustments_seconds` excludes the fade. Without a
glide the first two are equal, as before. Where song *i+1* starts
in the mix: `start[i+1] = start[i] + outgoing_start_seconds[i] +
(output_duration_seconds[i-1] - actual_overlap_seconds[i-1])`, with the last
term 0 for the first seam.

Deployment: needs the `rubberband` binary (`apt-get install rubberband-cli`
or `brew install rubberband`). No new Python packages.

### Loudness matching (`loudness.py`)

On by default in both `render_preview` and `render_mix`; routes need no
change. Each track gets one gain toward -14 LUFS (BS.1770 integrated
loudness), computed from that track alone, so a track's gain is identical in
its A/B previews and in the final mix. A boost never pushes a sample past
0.98 (no limiter on whole tracks), so a quiet track with loud peaks may stay
below target; that is noted in `safety_adjustments` and `limitations`.

`AppliedTransition.applied_gain_db` is `{"outgoing": dB, "incoming": dB}`,
the gains actually applied to the two tracks at that seam (0.0 when
`loudness_match=False`). The UI can show it as "track turned down 5.2 dB to
match".

**A/B preview leveling** (`preview.py`, on by default, `level_match=True`):
after the seam is rendered, the whole preview excerpt is scaled by one
uniform gain to `PREVIEW_TARGET_LUFS` (-16 LUFS), so A and B play at the
same integrated loudness and the louder one can't win the vote just for
being louder. The seam itself is unchanged: the preview is exactly the mix's
seam times that one gain. The gain is recorded as
`applied_gain_db["preview_output"]` (absent with `level_match=False`; never
present in final-mix transitions). If a boost would pass the 0.98 sample
peak it stops short and a `preview level: ...` note goes in
`safety_adjustments`. Routes need no change.

### Claude mood (optional, `mood.py`)

```python
def mood_scores(tracks: list[Track]) -> dict[tuple[str, str], float] | None:
    """Directed mood compatibility in [0, 1] keyed by (from_id, to_id).
    Returns None on any failure or missing API key; never raises."""
```

Implemented in `backend/audio/mood.py`. One `claude-opus-5-5` call (low
effort, structured JSON output) rates every ordered pair. Claude sees each
song's filename plus BPM, key, length, melody share and section labels, and
is told to judge mood only, since tempo, key and energy are scored
separately. Results are cached per set of track ids, so `/order` with the
same songs (or `keep_order`) doesn't call Claude again. Failures aren't
cached. `/order` passes the result to `score_pairs`.

## Sahil: `backend/logic`

### `scoring.py`

```python
def score_pairs(
    tracks: list[Track],
    mood: dict[tuple[str, str], float] | None = None,
) -> list[PairScore]:
    """Directed scores for every ordered pair. Works without mood."""
```

### `ordering.py`

```python
def order_tracks(tracks: list[Track], scores: list[PairScore]) -> list[str]:
    """Return every track id exactly once, best path first."""
```

`POST /api/order` calls `score_pairs` then `order_tracks`. With `keep_order: true`
it keeps `track_ids` as the user arranged them and explains that order with
`explain_user_order`, scored against Mixr's pick (`suggested_track_ids`).

### `calibration.py`

```python
def calibration_overlaps() -> tuple[float, float]:
    """(short_seconds, long_seconds) to compare in an A/B trial."""

def update_preference(preference: Preference, vote: Vote) -> Preference:
    """Short vote: p = max(-1, p - 0.5). Long vote: p = min(1, p + 0.5)."""

def recommended_overlap_seconds(overlap_preference: float) -> float:
    """clamp(base + p * range, safe_min, safe_max)."""
```

`POST /api/calibration/vote` calls `update_preference`. `POST /api/mix`
passes the preference to `windows.plan_windows` (p = 0.0 when the user
skipped calibration), which picks every seam's blend length for its own pair
of songs and leans each choice by p. `recommended_overlap_seconds` is no
longer used by the routes.

## Musical cue points and DJ set mode

**Problem.** Today every blend is the last seconds of song A over the first
seconds of song B, wherever those happen to fall musically. Dance remixes
have DJ-friendly intros and outros, so that works. Pop, Bollywood and rap
usually don't. Measured on "Sheila Ki Jawani" → "Tell Her You Know": Sheila
stays at full volume with lead melody and vocals until its last 2 s, and Tell
Her has melody from its first second. The blend cut one melody off mid-line
and crashed it into another, which sounds abrupt and random.

**Fix.** Choose *where* each song is left and entered, the way a DJ does:
leave A at the end of a vocal or melody phrase, enter B where it's
instrumental, and hand the melody over so the two never sing at once. Each
track plays a **window** `[start, end]` of itself, and the existing
transitions blend the windows.

- **Full mode** (default): windows stay as long as possible. Each song is
  trimmed only as much as needed to reach a clean cue point, and every song
  still plays nearly in full.
- **Set mode**: windows are sized to hit a target set length (best section
  of each song), so five songs become an 8-10 minute set.

Both modes share one code path; full mode is just "no target length".

### Shared types (`schemas.py`)

```python
class Section(BaseModel):
    start_seconds: float           # on a phrase start
    end_seconds: float
    energy: float                  # mean loudness of the section, 0-1 relative to the track's loudest section
    label: Optional[str] = None    # "intro" | "verse" | "build" | "chorus" | "drop" | "break" | "outro"; None unless confident

class Track(BaseModel):
    ...                            # existing fields unchanged
    phrase_starts_seconds: list[float] = []  # every 8 or 16 bars on the beat grid; [] when there are no beats
    sections: list[Section] = []             # contiguous, cover the whole track, boundaries on phrase starts
    melody_activity: list[float] = []        # 0-1 per energy_hop_seconds: how much lead melody/vocal is present

class TrackWindow(BaseModel):
    track_id: str
    start_seconds: float           # in the source track
    end_seconds: float
    reason: Optional[str] = None   # e.g. "the drop at 1:12 (loudest section)"

class MixManifest(BaseModel):
    ...                            # existing fields unchanged
    mode: Literal["full", "set"] = "full"
    windows: list[TrackWindow] = []          # set mode: the windows actually played, same order as ordered_track_ids
```

With windows, every time in `AppliedTransition` (`outgoing_start_seconds` and
friends) is measured from the start of the **window**, not the source track,
so the existing timeline math is unchanged. Add `window.start_seconds` to get
the position in the source track.

### Aarav: analysis and rendering (implemented)

- `analyze_track` fills `melody_activity` (`cues.py`): harmonic part of an
  HPSS split, energy in 300-3400 Hz, as amplitude relative to the track's
  own 95th percentile, clipped to 0-1, one value per `energy_hop_seconds`
  (same grid as `energy_rms`). 1.0 = as melodic as this song gets, 0.5 =
  about 6 dB quieter. Sustained pads and chords in that band count too: it
  is a proxy, not vocal detection.
- `analyze_track` fills `phrase_starts_seconds`: every 8 bars, on beats from
  `beats_seconds`. The downbeat is the beat phase with the hardest low end
  (plus chord changes); the phrase phase is the bar offset where chroma,
  loudness and melody change most. Without enough beats (fewer than 8),
  both lists stay empty and a note says so.
- `analyze_track` fills `sections`: consecutive phrases merged while they
  sound alike. A phrase start begins a new section when the change there is
  > 1.25x the track's median change or the loudness steps by >= 0.2, and no
  section is longer than 2 phrases (16 bars, ~30 s at 128 BPM; the first
  section also holds any lead-in before the first phrase). The first starts
  at 0, the last ends at the duration, all other boundaries are phrase
  starts. Labels only where the shape is obvious, everything else `None`:
  - `"intro"` / `"outro"`: quiet first / last section (energy < 0.6)
  - `"break"`: quiet middle section (energy < 0.5)
  - `"drop"`: loud (>= 0.8) section the music lifts into (>= 0.2 above the
    section before), with little lead melody (mean `melody_activity` < 0.5)
  - `"chorus"`: loud melodic section that is lifted into, or comes back
    later with something >= 0.2 quieter in between
  - a loud section right after a chorus / drop that sounds the same keeps
    its label (one long chorus split by the 16-bar cap)
  `"verse"` and `"build"` are never set: energy alone can't tell a soft
  verse from a break reliably.
- `analyze_track` stores `metadata["fade_out_seconds"]` and
  `metadata["fade_in_seconds"]`: the `edges.py` fade lengths at the song's
  real end and start (0.0 when there is no fade). `choose_windows` reads
  them, and `render_mix` / `render_preview` pass them to `render_transition`
  for a side that plays to its real end / from its real start, so both use
  the identical value. When a window cuts a side (more than 0.05 s from the
  edge), the renderer measures the cut audio instead, as before. No schema
  change.
- Cue analysis adds about 5-10 s per song. If it fails, the track still
  analyzes; the three fields stay empty and a note says why.
- **Vocal swap** (`bands.py`), in every seam, preview and final, in both
  modes: when both songs carry melody at the seam (mean `melody_activity` of
  A's last 8 s and of B's first 8 s both >= 0.5, measured outside any
  fade-out tail / fade-in head) and the beats match (or neither song has a
  beat), the blend's mids swap over one beat ending on the bass swap, with
  B's mids at -14 dB before it, so two vocals are never loud together.
  Recorded as `applied_eq` containing `"mids (vocals) swapped"` and a note
  `"both sides have lead melody at the seam: ..."`.
- **Echo-out** (`echo.py`): every seam between tempos over 15% apart where
  both sides have a beat, and the same vocal clash when the beats can't be
  matched (even with a beatless side). A's fade-out is cut, A plays its
  last 1, 2 or 4 bars (the closest to the requested overlap, at most half
  of A) with its lows ramped out, its last beat echoes 4, 6 or 8 times on a
  1-beat delay (high-passed at 300 Hz, fading to the same level), and B
  starts so its first beat lands on A's last bar line (with more than a
  beat of lead-in, its sound starts on the bar line, any quiet start under
  A), so there is never a silent beat. Every sample of both songs still
  plays. Tune trials of 8 and 24 s get echo-outs within a second or so of
  each other, so the Tune step skips these seams (audible gap under 2 s). Recorded as: `applied_eq` starting with `"echo-out"`, notes
  `"melody at the seam: outgoing x, incoming y"` and `"requested overlap
  sets the echo-out's length instead of a blend: ..."`,
  `overlap_changed_for_safety = true` only for a vocal clash, and a
  manifest limitation. The echo tail rings under B, so `output_duration_seconds -
  actual_overlap_seconds` is where B's sample 0 plays inside the seam; the
  timeline formula above already handles that, same as a glide.
  So once `choose_windows` hands off melodies cleanly, echo-out only kicks
  in where no clean cue exists. Until then it also fixes clashes like
  Sheila -> Tell Her in full mode with no trimming.
- `render_mix(..., windows=None, mode=None)`: `windows=None` is full mode,
  exactly as before. With windows (validated: one per track, same order and
  ids, `0 <= start < end`, at least 4 s), each track is cut to its window and
  its beats and melody move with it. A first window that starts mid-song
  fades in over 1 beat, a last one that ends mid-song fades out over 8
  beats; other cuts sit inside a seam. `manifest.windows` holds the windows
  actually used (clamped to the decoded audio), and
  `source_durations_seconds` are **window lengths**, so the timeline math is
  unchanged. `mode` is recorded in the manifest; it defaults to `"set"` with
  windows and `"full"` without, so pass `mode="full"` when full mode calls
  `choose_windows`.
- `render_preview(..., outgoing_window=None, incoming_window=None)`: the
  same cutting, so the Tune step previews the seam the mix will contain.
  Pass the same windows the mix will use.

### Sahil: choosing cue points (windows) (implemented, `backend/logic/windows.py`)

```python
def choose_windows(
    tracks: list[Track],            # final order
    overlap_seconds: list[float],   # requested overlap per seam, len(tracks) - 1
    target_seconds: float | None = None,  # None = full mode (keep songs as long as possible)
) -> list[TrackWindow]:
    """One window per track, same order. reason says why each cue was picked."""

def plan_windows(
    tracks: list[Track],
    overlap_preference: float = 0.0,      # -1 short .. +1 long, from the A/B votes
    target_seconds: float | None = None,
    chroma: list[Chroma | None] | None = None,
    transition_length: Literal["auto", "longer", "extended"] = "auto",
) -> tuple[list[TrackWindow], list[float]]:
    """Windows plus a blend length per seam, chosen for each pair of songs."""
```

**Blend length per pair** (`plan_windows`, what the routes use): nothing
is a fixed ladder; every seam's length comes from the two songs.

- *Where it can land*: for each entry point in B, the events after it:
  vocal entries (melody coming in after 4 s without one), `chorus`/`drop`/
  `break`/`outro` section starts, energy lifts (+0.15 at a bar line), drops,
  phrase starts and half phrases, scored chorus/drop 1.0, vocal 0.9, lift
  and section 0.7, phrase and break 0.5, half phrase 0.25 (events within
  ~a beat merge, stacking +0.15). Each is counted in whole bars of matched
  beats from B's first beat, at least 8 beats, up to the setting's cap.
- *What A can carry*: the blend may not start before A's last vocal line
  starts, nor before a chorus, drop or energy lift in A's last stretch, and
  never runs past A's fade (it ends where the fade starts). Where both
  songs carry melody at the seam (echo.py's 8 s test), or the two melodies
  would sing over each other across the blend (stack >= 0.55), it stays at
  most 8 s whatever the setting or votes.
- *Choosing*: every allowed landing is scored with the seam terms below
  over its own length, plus 0.1 x the landing's strength, plus
  `0.1 x reach x (0.5 + 2 x (harmony - 0.5) - stack + p)`, where reach runs
  0 at 4 s to 1 at 32 s (and on past it for extended blends), stack is how
  much melody both songs carry over the blend (plus 0.5 where the vocals
  would have to swap) and p is the votes' preference plus the setting's
  lean. With no event in reach, any whole-bar length A can carry competes
  the same way. A length that won't fit in half of either window drops to the best
  one that does.
- *Tempos too far apart*: a beatless blend as long as the beatless stretch
  (A's ending or B's opening, from the stored pulse curve; not where both
  carry melody), else an echo-out of up to 2 bars (4 with Longer/Extended)
  that A can carry, scored 0.95x / 0.8x of a blend.
- Each reason says how long, where it lands and what stopped a later
  landing: "Blends into X over 24 s (48 beats) and lands as its vocals
  come in at 1:12.", "...lands on its chorus at 2:03; a later landing
  would start the blend before its last vocal line at 3:41.", "Their tempos
  (128 and 91 BPM) are too far apart to beat-match and both have a beat, so
  it echoes out over its last 2 bars ...".

**Transition length setting** (`transition_length`, `windows.BLEND_SIZES`):
`"auto"` (default) caps landings at 32 s with no lean. `"longer"` adds 1.0
to p and lets a blend land up to 40 s in; `"extended"` adds 1.5 and up to
60 s, halving the stack penalty in the length bonus. Both allow 4-bar
echo-outs, and set mode budgets 16, 24 or 40 s per seam before the real
lengths are known (a last pass mixes window lengths when every window
moving a whole phrase at once would step over the target). On the One More
Time pair, Auto lands on its chorus 31 s in and Extended on the same chorus
47 s in. The 8 s cap for two melodies and the fit-in-half-a-window rule
still apply; in set mode the target length wins, so a short set caps its
blends to fit. Render with
`overlap_limits(transition_length)`, which raises only
`SafetyLimits.max_overlap_seconds`; the default stays 32 s. The A/B trial
compares 8/24 s (auto), 12/36 s (longer) or 24/56 s (extended).

Every window starts and ends on a phrase start (or a beat when there are no
phrases) and never exceeds the track. For each seam, score candidate
(A's end, B's start) pairs:

- **Melody handoff (most important):** A's `melody_activity` should be
  falling or low over the blend (a vocal line just finished), and B's low at
  its start (instrumental intro or groove), so only one melody plays at a time.
  Never end A in the middle of a high-melody stretch if a phrase boundary
  where it drops is available.
- **Energy continuity:** A's energy at its end close to B's energy at its start.
- **Structure:** prefer leaving A after a chorus/drop ends and entering B at
  an intro or before a build, using `sections` when labeled.
- **Length:** full mode prefers the latest good end in A and the earliest
  good start in B (trim as little as possible). Set mode needs windows of at
  least 45 s that include each track's best section, with
  sum(window lengths) - sum(overlaps) within about 10% of `target_seconds`.

If no candidate scores well, return the least-bad one and say so in
`reason`, so the renderer can use the short-cut fallback. The order
explanation can mention the chosen cues once windows exist.

How the implementation decides (constants at the top of the module):

- Candidates are phrase starts, else every 4th beat (bar lines), else a
  4 s grid, plus the song's own start and end. A cue at the song's own end
  or start is judged past its fade-out / quiet lead-in (from `energy_rms`),
  where `transition.py` actually places the blend.
- Melody and energy are measured over the overlap, or over echo.py's 8 s
  if that's longer. Seam quality = 0.25 handoff x groove + 0.15 groove +
  0.15 flow + 0.15 payoff + 0.05 energy match + 0.10 structure + 0.15
  harmony.
  - Handoff penalizes both melodies at once most, then A still singing,
    then B busy in the blend's second half (where its mids come up); a
    falling A line earns a bonus; both sides at echo.py's clash level cost
    extra.
  - Harmony (`backend/audio/harmony.py`): mean chroma of A's last overlap
    vs B's first, each centered on its own mean, cosine mapped to 0-1. The
    route computes chroma from the audio (cached per file, ~1 s per song)
    and passes it as `choose_windows(..., chroma=)`; without it the term is
    a neutral 0.5. Reasons call out matches >= 0.75 and rubs < 0.4.
  - Groove: both songs near their normal loudness during the blend. It also
    scales the handoff, because quiet melody over a quiet blend is dead air
    (an outro running into a sparse intro), not a clean handoff.
  - Flow: seconds with no melody from either song across the seam, from A's
    last melodic moment to B's first. Up to 8 s is fine; 32 s scores 0. This
    stops a vocal ending from running into a long instrumental intro.
  - Payoff: B's energy and melody in the 16 s right after the blend, plus
    its energy jump on the landing beat (scored like a drop).
- Drop transitions: a drop is a phrase start where the energy over the next
  8 s jumps (0.3 of the song's loudness = full strength) up to near full.
  For each drop of the incoming song, a build start a whole number of bars
  (about the overlap, at most 24 s) before it is added as a candidate
  start, usable only for the drop (a blend can't enter there). A drop seam
  scores 0.2 + 0.25 x strength + 0.1 x A's energy as it leaves + 0.1 x
  structure (at most ~0.65), and replaces the blend where it scores higher,
  so it only takes over from weak blends: clean blends score 0.75-0.9.
  Drops between tempos over 15% apart (a cut with a beat of silence) are
  scaled by 0.8, so they only win over an echo-out or a beatless blend
  where B has a strong drop. The incoming window then sets
  `drop_seconds`; mix.py and preview.py turn that into
  `TransitionRequest(style="drop", requested_overlap_seconds=build)`.
  backend/audio/drop.py renders it: the outgoing song gets a resonant
  high-pass sweep (40 Hz to 1.8 kHz, with makeup gain) and the incoming drop
  lands with full bass. With tempos close enough to glide, the build plays
  under the sweep without its bass and the sweep runs right up to the drop
  (layered, the beat never stops); otherwise the outgoing's last bar rolls
  (1, 1/2, 1/4, 1/8 beat), the beat before the drop is silent, and the
  incoming song starts cold on its drop (cut). Both A/B previews of a drop
  seam sound the same.
- Full mode: the first song starts at 0 and the last plays to its end; each
  trimmed second costs 0.003 (a sparse second, below 0.7 of the song's
  normal loudness, proportionally less) and no edge loses more than 30% of
  its song.
- Set mode: per-song target lengths split `target_seconds + sum(overlaps)`
  evenly (bounded by each song's length and the 45 s floor), cost = relative
  length error + 0.15 x the part of the best section left out (counted
  from where the incoming blend lands), so a clearly better-matched pair
  of parts can win over keeping the loudest section. The best
  section is a `chorus`/`drop` section if ever labeled, else the loudest
  30 s of `energy_rms` snapped out to the cue grid. Up to 4 passes re-aim
  the targets until the set is within 5% (10% is accepted).
- A dynamic program over the running order maximizes total seam quality
  minus the length costs, so a middle song's entry and exit are picked
  together. Every window is at least twice each adjoining overlap (the
  renderer caps an overlap at half a window), so head and tail seams never
  collide; songs too short for that play whole.

### Shaan: API and UI

- `POST /api/mix` and `POST /api/calibration/preview` accept
  `mode: "full" | "set"` (default `"full"`) and `target_seconds` (set mode
  only, 120-1200, default 90 s per track) and `transition_length:
  "auto" | "longer" | "extended"` (default `"auto"`). The route always calls
  `plan_windows` (with `target_seconds=None` in full mode) and passes the
  blend lengths it returns to `render_mix`, one per seam.
- UI: a "Full songs / DJ set" toggle with a set-length picker and an
  "Auto / Longer / Extended" transition-length picker on the running-order
  screen; the setting flows into both the A/B previews and the final mix,
  and a blend over 32 s reads "extended blend" on its card (an echo-out
  reads "Echo → downbeat", a beatless blend "No beat to clash"). The timeline shows only the played part of each song,
  and the tracklist shows which part was used (e.g. "1:12-2:45").
- Preview cue points use the neutral preference (`plan_windows(p=0)`),
  since the votes aren't in yet; the A/B trial itself still compares the
  fixed short and long overlaps. The mix plans with the voted preference, so
  a strong lean can shift a cue slightly.

## HTTP API (Shaan)

| Endpoint | Request | Response |
| --- | --- | --- |
| `POST /api/tracks` | multipart `files` (2–5) | `{track_ids, tracks}` |
| `GET /api/tracks/{id}` | | `Track` |
| `GET /api/tracks/{id}/audio` | | audio file |
| `POST /api/analyze` | `{track_ids}` | `{tracks}` |
| `POST /api/order` | `{track_ids, keep_order?}` | `{ordered_track_ids, suggested_track_ids, scores, explanation}` |
| `POST /api/calibration/preview` | `{ordered_track_ids, trial, mode?, target_seconds?, transition_length?}` | `{trial_id, from_id, to_id, a, b}`, each `{audio_url}` |
| `GET /api/calibration/previews/{trial_id}/{a\|b}` | | preview WAV |
| `POST /api/calibration/vote` | `{trial_id, chosen, preference}` | `{preference, chosen_overlap_seconds, other_overlap_seconds, chose_longer}` |
| `POST /api/mix` | `{ordered_track_ids, overlap_preference, calibrated, mode?, target_seconds?, transition_length?}` | `{mix_id, audio_url, manifest}` |
| `GET /api/mixes/{id}` | | `{mix_id, audio_url, manifest}` |
| `GET /api/mixes/{id}/audio` | | mix WAV |

The server randomizes which of short/long is A, so the UI can hide the
labels until after the vote.
