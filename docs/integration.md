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
- **Falls back to a short ~2-beat blend** ending on an incoming beat when the
  tempos differ by more than 12% after half/double folding (e.g. 90 BPM rap
  into 128 BPM house), or when Rubber Band is not installed. The requested
  overlap is not used there, so both A/B previews of such a seam are the same
  length.
- **Swaps frequency bands**: lows (<200 Hz) at the midpoint, highs (>5 kHz)
  at ~30%, mids crossfade with `fade_style`. Described in `applied_eq`.

Field meanings with a glide (no schema change):

| Field | Meaning |
| --- | --- |
| `actual_overlap_seconds` | how much of the incoming track plays inside the blend (where the next track resumes) |
| `output_duration_seconds` (final mode) | how long the blend lasts in the mix |
| `outgoing_start_seconds` | where the blend starts in the outgoing track (unchanged) |

Without a glide the first two are equal, as before. Where song *i+1* starts
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

### Claude mood (optional)

```python
def mood_scores(tracks: list[Track]) -> dict[tuple[str, str], float] | None:
    """Directed mood compatibility in [0, 1] keyed by (from_id, to_id).
    Returns None on any failure or missing API key; never raises."""
```

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

`POST /api/order` calls `score_pairs` then `order_tracks`.

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
calls `recommended_overlap_seconds` once per seam, with p = 0.0 when the
user skipped calibration.

## Musical cue points and DJ set mode (proposal)

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

### Aarav: analysis and rendering

- `analyze_track` also fills `phrase_starts_seconds` and `sections`, using the
  beat grid and the energy envelope (plus chroma self-similarity if it helps).
  Without beats, both stay empty and cue selection falls back to time-based cuts.
- `analyze_track` fills `melody_activity`. A cheap first version that already
  separates the Sheila/Tell Her cases: harmonic part of HPSS
  (`librosa.effects.hpss`), energy in roughly the 300-3400 Hz band, normalized
  per track. Upgrade path if it isn't reliable enough: the vocals stem from
  Demucs (heavier dependency, 30-60 s per song).
- When no clean cue exists on one side (e.g. vocals right up to the end), the
  seam should not overlap two melodies: let A finish its phrase, echo or
  filter out its last bar, and drop B on a downbeat. A short, intentional cut
  beats a long clash.
- `render_mix(..., windows: list[TrackWindow] | None = None)`: `None` means
  full mode, exactly as today. With windows, slice each track to its window
  and shift its beats by `-start_seconds` before the seams. Trimming must not
  click: the first window gets a short fade-in (about 1 beat) if it doesn't
  start at 0, and the last window a fade-out (about 2 bars) if it doesn't end
  at the track end. Return the windows actually used in `manifest.windows`,
  with `mode="set"`.
- `render_preview(..., outgoing_window=None, incoming_window=None)`: the same
  slicing, so the Tune step previews the seam the set will actually contain.

### Sahil: choosing cue points (windows)

```python
def choose_windows(
    tracks: list[Track],            # final order
    overlap_seconds: list[float],   # requested overlap per seam, len(tracks) - 1
    target_seconds: float | None = None,  # None = full mode (keep songs as long as possible)
) -> list[TrackWindow]:
    """One window per track, same order. reason says why each cue was picked."""
```

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

### Shaan: API and UI

- `POST /api/mix` and `POST /api/calibration/preview` accept
  `mode: "full" | "set"` (default `"full"`) and `target_seconds` (set mode
  only, 120-1200, default 90 s per track). The route always calls
  `choose_windows` (with `target_seconds=None` in full mode) using the same
  overlaps it passes to `render_mix`.
- UI: a "Full songs / DJ set" toggle with a set-length picker on the
  running-order screen. The timeline shows only the played part of each song,
  and the tracklist shows which part was used (e.g. "1:12-2:45").
- Until `choose_windows` and the windowed `render_mix` land, full mode keeps
  today's behavior and set mode returns `501`, shown as "coming soon".

## HTTP API (Shaan)

| Endpoint | Request | Response |
| --- | --- | --- |
| `POST /api/tracks` | multipart `files` (2–5) | `{track_ids, tracks}` |
| `GET /api/tracks/{id}` | | `Track` |
| `GET /api/tracks/{id}/audio` | | audio file |
| `POST /api/analyze` | `{track_ids}` | `{tracks}` |
| `POST /api/order` | `{track_ids}` | `{ordered_track_ids, scores, explanation}` |
| `POST /api/calibration/preview` | `{ordered_track_ids, trial, mode?, target_seconds?}` | `{trial_id, from_id, to_id, a, b}`, each `{audio_url}` |
| `GET /api/calibration/previews/{trial_id}/{a\|b}` | | preview WAV |
| `POST /api/calibration/vote` | `{trial_id, chosen, preference}` | `{preference, chosen_overlap_seconds, other_overlap_seconds, chose_longer}` |
| `POST /api/mix` | `{ordered_track_ids, overlap_preference, calibrated, mode?, target_seconds?}` | `{mix_id, audio_url, manifest}` |
| `GET /api/mixes/{id}` | | `{mix_id, audio_url, manifest}` |
| `GET /api/mixes/{id}/audio` | | mix WAV |

The server randomizes which of short/long is A, so the UI can hide the
labels until after the vote.
