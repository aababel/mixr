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

## HTTP API (Shaan)

| Endpoint | Request | Response |
| --- | --- | --- |
| `POST /api/tracks` | multipart `files` (3–5) | `{track_ids, tracks}` |
| `GET /api/tracks/{id}` | | `Track` |
| `GET /api/tracks/{id}/audio` | | audio file |
| `POST /api/analyze` | `{track_ids}` | `{tracks}` |
| `POST /api/order` | `{track_ids}` | `{ordered_track_ids, scores}` |
| `POST /api/calibration/preview` | `{from_id, to_id}` | `{a, b}`, each `{audio_url, requested_overlap_seconds, applied}` |
| `POST /api/calibration/vote` | `{preference, vote}` | `Preference` |
| `POST /api/mix` | `{ordered_track_ids, overlap_preference}` | `{mix_id, audio_url, manifest}` |
| `GET /api/mixes/{id}` | | `{mix_id, audio_url, manifest}` |

The server randomizes which of short/long is A, so the UI can hide the
labels until after the vote.
