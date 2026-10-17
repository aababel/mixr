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

Used by `POST /api/mix`.

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
