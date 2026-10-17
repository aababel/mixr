"""Pairwise compatibility scoring (owner: Sahil).

Every sub-score is in [0, 1], or None when the inputs needed for it are
missing. A missing sub-score counts as NEUTRAL in the total, so the default
mix works without Claude (mood) or reliable key detection (harmonic), and
missing data is never a bonus over real data.
"""
import math
from typing import Optional

from backend.api.schemas import PairScore, Track

# Starting weights from the project brief. Tune by ear, not by spreadsheet.
WEIGHTS = {"harmonic": 0.40, "tempo": 0.35, "energy": 0.15, "mood": 0.10}

# Above this BPM gap (as a fraction) the tempo score bottoms out at 0.
MAX_TEMPO_GAP = 0.08

# How much audio at each seam to compare for energy.
ENERGY_WINDOW_SECONDS = 30.0

# Stand-in for any sub-score we couldn't compute.
NEUTRAL = 0.5


def _clean(score: Optional[float]) -> Optional[float]:
    """Clamp to [0, 1]; treat NaN/inf as missing."""
    if score is None or not math.isfinite(score):
        return None
    return min(1.0, max(0.0, score))


def parse_camelot(key: str) -> Optional[tuple[int, str]]:
    """'8A' -> (8, 'A'). Returns None for anything that isn't 1A-12B."""
    key = key.strip().upper()
    if len(key) < 2 or key[-1] not in ("A", "B") or not key[:-1].isdecimal():
        return None
    number = int(key[:-1])
    if not 1 <= number <= 12:
        return None
    return number, key[-1]


def harmonic_score(from_key: Optional[str], to_key: Optional[str]) -> Optional[float]:
    """Camelot wheel compatibility."""
    if from_key is None or to_key is None:
        return None
    a, b = parse_camelot(from_key), parse_camelot(to_key)
    if a is None or b is None:
        return None

    # Steps around the 12-position wheel, wrapping (12 -> 1 is one step).
    diff = abs(a[0] - b[0])
    steps = min(diff, 12 - diff)
    same_letter = a[1] == b[1]

    if steps == 0:
        return 1.0 if same_letter else 0.9  # same key / relative major-minor
    if steps == 1:
        return 0.9 if same_letter else 0.6  # neighbor / diagonal
    if steps == 2 and same_letter:
        return 0.5  # "energy boost" move, usable but noticeable
    return max(0.0, 0.4 - 0.05 * steps)


def tempo_score(from_bpm: Optional[float], to_bpm: Optional[float]) -> Optional[float]:
    """Closeness of BPM, treating half/double time as compatible. Symmetric."""
    if from_bpm is None or to_bpm is None or from_bpm <= 0 or to_bpm <= 0:
        return None  # librosa reports 0 when it finds no beat
    # A 64 BPM and a 128 BPM track mix fine, so compare against whichever of
    # to_bpm, 2x, or 0.5x is closest. analyze_track already folds tempos into
    # 90-180 BPM; this is the backstop for values that weren't folded.
    candidates = (to_bpm, to_bpm * 2, to_bpm / 2)
    gap = min(abs(from_bpm - c) / max(from_bpm, c) for c in candidates)
    return _clean(1.0 - gap / MAX_TEMPO_GAP)


def _mean(values: list[float]) -> float:
    return sum(values) / len(values)


def energy_score(outgoing: Track, incoming: Track) -> Optional[float]:
    """How well the end of `outgoing` matches the start of `incoming` in loudness.

    Expects linear RMS amplitudes (librosa.feature.rms), not dB.
    """
    if not outgoing.energy_hop_seconds or not incoming.energy_hop_seconds:
        return None

    out_frames = max(1, int(ENERGY_WINDOW_SECONDS / outgoing.energy_hop_seconds))
    in_frames = max(1, int(ENERGY_WINDOW_SECONDS / incoming.energy_hop_seconds))
    tail = [v for v in outgoing.energy_rms[-out_frames:] if math.isfinite(v)]
    head = [v for v in incoming.energy_rms[:in_frames] if math.isfinite(v)]
    if not tail or not head or min(tail) < 0 or min(head) < 0:
        return None  # empty, all-NaN, or dB values: can't compare

    tail_mean, head_mean = _mean(tail), _mean(head)
    louder = max(tail_mean, head_mean)
    if louder == 0:
        return 1.0  # both silent at the seam
    return _clean(1.0 - abs(tail_mean - head_mean) / louder)


def combine(subscores: dict[str, Optional[float]]) -> float:
    """Weighted sum, with missing (or invalid) sub-scores counted as NEUTRAL."""
    total = 0.0
    for name, weight in WEIGHTS.items():
        score = _clean(subscores.get(name))
        total += weight * (NEUTRAL if score is None else score)
    return total


def score_pair(outgoing: Track, incoming: Track, mood: Optional[float] = None) -> PairScore:
    subscores = {
        "harmonic": harmonic_score(outgoing.key_camelot, incoming.key_camelot),
        "tempo": tempo_score(outgoing.bpm, incoming.bpm),
        "energy": energy_score(outgoing, incoming),
        "mood": _clean(mood),
    }
    return PairScore(
        from_id=outgoing.id,
        to_id=incoming.id,
        total=combine(subscores),
        **subscores,
    )


def score_matrix(
    tracks: list[Track],
    mood_scores: Optional[dict[tuple[str, str], float]] = None,
) -> dict[tuple[str, str], PairScore]:
    """Score every ordered pair (A->B and B->A differ because energy is directional).

    mood_scores comes from the Claude integration, keyed by (from_id, to_id),
    with values in [0, 1]. Pass None, or leave pairs out, when Claude is
    unavailable.
    """
    mood_scores = mood_scores or {}
    matrix = {}
    for a in tracks:
        for b in tracks:
            if a.id != b.id:
                matrix[(a.id, b.id)] = score_pair(a, b, mood_scores.get((a.id, b.id)))
    return matrix


def score_pairs(
    tracks: list[Track],
    mood: Optional[dict[tuple[str, str], float]] = None,
) -> list[PairScore]:
    """Directed scores for every ordered pair. Works without mood.

    Entry point from docs/integration.md; mood comes from mood_scores() in
    backend/audio and may be None.
    """
    return list(score_matrix(tracks, mood).values())
