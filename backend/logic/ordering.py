"""Track ordering (owner: Sahil).

The demo uses 3-5 tracks, so we try every order (at most 120) and keep the
best. Above BRUTE_FORCE_LIMIT we fall back to a greedy nearest-neighbor pass.
"""
from itertools import permutations
from typing import Sequence

from backend.api.schemas import PairScore, Track
from backend.logic.scoring import NEUTRAL

BRUTE_FORCE_LIMIT = 8  # 8! = 40,320 orders, still instant

# An order must beat the current best by more than this to replace it, so
# float rounding never overrides the user's order on a real tie.
TIE_TOLERANCE = 1e-9


def order_score(order: Sequence[str], matrix: dict[tuple[str, str], PairScore]) -> float:
    """Sum of transition totals along an order."""
    return sum(matrix[(a, b)].total for a, b in zip(order, order[1:]))


def order_tracks(tracks: list[Track], scores: list[PairScore]) -> list[str]:
    """Return every track id exactly once, best path first.

    Entry point from docs/integration.md. Pairs missing from `scores` count
    as NEUTRAL rather than crashing.
    """
    ids = [t.id for t in tracks]
    matrix = {(s.from_id, s.to_id): s for s in scores}
    for a in ids:
        for b in ids:
            if a != b and (a, b) not in matrix:
                matrix[(a, b)] = PairScore(from_id=a, to_id=b, total=NEUTRAL)
    return best_order(ids, matrix)


def best_order(track_ids: list[str], matrix: dict[tuple[str, str], PairScore]) -> list[str]:
    """Order that maximizes the summed transition scores.

    The user's original order wins unless something is strictly better.
    """
    if len(set(track_ids)) != len(track_ids):
        raise ValueError(f"track ids must be unique, got {track_ids}")
    if len(track_ids) > BRUTE_FORCE_LIMIT:
        return _greedy(track_ids, matrix)

    # permutations() yields the original order first, so it holds ties.
    best, best_score = None, float("-inf")
    for perm in permutations(track_ids):
        score = order_score(perm, matrix)
        if best is None or score > best_score + TIE_TOLERANCE:
            best, best_score = perm, score
    return list(best)


def _greedy(track_ids: list[str], matrix: dict[tuple[str, str], PairScore]) -> list[str]:
    """Start from each track, always jump to the best unused next track; keep the best run."""
    best, best_score = list(track_ids), order_score(track_ids, matrix)
    for start in track_ids:
        order, remaining = [start], [t for t in track_ids if t != start]
        while remaining:
            nxt = max(remaining, key=lambda t: matrix[(order[-1], t)].total)
            order.append(nxt)
            remaining.remove(nxt)
        score = order_score(order, matrix)
        if score > best_score + TIE_TOLERANCE:
            best, best_score = order, score
    return best
