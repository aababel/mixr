import pytest

from backend.api.schemas import PairScore, Track
from backend.logic.ordering import best_order, order_score, order_tracks


def matrix_from(totals: dict[tuple[str, str], float]):
    return {pair: PairScore(from_id=pair[0], to_id=pair[1], total=t) for pair, t in totals.items()}


def full_matrix(ids, good_pairs, good=1.0, bad=0.1):
    return matrix_from({(a, b): good if (a, b) in good_pairs else bad for a in ids for b in ids if a != b})


def test_finds_the_only_good_chain():
    ids = ["a", "b", "c", "d"]
    matrix = full_matrix(ids, {("c", "a"), ("a", "d"), ("d", "b")})
    assert best_order(ids, matrix) == ["c", "a", "d", "b"]


def test_keeps_input_order_on_ties():
    ids = ["a", "b", "c"]
    matrix = full_matrix(ids, set(), bad=0.5)
    assert best_order(ids, matrix) == ids


def test_float_rounding_does_not_break_ties():
    # Regression: on Python 3.11, 0.1+0.2+0.3 > 0.3+0.2+0.1, so the reversed
    # order used to "win" a real tie.
    ids = ["a", "b", "c", "d"]
    totals = {("a", "b"): 0.3, ("b", "c"): 0.2, ("c", "d"): 0.1}
    totals.update({(b, a): t for (a, b), t in totals.items()})
    rest = {(a, b): 0.0 for a in ids for b in ids if a != b and (a, b) not in totals}
    assert best_order(ids, matrix_from({**totals, **rest})) == ids


def test_small_inputs():
    assert best_order([], {}) == []
    assert best_order(["a"], {}) == ["a"]
    matrix = matrix_from({("a", "b"): 0.2, ("b", "a"): 0.9})
    assert best_order(["a", "b"], matrix) == ["b", "a"]


def test_duplicate_ids_raise_clear_error():
    with pytest.raises(ValueError, match="unique"):
        best_order(["a", "b", "a"], {})


def test_greedy_fallback_for_large_playlists():
    ids = [str(i) for i in range(10)]
    chain = {(ids[i], ids[i + 1]) for i in range(9)}
    matrix = full_matrix(ids, chain)
    order = best_order(list(reversed(ids)), matrix)
    assert order_score(order, matrix) == order_score(ids, matrix)


def test_greedy_never_returns_worse_than_input():
    # Regression: every track's single best next step is "9", a dead end with
    # nothing good after it, so every greedy run (6.5) scores worse than the
    # user's order (7.2). The old code returned the 6.5 order.
    ids = [str(i) for i in range(10)]
    totals = {(a, b): 0.0 for a in ids for b in ids if a != b}
    for i in range(9):
        totals[(ids[i], ids[i + 1])] = 0.8  # the input order is a solid chain
    for i in range(8):
        totals[(ids[i], "9")] = 0.9  # tempting but leads nowhere
    matrix = matrix_from(totals)
    order = best_order(ids, matrix)
    assert order_score(order, matrix) == pytest.approx(order_score(ids, matrix))
    assert sorted(order) == sorted(ids)


def test_order_tracks_matches_integration_contract():
    tracks = [Track(id=i, filename=i, duration_seconds=1, sample_rate=1) for i in "abcd"]
    matrix = full_matrix("abcd", {("c", "a"), ("a", "d"), ("d", "b")})
    assert order_tracks(tracks, list(matrix.values())) == ["c", "a", "d", "b"]


def test_order_tracks_survives_missing_scores():
    # The placeholder /api/order route passes scores=[]; that must not crash.
    tracks = [Track(id=i, filename=i, duration_seconds=1, sample_rate=1) for i in "abc"]
    assert order_tracks(tracks, []) == ["a", "b", "c"]
