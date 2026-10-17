import pytest

from backend.api.schemas import Preference, Vote
from backend.audio.transition import DEFAULT_SAFETY
from backend.logic.calibration import (
    calibration_overlaps,
    recommended_overlap_seconds,
    update_preference,
)


def vote(chosen, a=4.0, b=12.0):
    return Vote(from_id="x", to_id="y", a_requested_seconds=a, b_requested_seconds=b, chosen=chosen)


def test_overlaps_are_short_then_long_and_safe():
    short, long = calibration_overlaps()
    assert short < long
    assert DEFAULT_SAFETY.min_overlap_seconds <= short
    assert long <= DEFAULT_SAFETY.max_overlap_seconds


def test_short_vote_moves_toward_minus_one():
    p = update_preference(Preference(), vote("A"))  # A = 4s = short
    assert p.overlap_preference == -0.5
    assert len(p.votes) == 1


def test_long_vote_moves_toward_plus_one():
    assert update_preference(Preference(), vote("B")).overlap_preference == 0.5


def test_short_is_judged_by_seconds_not_by_letter():
    # Server shuffles: here A is the LONG one, so picking A is a long vote.
    assert update_preference(Preference(), vote("A", a=12.0, b=4.0)).overlap_preference == 0.5


def test_preference_is_clamped():
    p = Preference()
    for _ in range(5):
        p = update_preference(p, vote("B"))
    assert p.overlap_preference == 1.0
    assert len(p.votes) == 5


def test_equal_overlaps_record_vote_without_moving():
    p = update_preference(Preference(overlap_preference=0.5), vote("A", a=8.0, b=8.0))
    assert p.overlap_preference == 0.5
    assert len(p.votes) == 1


def test_update_does_not_mutate_input():
    original = Preference()
    update_preference(original, vote("A"))
    assert original.overlap_preference == 0.0
    assert original.votes == []


def test_recommended_overlap_maps_preference_to_seconds():
    short, long = calibration_overlaps()
    assert recommended_overlap_seconds(-1.0) == short
    assert recommended_overlap_seconds(1.0) == long
    assert recommended_overlap_seconds(0.0) == pytest.approx((short + long) / 2)  # skipped calibration


@pytest.mark.parametrize("p", [-50.0, 50.0, float("nan"), float("inf")])
def test_recommended_overlap_stays_safe(p):
    seconds = recommended_overlap_seconds(p)
    assert DEFAULT_SAFETY.min_overlap_seconds <= seconds <= DEFAULT_SAFETY.max_overlap_seconds


def test_calibration_range_fits_inside_renderer_safety_limits():
    # If Aarav's SafetyLimits ever tighten past 4-12 s, every preference would
    # clamp to the same overlap and calibration would silently do nothing.
    short, long = calibration_overlaps()
    assert DEFAULT_SAFETY.min_overlap_seconds <= short < long <= DEFAULT_SAFETY.max_overlap_seconds
    assert recommended_overlap_seconds(-1.0) < recommended_overlap_seconds(0.0) < recommended_overlap_seconds(1.0)
