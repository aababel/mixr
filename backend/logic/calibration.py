"""A/B overlap calibration (owner: Sahil).

The user hears the same seam with a short and a long overlap and picks one.
Each vote nudges overlap_preference toward -1 (short) or +1 (long). The mix
picks each seam's blend length for its own pair of songs (windows.plan_windows);
the preference leans every one of those choices longer or shorter.
Calibration is optional: p = 0.0 leaves the choice to the songs.
"""
import math

from backend.api.schemas import Preference, Vote
from backend.audio.transition import DEFAULT_SAFETY

# Must sit inside the renderer's SafetyLimits (1-32 s), or the clamp in
# recommended_overlap_seconds would flatten every preference to one value.
# A continuous mix needs blends of 8-16 bars: 16 s is ~8 bars at 120 BPM.
SHORT_OVERLAP_SECONDS = 8.0
LONG_OVERLAP_SECONDS = 24.0
VOTE_STEP = 0.5

# p = -1 -> SHORT, p = 0 -> halfway, p = +1 -> LONG.
BASE_OVERLAP_SECONDS = (SHORT_OVERLAP_SECONDS + LONG_OVERLAP_SECONDS) / 2
OVERLAP_RANGE_SECONDS = (LONG_OVERLAP_SECONDS - SHORT_OVERLAP_SECONDS) / 2


# The A/B pair for each transition length setting (windows.BLEND_SIZES), each
# inside that setting's cap, so the trial compares lengths the mix can play.
TRIAL_OVERLAPS = {
    "auto": (SHORT_OVERLAP_SECONDS, LONG_OVERLAP_SECONDS),
    "longer": (12.0, 36.0),
    "extended": (24.0, 56.0),
}


def calibration_overlaps(transition_length: str = "auto") -> tuple[float, float]:
    """(short_seconds, long_seconds) to compare in an A/B trial."""
    return TRIAL_OVERLAPS[transition_length]


def update_preference(preference: Preference, vote: Vote) -> Preference:
    """Short vote: p = max(-1, p - 0.5). Long vote: p = min(1, p + 0.5).

    A and B are shuffled by the server, so "short" means whichever of the two
    requested overlaps was smaller. A trial with equal overlaps is recorded
    but doesn't move p.
    """
    chosen, other = (
        (vote.a_requested_seconds, vote.b_requested_seconds)
        if vote.chosen == "A"
        else (vote.b_requested_seconds, vote.a_requested_seconds)
    )
    p = preference.overlap_preference
    if chosen < other:
        p = max(-1.0, p - VOTE_STEP)
    elif chosen > other:
        p = min(1.0, p + VOTE_STEP)
    return preference.model_copy(
        update={"overlap_preference": p, "votes": [*preference.votes, vote]}
    )


def recommended_overlap_seconds(overlap_preference: float) -> float:
    """clamp(base + p * range, safe_min, safe_max).

    The renderer clamps again per seam (short tracks get less), so this only
    has to stay inside the global safety limits.
    """
    p = overlap_preference if math.isfinite(overlap_preference) else 0.0
    p = min(1.0, max(-1.0, p))
    seconds = BASE_OVERLAP_SECONDS + p * OVERLAP_RANGE_SECONDS
    return min(DEFAULT_SAFETY.max_overlap_seconds, max(DEFAULT_SAFETY.min_overlap_seconds, seconds))
