"""A/B overlap calibration (owner: Sahil).

The user hears the same seam with a short and a long overlap and picks one.
Each vote nudges overlap_preference toward -1 (short) or +1 (long), and that
preference sets the overlap requested for every seam in the final mix.
Calibration is optional: p = 0.0 gives the default overlap.
"""
import math

from backend.api.schemas import Preference, Vote
from backend.audio.transition import DEFAULT_SAFETY

# Must sit inside the renderer's SafetyLimits (1-16 s), or the clamp in
# recommended_overlap_seconds would flatten every preference to one value.
SHORT_OVERLAP_SECONDS = 4.0
LONG_OVERLAP_SECONDS = 12.0
VOTE_STEP = 0.5

# p = -1 -> SHORT, p = 0 -> halfway, p = +1 -> LONG.
BASE_OVERLAP_SECONDS = (SHORT_OVERLAP_SECONDS + LONG_OVERLAP_SECONDS) / 2
OVERLAP_RANGE_SECONDS = (LONG_OVERLAP_SECONDS - SHORT_OVERLAP_SECONDS) / 2


def calibration_overlaps() -> tuple[float, float]:
    """(short_seconds, long_seconds) to compare in an A/B trial."""
    return SHORT_OVERLAP_SECONDS, LONG_OVERLAP_SECONDS


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
