"""Plain-English explanations of the chosen order (owner: Sahil).

Every phrase is backed by something the code measured. A sub-score that
wasn't computed (no key, no BPM, no Claude) is left out rather than guessed.

Returned as structured data (summary + per-transition reasons) so the UI
never has to parse strings; it looks up song names by track id itself.
"""
from typing import Optional, Sequence

from pydantic import BaseModel

from backend.api.schemas import PairScore, Track
from backend.logic.ordering import order_score
from backend.logic.scoring import MAX_TEMPO_GAP, camelot_steps, tempo_gap

# Tempo gap (fraction) still described as "matching". At MAX_TEMPO_GAP or
# more the tempo sub-score is 0, so that's described as a jump.
TEMPO_MATCH_GAP = 0.02
# Energy sub-score at or above this reads as "carries over", below LOW as a jump.
ENERGY_SMOOTH = 0.8
ENERGY_LOW = 0.5
# Claude mood at or above GOOD is a match, at or below POOR a shift.
MOOD_GOOD = 0.7
MOOD_POOR = 0.3

NO_DATA_REASON = "not enough analysis data to compare"
NOT_ENOUGH_TRACKS = "Not enough scored tracks to explain an order."


class TransitionExplanation(BaseModel):
    from_id: str
    to_id: str
    reasons: list[str]  # never empty; NO_DATA_REASON when nothing was measured
    score: float  # PairScore.total, 0..1


class OrderExplanation(BaseModel):
    summary: str
    transitions: list[TransitionExplanation]  # in the chosen order


def _key_reason(from_key: Optional[str], to_key: Optional[str]) -> Optional[str]:
    relation = camelot_steps(from_key, to_key)
    if relation is None:
        return None
    steps, same_letter = relation
    pair = f"{from_key} → {to_key}"
    if steps == 0:
        return f"same key ({from_key})" if same_letter else f"relative major/minor keys ({pair})"
    if steps == 1:
        return f"neighboring keys ({pair})" if same_letter else f"closely related keys ({pair})"
    if steps == 2 and same_letter:
        return f"a two-step key lift ({pair})"
    return f"keys clash ({pair})"


def _tempo_reason(from_bpm: Optional[float], to_bpm: Optional[float]) -> Optional[str]:
    measured = tempo_gap(from_bpm, to_bpm)
    if measured is None:
        return None
    gap, half_double = measured
    if gap <= TEMPO_MATCH_GAP:
        if half_double:
            return f"half/double-time match ({from_bpm:.0f} ↔ {to_bpm:.0f} BPM)"
        return f"matching tempo ({from_bpm:.0f} vs {to_bpm:.0f} BPM)"
    bpms = f"{from_bpm:.0f} vs {to_bpm:.0f} BPM"
    if gap >= MAX_TEMPO_GAP:
        return f"big tempo jump ({bpms})"
    return f"tempo differs by {gap:.0%} ({bpms})"


def _energy_reason(energy: Optional[float]) -> Optional[str]:
    if energy is None:
        return None
    if energy >= ENERGY_SMOOTH:
        return "loudness carries over at the seam"
    if energy < ENERGY_LOW:
        return "noticeable loudness jump at the seam"
    return None


def _mood_reason(mood: Optional[float]) -> Optional[str]:
    if mood is None:
        return None
    if mood >= MOOD_GOOD:
        return "Claude rated the moods a good match"
    if mood <= MOOD_POOR:
        return "Claude flagged a mood shift"
    return None


def explain_transition(outgoing: Track, incoming: Track, score: PairScore) -> TransitionExplanation:
    reasons = [
        r
        for r in (
            _key_reason(outgoing.key_camelot, incoming.key_camelot),
            _tempo_reason(outgoing.bpm, incoming.bpm),
            _energy_reason(score.energy),
            _mood_reason(score.mood),
        )
        if r
    ]
    return TransitionExplanation(
        from_id=outgoing.id,
        to_id=incoming.id,
        reasons=reasons or [NO_DATA_REASON],
        score=score.total,
    )


def _average_flow(order: Sequence[str], matrix: dict[tuple[str, str], PairScore]) -> Optional[float]:
    if len(order) < 2:
        return None
    return order_score(order, matrix) / (len(order) - 1)


def explain_order(
    tracks: list[Track],
    scores: list[PairScore],
    original_ids: list[str],
    ordered_ids: list[str],
) -> OrderExplanation:
    """A summary of the reorder plus one explanation per transition in the chosen order."""
    by_id = {t.id: t for t in tracks}
    matrix = {(s.from_id, s.to_id): s for s in scores}
    if (
        len(ordered_ids) < 2
        or any(i not in by_id for i in ordered_ids)
        or any(p not in matrix for p in zip(ordered_ids, ordered_ids[1:]))
    ):
        return OrderExplanation(summary=NOT_ENOUGH_TRACKS, transitions=[])

    # No "before" figure if the uploaded order has unscored pairs.
    if all(p in matrix for p in zip(original_ids, original_ids[1:])):
        before = _average_flow(original_ids, matrix)
    else:
        before = None
    after = _average_flow(ordered_ids, matrix)

    if list(ordered_ids) == list(original_ids):
        summary = f"Kept your order: it already had the smoothest transitions (average score {after:.0%})."
    elif before is not None:
        summary = f"Reordered for smoother transitions: average score {before:.0%} → {after:.0%}."
    else:
        summary = f"Reordered for smoother transitions (average score {after:.0%})."

    return OrderExplanation(
        summary=summary,
        transitions=[
            explain_transition(by_id[a], by_id[b], matrix[(a, b)])
            for a, b in zip(ordered_ids, ordered_ids[1:])
        ],
    )


def explain_user_order(
    tracks: list[Track],
    scores: list[PairScore],
    suggested_ids: list[str],
    ordered_ids: list[str],
) -> OrderExplanation:
    """The user's own order, explained as-is and scored against Mixr's pick."""
    explanation = explain_order(tracks, scores, suggested_ids, ordered_ids)
    if not explanation.transitions:
        return explanation
    matrix = {(s.from_id, s.to_id): s for s in scores}
    after = _average_flow(ordered_ids, matrix)
    if list(ordered_ids) == list(suggested_ids):
        summary = f"Your order is Mixr's pick too (average score {after:.0%})."
    elif all(p in matrix for p in zip(suggested_ids, suggested_ids[1:])):
        best = _average_flow(suggested_ids, matrix)
        if after >= best - 1e-9:
            summary = f"Your order flows as well as Mixr's pick (average score {after:.0%})."
        else:
            summary = f"Your order: average score {after:.0%} (Mixr's pick scores {best:.0%})."
    else:
        summary = f"Your order (average score {after:.0%})."
    return explanation.model_copy(update={"summary": summary})
