"""Claude mood compatibility between songs (owner: Aarav).

One Claude call per set of songs rates every ordered pair: how well the
mood of the incoming song follows the outgoing one (genre, emotional tone,
vibe). Tempo, key and loudness are scored separately in backend/logic/scoring.py,
so the prompt tells Claude to judge mood only and gives the audio features
as context, not as things to score.

Claude only sees what the app knows about each song: its filename (often
"Artist - Title", which Claude may recognize) and the analysis features.
No audio is sent.

Mood is optional. mood_scores returns None when ANTHROPIC_API_KEY is unset
or anything goes wrong (network, refusal, bad output), and scoring then
treats mood as neutral. Results are cached per set of track ids, so
reordering the same songs doesn't call Claude again.
"""
import json
import logging
import os
from pathlib import Path
from typing import Optional

import anthropic

from backend.api.schemas import Track

log = logging.getLogger(__name__)

MODEL = "claude-opus-5-5"
EFFORT = "low"  # a quick judgment call that sits in the /order request path
MAX_TOKENS = 4000
TIMEOUT_SECONDS = 30.0
MAX_RETRIES = 1

SYSTEM = (
    "You are an experienced DJ judging whether songs flow into each other by mood: "
    "genre, emotional tone, vibe and intensity of feeling. Tempo, musical key and "
    "loudness are already scored separately, so do NOT reward or punish a pair for "
    "them; use them only as hints about each song's style. Songs are identified by "
    "their uploaded filename, which may contain the artist and title. "
    "Rate each ordered pair from 0 to 1: 1 means the second song feels like a natural "
    "next step after the first, 0.5 means neutral or unsure, 0 means a jarring mood "
    "clash. Order matters (a calm song into a hype one can rate differently from the "
    "reverse). Rate every ordered pair you are asked for, exactly once."
)

OUTPUT_SCHEMA = {
    "type": "object",
    "properties": {
        "pairs": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "from": {"type": "string"},
                    "to": {"type": "string"},
                    "mood": {"type": "number"},
                },
                "required": ["from", "to", "mood"],
                "additionalProperties": False,
            },
        }
    },
    "required": ["pairs"],
    "additionalProperties": False,
}

_cache: dict[tuple[str, ...], dict[tuple[str, str], float]] = {}


def _describe(label: str, track: Track) -> str:
    """One song as the prompt sees it: name plus the features that hint at style."""
    parts = [f'{label}: "{Path(track.filename).stem}"']
    if track.bpm:
        parts.append(f"{track.bpm:.0f} BPM")
    if track.key_camelot:
        parts.append(f"key {track.key_camelot}")
    parts.append(f"{track.duration_seconds / 60:.1f} min")
    if track.melody_activity:
        share = sum(track.melody_activity) / len(track.melody_activity)
        parts.append(f"lead melody/vocals in {share:.0%} of the song")
    labels = [s.label for s in track.sections if s.label]
    if labels:
        parts.append("sections: " + ", ".join(labels))
    return ", ".join(parts)


def _prompt(labels: dict[str, Track]) -> str:
    songs = "\n".join(_describe(label, t) for label, t in labels.items())
    pairs = ", ".join(f"{a}->{b}" for a in labels for b in labels if a != b)
    return f"Songs:\n{songs}\n\nRate these ordered pairs: {pairs}"


def _parse(text: str, labels: dict[str, Track]) -> dict[tuple[str, str], float]:
    """Keep only well-formed ratings of known, distinct songs; first rating wins."""
    scores: dict[tuple[str, str], float] = {}
    for pair in json.loads(text).get("pairs", []):
        a, b, mood = pair.get("from"), pair.get("to"), pair.get("mood")
        if a in labels and b in labels and a != b and isinstance(mood, (int, float)):
            key = (labels[a].id, labels[b].id)
            scores.setdefault(key, min(1.0, max(0.0, float(mood))))
    return scores


def _ask_claude(prompt: str) -> Optional[str]:
    client = anthropic.Anthropic(timeout=TIMEOUT_SECONDS, max_retries=MAX_RETRIES)
    response = client.beta.messages.create(
        model=MODEL,
        max_tokens=MAX_TOKENS,
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
        output_config={"effort": EFFORT, "format": {"type": "json_schema", "schema": OUTPUT_SCHEMA}},
        system=SYSTEM,
        messages=[{"role": "user", "content": prompt}],
    )
    if response.stop_reason != "end_turn":
        log.warning("mood: Claude stopped with %s; skipping mood", response.stop_reason)
        return None
    return next((b.text for b in response.content if b.type == "text"), None)


def mood_scores(tracks: list[Track]) -> Optional[dict[tuple[str, str], float]]:
    """Directed mood compatibility in [0, 1] keyed by (from_id, to_id).
    Returns None on any failure or missing API key; never raises."""
    if len(tracks) < 2 or not os.getenv("ANTHROPIC_API_KEY"):
        return None
    cache_key = tuple(sorted(t.id for t in tracks))
    if cache_key in _cache:
        return _cache[cache_key]

    # Short labels instead of track ids: easier for Claude to echo back exactly.
    labels = {chr(ord("A") + i): t for i, t in enumerate(tracks)}
    try:
        text = _ask_claude(_prompt(labels))
        scores = _parse(text, labels) if text else {}
    except Exception as e:  # mood is optional: never break ordering
        log.warning("mood: Claude call failed (%s); skipping mood", e)
        return None
    if not scores:
        return None
    _cache[cache_key] = scores
    return scores
