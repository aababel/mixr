import json

import pytest

from backend.api.schemas import Section, Track
from backend.audio import mood
from backend.logic.scoring import score_pairs


def make_track(id, filename=None, bpm=None, key=None):
    return Track(
        id=id,
        filename=filename or f"{id}.mp3",
        duration_seconds=180.0,
        sample_rate=44100,
        bpm=bpm,
        key_camelot=key,
    )


@pytest.fixture(autouse=True)
def fresh(monkeypatch):
    """Clean cache, a fake key, and no real Claude call unless a test fakes one."""
    monkeypatch.setattr(mood, "_cache", {})
    monkeypatch.setenv("ANTHROPIC_API_KEY", "test-key")

    def no_network(prompt):
        raise AssertionError("tests must fake _ask_claude")

    monkeypatch.setattr(mood, "_ask_claude", no_network)


def fake_claude(monkeypatch, pairs):
    calls = []

    def ask(prompt):
        calls.append(prompt)
        return json.dumps({"pairs": pairs})

    monkeypatch.setattr(mood, "_ask_claude", ask)
    return calls


def test_scores_map_labels_back_to_track_ids(monkeypatch):
    tracks = [make_track("t1"), make_track("t2")]
    fake_claude(monkeypatch, [{"from": "A", "to": "B", "mood": 0.9}, {"from": "B", "to": "A", "mood": 0.2}])
    assert mood.mood_scores(tracks) == {("t1", "t2"): 0.9, ("t2", "t1"): 0.2}


def test_bad_ratings_are_dropped_and_values_clamped(monkeypatch):
    tracks = [make_track("t1"), make_track("t2")]
    fake_claude(
        monkeypatch,
        [
            {"from": "A", "to": "B", "mood": 1.7},
            {"from": "A", "to": "B", "mood": 0.1},  # duplicate: first wins
            {"from": "A", "to": "A", "mood": 0.5},  # self pair
            {"from": "A", "to": "Z", "mood": 0.5},  # unknown song
            {"from": "B", "to": "A", "mood": "high"},  # not a number
        ],
    )
    assert mood.mood_scores(tracks) == {("t1", "t2"): 1.0}


def test_no_key_means_no_call(monkeypatch):
    monkeypatch.delenv("ANTHROPIC_API_KEY")
    assert mood.mood_scores([make_track("t1"), make_track("t2")]) is None


def test_fewer_than_two_songs_means_no_call():
    assert mood.mood_scores([make_track("t1")]) is None


@pytest.mark.parametrize("reply", [None, "not json", json.dumps({"pairs": []})])
def test_unusable_replies_give_none(monkeypatch, reply):
    monkeypatch.setattr(mood, "_ask_claude", lambda prompt: reply)
    assert mood.mood_scores([make_track("t1"), make_track("t2")]) is None


def test_errors_never_raise(monkeypatch):
    def boom(prompt):
        raise ConnectionError("offline")

    monkeypatch.setattr(mood, "_ask_claude", boom)
    assert mood.mood_scores([make_track("t1"), make_track("t2")]) is None


def test_same_songs_in_any_order_call_claude_once(monkeypatch):
    calls = fake_claude(monkeypatch, [{"from": "A", "to": "B", "mood": 0.8}])
    a, b = make_track("t1"), make_track("t2")
    first = mood.mood_scores([a, b])
    assert mood.mood_scores([b, a]) == first
    assert len(calls) == 1


def test_failures_are_not_cached(monkeypatch):
    monkeypatch.setattr(mood, "_ask_claude", lambda prompt: None)
    tracks = [make_track("t1"), make_track("t2")]
    assert mood.mood_scores(tracks) is None
    calls = fake_claude(monkeypatch, [{"from": "A", "to": "B", "mood": 0.8}])
    assert mood.mood_scores(tracks) == {("t1", "t2"): 0.8}
    assert len(calls) == 1


def test_prompt_names_songs_and_asks_for_every_ordered_pair():
    track = make_track("t1", filename="Daft Punk - One More Time.mp3", bpm=122.6, key="8A")
    track.sections = [Section(start_seconds=0, end_seconds=180, energy=1.0, label="chorus")]
    prompt = mood._prompt({"A": track, "B": make_track("t2"), "C": make_track("t3")})
    assert '"Daft Punk - One More Time"' in prompt
    assert "123 BPM" in prompt and "key 8A" in prompt and "chorus" in prompt
    for pair in ("A->B", "B->A", "A->C", "C->A", "B->C", "C->B"):
        assert pair in prompt


def test_mood_feeds_the_pair_score(monkeypatch):
    tracks = [make_track("t1"), make_track("t2")]
    fake_claude(monkeypatch, [{"from": "A", "to": "B", "mood": 1.0}, {"from": "B", "to": "A", "mood": 0.0}])
    scores = {(s.from_id, s.to_id): s for s in score_pairs(tracks, mood.mood_scores(tracks))}
    assert scores[("t1", "t2")].mood == 1.0
    assert scores[("t1", "t2")].total > scores[("t2", "t1")].total
