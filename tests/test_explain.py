import pytest

from backend.api.schemas import PairScore, Track
from backend.logic.explain import explain_order, explain_transition
from backend.logic.ordering import order_tracks
from backend.logic.scoring import harmonic_score, score_pairs, tempo_score


def make_track(id, bpm=None, key=None, energy=None):
    return Track(
        id=id,
        filename=f"{id}.mp3",
        duration_seconds=180.0,
        sample_rate=44100,
        bpm=bpm,
        key_camelot=key,
        energy_rms=energy or [],
        energy_hop_seconds=0.5 if energy else None,
    )


def score(a, b, total=0.8, energy=None, mood=None):
    return PairScore(from_id=a, to_id=b, total=total, energy=energy, mood=mood)


def test_transition_names_each_measured_reason():
    a = make_track("Song A", bpm=128, key="8A")
    b = make_track("Song B", bpm=126, key="9A")
    line = explain_transition(a, b, score("Song A", "Song B", total=0.91, energy=0.9, mood=0.8))
    assert line.startswith("Song A → Song B: ")
    assert "neighboring keys (8A → 9A)" in line
    assert "matching tempo (128 vs 126 BPM)" in line
    assert "loudness carries over" in line
    assert "Claude rated the moods a good match" in line
    assert line.endswith("(score 91%)")


def test_key_relationships():
    def key_text(k1, k2):
        return explain_transition(make_track("a", key=k1), make_track("b", key=k2), score("a", "b"))

    assert "same key (8A)" in key_text("8A", "8A")
    assert "relative major/minor keys (8A → 8B)" in key_text("8A", "8B")
    assert "neighboring keys (12A → 1A)" in key_text("12A", "1A")
    assert "closely related keys (8A → 9B)" in key_text("8A", "9B")
    assert "keys clash (8A → 2B)" in key_text("8A", "2B")


def test_tempo_wording():
    def tempo_text(b1, b2):
        return explain_transition(make_track("a", bpm=b1), make_track("b", bpm=b2), score("a", "b"))

    # Neutral wording: 64 -> 128 is double time and 128 -> 64 is half time.
    assert "half/double-time match (64 ↔ 128 BPM)" in tempo_text(64, 128)
    assert "half/double-time match (128 ↔ 64 BPM)" in tempo_text(128, 64)
    assert "tempo differs by 5% (126 vs 120 BPM)" in tempo_text(126, 120)
    assert "big tempo jump (120 vs 90 BPM)" in tempo_text(120, 90)


@pytest.mark.parametrize("bad", [float("nan"), float("inf"), 0.0, -5.0])
def test_unusable_bpm_is_left_out(bad):
    # Regression: NaN BPM used to print "tempo differs by nan%".
    line = explain_transition(make_track("a", bpm=bad), make_track("b", bpm=128), score("a", "b"))
    assert "BPM" not in line and "nan" not in line


def test_wording_agrees_with_the_score():
    # Regression: explain.py had its own copy of the key/tempo math, so the
    # two could drift. Now the text must line up with the sub-scores.
    keys = [f"{n}{l}" for n in range(1, 13) for l in "AB"]
    for k1 in keys:
        for k2 in keys:
            line = explain_transition(make_track("a", key=k1), make_track("b", key=k2), score("a", "b"))
            h = harmonic_score(k1, k2)
            assert ("keys clash" in line) == (h < 0.5), (k1, k2, h, line)
    for b1 in range(60, 200, 7):
        for b2 in range(60, 200, 5):
            line = explain_transition(make_track("a", bpm=b1), make_track("b", bpm=b2), score("a", "b"))
            t = tempo_score(b1, b2)
            if "matching tempo" in line or "half/double-time match" in line:
                assert t >= 0.75, (b1, b2, t)
            if "big tempo jump" in line:
                assert t == 0.0, (b1, b2, t)


def test_order_with_unknown_track_id_does_not_crash():
    # Regression: an id with a score but no Track used to raise KeyError.
    scores = [score("a", "ghost"), score("ghost", "a")]
    assert explain_order([make_track("a")], scores, ["a", "ghost"], ["a", "ghost"]) == [
        "Not enough scored tracks to explain an order."
    ]


def test_unmeasured_things_are_never_claimed():
    # No key, no BPM, no energy, no Claude: no musical claims at all.
    line = explain_transition(make_track("a"), make_track("b"), score("a", "b", total=0.5))
    assert "not enough analysis data" in line
    for claim in ("key", "tempo", "BPM", "loudness", "Claude"):
        assert claim not in line


def test_mood_line_only_when_claude_answered():
    a, b = make_track("a", bpm=128), make_track("b", bpm=128)
    assert "Claude" not in explain_transition(a, b, score("a", "b", mood=None))
    assert "Claude flagged a mood shift" in explain_transition(a, b, score("a", "b", mood=0.1))
    assert "Claude" not in explain_transition(a, b, score("a", "b", mood=0.5))  # no strong opinion


def test_energy_jump_is_called_out():
    a, b = make_track("a"), make_track("b")
    assert "noticeable loudness jump" in explain_transition(a, b, score("a", "b", energy=0.2))


def test_summary_when_reordered():
    tracks = [make_track(i) for i in "abc"]
    scores = [score("a", "b", 0.2), score("b", "c", 0.4), score("b", "a", 0.9), score("a", "c", 0.7)]
    lines = explain_order(tracks, scores, ["a", "b", "c"], ["b", "a", "c"])
    assert lines[0] == "Reordered for smoother transitions: average score 30% → 80%."
    assert lines[1].startswith("b → a:")
    assert lines[2].startswith("a → c:")
    assert len(lines) == 3


def test_summary_when_order_kept():
    tracks = [make_track(i) for i in "ab"]
    lines = explain_order(tracks, [score("a", "b", 0.75)], ["a", "b"], ["a", "b"])
    assert lines[0] == "Kept your order: it already had the smoothest transitions (average score 75%)."


def test_too_few_tracks_or_missing_scores():
    assert explain_order([make_track("a")], [], ["a"], ["a"]) == ["Not enough scored tracks to explain an order."]
    tracks = [make_track(i) for i in "ab"]
    assert explain_order(tracks, [], ["a", "b"], ["a", "b"]) == ["Not enough scored tracks to explain an order."]


def test_filename_extension_is_dropped():
    t = Track(id="x1", filename="Daft Punk - One More Time.mp3", duration_seconds=1, sample_rate=1)
    u = Track(id="x2", filename="Track 2.wav", duration_seconds=1, sample_rate=1)
    assert explain_transition(t, u, score("x1", "x2")).startswith("Daft Punk - One More Time → Track 2:")


def test_end_to_end_with_real_scoring_and_ordering():
    tracks = [
        make_track("opener", bpm=128, key="8A", energy=[0.3] * 60 + [0.7] * 60),
        make_track("peak", bpm=126, key="9A", energy=[0.7] * 120),
        make_track("clash", bpm=90, key="2B", energy=[0.1] * 120),
    ]
    scores = score_pairs(tracks)
    original = ["clash", "opener", "peak"]
    ordered = order_tracks([tracks[2], tracks[0], tracks[1]], scores)
    lines = explain_order(tracks, scores, original, ordered)
    assert lines[0].startswith("Reordered") or lines[0].startswith("Kept")
    assert len(lines) == len(ordered)  # summary + one line per transition
    assert "opener → peak: neighboring keys (8A → 9A); matching tempo" in "\n".join(lines)
