import numpy as np
import pytest
import soundfile as sf

from backend.api.schemas import Track
from backend.audio.analysis import analyze_track
from backend.logic.scoring import (
    NEUTRAL,
    WEIGHTS,
    camelot_steps,
    combine,
    energy_score,
    harmonic_score,
    parse_camelot,
    score_matrix,
    score_pair,
    score_pairs,
    tempo_gap,
    tempo_score,
)


def make_track(id, bpm=None, key=None, energy=None, hop=None):
    return Track(
        id=id,
        filename=f"{id}.mp3",
        duration_seconds=180.0,
        sample_rate=44100,
        bpm=bpm,
        key_camelot=key,
        energy_rms=energy or [],
        energy_hop_seconds=hop,
    )


def test_parse_camelot():
    assert parse_camelot("8A") == (8, "A")
    assert parse_camelot("12b") == (12, "B")
    assert parse_camelot("13A") is None
    assert parse_camelot("C major") is None
    assert parse_camelot("²A") is None  # unicode digit used to raise ValueError


def test_harmonic_rules():
    assert harmonic_score("8A", "8A") == 1.0
    assert harmonic_score("8A", "8B") == 0.9   # relative major/minor
    assert harmonic_score("8A", "9A") == 0.9   # neighbor
    assert harmonic_score("12A", "1A") == 0.9  # wraps around the wheel
    assert harmonic_score("8A", "2A") < harmonic_score("8A", "10A")


def test_harmonic_unknown_key_is_none_not_zero():
    assert harmonic_score(None, "8A") is None
    assert harmonic_score("garbage", "8A") is None


def test_tempo():
    assert tempo_score(128, 128) == 1.0
    assert tempo_score(128, 64) == 1.0  # half time counts as a match
    assert tempo_score(120, 128) < tempo_score(124, 128)
    assert tempo_score(90, 140) == 0.0
    assert tempo_score(None, 128) is None


def test_tempo_is_symmetric():
    assert tempo_score(120, 128) == pytest.approx(tempo_score(128, 120))
    assert tempo_score(64, 128) == pytest.approx(tempo_score(128, 64))


def test_tempo_zero_or_negative_bpm_is_none():
    assert tempo_score(0.0, 128) is None  # librosa's "no beat found"
    assert tempo_score(-120, 128) is None


def test_energy_compares_tail_to_head():
    loud_ending = make_track("a", energy=[0.1] * 10 + [0.8] * 10, hop=3.0)
    loud_start = make_track("b", energy=[0.8] * 10 + [0.1] * 10, hop=3.0)
    quiet_start = make_track("c", energy=[0.1] * 20, hop=3.0)
    assert energy_score(loud_ending, loud_start) == pytest.approx(1.0)
    assert energy_score(loud_ending, quiet_start) < 0.5


def test_energy_missing_data_is_none():
    assert energy_score(make_track("a"), make_track("b")) is None


def test_energy_rejects_nan_and_db_values():
    ok = make_track("ok", energy=[0.5] * 5, hop=1.0)
    all_nan = make_track("nan", energy=[float("nan")] * 5, hop=1.0)
    in_db = make_track("db", energy=[-20.0] * 5, hop=1.0)
    assert energy_score(all_nan, ok) is None
    assert energy_score(in_db, ok) is None
    # A few NaN frames among real ones are skipped, not fatal.
    some_nan = make_track("some", energy=[0.5, float("nan"), 0.5], hop=1.0)
    assert energy_score(some_nan, ok) == pytest.approx(1.0)


def test_combine_treats_missing_as_neutral():
    assert combine({"harmonic": None, "tempo": None, "energy": None, "mood": None}) == NEUTRAL
    # Only tempo known: missing weights count as NEUTRAL, not as perfect.
    expected = 0.35 * 1.0 + 0.65 * NEUTRAL
    assert combine({"harmonic": None, "tempo": 1.0, "energy": None, "mood": None}) == pytest.approx(expected)


def test_missing_data_is_not_a_bonus():
    # Regression: unknown key used to outscore a known, good key.
    unknown = combine({"harmonic": None, "tempo": 1.0, "energy": None, "mood": None})
    known_good = combine({"harmonic": 0.9, "tempo": 1.0, "energy": None, "mood": None})
    assert known_good > unknown

    # Same for mood: a pair Claude failed on shouldn't beat one it rated well.
    failed = combine({"harmonic": 0.9, "tempo": 1.0, "energy": 1.0, "mood": None})
    rated_well = combine({"harmonic": 0.9, "tempo": 1.0, "energy": 1.0, "mood": 0.9})
    assert rated_well > failed


@pytest.mark.parametrize("bad", [5.0, -2.0, float("nan"), float("inf")])
def test_out_of_range_inputs_keep_total_in_bounds(bad):
    total = combine({"harmonic": bad, "tempo": bad, "energy": bad, "mood": bad})
    assert 0.0 <= total <= 1.0


def test_out_of_range_mood_is_clamped_on_the_pair():
    a, b = make_track("a", bpm=128), make_track("b", bpm=128)
    assert score_pair(a, b, mood=7).mood == 1.0
    assert score_pair(a, b, mood=float("nan")).mood is None


def test_score_pair_works_without_claude():
    a = make_track("a", bpm=128, key="8A")
    b = make_track("b", bpm=128, key="9A")
    score = score_pair(a, b)
    assert score.mood is None
    assert 0.0 <= score.total <= 1.0


def test_matrix_covers_every_ordered_pair():
    tracks = [make_track(i, bpm=120) for i in "abc"]
    matrix = score_matrix(tracks, mood_scores={("a", "b"): 0.2})
    assert len(matrix) == 6
    assert ("a", "a") not in matrix
    assert matrix[("a", "b")].mood == 0.2
    assert matrix[("b", "a")].mood is None


def test_score_pairs_returns_one_score_per_ordered_pair():
    tracks = [make_track(i, bpm=120) for i in "abc"]
    scores = score_pairs(tracks, mood={("a", "b"): 0.9})
    assert len(scores) == 6
    assert {(s.from_id, s.to_id) for s in scores} == {(a, b) for a in "abc" for b in "abc" if a != b}
    assert next(s for s in scores if (s.from_id, s.to_id) == ("a", "b")).mood == 0.9
    assert score_pairs(tracks, None)[0].mood is None  # mood_scores() returns None on failure


def test_scores_real_analyze_track_output(tmp_path):
    # Aarav's analyze_track -> score_pairs, on synthetic audio.
    sr = 22050
    analyzed = []
    for id, bpm in (("a", 120), ("b", 124)):
        y = np.zeros(sr * 20, dtype=np.float32)
        for beat in np.arange(0, 20, 60 / bpm):
            i = int(beat * sr)
            y[i:i + 600] += np.random.default_rng(i).standard_normal(600).astype(np.float32) * 0.5
        path = tmp_path / f"{id}.wav"
        sf.write(path, y, sr)
        analyzed.append(analyze_track(path, make_track(id)))

    assert all(t.analysis_status == "ok" for t in analyzed)
    for s in score_pairs(analyzed):
        assert s.tempo is not None and s.energy is not None
        assert 0.0 <= s.total <= 1.0


def test_camelot_steps_shared_helper():
    assert camelot_steps("8A", "9A") == (1, True)
    assert camelot_steps("12A", "1B") == (1, False)
    assert camelot_steps("8A", None) is None
    assert camelot_steps("8A", "bogus") is None


def test_tempo_gap_shared_helper():
    assert tempo_gap(128, 128) == (0.0, False)
    assert tempo_gap(64, 128) == (0.0, True)
    assert tempo_gap(float("nan"), 128) is None
    assert tempo_score(float("nan"), 128) is None
    assert tempo_score(128, float("inf")) is None


def test_weights_sum_to_one_so_neutral_stays_neutral():
    # Regression guard: retuning the weights must keep an all-missing pair at
    # exactly NEUTRAL, or unscored pairs would tilt the ordering.
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)
    assert combine({}) == pytest.approx(NEUTRAL)
