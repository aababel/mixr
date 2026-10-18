import numpy as np
import pytest

from backend.api.schemas import Section, Track
from backend.audio.windows import check_windows
from backend.logic.windows import (
    SET_MIN_WINDOW_SECONDS,
    SET_TOLERANCE,
    TIME_GRID_SECONDS,
    best_section,
    choose_windows,
)

HOP = 0.5
BPM = 120.0  # 0.5 s beats, 16 s phrases


def make_track(id, duration=192.0, *, beats=True, phrases=True, melody=None, energy=None,
               sections=None):
    """Synthetic analyzed track. melody / energy: list of (start, end, value)
    spans over a default of 0.5 (melody) and 0.8 (energy)."""
    n = int(duration / HOP)
    beat_list = list(np.arange(0.0, duration, 60.0 / BPM)) if beats else []
    phrase_list = list(np.arange(0.0, duration, 16.0)) if beats and phrases else []

    def curve(spans, default):
        if spans is None:
            return []
        arr = np.full(n, default)
        for t0, t1, v in spans:
            arr[int(t0 / HOP):int(t1 / HOP)] = v
        return arr.tolist()

    return Track(
        id=id,
        filename=f"{id}.mp3",
        duration_seconds=duration,
        sample_rate=44100,
        bpm=BPM if beats else None,
        beats_seconds=beat_list,
        energy_rms=curve(energy, 0.8) if energy is not None else ([0.8] * n),
        energy_hop_seconds=HOP,
        phrase_starts_seconds=phrase_list,
        sections=sections or [],
        melody_activity=curve(melody, 0.5),
        analysis_status="ok",
    )


def lengths(windows):
    return [w.end_seconds - w.start_seconds for w in windows]


def played(windows, overlaps):
    return sum(lengths(windows)) - sum(overlaps)


def assert_on_grid(track, windows_or_window):
    w = windows_or_window
    grid = {0.0, track.duration_seconds, *[round(p, 3) for p in track.phrase_starts_seconds]}
    assert round(w.start_seconds, 3) in grid
    assert round(w.end_seconds, 3) in grid


def test_empty_and_single_track():
    assert choose_windows([], []) == []
    t = make_track("a")
    [w] = choose_windows([t], [])
    assert (w.start_seconds, w.end_seconds) == (0.0, t.duration_seconds)


def test_overlap_count_must_match():
    with pytest.raises(ValueError):
        choose_windows([make_track("a"), make_track("b")], [])


def test_full_mode_without_cue_data_keeps_whole_songs():
    tracks = [make_track(i, beats=False, melody=None) for i in "abc"]
    for t in tracks:
        t.energy_rms = []
    windows = choose_windows(tracks, [8.0, 8.0])
    assert [(w.start_seconds, w.end_seconds) for w in windows] == [(0.0, 192.0)] * 3
    assert all(w.reason for w in windows)
    check_windows(tracks, windows)


def test_full_mode_leaves_on_a_quiet_phrase_when_the_next_song_sings_at_once():
    # A sings until the very end, except the phrase ending at 160 where its
    # vocal line stops. B sings from its first second.
    a = make_track("a", melody=[(0, 192, 0.9), (148, 160, 0.1)])
    b = make_track("b", melody=[(0, 192, 0.9)])
    wa, wb = choose_windows([a, b], [8.0])
    assert (wa.start_seconds, wa.end_seconds) == (0.0, 160.0)
    assert (wb.start_seconds, wb.end_seconds) == (0.0, 192.0)
    assert "melody" in wa.reason and "2:40" in wa.reason


def test_full_mode_enters_on_the_instrumental_phrase_before_the_vocal():
    # B sings from its first second, then has an instrumental phrase from 16
    # to 32: entering there puts A's last line over B's instrumental and B's
    # vocal right after the blend.
    a = make_track("a", melody=[(0, 192, 0.9), (148, 160, 0.1)])
    b = make_track("b", melody=[(0, 192, 0.9), (16, 32, 0.1)])
    wa, wb = choose_windows([a, b], [8.0])
    assert (wb.start_seconds, wb.end_seconds) == (16.0, 192.0)
    assert wa.end_seconds == 192.0, "no 20 s melody gap from leaving on A's quiet phrase"
    assert "0:16" in wb.reason and "melody" in wb.reason


def test_full_mode_skips_a_long_instrumental_intro_after_a_vocal():
    # A sings into its ending; B is loud but has no melody for its first
    # 48 s. Coming in at 0 leaves ~40 s without melody: enter one phrase
    # before B's melody instead.
    a = make_track("a", melody=[(0, 192, 0.8)])
    b = make_track("b", melody=[(0, 48, 0.1), (48, 192, 0.8)])
    wa, wb = choose_windows([a, b], [8.0])
    assert wb.start_seconds == 32.0
    assert "after the blend" in wb.reason
    assert "without melody" not in wa.reason


def test_full_mode_prefers_the_latest_good_end_and_earliest_good_start():
    # Quiet melody at several phrase ends: keep as much of each song as possible.
    a = make_track("a", melody=[(0, 192, 0.9), (100, 112, 0.1), (164, 176, 0.1)])
    b = make_track("b", melody=[(0, 192, 0.9), (16, 32, 0.1), (48, 64, 0.1)])
    wa, wb = choose_windows([a, b], [8.0])
    assert wa.end_seconds == 176.0
    assert wb.start_seconds == 16.0


def test_full_mode_trims_little_when_songs_already_hand_over():
    # A's melody fades out over its last phrase, B opens instrumental.
    a = make_track("a", melody=[(0, 176, 0.8), (176, 192, 0.1)])
    b = make_track("b", melody=[(0, 16, 0.1), (16, 192, 0.8)])
    wa, wb = choose_windows([a, b], [8.0])
    assert (wa.start_seconds, wa.end_seconds) == (0.0, 192.0)
    assert (wb.start_seconds, wb.end_seconds) == (0.0, 192.0)


def test_full_mode_judges_a_natural_end_before_its_fade_out():
    # A fades to silence over its last 6 s with the melody still busy right
    # before; the renderer blends before the fade, so that's where it counts.
    a = make_track("a", melody=[(0, 192, 0.9), (148, 160, 0.1), (186, 192, 0.0)],
                   energy=[(186, 192, 0.01)])
    b = make_track("b", melody=[(0, 192, 0.1)])
    wa, _ = choose_windows([a, b], [8.0])
    assert wa.end_seconds == 160.0


def test_windows_fall_back_to_bars_then_time():
    a = make_track("a", phrases=False, melody=[(0, 192, 0.9), (150, 158, 0.1)])
    b = make_track("b", phrases=False, melody=[(0, 192, 0.9), (10, 30, 0.1)])
    wa, wb = choose_windows([a, b], [8.0])
    bars = {round(t, 3) for t in a.beats_seconds[::4]} | {0.0, 192.0}
    assert wa.end_seconds in bars and wb.start_seconds in bars
    assert wa.end_seconds < 192.0 and wb.start_seconds > 0.0
    assert "bar start" in wa.reason

    c = make_track("c", beats=False, melody=[(0, 192, 0.9), (150, 158, 0.1)])
    d = make_track("d", beats=False, melody=[(0, 192, 0.9), (10, 30, 0.1)])
    wc, wd = choose_windows([c, d], [8.0])
    for t in (wc.end_seconds, wd.start_seconds):
        assert t in (0.0, 192.0) or t % TIME_GRID_SECONDS == 0
    assert wc.end_seconds < 192.0 and wd.start_seconds > 0.0


def test_poor_seam_says_it_is_the_least_bad_cue():
    a = make_track("a", melody=[(0, 192, 0.95)])
    b = make_track("b", melody=[(0, 192, 0.95)])
    wa, wb = choose_windows([a, b], [8.0])
    assert "least-bad" in wa.reason and "echo out" in wa.reason
    check_windows([a, b], [wa, wb])


@pytest.mark.parametrize("n,target", [(2, 180.0), (3, 300.0), (5, 450.0), (5, 600.0)])
def test_set_mode_hits_the_target(n, target):
    tracks = [make_track(f"t{i}", duration=240.0, energy=[(96 + 16 * (i % 2), 144 + 16 * (i % 2), 1.0)])
              for i in range(n)]
    overlaps = [8.0] * (n - 1)
    windows = choose_windows(tracks, overlaps, target)
    check_windows(tracks, windows)
    assert abs(played(windows, overlaps) - target) <= SET_TOLERANCE * target
    for t, w in zip(tracks, windows):
        assert w.end_seconds - w.start_seconds >= SET_MIN_WINDOW_SECONDS
        assert 0.0 <= w.start_seconds < w.end_seconds <= t.duration_seconds
        assert_on_grid(t, w)
        b0, b1 = best_section(t)
        assert w.start_seconds <= b0 and w.end_seconds >= b1, "keeps the loudest part"
        assert "loudest part" in w.reason


def test_set_mode_target_too_small_keeps_45_second_windows():
    tracks = [make_track(f"t{i}") for i in range(5)]
    overlaps = [8.0] * 4
    windows = choose_windows(tracks, overlaps, 120.0)
    assert all(length >= SET_MIN_WINDOW_SECONDS for length in lengths(windows))
    assert played(windows, overlaps) > 120.0 * (1 + SET_TOLERANCE)
    assert "instead of the 2:00 target" in windows[0].reason
    assert "at least 45 s" in windows[0].reason


def test_set_mode_target_too_large_plays_whole_songs():
    tracks = [make_track(f"t{i}", duration=150.0) for i in range(2)]
    windows = choose_windows(tracks, [8.0], 1200.0)
    assert [(w.start_seconds, w.end_seconds) for w in windows] == [(0.0, 150.0)] * 2
    assert "too short to fill it" in windows[0].reason


def test_short_tracks_play_whole_and_seams_never_collide():
    # 20 s middle song with 12 s overlaps on both sides: it can't fit two
    # full seams, so it plays whole and says so.
    tracks = [make_track("a"), make_track("b", duration=20.0), make_track("c")]
    windows = choose_windows(tracks, [12.0, 12.0])
    assert (windows[1].start_seconds, windows[1].end_seconds) == (0.0, 20.0)
    assert "Too short to trim" in windows[1].reason
    check_windows(tracks, windows)

    # Long enough songs: every window is at least twice each adjoining overlap.
    tracks = [make_track(i, duration=64.0, melody=[(0, 64, 0.9), (20, 30, 0.1)]) for i in "abcde"]
    overlaps = [12.0, 4.0, 12.0, 8.0]
    for target in (None, 200.0):
        windows = choose_windows(tracks, overlaps, target)
        check_windows(tracks, windows)
        for i, length in enumerate(lengths(windows)):
            head = overlaps[i - 1] if i else 0.0
            tail = overlaps[i] if i < len(overlaps) else 0.0
            assert length >= 2 * max(head, tail)


def test_set_mode_works_without_phrases_or_beats():
    tracks = [make_track("a", beats=False), make_track("b", phrases=False), make_track("c", beats=False)]
    windows = choose_windows(tracks, [8.0, 8.0], 240.0)
    check_windows(tracks, windows)
    assert abs(played(windows, [8.0, 8.0]) - 240.0) <= SET_TOLERANCE * 240.0


def test_set_mode_enters_in_a_break_so_the_next_section_lands_as_a_leaves():
    sections = [Section(start_seconds=0, end_seconds=96, energy=0.9),
                Section(start_seconds=96, end_seconds=128, energy=0.4, label="break"),
                Section(start_seconds=128, end_seconds=240, energy=1.0)]
    a = make_track("a", duration=240.0, melody=[(0, 240, 0.9), (196, 208, 0.1)])
    b = make_track("b", duration=240.0, sections=sections,
                   melody=[(0, 240, 0.9), (96, 128, 0.2)], energy=[(96, 128, 0.4), (128, 176, 1.0)])
    wa, wb = choose_windows([a, b], [8.0], 200.0)
    assert wb.start_seconds == 112.0, "last phrase of the break: B's melody kicks in after the blend"
    assert "after the blend" in wb.reason


def test_full_mode_skips_an_outro_running_into_a_sparse_intro():
    # A's last 32 s and B's first 32 s are thin (quiet, no melody). Blending
    # them is a handoff with nothing playing: trim into the music instead.
    a = make_track("a", melody=[(0, 192, 0.8), (160, 192, 0.1)], energy=[(160, 192, 0.25)])
    b = make_track("b", melody=[(0, 32, 0.1), (32, 192, 0.8)], energy=[(0, 32, 0.25)])
    wa, wb = choose_windows([a, b], [8.0])
    assert wa.end_seconds <= 160.0 or wb.start_seconds >= 32.0
    assert "sparse" not in wa.reason


def test_best_section_prefers_labeled_chorus_then_loudest_stretch():
    t = make_track("a", energy=[(64, 96, 1.0)])
    assert best_section(t) == (64.0, 96.0)
    t.sections = [Section(start_seconds=0, end_seconds=32, energy=0.5),
                  Section(start_seconds=32, end_seconds=192, energy=1.0, label="chorus")]
    assert best_section(t) == (32.0, 192.0)
    bare = make_track("b")
    bare.energy_rms = []
    assert best_section(bare) is None
