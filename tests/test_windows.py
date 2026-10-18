import numpy as np
import pytest

from backend.api.schemas import Section, Track
from backend.audio.analysis import block_rms
from backend.audio.echo import melody_clash
from backend.audio.edges import MAX_EDGE_SECONDS, fade_in_seconds, fade_out_seconds
from backend.audio.windows import check_windows
from backend.logic.windows import (
    FADE_IN_KEY,
    FADE_OUT_KEY,
    SET_MIN_WINDOW_SECONDS,
    SET_TOLERANCE,
    TIME_GRID_SECONDS,
    WEIGHTS,
    BLEND_SIZES,
    CLASH_BLEND_SECONDS,
    MAX_BLEND_SECONDS,
    _cues,
    _length_text,
    _seam_choice,
    best_section,
    choose_windows,
    find_drops,
    plan_windows,
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
    if w.drop_seconds is not None:  # a drop entry starts its build on a beat
        assert round(w.drop_seconds, 3) in grid
        grid |= {round(b, 3) for b in track.beats_seconds}
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
    # Quiet melody at several phrase ends: keep as much of each song as
    # possible. A's instrumental end already hands over, and B's quiet
    # phrases would only stretch the gap without melody to 20 s.
    a = make_track("a", melody=[(0, 192, 0.9), (100, 112, 0.1), (164, 176, 0.1)])
    b = make_track("b", melody=[(0, 192, 0.9), (16, 32, 0.1), (48, 64, 0.1)])
    wa, wb = choose_windows([a, b], [8.0])
    assert wa.end_seconds == 176.0
    assert wb.start_seconds == 0.0


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
    a = make_track("a", phrases=False, melody=[(0, 192, 0.9), (172, 182, 0.1)])
    b = make_track("b", phrases=False, melody=[(0, 192, 0.9), (10, 30, 0.1)])
    wa, wb = choose_windows([a, b], [8.0])
    bars = {round(t, 3) for t in a.beats_seconds[::4]} | {0.0, 192.0}
    assert wa.end_seconds in bars and wb.start_seconds in bars
    assert wa.end_seconds < 192.0 and wb.start_seconds > 0.0
    assert "bar start" in wa.reason

    c = make_track("c", beats=False, melody=[(0, 192, 0.9), (172, 182, 0.1)])
    d = make_track("d", beats=False, melody=[(0, 192, 0.9), (10, 30, 0.1)])
    wc, wd = choose_windows([c, d], [8.0])
    for t in (wc.end_seconds, wd.start_seconds):
        assert t in (0.0, 192.0) or t % TIME_GRID_SECONDS == 0
    assert wc.end_seconds < 192.0 and wd.start_seconds > 0.0


def test_poor_seam_says_it_is_the_least_bad_cue():
    a = make_track("a", melody=[(0, 192, 0.95)])
    b = make_track("b", melody=[(0, 192, 0.95)])
    wa, wb = choose_windows([a, b], [8.0])
    assert "least-bad" in wa.reason and "vocals swap on a beat" in wa.reason
    check_windows([a, b], [wa, wb])
    b.bpm = 90.0  # too far from 120 to beat-match
    wa, _ = choose_windows([a, b], [8.0])
    assert "least-bad" in wa.reason and "echoes it out" in wa.reason


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
    # B's break is no quieter than the rest (no drop after it), only its
    # melody dips: enter in the break's last phrase.
    sections = [Section(start_seconds=0, end_seconds=96, energy=0.9),
                Section(start_seconds=96, end_seconds=128, energy=0.8, label="break"),
                Section(start_seconds=128, end_seconds=240, energy=0.9)]
    a = make_track("a", duration=240.0, melody=[(0, 240, 0.9), (196, 208, 0.1)])
    b = make_track("b", duration=240.0, sections=sections, melody=[(0, 240, 0.9), (96, 128, 0.2)],
                   energy=[(128, 176, 0.9)])
    wa, wb = choose_windows([a, b], [8.0], 200.0)
    assert wb.drop_seconds is None
    assert wb.start_seconds == 112.0, "last phrase of the break: B's melody kicks in after the blend"
    assert "after the blend" in wb.reason


def test_set_mode_blends_when_a_clean_handoff_exists_despite_a_drop():
    sections = [Section(start_seconds=0, end_seconds=96, energy=0.9),
                Section(start_seconds=96, end_seconds=128, energy=0.4, label="break"),
                Section(start_seconds=128, end_seconds=240, energy=1.0)]
    a = make_track("a", duration=240.0, melody=[(0, 240, 0.9), (196, 208, 0.1)])
    b = make_track("b", duration=240.0, sections=sections,
                   melody=[(0, 240, 0.9), (96, 128, 0.2)], energy=[(96, 128, 0.4), (128, 176, 1.0)])
    wa, wb = choose_windows([a, b], [8.0], 200.0)
    assert find_drops(b) == [(128.0, 1.0)]
    assert wb.drop_seconds is None, "a long blend keeps the set flowing; the drop is a fallback"
    check_windows([a, b], [wa, wb])


def test_set_mode_lands_the_drop_when_every_blend_clashes():
    a = make_track("a", duration=240.0, melody=[(0, 240, 0.95)])
    b = make_track("b", duration=240.0, melody=[(0, 240, 0.95)],
                   energy=[(0, 240, 0.9), (96, 128, 0.4), (128, 176, 1.0)])
    wa, wb = choose_windows([a, b], [8.0], 200.0)
    assert wb.drop_seconds == 128.0
    assert wb.start_seconds == 120.0, "an 8 s build: 4 bars at 120 BPM"
    assert "drop at 2:08" in wb.reason and "filtered up over b's build" in wa.reason
    check_windows([a, b], [wa, wb])


def test_full_mode_skips_an_outro_running_into_a_sparse_intro():
    # A's last 32 s and B's first 32 s are thin (quiet, no melody). Blending
    # them is a handoff with nothing playing: trim into the music, or build
    # into B's drop where its music starts.
    a = make_track("a", melody=[(0, 192, 0.8), (160, 192, 0.1)], energy=[(160, 192, 0.25)])
    b = make_track("b", melody=[(0, 32, 0.1), (32, 192, 0.8)], energy=[(0, 32, 0.25)])
    wa, wb = choose_windows([a, b], [8.0])
    assert wa.end_seconds <= 160.0 or wb.start_seconds >= 32.0 or wb.drop_seconds == 32.0
    assert "sparse" not in wa.reason


def test_find_drops_and_blends_without_them():
    t = make_track("t", energy=[(0, 192, 0.5), (64, 192, 1.0)])
    assert find_drops(t) == [(64.0, 1.0)]
    assert find_drops(make_track("flat")) == []
    # Flat energy: no drop anywhere, so every seam is a blend.
    windows = choose_windows([make_track(i) for i in "abc"], [8.0, 8.0], 200.0)
    assert all(w.drop_seconds is None for w in windows)


def test_best_section_prefers_labeled_chorus_then_loudest_stretch():
    t = make_track("a", energy=[(64, 96, 1.0)])
    assert best_section(t) == (64.0, 96.0)
    t.sections = [Section(start_seconds=0, end_seconds=32, energy=0.5),
                  Section(start_seconds=32, end_seconds=192, energy=1.0, label="chorus")]
    assert best_section(t) == (32.0, 192.0)
    bare = make_track("b")
    bare.energy_rms = []
    assert best_section(bare) is None


def _renderer_clash(a, wa, b, wb):
    """What transition.py will decide, computed independently: cut melody the
    way audio/windows.play does and ask echo.melody_clash (flat energy here,
    so there are no fade edges)."""
    def cut(t, w):
        return t.melody_activity[int(w.start_seconds / HOP):int(np.ceil(w.end_seconds / HOP))]
    clash, _, _ = melody_clash(cut(a, wa), cut(b, wb), HOP, wa.end_seconds - wa.start_seconds)
    return clash


def _flags_clash(reason):
    """The reason names the renderer's clash fix: a vocal swap when the tempos
    blend, echo-out when they don't."""
    return "echo" in reason or "vocals swap" in reason


def test_echo_out_reason_uses_the_renderers_8_second_check():
    # Regression: with a 12 s overlap the old code averaged A's melody over the
    # last 12 s (0.47, under the 0.5 clash level) while echo.py checks only the
    # last 8 s (0.6), so the renderer echoed out but the reason didn't say so.
    # Songs shorter than 2x the overlap play whole, so the seam is fixed.
    a = make_track("a", duration=20.0, melody=[(0, 12, 0.2), (12, 20, 0.6)])
    b = make_track("b", duration=20.0, melody=[(0, 20, 0.6)])
    wa, wb = choose_windows([a, b], [12.0])
    assert (wa.start_seconds, wa.end_seconds, wb.start_seconds) == (0.0, 20.0, 0.0)
    assert _renderer_clash(a, wa, b, wb)
    assert _flags_clash(wa.reason)


@pytest.mark.parametrize("overlap", [4.0, 8.0, 12.0, 16.0])
@pytest.mark.parametrize("a_melody,b_melody", [
    ([(0, 192, 0.9), (148, 160, 0.1)], [(0, 192, 0.9), (16, 32, 0.1)]),  # clean handoff exists
    ([(0, 192, 0.95)], [(0, 192, 0.95)]),                                # always clashes
    ([(0, 184, 0.3), (184, 192, 0.7)], [(0, 192, 0.55)]),                # near the threshold
])
def test_echo_out_reason_always_matches_the_renderer(overlap, a_melody, b_melody):
    a = make_track("a", melody=a_melody)
    b = make_track("b", melody=b_melody)
    wa, wb = choose_windows([a, b], [overlap])
    assert _flags_clash(wa.reason) == _renderer_clash(a, wa, b, wb), wa.reason


def test_seam_weights_sum_to_one():
    # quality is a weighted sum compared against POOR_SEAM and NEUTRAL, so
    # retuning the weights must keep them summing to 1.0.
    assert sum(WEIGHTS.values()) == pytest.approx(1.0)


def test_best_section_ignores_nan_energy():
    # Regression: one NaN frame made np.argmax return the first NaN-tainted
    # window, so the "best part" landed near the song's start.
    t = make_track("t", duration=192.0, energy=[(0, 192, 0.3), (112, 144, 1.0)])
    t.energy_rms[20] = float("nan")
    start, end = best_section(t)
    assert start >= 96.0 and end <= 160.0


def test_stored_fade_lengths_replace_the_energy_estimate():
    # A song with a real 6 s fade-out. The renderer's edges.py measures it from
    # the audio; the energy_rms estimate reads it about 40% short. When
    # analysis stores edges.py's numbers, windows.py must use exactly those.
    sr = 22050
    t = np.arange(int(90 * sr)) / sr
    y = (0.3 * np.sin(2 * np.pi * 220 * t) * np.clip((90 - t) / 6, 0, 1) ** 2
         * np.clip(t / 5, 0, 1) ** 2).astype(np.float32)
    track = Track(id="f", filename="f.wav", duration_seconds=90.0, sample_rate=sr,
                  energy_rms=block_rms(y, sr, HOP), energy_hop_seconds=HOP)
    estimated = _cues(track)
    track.metadata = {FADE_OUT_KEY: fade_out_seconds(y, sr), FADE_IN_KEY: fade_in_seconds(y, sr)}
    measured = _cues(track)
    assert measured.tail == pytest.approx(fade_out_seconds(y, sr))
    assert measured.head == pytest.approx(fade_in_seconds(y, sr))
    assert measured.tail > estimated.tail  # the estimate it replaces reads short


@pytest.mark.parametrize("bad", [None, "6", float("nan"), -1.0, True])
def test_unusable_stored_fade_falls_back_to_the_estimate(bad):
    track = make_track("a", energy=[(186, 192, 0.01)])
    estimated = _cues(track).tail
    track.metadata = {FADE_OUT_KEY: bad}
    assert _cues(track).tail == estimated


def test_stored_fade_is_capped_like_the_renderer():
    track = make_track("a")
    track.metadata = {FADE_OUT_KEY: 30.0}
    assert _cues(track).tail == MAX_EDGE_SECONDS


def test_stored_fade_moves_where_the_seam_is_judged():
    # A's melody is quiet for the 8 s before a 10 s fade and busy inside it.
    # With the stored 10 s fade the seam is judged before the fade (quiet):
    # no echo-out. Judged at the energy estimate's shorter fade, it would see
    # the busy melody inside the fade.
    a = make_track("a", duration=20.0, melody=[(0, 2, 0.9), (2, 10, 0.1), (10, 20, 0.9)])
    b = make_track("b", duration=20.0, melody=[(0, 20, 0.9)])
    a.metadata = {FADE_OUT_KEY: 10.0}
    wa, _ = choose_windows([a, b], [12.0])
    assert not _flags_clash(wa.reason)
    a.metadata = {FADE_OUT_KEY: 0.0}
    wa, _ = choose_windows([a, b], [12.0])
    assert _flags_clash(wa.reason)


def _clean_pair():
    """A's melody stops 42 s before its end, B's starts 40 s in: room for a long blend."""
    return (make_track("ca", melody=[(0, 150, 0.8), (150, 192, 0.05)]),
            make_track("cb", melody=[(0, 40, 0.05), (40, 192, 0.8)]))


def _clashing_pair():
    """Both sing right up to the seam: any long blend stacks the two vocals."""
    return (make_track("xa", melody=[(0, 186, 0.9), (186, 192, 0.1)]),
            make_track("xb", melody=[(0, 6, 0.1), (6, 192, 0.9)]))


def test_each_seam_gets_a_blend_length_for_its_own_songs():
    ca, cb = _clean_pair()
    xa, xb = _clashing_pair()
    windows, overlaps = plan_windows([ca, cb, xa, xb])
    # Clean handoff: B comes in at 0:16 and the blend lands as its vocals start at 0:40.
    assert overlaps[0] == 24.0
    assert windows[1].start_seconds == 16.0
    assert "Blends into cb over 24 s (48 beats) and lands as its vocals come in at 0:40." in windows[0].reason
    # cb and xa, xa and xb sing over each other: short, xb landing on its vocal entry.
    assert overlaps[1] <= CLASH_BLEND_SECONDS and overlaps[2] <= CLASH_BLEND_SECONDS
    assert "both songs carry melody at the seam, so it stays short" in windows[1].reason
    assert "lands as its vocals come in at 0:06" in windows[2].reason
    check_windows([ca, cb, xa, xb], windows)


def test_blend_lands_on_the_incoming_chorus():
    sec = lambda t0, t1, label: Section(start_seconds=t0, end_seconds=t1, label=label, energy=0.5)
    a = make_track("a", melody=[(0, 150, 0.8), (150, 192, 0.05)])
    b = make_track("b", melody=[(0, 192, 0.05)],
                   sections=[sec(0, 32, "intro"), sec(32, 96, "chorus"), sec(96, 192, "verse")])
    windows, overlaps = plan_windows([a, b])
    assert windows[1].start_seconds + overlaps[0] == 32.0
    assert "lands on its chorus at 0:32" in windows[0].reason


def test_outgoing_vocal_line_limits_how_early_the_blend_starts():
    # Leaving at A's own end with B in at 0:16: B's vocals at 0:40 are a
    # 24 s landing, unless A starts a new vocal line 16 s before its end.
    b = make_track("b", melody=[(0, 40, 0.05), (40, 192, 0.8)])
    cb = _cues(b)
    j = list(cb.grid).index(16.0)
    picks = []
    for melody in ([(0, 192, 0.05)], [(0, 176, 0.05), (176, 192, 0.6)]):
        ca = _cues(make_track("a", melody=melody))
        choice = _seam_choice(ca, cb, BLEND_SIZES["auto"], 0.0, {})
        i = len(ca.grid) - 1
        picks.append((choice.at(i, j), _length_text(ca, cb, choice.at(i, j), choice, i, j, False, 32.0)))
    (free, free_text), (held, held_text) = picks
    assert free.request(j) == 24.0 and free.land_what[j] == "vocal"
    assert held.request(j) <= 16.5
    assert "a later landing would start the blend before its last vocal line at 2:56" in held_text


def test_preference_and_setting_lean_blend_lengths_without_overriding_the_songs():
    ca, cb = _clean_pair()
    xa, xb = _clashing_pair()
    for setting in BLEND_SIZES:
        clean = [plan_windows([ca, cb], p, transition_length=setting)[1][0] for p in (-1.0, 0.0, 1.0)]
        assert clean == sorted(clean)
        for p in (0.0, 1.0):  # even a long preference keeps a vocal clash short
            assert plan_windows([xa, xb], p, transition_length=setting)[1][0] <= CLASH_BLEND_SECONDS
    extended = [plan_windows([ca, cb], p, transition_length="extended")[1][0] for p in (-1.0, 1.0)]
    assert extended[0] < extended[1]
    by_setting = [plan_windows([ca, cb], transition_length=s)[1][0] for s in ("auto", "longer")]
    assert by_setting == [24.0, 40.0]  # Longer lands on the same vocal entry from B's start


def test_unmatched_tempos_echo_out_when_both_sides_have_a_beat():
    a, b = make_track("a"), _at_bpm(make_track("b"), 90.0)
    for setting, bars in (("auto", 2), ("extended", 4)):
        windows, overlaps = plan_windows([a, b], transition_length=setting)
        assert overlaps == [bars * 4 * 0.5]  # A's bars at 120 BPM
        assert f"echoes out over its last {bars} bars" in windows[0].reason
        assert "comes in on its downbeat under the echo" in windows[0].reason


def test_unmatched_tempos_blend_over_a_beatless_intro():
    a, b = make_track("a"), _at_bpm(make_track("b"), 90.0)
    b.metadata = {"pulse": [0.5] * 40 + [3.0] * 152, "pulse_hop_seconds": 1.0}  # no kick for 40 s
    windows, overlaps = plan_windows([a, b])
    assert overlaps == [MAX_BLEND_SECONDS] and windows[1].start_seconds == 0.0
    assert "b opens without a beat" in windows[0].reason
    assert "lands ahead of its beat, which comes in at 0:40" in windows[0].reason
    windows, overlaps = plan_windows([a, b], transition_length="extended")
    assert overlaps == [40.0]
    assert "lands as its beat comes in at 0:40" in windows[0].reason


def test_blend_fits_inside_short_windows():
    a, b = make_track("a", duration=40.0), make_track("b", duration=40.0)
    windows, overlaps = plan_windows([a, b], 1.0)
    shortest = min(lengths(windows))
    assert overlaps[0] <= shortest / 2
    assert "fits these windows" in windows[0].reason


def test_set_mode_counts_the_chosen_blends_toward_the_target():
    ca, cb = _clean_pair()
    xa, xb = _clashing_pair()
    tracks = [ca, cb, xa, xb]
    windows, overlaps = plan_windows(tracks, 0.0, 300.0)
    assert abs(played(windows, overlaps) - 300.0) <= SET_TOLERANCE * 300.0


def _at_bpm(track, bpm):
    track.bpm = bpm
    track.beats_seconds = list(np.arange(0.0, track.duration_seconds, 60.0 / bpm))
    return track


def test_extended_reaches_past_32_s_only_where_the_songs_allow():
    ca = _at_bpm(make_track("ca", 240.0, melody=[(0, 170, 0.8), (170, 240, 0.05)]), 132.0)
    cb = _at_bpm(make_track("cb", 240.0, melody=[(0, 70, 0.05), (70, 240, 0.8)]), 132.0)
    xa = _at_bpm(make_track("xa", 240.0, melody=[(0, 234, 0.9), (234, 240, 0.1)]), 132.0)
    xb = _at_bpm(make_track("xb", 240.0, melody=[(0, 6, 0.1), (6, 240, 0.9)]), 132.0)

    auto = plan_windows([ca, cb])[1][0]
    windows, overlaps = plan_windows([ca, cb], transition_length="extended")
    assert auto <= MAX_BLEND_SECONDS < overlaps[0]
    assert "Extended blend into cb over 53 s (116 beats) and lands as its vocals come in" in windows[0].reason
    check_windows([ca, cb], windows)

    # A vocal clash still keeps the blend well short of the cap.
    assert plan_windows([xa, xb], transition_length="extended")[1][0] < MAX_BLEND_SECONDS


def test_extended_blends_shrink_to_fit_their_windows():
    a = _at_bpm(make_track("a", 80.0, melody=[(0, 30, 0.8), (30, 80, 0.05)]), 132.0)
    b = _at_bpm(make_track("b", 80.0, melody=[(0, 50, 0.05), (50, 80, 0.8)]), 132.0)
    windows, overlaps = plan_windows([a, b], transition_length="extended")
    assert overlaps[0] <= min(lengths(windows)) / 2
    assert "fits these windows" in windows[0].reason


def test_extended_set_mode_still_hits_the_target():
    ca, cb = _clean_pair()
    xa, xb = _clashing_pair()
    tracks = [_at_bpm(t, 132.0) for t in (ca, cb, xa, xb)]
    windows, overlaps = plan_windows(tracks, 0.0, 300.0, transition_length="extended")
    assert abs(played(windows, overlaps) - 300.0) <= SET_TOLERANCE * 300.0
    assert all(o <= min(lengths(windows[i:i + 2])) / 2 for i, o in enumerate(overlaps))


def test_votes_shorten_and_lengthen_blends_inside_every_setting():
    # Regression: the votes only reached the pick through LENGTH_WEIGHT's
    # small length bonus, so in most seams a short or long vote changed
    # nothing, and inside "longer" / "extended" a short vote never could.
    ca, cb = _clean_pair()
    longer = [plan_windows([ca, cb], p, transition_length="longer")[1][0] for p in (-1.0, 0.0)]
    extended = [plan_windows([ca, cb], p, transition_length="extended")[1][0] for p in (-1.0, 0.0, 1.0)]
    assert longer[0] < longer[1]
    assert extended[0] < extended[1] < extended[2]


def test_vote_lean_only_changes_blend_length_never_the_cuts(monkeypatch):
    # The lean picks among one cell's lengths; the cell keeps its unleaned
    # score, so cut points and drops are exactly what they'd be without it.
    import backend.logic.windows as windows

    def cuts(a, b, p, setting):
        ws, _ = plan_windows([a, b], p, transition_length=setting)
        return [(w.start_seconds, w.end_seconds, w.drop_seconds) for w in ws]

    pairs = [_clean_pair(), _clashing_pair()]
    for a, b in pairs:
        for setting in BLEND_SIZES:
            for p in (-1.0, 0.0, 1.0):
                leaned = cuts(a, b, p, setting)
                monkeypatch.setattr(windows, "LENGTH_LEAN", 0.0)
                assert cuts(a, b, p, setting) == leaned
                monkeypatch.undo()


def test_vote_lean_never_lengthens_a_vocal_clash():
    xa, xb = _clashing_pair()
    for setting in BLEND_SIZES:
        assert plan_windows([xa, xb], 1.0, transition_length=setting)[1][0] <= CLASH_BLEND_SECONDS
