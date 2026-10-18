import numpy as np
import pytest
import soundfile as sf

from backend.api.schemas import Track, TrackWindow, TransitionRequest
from backend.audio.analysis import ENERGY_HOP_SECONDS, analyze_track
from backend.audio.echo import MELODY_CLASH
from backend.audio.loading import RENDER_SR, load_render_audio
from backend.audio.mix import render_mix
from backend.audio.preview import render_preview
from backend.audio.transition import render_transition
from backend.audio.windows import check_windows

SR = RENDER_SR
Q = 1.5 / 32768


# ---------------------------------------------------------------- analysis

def drums_then_melody(sr=22050, bpm=120.0, intro_bars=16, melody_bars=16):
    """Kick + click on every beat; a stepwise lead melody only after the intro."""
    p = 60.0 / bpm
    dur = (intro_bars + melody_bars) * 4 * p
    n = int(dur * sr)
    y = np.zeros(n, dtype=np.float64)
    k = np.arange(int(0.12 * sr)) / sr
    kick = np.sin(2 * np.pi * 60 * k) * np.exp(-k * 25)
    t = np.arange(400) / sr
    click = np.sin(2 * np.pi * 2000 * t) * np.exp(-t * 80)
    for j, b in enumerate(np.arange(0.0, dur - 0.2, p)):
        i = int(b * sr)
        y[i:i + kick.size] += kick[:n - i] * (0.9 if j % 4 == 0 else 0.6)
        y[i:i + click.size] += click[:n - i] * 0.3
    notes = [440, 494, 523, 587, 659, 587, 523, 494]
    start = intro_bars * 4 * p
    for j, b in enumerate(np.arange(start, dur, p / 2)):
        i, m = int(b * sr), int(p / 2 * sr)
        seg = np.arange(min(m, n - i)) / sr
        y[i:i + seg.size] += 0.2 * np.sin(2 * np.pi * notes[j % 8] * seg)
    return y.astype(np.float32), start, dur


def test_melody_phrases_and_sections(tmp_path):
    y, melody_start, dur = drums_then_melody()
    p = tmp_path / "dm.wav"
    sf.write(p, y, 22050)
    tr = analyze_track(p, Track(id="dm", filename="dm.wav", duration_seconds=dur, sample_rate=22050))
    assert tr.analysis_status == "ok", tr.analysis_error

    mel = np.array(tr.melody_activity)
    assert len(mel) == len(tr.energy_rms)
    assert 0.0 <= mel.min() and mel.max() <= 1.0
    i = int(melody_start / ENERGY_HOP_SECONDS)
    assert mel[2:i - 2].mean() < 0.2  # drums only
    assert mel[i + 2:].mean() > 0.7  # lead melody

    phrases = tr.phrase_starts_seconds
    assert len(phrases) >= 2
    assert np.diff(phrases) == pytest.approx(8 * 4 * 0.5, abs=0.3)  # 8 bars at 120 BPM
    assert all(any(abs(ph - b) < 1e-3 for b in tr.beats_seconds) for ph in phrases)

    secs = tr.sections
    assert secs[0].start_seconds == 0.0
    assert secs[-1].end_seconds == pytest.approx(dur, abs=0.01)
    assert all(a.end_seconds == b.start_seconds for a, b in zip(secs, secs[1:]))
    assert all(0.0 <= s.energy <= 1.0 for s in secs)
    assert max(s.energy for s in secs) == 1.0
    # the melody entrance is a section boundary
    assert any(abs(s.start_seconds - melody_start) < 1.1 for s in secs[1:])


def test_no_beats_means_no_phrases(tmp_path):
    t = np.arange(int(6 * 22050)) / 22050
    y = (0.2 * np.sin(2 * np.pi * 440 * t)).astype(np.float32)
    p = tmp_path / "tone.wav"
    sf.write(p, y, 22050)
    tr = analyze_track(p, Track(id="t", filename="tone.wav", duration_seconds=6, sample_rate=22050))
    assert tr.analysis_status == "ok"
    if not tr.beats_seconds:
        assert tr.phrase_starts_seconds == [] and tr.sections == []
    assert tr.melody_activity


# ---------------------------------------------------------------- echo-out

def make(tmp_path, name, dur, bpm, seed, melody=1.0, first=0.2):
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    y = rng.uniform(-0.3, 0.3, (n, 2)).astype(np.float32)
    beats = [float(b) for b in np.arange(first, dur - 0.05, 60.0 / bpm)]
    p = tmp_path / f"{name}.wav"
    sf.write(p, y, SR, subtype="FLOAT")
    blocks = int(np.ceil(dur / ENERGY_HOP_SECONDS))
    tr = Track(id=name, filename=p.name, duration_seconds=dur, sample_rate=SR, bpm=bpm,
               beats_seconds=beats, analysis_status="ok", energy_hop_seconds=ENERGY_HOP_SECONDS,
               energy_rms=[0.2] * blocks, melody_activity=[melody] * blocks)
    return p, tr


def seam(out_p, out_t, in_p, in_t, overlap, mode="final"):
    return render_transition(
        load_render_audio(out_p), load_render_audio(in_p), SR,
        out_t.beats_seconds, in_t.beats_seconds,
        TransitionRequest(from_id=out_t.id, to_id=in_t.id, requested_overlap_seconds=overlap),
        mode, outgoing_melody=out_t.melody_activity, incoming_melody=in_t.melody_activity,
        melody_hop_seconds=ENERGY_HOP_SECONDS,
    )


def test_vocal_swap_when_both_sides_are_melodic(tmp_path):
    (pa, a), (pb, b) = make(tmp_path, "a", 20, 120, 1), make(tmp_path, "b", 20, 120, 2)
    audio, t = seam(pa, a, pb, b, 6.0)
    assert t.applied_eq.startswith("3-band swap") and "mids (vocals) swapped" in t.applied_eq
    assert any("vocals swap on a beat" in n for n in t.safety_adjustments)
    assert not t.overlap_changed_for_safety
    assert np.isfinite(audio).all() and np.abs(audio).max() <= 1.0


def test_echo_out_when_both_sides_are_melodic_and_tempos_dont_match(tmp_path):
    (pa, a), (pb, b) = make(tmp_path, "a", 20, 120, 1), make(tmp_path, "b", 20, 90, 2)
    audio, t = seam(pa, a, pb, b, 6.0)
    assert t.applied_eq.startswith("echo-out")
    assert t.overlap_changed_for_safety
    assert any("requested overlap not used" in n for n in t.safety_adjustments)
    # A's last beat is at 19.7 s, so its last bar is 17.7-19.7 s; B's first beat
    # (0.2 s in) lands on that bar line, so B's sample 0 plays 1.8 s into the seam
    # (the grid is fitted to noise here, so allow a few tens of ms)
    assert t.outgoing_start_seconds == pytest.approx(17.7, abs=0.25)
    offset = t.output_duration_seconds - t.actual_overlap_seconds
    assert offset == pytest.approx(1.8, abs=0.25)
    assert t.output_duration_seconds >= 2.0 + 4 * 0.5 - 0.1  # 4 echo repeats ring under B
    assert np.isfinite(audio).all() and np.abs(audio).max() <= 1.0


def test_echo_out_ab_previews_are_identical(tmp_path):
    (pa, a), (pb, b) = make(tmp_path, "a", 20, 120, 1), make(tmp_path, "b", 20, 90, 2)
    a2, ta = seam(pa, a, pb, b, 2.0, "preview")
    b2, tb = seam(pa, a, pb, b, 8.0, "preview")
    assert ta.actual_overlap_seconds == tb.actual_overlap_seconds
    np.testing.assert_array_equal(a2, b2)


def test_blend_when_one_side_is_instrumental(tmp_path):
    (pa, a), (pb, b) = make(tmp_path, "a", 20, 120, 1), make(tmp_path, "b", 20, 120, 2, melody=0.1)
    _, t = seam(pa, a, pb, b, 4.0)
    assert t.applied_eq.startswith("3-band swap")
    assert any(f"incoming 0.10 (echo-out when both >= {MELODY_CLASH})" in n
               for n in t.safety_adjustments)


def test_mix_with_echo_out_keeps_every_sample(tmp_path):
    files = [make(tmp_path, "a", 20, 120, 1), make(tmp_path, "b", 25, 90, 2), make(tmp_path, "c", 18, 120, 3)]
    paths, tracks = [p for p, _ in files], [t for _, t in files]
    out = tmp_path / "mix.wav"
    m = render_mix("m", tracks, paths, [4.0, 6.0], out, loudness_match=False)
    mix, _ = sf.read(out, dtype="float32")
    src = [load_render_audio(p) for p in paths]
    o_start = [round(t.outgoing_start_seconds * SR) for t in m.transitions]
    ov = [round(t.actual_overlap_seconds * SR) for t in m.transitions]
    seams = [round(t.output_duration_seconds * SR) for t in m.transitions]
    assert all(t.applied_eq.startswith("echo-out") for t in m.transitions)
    used = sum((s.shape[0] - o) + v for s, o, v in zip(src, o_start, ov))
    assert mix.shape[0] == sum(s.shape[0] for s in src) - used + sum(seams)
    np.testing.assert_allclose(mix[:o_start[0]], src[0][:o_start[0]], atol=Q)
    np.testing.assert_allclose(mix[-(src[2].shape[0] - ov[1]):], src[2][ov[1]:], atol=Q)
    assert any(s.startswith("echo-out instead of a blend") for s in m.limitations)


# ---------------------------------------------------------------- windows

def test_mix_plays_only_the_windows(tmp_path):
    files = [make(tmp_path, "a", 30, 120, 1, 0.1), make(tmp_path, "b", 30, 124, 2, 0.1),
             make(tmp_path, "c", 30, 128, 3, 0.1)]
    paths, tracks = [p for p, _ in files], [t for _, t in files]
    windows = [TrackWindow(track_id="a", start_seconds=4.2, end_seconds=24.2),
               TrackWindow(track_id="b", start_seconds=2.0, end_seconds=28.0),
               TrackWindow(track_id="c", start_seconds=0.0, end_seconds=20.0)]
    out = tmp_path / "mix.wav"
    m = render_mix("w", tracks, paths, [4.0, 4.0], out, loudness_match=False, windows=windows)
    mix, _ = sf.read(out, dtype="float32")
    src = [load_render_audio(p) for p in paths]

    assert m.mode == "set"
    assert [(w.start_seconds, w.end_seconds) for w in m.windows] == [(4.2, 24.2), (2.0, 28.0), (0.0, 20.0)]
    assert m.source_durations_seconds == pytest.approx([20.0, 26.0, 20.0])
    assert all(t.outgoing_start_seconds < d for t, d in zip(m.transitions, [20.0, 26.0]))

    s0 = int(4.2 * SR)
    fade = int(0.5 * SR)  # 1 beat at 120 BPM
    o0 = round(m.transitions[0].outgoing_start_seconds * SR)
    np.testing.assert_allclose(mix[fade:o0], src[0][s0 + fade:s0 + o0], atol=Q)
    assert np.abs(mix[:50]).max() < np.abs(src[0][s0:s0 + 50]).max()  # faded in
    # last window ends at 20 s of a 30 s song: faded out to silence
    assert np.abs(mix[-20:]).max() < 0.01
    assert any("faded in" in s for s in m.limitations) and any("faded out" in s for s in m.limitations)
    assert m.output_duration_seconds == pytest.approx(mix.shape[0] / SR)


def test_full_mode_without_windows_is_unchanged(tmp_path):
    files = [make(tmp_path, "a", 20, 120, 1, 0.1), make(tmp_path, "b", 20, 120, 2, 0.1)]
    paths, tracks = [p for p, _ in files], [t for _, t in files]
    m = render_mix("f", tracks, paths, [4.0], tmp_path / "m.wav", loudness_match=False)
    assert m.mode == "full" and m.windows == []
    assert m.source_durations_seconds == pytest.approx([20.0, 20.0])


def test_preview_uses_the_windows(tmp_path):
    (pa, a), (pb, b) = make(tmp_path, "a", 30, 120, 1, 0.1), make(tmp_path, "b", 30, 120, 2, 0.1)
    wa = TrackWindow(track_id="a", start_seconds=0.0, end_seconds=22.2)
    wb = TrackWindow(track_id="b", start_seconds=3.0, end_seconds=30.0)
    req = TransitionRequest(from_id="a", to_id="b", requested_overlap_seconds=4.0)
    out = tmp_path / "p.wav"
    t = render_preview(pa, pb, a, b, req, out, loudness_match=False,
                       outgoing_window=wa, incoming_window=wb)

    A, B = load_render_audio(pa)[:int(22.2 * SR)], load_render_audio(pb)[int(3.0 * SR):]
    expected, te = render_transition(
        A, B, SR, [x for x in a.beats_seconds if x < 22.2], [x - 3.0 for x in b.beats_seconds if x >= 3.0],
        req, "preview")
    got, _ = sf.read(out, dtype="float32")
    assert t.outgoing_start_seconds == pytest.approx(te.outgoing_start_seconds)
    assert t.actual_overlap_seconds == pytest.approx(te.actual_overlap_seconds)
    np.testing.assert_allclose(got, np.clip(expected, -1, 1), atol=Q)


def test_bad_windows_are_rejected():
    tracks = [Track(id="a", filename="a", duration_seconds=10, sample_rate=SR),
              Track(id="b", filename="b", duration_seconds=10, sample_rate=SR)]
    ok = TrackWindow(track_id="a", start_seconds=0, end_seconds=10)
    with pytest.raises(ValueError):
        check_windows(tracks, [ok])
    with pytest.raises(ValueError):
        check_windows(tracks, [ok, TrackWindow(track_id="x", start_seconds=0, end_seconds=10)])
    with pytest.raises(ValueError):
        check_windows(tracks, [ok, TrackWindow(track_id="b", start_seconds=5, end_seconds=6)])


# ---------------------------------------------------------------- section labels

def pop_song(sr=22050, bpm=120.0):
    """intro (drums) / verse (soft melody) / chorus (loud melody + chords) x2 /
    break (pad) / chorus / outro. Returns audio and [(name, start, end)]."""
    plan = (("intro", 16), ("verse", 16), ("chorus", 16), ("verse", 16), ("chorus", 16),
            ("break", 8), ("chorus", 16), ("outro", 8))
    p = 60.0 / bpm
    k = np.arange(int(0.12 * sr)) / sr
    kick = np.sin(2 * np.pi * 60 * k) * np.exp(-k * 25)
    tc = np.arange(400) / sr
    click = np.sin(2 * np.pi * 2000 * tc) * np.exp(-tc * 80)
    level = {"intro": 0.5, "verse": 0.6, "chorus": 1.0, "break": 0.0, "outro": 0.4}
    segs, marks, t0 = [], [], 0.0
    for name, bars in plan:
        dur = bars * 4 * p
        n = int(round(dur * sr))
        y, t = np.zeros(n), np.arange(n) / sr
        for j, b in enumerate(np.arange(0, dur - 1e-6, p)):
            i = int(b * sr)
            y[i:i + kick.size] += kick[:n - i] * level[name] * (1.0 if j % 4 == 0 else 0.7)
            y[i:i + click.size] += click[:n - i] * 0.3 * level[name]
        if name == "verse":
            for j, b in enumerate(np.arange(0, dur, p)):
                i, m = int(b * sr), int(p * sr)
                seg = np.arange(min(m, n - i)) / sr
                y[i:i + seg.size] += 0.12 * np.sin(2 * np.pi * [330, 392, 440, 392][j % 4] * seg)
        if name == "chorus":
            notes = [523, 587, 659, 784, 659, 587, 523, 494]
            for j, b in enumerate(np.arange(0, dur, p / 2)):
                i, m = int(b * sr), int(p / 2 * sr)
                seg = np.arange(min(m, n - i)) / sr
                y[i:i + seg.size] += 0.22 * np.sin(2 * np.pi * notes[j % 8] * seg)
            y += 0.08 * sum(np.sin(2 * np.pi * f * t) for f in (262, 330, 392))
        if name == "break":
            y += 0.05 * (np.sin(2 * np.pi * 220 * t) + np.sin(2 * np.pi * 277 * t))
        segs.append(y)
        marks.append((name, t0, t0 + dur))
        t0 += dur
    return np.concatenate(segs).astype(np.float32), marks


def test_sections_are_fine_grained_and_label_choruses(tmp_path):
    y, marks = pop_song()
    p = tmp_path / "pop.wav"
    sf.write(p, y, 22050)
    dur = len(y) / 22050
    tr = analyze_track(p, Track(id="pop", filename="pop.wav", duration_seconds=dur, sample_rate=22050))
    secs = tr.sections
    phrase = 8 * 4 * 0.5
    # contiguous, whole track, and no section longer than 16 bars (plus the lead-in before the first phrase)
    assert secs[0].start_seconds == 0.0 and secs[-1].end_seconds == pytest.approx(dur, abs=0.01)
    assert all(a.end_seconds == b.start_seconds for a, b in zip(secs, secs[1:]))
    lead_in = tr.phrase_starts_seconds[0]
    assert all(s.end_seconds - s.start_seconds <= 2 * phrase + 1.0 + (lead_in if i == 0 else 0)
               for i, s in enumerate(secs))

    def label_at(t):
        return next(s.label for s in secs if s.start_seconds <= t < s.end_seconds)

    choruses = [(a, b) for n, a, b in marks if n == "chorus"]
    assert sum(label_at((a + b) / 2) == "chorus" for a, b in choruses) >= 2
    assert label_at(8.0) == "intro"
    assert label_at(168.0) == "break"
    verses = [(a, b) for n, a, b in marks if n == "verse"]
    assert all(label_at((a + b) / 2) not in ("chorus", "drop") for a, b in verses)


def test_flat_song_gets_no_chorus_or_drop(tmp_path):
    y, _, _ = drums_then_melody(intro_bars=0, melody_bars=64)
    p = tmp_path / "flat.wav"
    sf.write(p, y, 22050)
    tr = analyze_track(p, Track(id="f", filename="flat.wav", duration_seconds=len(y) / 22050,
                                sample_rate=22050))
    assert len(tr.sections) >= 3  # 64 bars can't be one section any more
    assert not any(s.label in ("chorus", "drop") for s in tr.sections)
