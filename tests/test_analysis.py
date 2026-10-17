import numpy as np
import pytest
import soundfile as sf

from backend.api.schemas import Track
from backend.audio.analysis import ENERGY_HOP_SECONDS, analyze_track

SR = 22050


def make_track(tmp_path, y, name="t.wav"):
    p = tmp_path / name
    sf.write(p, y, SR)
    return p, Track(id="t1", filename=name, duration_seconds=len(y) / SR, sample_rate=SR)


def click_signal(dur=20.0, bpm=120.0, first=0.3):
    n = int(dur * SR)
    y = np.zeros(n, dtype=np.float32)
    t = np.arange(400) / SR
    click = (np.sin(2 * np.pi * 1000 * t) * np.exp(-t * 60)).astype(np.float32) * 0.8
    for b in np.arange(first, dur - 0.05, 60.0 / bpm):
        i = int(b * SR)
        y[i:i + 400] += click[: n - i]
    return y


def chord_signal(freqs, dur=12.0, amp=0.15):
    t = np.arange(int(dur * SR)) / SR
    y = sum(np.sin(2 * np.pi * f * t) for f in freqs) * amp
    return y.astype(np.float32)


def test_bpm_and_beats_on_click_track(tmp_path):
    p, tr = make_track(tmp_path, click_signal(bpm=120))
    out = analyze_track(p, tr)
    assert out.analysis_status == "ok"
    assert out.bpm == pytest.approx(120, rel=0.01)
    gaps = np.diff(out.beats_seconds)
    assert np.median(gaps) == pytest.approx(0.5, abs=0.02)
    assert all(0 <= b <= tr.duration_seconds for b in out.beats_seconds)


def test_energy_profile(tmp_path):
    loud = chord_signal([220.0], dur=10, amp=0.5)
    quiet = chord_signal([220.0], dur=10, amp=0.05)
    p, tr = make_track(tmp_path, np.concatenate([quiet, loud]))
    out = analyze_track(p, tr)
    assert out.energy_hop_seconds == ENERGY_HOP_SECONDS
    assert len(out.energy_rms) == pytest.approx(20 / ENERGY_HOP_SECONDS, abs=1)
    assert np.mean(out.energy_rms[-10:]) > 5 * np.mean(out.energy_rms[:10])
    assert out.metadata["outro_energy"] > out.metadata["intro_energy"]


def test_c_major_chord_gives_8b(tmp_path):
    # C major triad over two octaves
    p, tr = make_track(tmp_path, chord_signal([130.81, 164.81, 196.0, 261.63, 329.63, 392.0]))
    out = analyze_track(p, tr)
    assert out.key_camelot == "8B"
    assert out.metadata["key_name"] == "C major"


def test_a_minor_chord_gives_8a(tmp_path):
    p, tr = make_track(tmp_path, chord_signal([110.0, 130.81, 164.81, 220.0, 261.63, 329.63]))
    out = analyze_track(p, tr)
    assert out.key_camelot == "8A"


def test_silence_is_ok_without_beats_or_key(tmp_path):
    p, tr = make_track(tmp_path, np.zeros(SR * 5, dtype=np.float32))
    out = analyze_track(p, tr)
    assert out.analysis_status == "ok"
    assert out.bpm is None and out.beats_seconds == []
    assert out.key_camelot is None
    assert out.metadata["analysis_notes"]


def test_bad_file_sets_error_without_raising(tmp_path):
    p = tmp_path / "junk.mp3"
    p.write_bytes(b"not audio at all" * 100)
    tr = Track(id="x", filename="junk.mp3", duration_seconds=1.0, sample_rate=44100)
    out = analyze_track(p, tr)
    assert out.analysis_status == "error"
    assert out.analysis_error


def test_input_track_not_mutated_and_ids_kept(tmp_path):
    p, tr = make_track(tmp_path, click_signal(dur=8))
    out = analyze_track(p, tr)
    assert tr.analysis_status == "pending" and tr.bpm is None
    assert (out.id, out.filename, out.sample_rate) == (tr.id, tr.filename, tr.sample_rate)


def test_fold_doubles_half_time():
    from backend.audio.analysis import fold_tempo
    beats = [0.0, 0.9375, 1.875, 2.8125]  # 64 BPM
    bpm, out, note = fold_tempo(64.0, beats)
    assert bpm == 128.0
    assert out == [0.0, 0.4688, 0.9375, 1.4062, 1.875, 2.3438, 2.8125]
    assert "folded" in note


def test_fold_halves_double_time_and_leaves_normal_alone():
    from backend.audio.analysis import fold_tempo
    bpm, out, _ = fold_tempo(240.0, [0.0, 0.25, 0.5, 0.75, 1.0])
    assert bpm == 120.0 and out == [0.0, 0.5, 1.0]
    assert fold_tempo(128.0, [0.0, 0.46875]) == (128.0, [0.0, 0.4688], None)


def test_slow_click_track_folded_into_range(tmp_path):
    p, tr = make_track(tmp_path, click_signal(dur=30, bpm=64))
    out = analyze_track(p, tr)
    assert 90 <= out.bpm < 180
    assert out.metadata["bpm_detected"] < 90 or out.bpm == pytest.approx(128, rel=0.01)
