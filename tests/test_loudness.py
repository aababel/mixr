import numpy as np
import pytest
import soundfile as sf

from backend.api.schemas import Track, TransitionRequest
from backend.audio.loading import RENDER_SR, load_render_audio
from backend.audio.loudness import (
    DEFAULT_LOUDNESS,
    LoudnessSettings,
    _k_weighting,
    compute_track_gain,
    integrated_loudness,
    track_gain_for_path,
)
from backend.audio.mix import render_mix
from backend.audio.preview import PREVIEW_CONTEXT_SECONDS, render_preview
from backend.audio.transition import render_transition

SR = RENDER_SR
Q = 1.5 / 32768  # 16-bit output quantization tolerance
TARGET = DEFAULT_LOUDNESS.target_lufs


def noise(dur, rms, seed, channels=2):
    rng = np.random.default_rng(seed)
    y = rng.standard_normal((int(dur * SR), channels)) * rms
    return np.clip(y, -0.95, 0.95).astype(np.float32)


def write(tmp_path, name, y, bpm=120):
    p = tmp_path / f"{name}.wav"
    sf.write(p, y, SR, subtype="FLOAT")
    dur = y.shape[0] / SR
    beats = [float(b) for b in np.arange(0.2, dur - 0.05, 60.0 / bpm)]
    tr = Track(id=name, filename=p.name, duration_seconds=dur, sample_rate=SR,
               bpm=bpm, beats_seconds=beats, analysis_status="ok")
    return p, tr


# ---------------------------------------------------------------- measurement


def test_k_weighting_matches_bs1770_at_48k():
    (b1, a1), (b2, a2) = _k_weighting(48000)
    np.testing.assert_allclose(b1, [1.53512485958697, -2.69169618940638, 1.19839281085285], atol=1e-10)
    np.testing.assert_allclose(a1, [1.0, -1.69065929318241, 0.73248077421585], atol=1e-10)
    np.testing.assert_allclose(b2, [1.0, -2.0, 1.0])
    np.testing.assert_allclose(a2, [1.0, -1.99004745483398, 0.99007225036621], atol=1e-10)


@pytest.mark.parametrize("sr", [44100, 48000])
def test_full_scale_997hz_sine_in_one_channel_is_minus_3_01(sr):
    t = np.arange(sr * 10) / sr
    y = np.zeros((t.size, 2))
    y[:, 0] = np.sin(2 * np.pi * 997 * t)
    assert integrated_loudness(y, sr) == pytest.approx(-3.01, abs=0.02)


def test_loudness_scales_with_gain_and_ignores_silence():
    y = noise(10, 0.05, 1)
    base = integrated_loudness(y, SR)
    assert integrated_loudness(y * 0.5, SR) == pytest.approx(base - 6.02, abs=0.01)
    padded = np.concatenate([np.zeros((SR * 5, 2), np.float32), y, np.zeros((SR * 5, 2), np.float32)])
    assert integrated_loudness(padded, SR) == pytest.approx(base, abs=0.2)  # silence gated out


def test_silent_or_tiny_audio_is_unmeasurable():
    assert integrated_loudness(np.zeros((SR * 3, 2)), SR) is None
    assert integrated_loudness(noise(0.2, 0.1, 2), SR) is None
    g = compute_track_gain(np.zeros((SR * 3, 2), np.float32), SR)
    assert g.gain_db == 0.0 and g.measured_lufs is None and g.note


# ---------------------------------------------------------------- gain


def test_gain_reaches_target_when_safe():
    for rms in (0.05, 0.2):
        y = noise(10, rms, 3)
        g = compute_track_gain(y, SR)
        assert not g.peak_limited and g.note is None
        assert integrated_loudness(y * 10 ** (g.gain_db / 20), SR) == pytest.approx(TARGET, abs=0.01)


def test_boost_is_capped_by_peak_headroom():
    y = noise(10, 0.01, 4)
    y[SR] = 0.9  # one hot sample leaves almost no headroom
    g = compute_track_gain(y, SR)
    assert g.peak_limited and g.note and "peak headroom" in g.note
    assert 0.9 * 10 ** (g.gain_db / 20) <= DEFAULT_LOUDNESS.peak_ceiling + 1e-6


def test_gain_range_limit():
    y = noise(10, 0.001, 5)  # very quiet: wants far more than +12 dB
    g = compute_track_gain(y, SR, LoudnessSettings(peak_ceiling=1e9))
    assert g.gain_db == pytest.approx(12.0) and "range" in g.note


# ---------------------------------------------------------------- renderer integration


def test_render_transition_records_and_applies_gain():
    a, b = noise(20, 0.1, 6), noise(20, 0.1, 7)
    req = TransitionRequest(from_id="a", to_id="b", requested_overlap_seconds=4.0, beat_align=False)
    plain, ap0 = render_transition(a, b, SR, [], [], req, "final")
    gained, ap = render_transition(a, b, SR, [], [], req, "final",
                                   outgoing_gain_db=-6.0, incoming_gain_db=3.0)
    assert ap0.applied_gain_db == {"outgoing": 0.0, "incoming": 0.0}
    assert ap.applied_gain_db == {"outgoing": -6.0, "incoming": 3.0}
    n = gained.shape[0]
    fo, fi = np.cos((np.arange(n) + 0.5) / n * np.pi / 2), np.sin((np.arange(n) + 0.5) / n * np.pi / 2)
    expected = a[-n:] * (fo * 10 ** (-6 / 20))[:, None] + b[:n] * (fi * 10 ** (3 / 20))[:, None]
    np.testing.assert_allclose(gained, expected, atol=1e-5)


@pytest.fixture
def uneven(tmp_path):
    """A loud track and a quiet track, both able to reach the target."""
    return [write(tmp_path, "loud", noise(20, 0.25, 8)), write(tmp_path, "quiet", noise(20, 0.03, 9))]


def test_mix_levels_tracks_and_records_gain(tmp_path, uneven):
    (pl, tl), (pq, tq) = uneven
    m = render_mix("lm", [tl, tq], [pl, pq], [4.0], tmp_path / "mix.wav")
    mix, _ = sf.read(tmp_path / "mix.wav", dtype="float32")

    gl, gq = track_gain_for_path(pl), track_gain_for_path(pq)
    assert gl.gain_db < -5 and gq.gain_db > 5
    assert m.transitions[0].applied_gain_db == {"outgoing": gl.gain_db, "incoming": gq.gain_db}
    assert not any("no loudness normalization" in s for s in m.limitations)

    # Bodies of both tracks land at the target (seam excluded).
    o_start = round(m.transitions[0].outgoing_start_seconds * SR)
    ov = round(m.transitions[0].actual_overlap_seconds * SR)
    assert integrated_loudness(mix[:o_start], SR) == pytest.approx(TARGET, abs=0.1)
    assert integrated_loudness(mix[o_start + ov:], SR) == pytest.approx(TARGET, abs=0.1)

    # Bodies are exactly source * gain: content preserved, only level changed.
    src = load_render_audio(pl)
    np.testing.assert_allclose(mix[:o_start], src[:o_start] * 10 ** (gl.gain_db / 20), atol=Q + 1e-6)


def test_loudness_off_keeps_old_behaviour(tmp_path, uneven):
    (pl, tl), (pq, tq) = uneven
    m = render_mix("lm", [tl, tq], [pl, pq], [4.0], tmp_path / "mix.wav", loudness_match=False)
    assert m.transitions[0].applied_gain_db == {"outgoing": 0.0, "incoming": 0.0}
    assert any("no loudness normalization" in s for s in m.limitations)


def test_preview_applies_same_gain_as_mix(tmp_path, uneven):
    (pl, tl), (pq, tq) = uneven
    req = TransitionRequest(from_id="loud", to_id="quiet", requested_overlap_seconds=5.0)
    applied = render_preview(pl, pq, tl, tq, req, tmp_path / "prev.wav")
    prev, _ = sf.read(tmp_path / "prev.wav", dtype="float32")

    m = render_mix("lm", [tl, tq], [pl, pq], [5.0], tmp_path / "mix.wav")
    mix, _ = sf.read(tmp_path / "mix.wav", dtype="float32")
    assert applied.applied_gain_db == m.transitions[0].applied_gain_db
    assert applied.applied_gain_db["outgoing"] != 0.0

    ctx = round(PREVIEW_CONTEXT_SECONDS * SR)
    start = round(applied.outgoing_start_seconds * SR) - ctx
    np.testing.assert_allclose(prev, mix[start:start + len(prev)], atol=Q)


def test_peak_limited_track_is_reported(tmp_path):
    y = noise(20, 0.01, 10)
    y[SR] = 0.9
    pa, ta = write(tmp_path, "hot", y)
    pb, tb = write(tmp_path, "ok", noise(20, 0.1, 11))
    m = render_mix("pk", [ta, tb], [pa, pb], [4.0], tmp_path / "mix.wav")
    assert any("hot" in s and "peak headroom" in s for s in m.transitions[0].safety_adjustments)
    assert any("not reached for hot" in s for s in m.limitations)

    req = TransitionRequest(from_id="hot", to_id="ok", requested_overlap_seconds=4.0)
    ap = render_preview(pa, pb, ta, tb, req, tmp_path / "p.wav")
    assert ap.safety_adjustments == m.transitions[0].safety_adjustments


def test_single_track_mix_gets_gain(tmp_path, uneven):
    (pq, tq) = uneven[1]
    m = render_mix("one", [tq], [pq], [], tmp_path / "one.wav")
    mix, _ = sf.read(tmp_path / "one.wav", dtype="float32")
    assert m.transitions == []
    assert integrated_loudness(mix, SR) == pytest.approx(TARGET, abs=0.1)
