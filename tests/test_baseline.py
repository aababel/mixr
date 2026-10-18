"""Plain-crossfade baseline used only by the before/after demo."""
import numpy as np
import pytest
import soundfile as sf

from backend.api.schemas import Track, TransitionRequest
from backend.audio.baseline import plain_crossfade, render_plain_crossfade
from backend.audio.loading import RENDER_SR
from backend.audio.loudness import integrated_loudness, track_gain_for_path
from backend.audio.preview import PREVIEW_CONTEXT_SECONDS, PREVIEW_TARGET_LUFS, render_preview

SR = RENDER_SR


def noise(dur, rms, seed):
    rng = np.random.default_rng(seed)
    return np.clip(rng.standard_normal((int(dur * SR), 2)) * rms, -0.95, 0.95).astype(np.float32)


def write(tmp_path, name, y, bpm=120):
    p = tmp_path / f"{name}.wav"
    sf.write(p, y, SR, subtype="FLOAT")
    dur = y.shape[0] / SR
    beats = [float(b) for b in np.arange(0.2, dur - 0.05, 60.0 / bpm)]
    return p, Track(id=name, filename=p.name, duration_seconds=dur, sample_rate=SR,
                    bpm=bpm, beats_seconds=beats, analysis_status="ok")


def test_plain_crossfade_is_end_to_start_equal_power():
    out = np.ones((SR * 20, 2), dtype=np.float32) * 0.5
    inc = np.ones((SR * 20, 2), dtype=np.float32) * 0.25
    audio, ap = plain_crossfade(out, inc, SR, 4.0, context_seconds=2.0)
    n, ctx = 4 * SR, 2 * SR
    assert len(audio) == ctx + n + ctx
    assert ap["actual_overlap_seconds"] == pytest.approx(4.0)
    assert ap["outgoing_start_seconds"] == pytest.approx(16.0)
    assert ap["incoming_start_seconds"] == 0.0
    assert ap["beat_aligned"] is False and ap["tempo_change"] is None and ap["applied_eq"] is None
    np.testing.assert_allclose(audio[:ctx], 0.5)
    np.testing.assert_allclose(audio[-ctx:], 0.25)
    mid = ctx + n // 2  # equal power: both at cos(pi/4)
    assert audio[mid, 0] == pytest.approx((0.5 + 0.25) * np.cos(np.pi / 4), abs=1e-3)


def test_plain_crossfade_gains_and_short_track():
    out = np.ones((SR * 2, 2), dtype=np.float32) * 0.1
    inc = np.ones((SR * 10, 2), dtype=np.float32) * 0.1
    audio, ap = plain_crossfade(out, inc, SR, 5.0, context_seconds=1.0,
                                outgoing_gain_db=6.0, incoming_gain_db=0.0)
    assert ap["actual_overlap_seconds"] == pytest.approx(2.0)
    assert any("shortened" in s for s in ap["safety_adjustments"])
    assert audio[0, 0] == pytest.approx(0.1 * 10 ** (6 / 20), rel=1e-3)
    assert audio[-1, 0] == pytest.approx(0.1, rel=1e-3)


def test_plain_crossfade_guards_clipping():
    out = np.ones((SR * 10, 2), dtype=np.float32) * 0.9
    audio, ap = plain_crossfade(out, out.copy(), SR, 4.0, context_seconds=1.0)
    assert np.max(np.abs(audio)) <= 0.98 + 1e-6
    assert any("clipping" in s for s in ap["safety_adjustments"])


def test_before_and_after_match_loudness_and_track_gains(tmp_path):
    pa, ta = write(tmp_path, "a", noise(40, 0.05, 1))
    pb, tb = write(tmp_path, "b", noise(40, 0.2, 2))
    before = render_plain_crossfade(pa, pb, 8.0, tmp_path / "before.wav")
    req = TransitionRequest(from_id="a", to_id="b", requested_overlap_seconds=8.0)
    after = render_preview(pa, pb, ta, tb, req, tmp_path / "after.wav")

    assert before["applied_gain_db"]["outgoing"] == track_gain_for_path(pa).gain_db
    assert before["applied_gain_db"]["incoming"] == track_gain_for_path(pb).gain_db
    assert before["applied_gain_db"]["outgoing"] == after.applied_gain_db["outgoing"]

    b, _ = sf.read(tmp_path / "before.wav", dtype="float32")
    a, _ = sf.read(tmp_path / "after.wav", dtype="float32")
    assert integrated_loudness(b, SR) == pytest.approx(PREVIEW_TARGET_LUFS, abs=0.2)
    assert integrated_loudness(a, SR) == pytest.approx(PREVIEW_TARGET_LUFS, abs=0.2)
    ctx = PREVIEW_CONTEXT_SECONDS
    assert before["output_duration_seconds"] == pytest.approx(2 * ctx + 8.0, abs=1e-3)


def test_baseline_does_not_touch_cached_audio(tmp_path):
    from backend.audio.loading import load_render_audio
    pa, _ = write(tmp_path, "a", noise(20, 0.5, 3))
    pb, _ = write(tmp_path, "b", noise(20, 0.5, 4))
    before = load_render_audio(pa).copy()
    render_plain_crossfade(pa, pb, 4.0, tmp_path / "x.wav")
    np.testing.assert_array_equal(load_render_audio(pa), before)
