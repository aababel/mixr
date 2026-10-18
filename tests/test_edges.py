"""Fade-out tails / fade-in heads and how the renderer blends around them."""
import numpy as np
import pytest

from backend.api.schemas import TransitionRequest
from backend.audio.edges import fade_in_seconds, fade_out_seconds
from backend.audio.stretch import rubberband_available
from backend.audio.transition import render_transition, seam_sample_bounds

SR = 22050
needs_rb = pytest.mark.skipif(not rubberband_available(), reason="rubberband CLI not installed")


def beat_track(dur, bpm, seed=0, fade_in=0.0, fade_out=0.0):
    """Steady 'music' (noise bed + kicks on every beat) with optional linear fades in dB."""
    n = int(dur * SR)
    rng = np.random.default_rng(seed)
    a = (rng.standard_normal((n, 2)) * 0.05).astype(np.float32)
    beats = np.arange(0.1, dur - 0.05, 60.0 / bpm)
    for b in beats:
        i = int(b * SR)
        a[i:i + 400] += np.hanning(800)[400:, None].astype(np.float32) * 0.6
    t = np.arange(n) / SR
    gain_db = np.zeros(n)
    if fade_in:
        gain_db = np.minimum(gain_db, np.where(t < fade_in, -60 * (1 - t / fade_in), 0))
    if fade_out:
        left = dur - t
        gain_db = np.minimum(gain_db, np.where(left < fade_out, -60 * (1 - left / fade_out), 0))
    a *= (10 ** (gain_db / 20)).astype(np.float32)[:, None]
    return a, [float(b) for b in beats]


def req(ov):
    return TransitionRequest(from_id="a", to_id="b", requested_overlap_seconds=ov)


@pytest.mark.parametrize("fade", [2.0, 5.0, 8.0])
def test_detects_fade_out_and_fade_in(fade):
    a, _ = beat_track(40, 128, fade_out=fade)
    assert fade_out_seconds(a, SR) == pytest.approx(fade, abs=0.6)
    b, _ = beat_track(40, 128, fade_in=fade)
    assert fade_in_seconds(b, SR) == pytest.approx(fade, abs=0.6)


def test_no_edge_on_steady_track_hard_ending_or_quiet_intro():
    a, _ = beat_track(40, 128)
    assert fade_out_seconds(a, SR) == 0.0 and fade_in_seconds(a, SR) == 0.0
    # a quiet but steady intro section is music, not a fade
    quiet_intro = a.copy()
    quiet_intro[: 8 * SR] *= 10 ** (-10 / 20)
    assert fade_in_seconds(quiet_intro, SR) == 0.0
    assert fade_out_seconds(np.zeros((5 * SR, 2), np.float32), SR) == 0.0


def test_long_fade_is_capped():
    a, _ = beat_track(60, 128, fade_out=20.0)
    assert fade_out_seconds(a, SR) == pytest.approx(12.0)


def _rms_db(x):
    return 10 * np.log10(np.mean(x.astype(np.float64) ** 2) + 1e-12)


@needs_rb
def test_crossfade_finishes_before_the_outgoing_fade():
    out, ob = beat_track(60, 126, seed=1, fade_out=5.0)
    inc, ib = beat_track(60, 128, seed=2)
    seam, ap = render_transition(out, inc, SR, ob, ib, req(8.0), "final")
    note = next(s for s in ap.safety_adjustments if s.startswith("outgoing fades out"))
    assert "seam lengthened" in note
    # the seam covers the fade on top of the requested blend
    assert ap.output_duration_seconds > 8.0 + 4.0
    # the music never dips: every 0.5 s of the seam stays near the body level
    body = _rms_db(inc[20 * SR:30 * SR])
    h = SR // 2
    levels = [_rms_db(seam[i:i + h]) - body for i in range(0, seam.shape[0] - h, h)]
    assert min(levels) > -4.0, levels
    # beat alignment adjustments exclude the deliberate fade extension; what's
    # left is rounding to a whole phrase (seam_plan.PHRASE_FITS: up to +30%)
    assert abs(ap.beat_alignment_adjustments_seconds) < 0.35 * 8.0


@needs_rb
def test_fade_aware_seam_still_keeps_every_sample():
    out, ob = beat_track(50, 126, seed=3, fade_out=4.0)
    inc, ib = beat_track(50, 124, seed=4, fade_in=3.0)
    seam, ap = render_transition(out, inc, SR, ob, ib, req(6.0), "final")
    assert any("incoming fades in" in s for s in ap.safety_adjustments)
    o_start, consumed = seam_sample_bounds(ap, SR)
    mix = np.concatenate([out[:o_start], seam, inc[consumed:]])
    np.testing.assert_array_equal(mix[:o_start], out[:o_start])
    np.testing.assert_array_equal(mix[-(inc.shape[0] - consumed):], inc[consumed:])
    assert out.shape[0] - o_start <= out.shape[0] // 2 and consumed <= inc.shape[0] // 2


def test_short_blend_happens_before_the_outgoing_fade():
    out, ob = beat_track(40, 100, seed=5, fade_out=4.0)
    inc, ib = beat_track(40, 140, seed=6)  # 40% apart: short blend
    seam, ap = render_transition(out, inc, SR, ob, ib, req(8.0), "final")
    assert any(s.startswith("short blend") for s in ap.safety_adjustments)
    assert ap.actual_overlap_seconds == pytest.approx(4.0 + 2 * 60 / 140, abs=0.5)
    body = _rms_db(inc[20 * SR:30 * SR])
    h = int(1.2 * SR)  # two beats of either track, so kicks don't make the level jump
    levels = [_rms_db(seam[i:i + h]) - body for i in range(0, seam.shape[0] - h, SR // 4)]
    assert min(levels) > -4.0, levels


def test_preview_and_final_agree_with_edges():
    out, ob = beat_track(40, 100, seed=7, fade_out=4.0)
    inc, ib = beat_track(40, 140, seed=8)
    seam, _ = render_transition(out, inc, SR, ob, ib, req(8.0), "final")
    prev, _ = render_transition(out, inc, SR, ob, ib, req(8.0), "preview", 3.0)
    np.testing.assert_array_equal(prev[3 * SR:3 * SR + seam.shape[0]], seam)


def test_no_beat_align_ignores_edges():
    out, ob = beat_track(40, 128, seed=9, fade_out=4.0)
    inc, ib = beat_track(40, 128, seed=10)
    r = TransitionRequest(from_id="a", to_id="b", requested_overlap_seconds=6.0, beat_align=False)
    _, ap = render_transition(out, inc, SR, ob, ib, r, "final")
    assert ap.actual_overlap_seconds == pytest.approx(6.0, abs=1 / SR)
