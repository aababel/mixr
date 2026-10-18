import numpy as np
import pytest

from backend.api.schemas import TransitionRequest
from backend.audio.bands import MID_UNDER
from backend.audio.transition import SafetyLimits, render_transition, seam_sample_bounds

SR = 22050


def click_track(dur, bpm, first_beat=0.0, amp=0.5, channels=2, seed=0):
    """Low noise bed plus clicks on every beat. Returns (audio, beats_seconds)."""
    n = int(dur * SR)
    rng = np.random.default_rng(seed)
    a = (rng.standard_normal((n, channels)) * 0.01).astype(np.float32)
    period = 60.0 / bpm
    beats = list(np.arange(first_beat, dur - 0.01, period))
    for b in beats:
        i = int(round(b * SR))
        a[i:i + 200] += amp
    return a, [float(b) for b in beats]


def req(ov, **kw):
    return TransitionRequest(from_id="a", to_id="b", requested_overlap_seconds=ov, **kw)


def stitch(out, inc, seam, applied):
    o_start, ov = seam_sample_bounds(applied, SR)
    return np.concatenate([out[:o_start], seam, inc[ov:]])


@pytest.fixture
def pair():
    out, ob = click_track(30.0, 120, first_beat=0.25, seed=1)
    inc, ib = click_track(30.0, 120, first_beat=0.4, seed=2)
    return out, ob, inc, ib


def test_final_returns_only_seam(pair):
    out, ob, inc, ib = pair
    seam, ap = render_transition(out, inc, SR, ob, ib, req(4.0), "final")
    assert seam.shape[0] == round(ap.actual_overlap_seconds * SR)
    assert ap.output_duration_seconds == pytest.approx(ap.actual_overlap_seconds)
    assert ap.incoming_start_seconds == 0.0
    assert ap.outgoing_start_seconds + ap.actual_overlap_seconds == pytest.approx(30.0)


def test_stitch_preserves_all_content(pair):
    out, ob, inc, ib = pair
    seam, ap = render_transition(out, inc, SR, ob, ib, req(4.0), "final")
    o_start, ov = seam_sample_bounds(ap, SR)
    mix = stitch(out, inc, seam, ap)
    assert mix.shape[0] == out.shape[0] + inc.shape[0] - ov
    np.testing.assert_array_equal(mix[:o_start], out[:o_start])
    np.testing.assert_array_equal(mix[o_start + ov:], inc[ov:])


def test_preview_seam_is_bit_identical_to_final(pair):
    out, ob, inc, ib = pair
    seam, apf = render_transition(out, inc, SR, ob, ib, req(5.0), "final")
    prev, app = render_transition(out, inc, SR, ob, ib, req(5.0), "preview", 6.0)
    ctx = int(6.0 * SR)
    np.testing.assert_array_equal(prev[ctx:ctx + seam.shape[0]], seam)
    assert prev.shape[0] == 2 * ctx + seam.shape[0]
    assert app.actual_overlap_seconds == apf.actual_overlap_seconds
    assert app.mode == "preview" and apf.mode == "final"


def test_preview_context_clamped_at_track_edges():
    out, ob = click_track(4.0, 120)
    inc, ib = click_track(4.0, 120, seed=3)
    prev, ap = render_transition(out, inc, SR, ob, ib, req(1.5), "preview", 6.0)
    # pre-roll + seam + post-roll covers both tracks exactly once
    assert prev.shape[0] == out.shape[0] + inc.shape[0] - round(ap.actual_overlap_seconds * SR)


def test_same_tempo_matches_every_beat_without_warping(pair):
    out, ob, inc, ib = pair
    _, ap = render_transition(out, inc, SR, ob, ib, req(4.1), "final")
    # the grid is locked to the clicks themselves, so allow 0.5 ms of onset timing
    landed = ap.outgoing_start_seconds + ib[0]
    assert min(abs(landed - b) for b in ob) < 0.0005
    # same tempo: every later incoming beat lands on an outgoing beat too
    for b in ib[1:6]:
        assert min(abs(ap.outgoing_start_seconds + b - x) for x in ob) < 0.0005
    # whole bars (8 beats = 4 s at 120 BPM) plus pre-roll and tail, under one beat each
    assert abs(ap.actual_overlap_seconds - 4.0) <= 1.0
    assert ap.tempo_change is None
    assert ap.output_duration_seconds == pytest.approx(ap.actual_overlap_seconds)
    assert ap.beat_alignment_adjustments_seconds == pytest.approx(
        ap.actual_overlap_seconds - 4.1, abs=1e-9
    )
    assert ap.overlap_changed_for_safety is False
    assert any("beats matched over 8 beats" in s for s in ap.safety_adjustments)


def test_irregular_beats_fall_back_to_first_beat_snap(pair):
    out, ob, inc, ib = pair
    few = ib[:3]  # too few beats to fit a grid
    _, ap = render_transition(out, inc, SR, ob, few, req(4.1), "final")
    landed = ap.outgoing_start_seconds + few[0]
    assert min(abs(landed - b) for b in ob) < 1.5 / SR
    assert abs(ap.actual_overlap_seconds - 4.1) <= 0.5
    assert ap.tempo_change is None


def test_no_beat_align_keeps_requested(pair):
    out, ob, inc, ib = pair
    _, ap = render_transition(out, inc, SR, ob, ib, req(4.1, beat_align=False), "final")
    assert ap.actual_overlap_seconds == pytest.approx(4.1, abs=1 / SR)
    assert ap.beat_alignment_adjustments_seconds == 0.0


def test_missing_beats_skip_snap_and_record(pair):
    out, _, inc, _ = pair
    _, ap = render_transition(out, inc, SR, [], [], req(4.1), "final")
    assert ap.actual_overlap_seconds == pytest.approx(4.1, abs=1 / SR)
    assert any("beat alignment skipped" in s for s in ap.safety_adjustments)


def test_clamp_to_max(pair):
    out, ob, inc, ib = pair
    _, ap = render_transition(out, inc, SR, ob, ib, req(40.0, beat_align=False), "final")
    assert ap.requested_overlap_seconds == 40.0
    assert ap.actual_overlap_seconds == pytest.approx(15.0, abs=1 / SR)  # 50% of 30 s
    assert ap.overlap_changed_for_safety is True
    assert any("clamped" in s for s in ap.safety_adjustments)


def test_clamp_to_min(pair):
    out, ob, inc, ib = pair
    _, ap = render_transition(out, inc, SR, ob, ib, req(0.2, beat_align=False), "final")
    assert ap.actual_overlap_seconds == pytest.approx(1.0, abs=1 / SR)
    assert ap.overlap_changed_for_safety is True


def test_custom_safety_limits(pair):
    out, ob, inc, ib = pair
    s = SafetyLimits(max_overlap_seconds=3.0)
    _, ap = render_transition(out, inc, SR, ob, ib, req(8.0), "final", safety=s)
    assert ap.actual_overlap_seconds <= 3.0 + 1e-9


def test_beat_snap_never_leaves_safe_range(pair):
    out, ob, inc, ib = pair
    s = SafetyLimits(max_overlap_seconds=6.0)
    for r in [1.0, 2.0, 3.3, 5.9, 6.0, 12.0]:
        _, ap = render_transition(out, inc, SR, ob, ib, req(r), "final", safety=s)
        assert 1.0 - 1e-6 <= ap.actual_overlap_seconds <= 6.0 + 1e-6


@pytest.mark.parametrize("style", ["equal_power", "linear"])
def test_fade_shapes(style):
    """Mid-band content (1 kHz) follows the requested fade curve."""
    n = int(10 * SR)
    sine = (0.5 * np.sin(2 * np.pi * 1000 * np.arange(n) / SR)).astype(np.float32)[:, None]
    seam, ap = render_transition(
        sine, sine.copy(), SR, [], [], req(2.0, beat_align=False, fade_style=style), "final"
    )
    assert ap.fade_style == style
    m = seam.shape[0]
    rms = lambda x: float(np.sqrt(np.mean(x ** 2)))
    w = int(0.05 * SR)
    # outgoing plus a hint of the incoming at MID_UNDER
    assert rms(seam[:w]) == pytest.approx((1 + MID_UNDER) * 0.5 / np.sqrt(2), rel=0.03)
    assert rms(seam[-w // 5:]) == pytest.approx(0.5 / np.sqrt(2), rel=0.03)  # all incoming
    # identical in-phase inputs: amplitude = fade_out + fade_in. The mids cross
    # over in the second half, so its middle (3/4 of the way) is the fade midpoint
    gain = np.sqrt(2) if style == "equal_power" else 1.0
    q = 3 * m // 4
    assert rms(seam[q - w // 2:q + w // 2]) == pytest.approx(gain * 0.5 / np.sqrt(2), rel=0.03)
    # before that: all outgoing plus a hint of the incoming at MID_UNDER
    h = m // 4
    assert rms(seam[h - w // 2:h + w // 2]) == pytest.approx((1 + MID_UNDER) * 0.5 / np.sqrt(2), rel=0.03)


def test_limiter_fires_and_is_recorded():
    n = int(10 * SR)
    out = np.full((n, 2), 0.85, dtype=np.float32)
    inc = np.full((n, 2), 0.85, dtype=np.float32)
    seam, ap = render_transition(out, inc, SR, [], [], req(3.0, beat_align=False), "final")
    assert np.max(np.abs(seam)) < 1.0
    assert any("soft limiter" in s for s in ap.safety_adjustments)
    assert ap.applied_gain_db == {"outgoing": 0.0, "incoming": 0.0}
    assert ap.applied_eq.startswith("3-band swap") and ap.tempo_change is None


def test_quiet_seam_not_limited(pair):
    out, ob, inc, ib = pair
    _, ap = render_transition(out, inc, SR, ob, ib, req(4.0), "final")
    assert not any("limiter" in s for s in ap.safety_adjustments)


@pytest.mark.parametrize("layout", ["mono_1d", "frames", "channels_first"])
def test_input_layouts(layout):
    a, _ = click_track(10.0, 120, channels=2)
    b, _ = click_track(10.0, 120, channels=2, seed=4)
    if layout == "mono_1d":
        a, b = a[:, 0], b[:, 0]
    elif layout == "channels_first":
        a, b = a.T, b.T
    seam, _ = render_transition(a, b, SR, [], [], req(2.0, beat_align=False), "final")
    assert seam.dtype == np.float32
    assert seam.shape == (2 * SR, 1 if layout == "mono_1d" else 2)


def test_mono_upmixed_to_stereo():
    a, _ = click_track(10.0, 120, channels=1)
    b, _ = click_track(10.0, 120, channels=2)
    seam, ap = render_transition(a[:, 0], b, SR, [], [], req(2.0, beat_align=False), "final")
    assert seam.shape[1] == 2
    assert any("upmixed" in s for s in ap.safety_adjustments)
