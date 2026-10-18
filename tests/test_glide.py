"""Tempo glide, split-frequency swap and short-blend fallback."""
import numpy as np
import pytest

from backend.api.schemas import TransitionRequest
from backend.audio import transition as transition_mod
from backend.audio.bands import split_bands, swap_gains
from backend.audio.seam_plan import fit_grid, fold_ratio
from backend.audio.stretch import rubberband_available
from backend.audio.transition import render_transition, seam_sample_bounds

SR = 22050
needs_rb = pytest.mark.skipif(not rubberband_available(), reason="rubberband CLI not installed")


def clicks(dur, bpm, first=0.3, seed=0, quantize=False):
    """Quiet noise bed + 5 ms clicks on each beat. quantize=True rounds the
    reported beats to librosa's 512-sample hop at 22050 Hz, like real analysis."""
    n = int(dur * SR)
    a = (np.random.default_rng(seed).standard_normal((n, 2)) * 0.005).astype(np.float32)
    true = np.arange(first, dur - 0.05, 60.0 / bpm)
    for b in true:
        i = int(round(b * SR))
        a[i:i + 110] += 0.6
    beats = np.round(true * SR / 512) * 512 / SR if quantize else true
    return a, [float(b) for b in beats]


def onsets(x, thr=0.3, gap=0.15):
    e = np.abs(x).mean(axis=1)
    out, i = [], 0
    while i < len(e):
        if e[i] > thr:
            out.append(i / SR)
            i += int(gap * SR)
        else:
            i += 1
    return np.array(out)


def req(ov):
    return TransitionRequest(from_id="a", to_id="b", requested_overlap_seconds=ov)


# ---------------------------------------------------------------- beat grid


def test_fit_grid_survives_frame_quantization_and_missed_beats():
    true_p = 60 / 138
    beats = list(np.round((0.2 + np.arange(60) * true_p) * SR / 512) * 512 / SR)
    del beats[20]  # missed beat
    g = fit_grid(beats, 0, 100)
    assert g is not None
    assert g.period == pytest.approx(true_p, rel=1e-3)
    assert g.first == pytest.approx(0.2, abs=0.012)


def test_fit_grid_rejects_irregular_or_sparse_beats():
    rng = np.random.default_rng(0)
    assert fit_grid(list(np.cumsum(rng.uniform(0.3, 0.7, 30))), 0, 100) is None
    assert fit_grid([1.0, 1.5, 2.0], 0, 100) is None


@pytest.mark.parametrize("p_out,p_in,ratio,factor", [
    (60 / 128, 60 / 64, 1.0, 0.5),   # incoming detected at half tempo
    (60 / 87, 60 / 174, 1.0, 2.0),   # incoming detected at double tempo
    (60 / 138, 60 / 128, 128 / 138, 1.0),
])
def test_fold_ratio(p_out, p_in, ratio, factor):
    r, f = fold_ratio(p_out, p_in)
    assert r == pytest.approx(ratio) and f == factor


# ---------------------------------------------------------------- bands


def test_bands_sum_to_input_and_swaps_keep_power():
    x = np.random.default_rng(1).standard_normal((SR, 2))
    low, mid, high = split_bands(x, SR)
    np.testing.assert_allclose(low + mid + high, x, atol=1e-9)
    assert np.std(low) > 0 and np.std(high) > 0
    go, gi = swap_gains(1000, 500, 40)
    np.testing.assert_allclose(go ** 2 + gi ** 2, 1.0, atol=1e-12)
    assert go[0, 0] == 1 and gi[0, 0] == 0 and go[-1, 0] == 0 and gi[-1, 0] == 1
    assert gi[459, 0] == 0 and gi[500, 0] == 1  # ramp is [at - ramp, at)


def test_bass_and_highs_swap_at_the_right_time():
    n = 20 * SR
    t = np.arange(n) / SR
    bass_a = (0.3 * np.sin(2 * np.pi * 60 * t))[:, None].repeat(2, 1).astype(np.float32)
    hats_b = (0.3 * np.sin(2 * np.pi * 8000 * t))[:, None].repeat(2, 1).astype(np.float32)
    r = TransitionRequest(from_id="a", to_id="b", requested_overlap_seconds=4.0, beat_align=False)
    seam_a, ap = render_transition(bass_a, np.zeros_like(bass_a), SR, [], [], r, "final")
    seam_b, _ = render_transition(np.zeros_like(hats_b), hats_b, SR, [], [], r, "final")
    m = seam_a.shape[0]
    rms = lambda x: float(np.sqrt(np.mean(x ** 2)))
    # outgoing bass stays at full level until the midpoint, then is gone
    assert rms(seam_a[int(0.40 * m):int(0.48 * m)]) == pytest.approx(0.3 / np.sqrt(2), rel=0.05)
    assert rms(seam_a[int(0.52 * m):int(0.9 * m)]) < 0.003
    # incoming highs are silent until ~30%, then at full level
    assert rms(seam_b[int(0.05 * m):int(0.28 * m)]) < 0.003
    assert rms(seam_b[int(0.32 * m):int(0.6 * m)]) == pytest.approx(0.3 / np.sqrt(2), rel=0.05)
    assert "lows <200 Hz swapped at 2.00s (50%)" in ap.applied_eq
    assert "highs >5000 Hz swapped at 1.20s (30%)" in ap.applied_eq


# ---------------------------------------------------------------- tempo glide


@needs_rb
def test_glide_lands_every_beat_together():
    out, ob = clicks(40, 120, first=0.21, seed=1, quantize=True)
    inc, ib = clicks(40, 128, first=0.37, seed=2, quantize=True)
    seam, ap = render_transition(out, inc, SR, ob, ib, req(8.0), "final")
    assert ap.tempo_change == pytest.approx(128 / 120, rel=2e-3)
    assert any("tempo glide over 16 beats" in s for s in ap.safety_adjustments)
    # warp each side with the same plan (before the band swap) and compare click times
    lo, hi = transition_mod._safe_range(40, 40, transition_mod.DEFAULT_SAFETY)
    plan = transition_mod._plan_seam(out, inc, SR, ob, ib, req(8.0), 8.0, lo, hi, [])
    wo = transition_mod._warp_outgoing(out, plan, SR)
    wi = transition_mod._warp_incoming(inc, plan, SR)
    o_clicks, i_clicks = onsets(wo), onsets(wi)
    assert len(o_clicks) >= 15 and len(i_clicks) >= 15
    paired = [x for x in o_clicks if x > 0.01]  # skip a click cut at the seam start
    worst = max(np.min(np.abs(i_clicks - x)) for x in paired)
    assert worst < 0.006  # every beat within 6 ms (unwarped they drift ~31 ms per beat)


@needs_rb
def test_glide_seam_stitches_without_losing_content():
    out, ob = clicks(30, 124, seed=3)
    inc, ib = clicks(30, 118, seed=4)
    seam, ap = render_transition(out, inc, SR, ob, ib, req(6.0), "final")
    o_start, consumed = seam_sample_bounds(ap, SR)
    assert seam.shape[0] == round(ap.output_duration_seconds * SR)
    assert seam.shape[0] != consumed  # glided: seam length differs from incoming consumed
    assert consumed == round(ap.actual_overlap_seconds * SR)
    mix = np.concatenate([out[:o_start], seam, inc[consumed:]])
    assert mix.shape[0] == o_start + seam.shape[0] + inc.shape[0] - consumed
    np.testing.assert_array_equal(mix[:o_start], out[:o_start])
    np.testing.assert_array_equal(mix[-(inc.shape[0] - consumed):], inc[consumed:])
    # both sides stay inside the 50% safety cap
    assert out.shape[0] - o_start <= out.shape[0] // 2 and consumed <= inc.shape[0] // 2


@needs_rb
def test_glide_preview_matches_final():
    out, ob = clicks(30, 124, seed=3)
    inc, ib = clicks(30, 118, seed=4)
    seam, apf = render_transition(out, inc, SR, ob, ib, req(6.0), "final")
    prev, app = render_transition(out, inc, SR, ob, ib, req(6.0), "preview", 3.0)
    ctx = 3 * SR
    np.testing.assert_array_equal(prev[ctx:ctx + seam.shape[0]], seam)
    np.testing.assert_array_equal(prev[ctx + seam.shape[0]:],
                                  inc[round(apf.actual_overlap_seconds * SR):][:ctx])
    assert app.tempo_change == apf.tempo_change


def test_matching_tempos_are_not_warped(monkeypatch):
    def boom(*a, **k):
        raise AssertionError("should not warp")
    monkeypatch.setattr(transition_mod, "timemap_stretch", boom)
    out, ob = clicks(30, 126, seed=5)
    inc, ib = clicks(30, 126, first=0.1, seed=6)
    _, ap = render_transition(out, inc, SR, ob, ib, req(6.0), "final")
    assert ap.tempo_change is None
    assert any("beats matched" in s for s in ap.safety_adjustments)


# ---------------------------------------------------------------- short-blend fallback


@pytest.mark.parametrize("requested", [4.0, 12.0])
def test_far_apart_tempos_get_a_short_blend_on_an_incoming_beat(requested):
    out, ob = clicks(30, 90, seed=7)
    inc, ib = clicks(30, 128, first=0.05, seed=8)
    seam, ap = render_transition(out, inc, SR, ob, ib, req(requested), "final")
    assert ap.tempo_change is None
    assert 0.5 <= ap.actual_overlap_seconds <= 2.5 * 60 / 128 + 0.05  # about 2 beats
    assert min(abs(ap.actual_overlap_seconds - b) for b in ib) < 2 / SR  # ends on an incoming beat
    assert ap.output_duration_seconds == pytest.approx(ap.actual_overlap_seconds)
    assert any("differ by" in s and "no warp" in s for s in ap.safety_adjustments)
    assert any(s.startswith("short blend") for s in ap.safety_adjustments)


def test_missing_rubberband_falls_back_to_short_blend(monkeypatch):
    monkeypatch.setattr(transition_mod, "rubberband_available", lambda: False)
    out, ob = clicks(30, 120, seed=9)
    inc, ib = clicks(30, 128, seed=10)
    _, ap = render_transition(out, inc, SR, ob, ib, req(8.0), "final")
    assert ap.tempo_change is None
    assert ap.actual_overlap_seconds < 2.0
    assert any("Rubber Band is not installed" in s for s in ap.safety_adjustments)


def test_grid_locks_to_kicks_when_tracker_beats_jump_phase():
    """Real songs: the tracker can shift by a quarter beat or more mid-section
    (seen on the JJTBEATZ remixes). The grid must follow the kicks."""
    from backend.audio.seam_plan import kick_grid, beat_period
    out, true_beats = clicks(30, 128, first=0.25, seed=11)
    p = 60 / 128
    jumped = [b + (0.3 * p if b > 15 else 0.0) for b in true_beats]  # phase jump halfway
    assert fit_grid(jumped, 0, 30) is None or abs(
        (fit_grid(jumped, 0, 30).first - 0.25) % p) > 0.01  # tracked beats alone mislead
    g = kick_grid(out.mean(axis=1), SR, 0.0, beat_period(jumped, 0, 30))
    assert g is not None
    assert g.period == pytest.approx(p, rel=1e-3)
    offset = (g.first - 0.25 + p / 2) % p - p / 2
    assert abs(offset) < 0.006  # 4 ms phase bins


def test_render_matches_kicks_despite_phase_jump():
    out, ob = clicks(30, 126, first=0.25, seed=12)
    inc, ib = clicks(30, 126, first=0.1, seed=13)
    p = 60 / 126
    ib_jumped = [b + (0.3 * p if b > 4 else 0.0) for b in ib]
    _, ap = render_transition(out, inc, SR, ob, ib_jumped, req(6.0), "final")
    assert any("from kick onsets" in s for s in ap.safety_adjustments)
    for b in ib[1:8]:  # the TRUE incoming clicks land on outgoing clicks
        assert min(abs(ap.outgoing_start_seconds + b - x) for x in ob) < 0.006
