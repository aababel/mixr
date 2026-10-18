import numpy as np
import pytest

from backend.api.schemas import TrackWindow, TransitionRequest
from backend.audio.drop import highpass_sweep, roll
from backend.audio.transition import render_transition, seam_sample_bounds
from backend.audio.windows import with_entry

SR = 22050


def click_track(dur, bpm, first_beat=0.0, amp=0.5, seed=0):
    n = int(dur * SR)
    rng = np.random.default_rng(seed)
    a = (rng.standard_normal((n, 2)) * 0.05).astype(np.float32)
    period = 60.0 / bpm
    beats = list(np.arange(first_beat, dur - 0.01, period))
    for b in beats:
        i = int(round(b * SR))
        a[i:i + 200] += amp
    return a, [float(b) for b in beats]


def drop_req(build):
    return TransitionRequest(from_id="a", to_id="b", requested_overlap_seconds=build, style="drop")


def rms(x):
    return float(np.sqrt(np.mean(np.square(x)))) if x.size else 0.0


def test_layered_drop_builds_into_the_drop_without_a_break():
    out, ob = click_track(30.0, 120, seed=1)
    inc, ib = click_track(30.0, 120, seed=2)
    seam, ap = render_transition(out, inc, SR, ob, ib, drop_req(8.0), "final")
    assert ap.applied_eq.startswith("drop:") and "rolled" not in ap.applied_eq
    assert "build rises under the sweep" in ap.applied_eq
    assert "incoming drop at 8.00s" in ap.applied_eq  # 16 matched beats in, no warp
    p = int(0.5 * SR)
    drop_at = int(round(8.0 * SR))
    before = seam[drop_at - p:drop_at - int(0.01 * SR)]
    assert rms(before) > 0.05, "the beat before the drop keeps playing"
    assert rms(seam[drop_at:drop_at + p]) > 0.05, "B lands at full level"
    o_start, ov = seam_sample_bounds(ap, SR)
    assert o_start + seam.shape[0] == out.shape[0]
    assert ov == seam.shape[0]


def test_cut_drop_when_tempos_are_too_far_apart():
    out, ob = click_track(30.0, 100, seed=1)
    inc, ib = click_track(30.0, 140, seed=2)
    seam, ap = render_transition(out, inc, SR, ob, ib, drop_req(6.0), "final")
    assert "starts cold on its drop" in ap.applied_eq
    o_start, ov = seam_sample_bounds(ap, SR)
    lead = int(0.004 * SR)  # B's first few ms after its drop, faded in inside the seam
    assert ov == int(round(6.0 * SR)) + lead, "the build is skipped: B resumes at its drop"
    assert seam.shape[0] == pytest.approx(2 * 4 * 0.6 * SR + lead, abs=0.02 * SR), "2 bars of A"
    period = int(0.6 * SR)
    gap = seam[-lead - period + int(0.2 * SR):-lead - int(0.01 * SR)]
    assert rms(gap) < 0.02


def test_drop_style_wins_over_echo_out():
    out, ob = click_track(30.0, 120, seed=1)
    inc, ib = click_track(30.0, 120, seed=2)
    melody = [0.9] * 200
    _, ap = render_transition(out, inc, SR, ob, ib, drop_req(8.0), "final",
                              outgoing_melody=melody, incoming_melody=melody, melody_hop_seconds=0.5)
    assert ap.applied_eq.startswith("drop:")


def test_sweep_removes_lows_by_the_end_and_roll_repeats_slices():
    t = np.arange(int(2.0 * SR)) / SR
    low = np.stack([np.sin(2 * np.pi * 60 * t)] * 2, axis=1)
    y = highpass_sweep(low, SR, low.shape[0] // 2)
    assert rms(y[:int(0.2 * SR)]) > 0.5
    assert rms(y[-int(0.5 * SR):]) < 0.01
    a = np.random.default_rng(0).standard_normal((4 * 1000, 2))
    rolled, last = roll(a, 0, 1000, SR)
    assert rolled.shape == (3000, 2) and last.shape[0] == 125
    np.testing.assert_allclose(rolled[:1000], a[:1000])


def test_with_entry_turns_a_drop_window_into_a_drop_request():
    base = TransitionRequest(from_id="a", to_id="b", requested_overlap_seconds=12.0)
    w = TrackWindow(track_id="b", start_seconds=30.0, end_seconds=120.0, drop_seconds=38.0)
    r = with_entry(base, w)
    assert r.style == "drop" and r.requested_overlap_seconds == pytest.approx(8.0)
    assert with_entry(base, w.model_copy(update={"drop_seconds": None})) == base
    assert with_entry(base, None) == base
