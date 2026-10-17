import numpy as np
import pytest
import soundfile as sf

from backend.api.schemas import Track, TransitionRequest
from backend.audio.loading import RENDER_SR, load_render_audio
from backend.audio.mix import render_mix
from backend.audio.preview import PREVIEW_CONTEXT_SECONDS, render_preview
from backend.audio.transition import render_transition

SR = RENDER_SR
Q = 1.5 / 32768  # 16-bit output quantization tolerance


def make(tmp_path, name, dur, bpm, seed, channels=2, first=0.2):
    rng = np.random.default_rng(seed)
    n = int(dur * SR)
    y = rng.uniform(-0.3, 0.3, (n, channels)).astype(np.float32)
    beats = [float(b) for b in np.arange(first, dur - 0.05, 60.0 / bpm)]
    p = tmp_path / f"{name}.wav"
    sf.write(p, y if channels > 1 else y[:, 0], SR, subtype="FLOAT")
    tr = Track(id=name, filename=p.name, duration_seconds=dur, sample_rate=SR,
               bpm=bpm, beats_seconds=beats, analysis_status="ok")
    return p, tr


@pytest.fixture
def three(tmp_path):
    return [make(tmp_path, "a", 20, 120, 1), make(tmp_path, "b", 25, 124, 2), make(tmp_path, "c", 18, 128, 3)]


def test_mix_keeps_every_track_complete(tmp_path, three):
    paths, tracks = [p for p, _ in three], [t for _, t in three]
    out = tmp_path / "mix.wav"
    m = render_mix("m1", tracks, paths, [4.0, 6.0], out, loudness_match=False)

    mix, sr = sf.read(out, dtype="float32")
    src = [load_render_audio(p) for p in paths]
    ov = [round(t.actual_overlap_seconds * SR) for t in m.transitions]

    assert sr == SR and mix.shape[1] == 2
    assert mix.shape[0] == sum(s.shape[0] for s in src) - sum(ov)
    assert m.output_duration_seconds == pytest.approx(mix.shape[0] / SR)
    assert m.source_durations_seconds == pytest.approx([20, 25, 18])

    # Untouched stretches of each track appear verbatim, in order.
    a_end = src[0].shape[0] - ov[0]
    np.testing.assert_allclose(mix[:a_end], src[0][:a_end], atol=Q)
    b_start = a_end + ov[0]
    b_mid = src[1][ov[0]: src[1].shape[0] - ov[1]]
    np.testing.assert_allclose(mix[b_start:b_start + len(b_mid)], b_mid, atol=Q)
    np.testing.assert_allclose(mix[-(src[2].shape[0] - ov[1]):], src[2][ov[1]:], atol=Q)


def test_mix_seams_match_shared_renderer(tmp_path, three):
    paths, tracks = [p for p, _ in three], [t for _, t in three]
    out = tmp_path / "mix.wav"
    m = render_mix("m1", tracks, paths, [4.0, 6.0], out, loudness_match=False)
    mix, _ = sf.read(out, dtype="float32")
    a, b = load_render_audio(paths[0]), load_render_audio(paths[1])
    req = TransitionRequest(from_id="a", to_id="b", requested_overlap_seconds=4.0)
    seam, applied = render_transition(a, b, SR, tracks[0].beats_seconds, tracks[1].beats_seconds, req, "final")
    start = round(applied.outgoing_start_seconds * SR)
    np.testing.assert_allclose(mix[start:start + len(seam)], seam, atol=Q)
    assert m.transitions[0] == applied


def test_manifest_records_requests_and_flags(tmp_path, three):
    paths, tracks = [p for p, _ in three], [t for _, t in three]
    m = render_mix("m2", tracks, paths, [2.0, 40.0], tmp_path / "m.wav",
                   calibrated=True, overlap_preference=0.5)
    assert m.ordered_track_ids == ["a", "b", "c"]
    assert [t.requested_overlap_seconds for t in m.transitions] == [2.0, 40.0]
    assert m.transitions[1].overlap_changed_for_safety  # 40 s clamped to 50% of 18 s track
    assert m.transitions[1].actual_overlap_seconds <= 9.0 + 1e-6
    assert m.calibrated and m.overlap_preference == 0.5
    assert all(t.mode == "final" for t in m.transitions)


def test_mono_source_and_missing_beats(tmp_path):
    p1, t1 = make(tmp_path, "mono", 15, 120, 4, channels=1)
    p2, t2 = make(tmp_path, "s", 15, 120, 5)
    t1 = t1.model_copy(update={"beats_seconds": []})
    m = render_mix("m3", [t1, t2], [p1, p2], [3.0], tmp_path / "out.wav")
    mix, _ = sf.read(tmp_path / "out.wav")
    assert mix.shape[1] == 2
    assert any("no beat grid" in s for s in m.limitations)


def test_bad_inputs_raise(tmp_path, three):
    paths, tracks = [p for p, _ in three], [t for _, t in three]
    with pytest.raises(ValueError):
        render_mix("x", tracks, paths, [4.0], tmp_path / "x.wav")
    with pytest.raises(ValueError):
        render_mix("x", tracks, paths, [4.0, 0.0], tmp_path / "x.wav")
    with pytest.raises(ValueError):
        render_mix("x", tracks, paths, [4.0, 4.0], paths[0])


def test_preview_is_exact_excerpt_of_final_mix(tmp_path, three):
    (pa, ta), (pb, tb), _ = three
    req = TransitionRequest(from_id="a", to_id="b", requested_overlap_seconds=5.0)
    applied = render_preview(pa, pb, ta, tb, req, tmp_path / "prev.wav")
    prev, _ = sf.read(tmp_path / "prev.wav", dtype="float32")
    assert applied.mode == "preview"

    m = render_mix("m4", [ta, tb], [pa, pb], [5.0], tmp_path / "mix.wav")
    mix, _ = sf.read(tmp_path / "mix.wav", dtype="float32")
    assert m.transitions[0].actual_overlap_seconds == applied.actual_overlap_seconds

    ctx = round(PREVIEW_CONTEXT_SECONDS * SR)
    start = round(applied.outgoing_start_seconds * SR) - ctx
    np.testing.assert_allclose(prev, mix[start:start + len(prev)], atol=Q)


def test_ab_previews_differ_only_in_overlap(tmp_path, three):
    (pa, ta), (pb, tb), _ = three
    apps = [
        render_preview(pa, pb, ta, tb,
                       TransitionRequest(from_id="a", to_id="b", requested_overlap_seconds=s),
                       tmp_path / f"p{s}.wav")
        for s in (2.0, 6.0)
    ]
    a, b = (x.model_dump() for x in apps)
    differing = {k for k in a if a[k] != b[k]}
    assert differing <= {
        "requested_overlap_seconds", "actual_overlap_seconds", "outgoing_start_seconds",
        "beat_alignment_adjustments_seconds", "output_duration_seconds", "safety_adjustments",
    }
    assert b["actual_overlap_seconds"] - a["actual_overlap_seconds"] > 2.0
