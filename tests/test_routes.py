import io
import sys

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from backend.main import app
from tests import fake_audio


@pytest.fixture
def client(tmp_path, monkeypatch):
    monkeypatch.setenv("MIXR_STORAGE_DIR", str(tmp_path))
    return TestClient(app)


def wav_bytes(seconds=1.0, sr=22050):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    buf = io.BytesIO()
    sf.write(buf, 0.2 * np.sin(2 * np.pi * 440 * t), sr, format="WAV")
    return buf.getvalue()


def wav_files(n, seconds=1.0):
    return [("files", (f"song{i}.wav", wav_bytes(seconds), "audio/wav")) for i in range(n)]


def upload(client, n=3):
    res = client.post("/api/tracks", files=wav_files(n))
    assert res.status_code == 200, res.text
    return res.json()


def test_upload_returns_tracks_in_seconds(client, tmp_path):
    body = upload(client)
    assert len(body["track_ids"]) == 3
    track = body["tracks"][0]
    assert track["duration_seconds"] == pytest.approx(1.0)
    assert track["sample_rate"] == 22050
    assert track["analysis_status"] == "pending"
    assert len(list(tmp_path.iterdir())) == 3


@pytest.mark.parametrize("n", [2, 6])
def test_upload_rejects_wrong_count(client, n):
    assert client.post("/api/tracks", files=wav_files(n)).status_code == 400


def test_upload_rejects_unsupported_extension(client):
    files = wav_files(2) + [("files", ("notes.txt", b"hello", "text/plain"))]
    assert client.post("/api/tracks", files=files).status_code == 400


def test_upload_rejects_corrupt_audio_and_cleans_up(client, tmp_path):
    files = wav_files(2) + [("files", ("broken.wav", b"not really audio", "audio/wav"))]
    res = client.post("/api/tracks", files=files)
    assert res.status_code == 400
    assert "broken.wav" in res.json()["detail"]
    assert list(tmp_path.iterdir()) == []


def test_read_track(client):
    track_id = upload(client)["track_ids"][0]
    assert client.get(f"/api/tracks/{track_id}").json()["id"] == track_id
    assert client.get("/api/tracks/nope").status_code == 404


def test_read_track_audio(client):
    track_id = upload(client)["track_ids"][0]
    res = client.get(f"/api/tracks/{track_id}/audio")
    assert res.status_code == 200
    assert res.headers["content-type"] == "audio/wav"
    assert sf.info(io.BytesIO(res.content)).duration == pytest.approx(1.0)
    assert client.get("/api/tracks/nope/audio").status_code == 404


def test_analyze_and_order_stubs(client):
    ids = upload(client)["track_ids"]
    analyzed = client.post("/api/analyze", json={"track_ids": ids}).json()
    assert [t["id"] for t in analyzed["tracks"]] == ids
    ordered = client.post("/api/order", json={"track_ids": ids}).json()
    assert ordered == {"ordered_track_ids": ids, "scores": []}


def test_unknown_track_ids_404(client):
    assert client.post("/api/analyze", json={"track_ids": ["nope"]}).status_code == 404
    assert client.post("/api/order", json={"track_ids": ["nope"]}).status_code == 404


def test_unimplemented_stages_are_honest(client):
    assert client.post("/api/calibration/preview").status_code == 501
    assert client.post("/api/calibration/vote").status_code == 501


@pytest.fixture
def no_renderer(monkeypatch):
    monkeypatch.setitem(sys.modules, "backend.audio.mix", None)


@pytest.fixture
def fake_renderer(monkeypatch):
    monkeypatch.setitem(sys.modules, "backend.audio.mix", fake_audio.as_module())
    monkeypatch.setitem(sys.modules, "backend.logic.calibration", None)


def test_mix_without_renderer_is_501(client, no_renderer):
    ids = upload(client)["track_ids"]
    res = client.post("/api/mix", json={"ordered_track_ids": ids})
    assert res.status_code == 501
    assert "render_mix" in res.json()["detail"]


def test_mix_validates_tracks(client, fake_renderer):
    ids = upload(client)["track_ids"]
    assert client.post("/api/mix", json={"ordered_track_ids": ids[:1]}).status_code == 400
    assert client.post("/api/mix", json={"ordered_track_ids": [ids[0], ids[0]]}).status_code == 400
    assert client.post("/api/mix", json={"ordered_track_ids": [ids[0], "nope"]}).status_code == 404
    bad_pref = {"ordered_track_ids": ids, "overlap_preference": 2}
    assert client.post("/api/mix", json=bad_pref).status_code == 422


def test_mix_end_to_end(client, fake_renderer):
    ids = upload(client)["track_ids"]
    res = client.post("/api/mix", json={"ordered_track_ids": ids})
    assert res.status_code == 200, res.text
    body = res.json()
    manifest = body["manifest"]
    assert manifest["ordered_track_ids"] == ids
    assert len(manifest["transitions"]) == 2
    assert manifest["transitions"][0]["requested_overlap_seconds"] == 8.0
    assert manifest["transitions"][0]["overlap_changed_for_safety"] is True
    seam = manifest["transitions"][1]
    assert seam["outgoing_start_seconds"] == pytest.approx(1.0 - seam["actual_overlap_seconds"])

    assert client.get(f"/api/mixes/{body['mix_id']}").json() == body
    audio = client.get(body["audio_url"])
    assert audio.status_code == 200
    assert "attachment" in audio.headers["content-disposition"]
    duration = sf.info(io.BytesIO(audio.content)).duration
    assert duration == pytest.approx(manifest["output_duration_seconds"], abs=0.01)


def test_unknown_mix_404(client):
    assert client.get("/api/mixes/nope").status_code == 404
    assert client.get("/api/mixes/nope/audio").status_code == 404
