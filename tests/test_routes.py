import io

import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient

from backend.main import app


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
    assert client.post("/api/mix").status_code == 501
    assert client.post("/api/calibration/preview").status_code == 501
    assert client.post("/api/calibration/vote").status_code == 501
