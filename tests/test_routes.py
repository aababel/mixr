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


def wav_bytes(seconds=1.0, sr=22050, freq=440):
    t = np.linspace(0, seconds, int(sr * seconds), endpoint=False)
    buf = io.BytesIO()
    sf.write(buf, 0.2 * np.sin(2 * np.pi * freq * t), sr, format="WAV")
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


@pytest.mark.parametrize("n", [1, 6])
def test_upload_rejects_wrong_count(client, n):
    assert client.post("/api/tracks", files=wav_files(n)).status_code == 400


def test_upload_accepts_two_songs(client):
    assert len(upload(client, n=2)["track_ids"]) == 2


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


def click_wav_bytes(bpm, seconds=12.0, sr=22050):
    y = np.zeros(int(sr * seconds), dtype=np.float32)
    t = np.arange(400) / sr
    click = np.sin(2 * np.pi * 1000 * t) * np.exp(-t * 60) * 0.8
    for beat in np.arange(0.3, seconds - 0.05, 60.0 / bpm):
        i = int(beat * sr)
        y[i : i + 400] += click[: len(y) - i]
    buf = io.BytesIO()
    sf.write(buf, y, sr, format="WAV")
    return buf.getvalue()


def song_files(seconds=40.0):
    return [
        ("files", (f"song{i}.wav", wav_bytes(seconds, freq=freq), "audio/wav"))
        for i, freq in enumerate([330, 440, 550])
    ]


def test_analyze_fills_features_and_persists(client):
    files = [("files", (f"click{b}.wav", click_wav_bytes(b), "audio/wav")) for b in (120, 124, 128)]
    ids = client.post("/api/tracks", files=files).json()["track_ids"]
    tracks = client.post("/api/analyze", json={"track_ids": ids}).json()["tracks"]
    assert [t["id"] for t in tracks] == ids
    assert all(t["analysis_status"] == "ok" for t in tracks)
    assert tracks[0]["bpm"] == pytest.approx(120, rel=0.02)
    assert tracks[0]["beats_seconds"]
    assert client.get(f"/api/tracks/{ids[0]}").json()["bpm"] == tracks[0]["bpm"]


def test_order_returns_every_track_once_with_scores(client):
    ids = upload(client)["track_ids"]
    client.post("/api/analyze", json={"track_ids": ids})
    body = client.post("/api/order", json={"track_ids": ids}).json()
    assert sorted(body["ordered_track_ids"]) == sorted(ids)
    assert len(body["scores"]) == len(ids) * (len(ids) - 1)


def test_order_returns_explanation(client):
    ids = upload(client)["track_ids"]
    client.post("/api/analyze", json={"track_ids": ids})
    body = client.post("/api/order", json={"track_ids": ids}).json()
    explanation = body["explanation"]
    assert explanation["summary"].startswith(("Reordered", "Kept your order"))
    ordered = body["ordered_track_ids"]
    transitions = explanation["transitions"]
    assert [(t["from_id"], t["to_id"]) for t in transitions] == list(zip(ordered, ordered[1:]))
    assert all(t["reasons"] and 0.0 <= t["score"] <= 1.0 for t in transitions)


def test_unknown_track_ids_404(client):
    assert client.post("/api/analyze", json={"track_ids": ["nope"]}).status_code == 404
    assert client.post("/api/order", json={"track_ids": ["nope"]}).status_code == 404


def test_calibration_preview_and_vote(client):
    ids = client.post("/api/tracks", files=song_files()).json()["track_ids"]
    res = client.post("/api/calibration/preview", json={"ordered_track_ids": ids})
    assert res.status_code == 200, res.text
    trial = res.json()
    assert (trial["from_id"], trial["to_id"]) == (ids[0], ids[1])
    for variant in ("a", "b"):
        audio = client.get(trial[variant]["audio_url"])
        assert audio.status_code == 200
        assert sf.info(io.BytesIO(audio.content)).duration > 0

    vote = client.post(
        "/api/calibration/vote", json={"trial_id": trial["trial_id"], "chosen": "A"}
    ).json()
    p = vote["preference"]["overlap_preference"]
    assert p == (0.5 if vote["chose_longer"] else -0.5)
    assert len(vote["preference"]["votes"]) == 1
    assert abs(vote["chosen_overlap_seconds"] - vote["other_overlap_seconds"]) >= 2.0
    recorded = vote["preference"]["votes"][0]
    assert recorded["a_requested_seconds"] == pytest.approx(vote["chosen_overlap_seconds"])
    assert recorded["b_requested_seconds"] == pytest.approx(vote["other_overlap_seconds"])


def test_calibration_next_trial_uses_next_seam(client):
    ids = client.post("/api/tracks", files=song_files()).json()["track_ids"]
    trial = client.post(
        "/api/calibration/preview", json={"ordered_track_ids": ids, "trial": 1}
    ).json()
    assert (trial["from_id"], trial["to_id"]) == (ids[1], ids[2])


def test_calibration_never_repeats_a_compared_pair(client):
    ids = client.post("/api/tracks", files=song_files()).json()["track_ids"]
    first = [ids[0], ids[1]]
    trial = client.post(
        "/api/calibration/preview",
        json={"ordered_track_ids": ids, "trial": 2, "exclude_pairs": [first]},
    ).json()
    assert [trial["from_id"], trial["to_id"]] == [ids[1], ids[2]]

    res = client.post(
        "/api/calibration/preview",
        json={"ordered_track_ids": ids, "exclude_pairs": [first, [ids[1], ids[2]]]},
    )
    assert res.status_code == 422
    assert "No more song pairs" in res.json()["detail"]


def test_set_mode_is_not_available_yet(client):
    ids = upload(client)["track_ids"]
    body = {"ordered_track_ids": ids, "mode": "set", "target_seconds": 300}
    assert client.post("/api/mix", json=body).status_code == 501
    assert client.post("/api/calibration/preview", json=body).status_code == 501
    bad_length = {**body, "target_seconds": 30}
    assert client.post("/api/mix", json=bad_length).status_code == 422


def test_calibration_rejects_songs_too_short_to_compare(client):
    ids = upload(client)["track_ids"]
    res = client.post("/api/calibration/preview", json={"ordered_track_ids": ids})
    assert res.status_code == 422
    assert "wouldn't make an audible difference" in res.json()["detail"]


def test_calibration_unknown_trial_404(client):
    assert client.post(
        "/api/calibration/vote", json={"trial_id": "nope", "chosen": "A"}
    ).status_code == 404
    assert client.get("/api/calibration/previews/nope/a").status_code == 404


def test_mix_validates_tracks(client):
    ids = upload(client)["track_ids"]
    assert client.post("/api/mix", json={"ordered_track_ids": ids[:1]}).status_code == 400
    assert client.post("/api/mix", json={"ordered_track_ids": [ids[0], ids[0]]}).status_code == 400
    assert client.post("/api/mix", json={"ordered_track_ids": [ids[0], "nope"]}).status_code == 404
    bad_pref = {"ordered_track_ids": ids, "overlap_preference": 2}
    assert client.post("/api/mix", json=bad_pref).status_code == 422


def test_full_pipeline(client):
    ids = client.post("/api/tracks", files=song_files()).json()["track_ids"]
    client.post("/api/analyze", json={"track_ids": ids})
    ordered = client.post("/api/order", json={"track_ids": ids}).json()["ordered_track_ids"]

    res = client.post(
        "/api/mix",
        json={"ordered_track_ids": ordered, "overlap_preference": 0.5, "calibrated": True},
    )
    assert res.status_code == 200, res.text
    body = res.json()
    manifest = body["manifest"]
    assert manifest["ordered_track_ids"] == ordered
    assert manifest["calibrated"] is True
    assert manifest["overlap_preference"] == 0.5
    assert len(manifest["transitions"]) == 2

    assert client.get(f"/api/mixes/{body['mix_id']}").json() == body
    audio = client.get(body["audio_url"])
    assert audio.status_code == 200
    assert "attachment" in audio.headers["content-disposition"]
    duration = sf.info(io.BytesIO(audio.content)).duration
    assert duration == pytest.approx(manifest["output_duration_seconds"], abs=0.05)
    # each seam replaces the outgoing tail and incoming head it consumed
    ts = manifest["transitions"]
    consumed = sum((40.0 - t["outgoing_start_seconds"]) + t["actual_overlap_seconds"] for t in ts)
    seams = sum(t["output_duration_seconds"] for t in ts)
    assert duration == pytest.approx(3 * 40.0 - consumed + seams, abs=0.1)


def test_unknown_mix_404(client):
    assert client.get("/api/mixes/nope").status_code == 404
    assert client.get("/api/mixes/nope/audio").status_code == 404
