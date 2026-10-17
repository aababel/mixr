import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from backend.api.schemas import Preference, TransitionRequest
from backend.main import app


def test_health():
    client = TestClient(app)
    assert client.get("/api/health").json() == {"ok": True}


def test_transition_request_defaults():
    req = TransitionRequest(from_id="a", to_id="b", requested_overlap_seconds=4.0)
    assert req.fade_style == "equal_power"
    assert req.beat_align is True


def test_overlap_must_be_positive():
    with pytest.raises(ValidationError):
        TransitionRequest(from_id="a", to_id="b", requested_overlap_seconds=0)


def test_preference_is_bounded():
    with pytest.raises(ValidationError):
        Preference(overlap_preference=1.5)
