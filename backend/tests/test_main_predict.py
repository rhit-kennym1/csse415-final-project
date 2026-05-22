from unittest.mock import patch

from fastapi.testclient import TestClient

import backend.api as api
from backend.main import app


def test_predict_forwards_body_from_core():
    body = {
        "url": "https://example.com",
        "reachable": True,
        "predictions": [{"model": "Gradient Boosting", "verdict": "phishing"}],
        "safe_browsing": {"verdict": "safe", "threat_types": []},
        "features_display": [],
        "features_meta": {"approximated": []},
    }
    with patch.object(api, "run_prediction", return_value=(200, body)):
        client = TestClient(app)
        resp = client.post("/api/predict", json={"url": "https://example.com"})

    assert resp.status_code == 200
    assert resp.json() == body


def test_predict_forwards_error_status_from_core():
    with patch.object(api, "run_prediction",
                      return_value=(400, {"error": "invalid_url", "message": "bad"})):
        client = TestClient(app)
        resp = client.post("/api/predict", json={"url": "whatever"})

    assert resp.status_code == 400
    assert resp.json()["error"] == "invalid_url"


def test_predict_rejects_empty_url_via_validation():
    """Pydantic rejects an empty url before it reaches the core."""
    client = TestClient(app)
    resp = client.post("/api/predict", json={"url": ""})
    assert resp.status_code == 422
