from unittest.mock import patch

from fastapi.testclient import TestClient

import backend.main as main_mod
from backend.main import app


def test_predict_returns_full_response_shape():
    fake_features = {name: 0 for name in main_mod.FEATURE_NAMES}
    fake_safe = {"verdict": "safe", "threat_types": []}

    with patch.object(main_mod, "extract_features",
                      return_value=(fake_features, {"reachable": True})), \
         patch.object(main_mod, "check_url", return_value=fake_safe), \
         patch.object(main_mod, "_predict_with_models", return_value=[
             {"model": "Gradient Boosting", "verdict": "phishing"},
             {"model": "Random Forest",     "verdict": "phishing"},
             {"model": "XGBoost",           "verdict": "legitimate"},
             {"model": "SVM",               "verdict": "phishing"},
         ]):
        client = TestClient(app)
        resp = client.post("/api/predict", json={"url": "https://example.com"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["url"] == "https://example.com"
    assert body["reachable"] is True
    assert len(body["predictions"]) == 4
    assert {p["model"] for p in body["predictions"]} == {
        "Gradient Boosting", "Random Forest", "XGBoost", "SVM",
    }
    assert body["safe_browsing"] == fake_safe
    assert "approximated" in body["features_meta"]
    # The eight slide items are surfaced for display
    assert len(body["features_display"]) == 8
    assert {"label", "value", "signal"} <= set(body["features_display"][0].keys())


def test_predict_rejects_malformed_url():
    client = TestClient(app)
    resp = client.post("/api/predict", json={"url": "not-a-url"})
    assert resp.status_code == 400
    assert resp.json()["error"] == "invalid_url"


def test_predict_unreachable_site_hides_predictions():
    fake_features = {name: 0 for name in main_mod.FEATURE_NAMES}

    with patch.object(main_mod, "extract_features",
                      return_value=(fake_features, {"reachable": False})), \
         patch.object(main_mod, "check_url", return_value={"verdict": "safe", "threat_types": []}), \
         patch.object(main_mod, "_predict_with_models") as predict_mock:
        client = TestClient(app)
        resp = client.post("/api/predict", json={"url": "https://nonexistent.invalid"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["reachable"] is False
    assert "predictions" not in body
    predict_mock.assert_not_called()


def test_predict_safe_browsing_failure_still_returns_predictions():
    fake_features = {name: 0 for name in main_mod.FEATURE_NAMES}
    fake_safe = {"verdict": "unknown", "threat_types": [], "error": "no_api_key"}

    with patch.object(main_mod, "extract_features",
                      return_value=(fake_features, {"reachable": True})), \
         patch.object(main_mod, "check_url", return_value=fake_safe), \
         patch.object(main_mod, "_predict_with_models", return_value=[
             {"model": "Gradient Boosting", "verdict": "legitimate"},
         ] * 4):
        client = TestClient(app)
        resp = client.post("/api/predict", json={"url": "https://example.com"})

    assert resp.status_code == 200
    assert resp.json()["safe_browsing"]["verdict"] == "unknown"
