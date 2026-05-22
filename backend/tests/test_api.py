from unittest.mock import patch

import backend.api as api
from backend.features import FEATURE_NAMES


def test_validate_url_accepts_http_and_https():
    assert api.validate_url("https://example.com") is None
    assert api.validate_url("http://example.com/path") is None


def test_validate_url_rejects_non_http_scheme():
    assert api.validate_url("ftp://example.com") is not None
    assert api.validate_url("not-a-url") is not None


def test_validate_url_rejects_missing_host():
    assert api.validate_url("https://") is not None


def test_run_prediction_invalid_url_returns_400():
    status, body = api.run_prediction("not-a-url")
    assert status == 400
    assert body["error"] == "invalid_url"


def test_run_prediction_reachable_returns_full_body():
    fake_features = {name: 0 for name in FEATURE_NAMES}
    fake_safe = {"verdict": "safe", "threat_types": []}
    preds = [
        {"model": "Gradient Boosting", "verdict": "phishing"},
        {"model": "Random Forest", "verdict": "phishing"},
        {"model": "XGBoost", "verdict": "legitimate"},
        {"model": "SVM", "verdict": "phishing"},
    ]
    with patch.object(api, "extract_features",
                      return_value=(fake_features, {"reachable": True})), \
         patch.object(api, "check_url", return_value=fake_safe), \
         patch.object(api, "predict_with_models", return_value=preds):
        status, body = api.run_prediction("https://example.com")

    assert status == 200
    assert body["reachable"] is True
    assert body["url"] == "https://example.com"
    assert len(body["predictions"]) == 4
    assert body["safe_browsing"] == fake_safe
    assert len(body["features_display"]) == 8
    assert "approximated" in body["features_meta"]


def test_run_prediction_unreachable_hides_predictions():
    fake_features = {name: 0 for name in FEATURE_NAMES}
    with patch.object(api, "extract_features",
                      return_value=(fake_features, {"reachable": False})), \
         patch.object(api, "check_url", return_value={"verdict": "safe", "threat_types": []}), \
         patch.object(api, "predict_with_models") as predict_mock:
        status, body = api.run_prediction("https://nonexistent.invalid")

    assert status == 200
    assert body["reachable"] is False
    assert "predictions" not in body
    predict_mock.assert_not_called()
