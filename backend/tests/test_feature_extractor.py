from unittest.mock import patch, MagicMock

from backend.feature_extractor import extract_features
from backend.features import FEATURE_NAMES, APPROXIMATED_FEATURES


def _mock_response(html: str = "<html></html>", url: str = "https://example.com/"):
    resp = MagicMock()
    resp.text = html
    resp.url = url
    resp.status_code = 200
    resp.history = []
    return resp


def test_extract_features_returns_all_keys():
    with patch("backend.feature_extractor.requests.get") as get, \
         patch("backend.feature_extractor.fetch_whois", return_value=None), \
         patch("backend.feature_extractor.dns_record", return_value=1), \
         patch("backend.feature_extractor.ssl_final_state", return_value=1):
        get.return_value = _mock_response()
        result = extract_features("https://example.com/")
    assert set(result.keys()) == set(FEATURE_NAMES)


def test_extract_features_approximated_are_zero():
    with patch("backend.feature_extractor.requests.get") as get, \
         patch("backend.feature_extractor.fetch_whois", return_value=None), \
         patch("backend.feature_extractor.dns_record", return_value=1), \
         patch("backend.feature_extractor.ssl_final_state", return_value=1):
        get.return_value = _mock_response()
        result = extract_features("https://example.com/")
    for name in APPROXIMATED_FEATURES:
        assert result[name] == 0


def test_extract_features_http_failure_does_not_raise():
    """Unreachable site still returns a complete feature dict."""
    with patch("backend.feature_extractor.requests.get") as get, \
         patch("backend.feature_extractor.fetch_whois", return_value=None), \
         patch("backend.feature_extractor.dns_record", return_value=-1), \
         patch("backend.feature_extractor.ssl_final_state", return_value=-1):
        get.side_effect = Exception("connection refused")
        result = extract_features("https://nonexistent.invalid/")
    assert set(result.keys()) == set(FEATURE_NAMES)
    # Network-fetched HTML features must still be present
    assert "Iframe" in result


def test_extract_features_redirect_count():
    """Redirect feature uses the response history length."""
    resp = _mock_response()
    resp.history = [MagicMock(), MagicMock(), MagicMock(), MagicMock()]  # 4 hops
    with patch("backend.feature_extractor.requests.get", return_value=resp), \
         patch("backend.feature_extractor.fetch_whois", return_value=None), \
         patch("backend.feature_extractor.dns_record", return_value=1), \
         patch("backend.feature_extractor.ssl_final_state", return_value=1):
        result = extract_features("https://example.com/")
    assert result["Redirect"] == -1  # >=4 hops
