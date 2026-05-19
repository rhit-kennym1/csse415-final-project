import pytest
import responses

from backend.safe_browsing import check_url


@responses.activate
def test_safe_url_returns_safe():
    responses.add(
        responses.POST,
        "https://safebrowsing.googleapis.com/v4/threatMatches:find",
        json={},
        status=200,
    )
    result = check_url("https://example.com", api_key="fake")
    assert result == {"verdict": "safe", "threat_types": []}


@responses.activate
def test_phishing_url_returns_phishing():
    responses.add(
        responses.POST,
        "https://safebrowsing.googleapis.com/v4/threatMatches:find",
        json={
            "matches": [{"threatType": "SOCIAL_ENGINEERING"}]
        },
        status=200,
    )
    result = check_url("https://phish.example", api_key="fake")
    assert result["verdict"] == "phishing"
    assert "SOCIAL_ENGINEERING" in result["threat_types"]


@responses.activate
def test_malware_threat():
    responses.add(
        responses.POST,
        "https://safebrowsing.googleapis.com/v4/threatMatches:find",
        json={"matches": [{"threatType": "MALWARE"}]},
        status=200,
    )
    result = check_url("https://malware.example", api_key="fake")
    assert result["verdict"] == "malware"


def test_missing_api_key_returns_unknown(monkeypatch):
    # Other tests in the suite may load a real key from .env via main.py's
    # load_dotenv() call; clear it so we're testing the no-key path.
    monkeypatch.delenv("SAFE_BROWSING_API_KEY", raising=False)
    result = check_url("https://example.com", api_key=None)
    assert result["verdict"] == "unknown"
    assert "error" in result


@responses.activate
def test_api_error_returns_unknown():
    responses.add(
        responses.POST,
        "https://safebrowsing.googleapis.com/v4/threatMatches:find",
        status=500,
    )
    result = check_url("https://example.com", api_key="fake")
    assert result["verdict"] == "unknown"
    assert "error" in result
