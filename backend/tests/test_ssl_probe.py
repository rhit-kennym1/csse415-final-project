from datetime import datetime, timedelta, timezone
from unittest.mock import patch

from backend.extractors.ssl_probe import ssl_final_state


def _cert_with_notbefore(days_ago: int):
    not_before = datetime.now(timezone.utc) - timedelta(days=days_ago)
    return {"notBefore": not_before.strftime("%b %d %H:%M:%S %Y GMT")}


def test_ssl_http_url_returns_minus_one():
    assert ssl_final_state("http://example.com") == -1


def test_ssl_https_old_cert_returns_one():
    with patch("backend.extractors.ssl_probe._fetch_cert") as m:
        m.return_value = _cert_with_notbefore(days_ago=400)
        assert ssl_final_state("https://example.com") == 1


def test_ssl_https_new_cert_returns_zero():
    with patch("backend.extractors.ssl_probe._fetch_cert") as m:
        m.return_value = _cert_with_notbefore(days_ago=30)
        assert ssl_final_state("https://example.com") == 0


def test_ssl_https_handshake_fails_returns_minus_one():
    with patch("backend.extractors.ssl_probe._fetch_cert") as m:
        m.side_effect = OSError("handshake failed")
        assert ssl_final_state("https://example.com") == -1
