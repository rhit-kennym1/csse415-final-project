"""SSL certificate probe for SSLfinal_State feature."""
from __future__ import annotations

import socket
import ssl
from datetime import datetime, timezone
from urllib.parse import urlparse

_TIMEOUT_SECONDS = 5
_OLD_CERT_DAYS = 365


def _fetch_cert(host: str, port: int) -> dict:
    ctx = ssl.create_default_context()
    with socket.create_connection((host, port), timeout=_TIMEOUT_SECONDS) as sock:
        with ctx.wrap_socket(sock, server_hostname=host) as ssock:
            return ssock.getpeercert()


def ssl_final_state(url: str) -> int:
    parsed = urlparse(url)
    if parsed.scheme != "https":
        return -1
    host = parsed.hostname
    if not host:
        return -1
    port = parsed.port or 443
    try:
        cert = _fetch_cert(host, port)
    except (OSError, ssl.SSLError, socket.timeout):
        return -1

    not_before_str = cert.get("notBefore")
    if not not_before_str:
        return 0

    try:
        not_before = datetime.strptime(not_before_str, "%b %d %H:%M:%S %Y GMT")
        not_before = not_before.replace(tzinfo=timezone.utc)
    except ValueError:
        return 0

    age_days = (datetime.now(timezone.utc) - not_before).days
    return 1 if age_days >= _OLD_CERT_DAYS else 0
