"""URL-only feature extractors. No network calls; pure string analysis."""
from __future__ import annotations

import re
from urllib.parse import urlparse

_IP_PATTERN = re.compile(
    r"^(?:\d{1,3}\.){3}\d{1,3}$|^(?:0x[0-9a-f]+\.){3}0x[0-9a-f]+$",
    re.IGNORECASE,
)

_SHORTENERS = {
    "bit.ly", "goo.gl", "tinyurl.com", "ow.ly", "t.co", "is.gd",
    "buff.ly", "adf.ly", "bit.do", "soo.gd", "cutt.ly", "rebrand.ly",
    "shorturl.at", "tiny.cc", "rb.gy", "lnkd.in", "youtu.be",
}

_STANDARD_PORTS = {None, 80, 443}


def _hostname(url: str) -> str:
    return (urlparse(url).hostname or "").lower()


def having_ip_address(url: str) -> int:
    host = _hostname(url)
    return -1 if _IP_PATTERN.match(host) else 1


def url_length(url: str) -> int:
    n = len(url)
    if n < 54:
        return 1
    if n <= 75:
        return 0
    return -1


def shortening_service(url: str) -> int:
    return -1 if _hostname(url) in _SHORTENERS else 1


def having_at_symbol(url: str) -> int:
    return -1 if "@" in url else 1


def double_slash_redirecting(url: str) -> int:
    # Ignore the // in the scheme (positions ~5-7). A later // is suspicious.
    last_double_slash = url.rfind("//")
    return -1 if last_double_slash > 7 else 1


def prefix_suffix(url: str) -> int:
    return -1 if "-" in _hostname(url) else 1


def having_sub_domain(url: str) -> int:
    host = _hostname(url)
    if host.startswith("www."):
        host = host[4:]
    # Strip the TLD by removing the last dot-separated segment
    parts = host.split(".")
    if len(parts) <= 2:
        return 1
    subdomain_dots = len(parts) - 2
    if subdomain_dots == 1:
        return 0
    return -1


def https_token(url: str) -> int:
    host = _hostname(url)
    return -1 if "https" in host else 1


def port(url: str) -> int:
    p = urlparse(url).port
    return 1 if p in _STANDARD_PORTS else -1
