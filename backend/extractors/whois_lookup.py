"""WHOIS-based feature extractors. Caller fetches the record once and reuses it."""
from __future__ import annotations

import socket
from datetime import datetime
from typing import Any, Optional

import whois

_REGISTRATION_LONG_DAYS = 365
_DOMAIN_OLD_DAYS = 180  # ~6 months
_TIMEOUT_SECONDS = 5


def fetch_whois(host: str) -> Optional[Any]:
    """Best-effort WHOIS lookup. Returns None on any failure."""
    socket.setdefaulttimeout(_TIMEOUT_SECONDS)
    try:
        return whois.whois(host)
    except Exception:
        return None


def _first(value):
    """python-whois sometimes returns lists; take the first non-None entry."""
    if isinstance(value, list):
        for v in value:
            if v is not None:
                return v
        return None
    return value


def domain_registration_length(rec: Any) -> int:
    if rec is None:
        return -1
    expiry = _first(rec.expiration_date)
    if not isinstance(expiry, datetime):
        return -1
    days_left = (expiry - datetime.utcnow()).days
    return 1 if days_left >= _REGISTRATION_LONG_DAYS else -1


def age_of_domain(rec: Any) -> int:
    if rec is None:
        return -1
    created = _first(rec.creation_date)
    if not isinstance(created, datetime):
        return -1
    age_days = (datetime.utcnow() - created).days
    return 1 if age_days >= _DOMAIN_OLD_DAYS else -1


def abnormal_url(rec: Any, host: str) -> int:
    if rec is None:
        return -1
    names = rec.domain_name
    if names is None:
        return -1
    if isinstance(names, str):
        names = [names]
    host_l = host.lower()
    if any(host_l in (n or "").lower() or (n or "").lower() in host_l for n in names):
        return 1
    return -1
