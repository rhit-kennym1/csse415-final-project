"""Thin Google Safe Browsing v4 client."""
from __future__ import annotations

import os
from typing import Optional

import requests

_ENDPOINT = "https://safebrowsing.googleapis.com/v4/threatMatches:find"
_TIMEOUT_SECONDS = 5

_THREAT_TO_VERDICT = {
    "SOCIAL_ENGINEERING": "phishing",
    "MALWARE": "malware",
    "UNWANTED_SOFTWARE": "unwanted",
    "POTENTIALLY_HARMFUL_APPLICATION": "unwanted",
}


def check_url(url: str, api_key: Optional[str] = None) -> dict:
    if api_key is None:
        api_key = os.environ.get("SAFE_BROWSING_API_KEY")
    if not api_key:
        return {"verdict": "unknown", "threat_types": [], "error": "no_api_key"}

    payload = {
        "client": {"clientId": "csse415-phishing-demo", "clientVersion": "1.0.0"},
        "threatInfo": {
            "threatTypes": [
                "MALWARE", "SOCIAL_ENGINEERING",
                "UNWANTED_SOFTWARE", "POTENTIALLY_HARMFUL_APPLICATION",
            ],
            "platformTypes": ["ANY_PLATFORM"],
            "threatEntryTypes": ["URL"],
            "threatEntries": [{"url": url}],
        },
    }
    try:
        resp = requests.post(
            _ENDPOINT, params={"key": api_key},
            json=payload, timeout=_TIMEOUT_SECONDS,
        )
        resp.raise_for_status()
    except requests.RequestException as e:
        return {"verdict": "unknown", "threat_types": [], "error": str(e)}

    matches = resp.json().get("matches", [])
    if not matches:
        return {"verdict": "safe", "threat_types": []}

    threat_types = [m["threatType"] for m in matches]
    # Prefer phishing > malware > unwanted in the surfaced verdict
    for tt in ("SOCIAL_ENGINEERING", "MALWARE", "UNWANTED_SOFTWARE", "POTENTIALLY_HARMFUL_APPLICATION"):
        if tt in threat_types:
            return {"verdict": _THREAT_TO_VERDICT[tt], "threat_types": threat_types}

    return {"verdict": "unknown", "threat_types": threat_types}
