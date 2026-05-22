"""Top-level feature extraction: URL → dict of all 30 features."""
from __future__ import annotations

import logging
from urllib.parse import urlparse

import requests
from bs4 import BeautifulSoup

from backend.extractors.url_only import (
    having_ip_address, url_length, shortening_service, having_at_symbol,
    double_slash_redirecting, prefix_suffix, having_sub_domain,
    https_token, port,
)
from backend.extractors.html_based import (
    favicon, request_url, url_of_anchor, links_in_tags, sfh,
    submitting_to_email, iframe, right_click, popup_window, on_mouseover,
)
from backend.extractors.ssl_probe import ssl_final_state
from backend.extractors.whois_lookup import (
    fetch_whois, domain_registration_length, age_of_domain, abnormal_url,
)
from backend.extractors.dns_lookup import dns_record
from backend.features import FEATURE_NAMES, APPROXIMATED_FEATURES

log = logging.getLogger(__name__)

_USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
    "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
)
_HTTP_TIMEOUT = 5


def _redirect_score(n_hops: int) -> int:
    if n_hops <= 1:
        return 1
    if n_hops <= 3:
        return 0
    return -1


def _safe(fn, default, *args, **kwargs):
    try:
        return fn(*args, **kwargs)
    except Exception as e:
        log.debug("extractor %s failed: %s", fn.__name__, e)
        return default


def extract_features(url: str) -> tuple[dict[str, int], dict]:
    """Run all extractors. Any single failure falls back to neutral 0.

    Returns a (features, diagnostics) tuple. diagnostics["reachable"] is True
    when the page returned an HTTP response; False when the site could not be
    reached at all (DNS failure or connection error) — i.e. it likely does
    not exist.
    """
    host = (urlparse(url).hostname or "").lower()
    features: dict[str, int] = {}

    # URL-only — no network
    features["having_IPhaving_IP_Address"] = _safe(having_ip_address, 0, url)
    features["URLURL_Length"] = _safe(url_length, 0, url)
    features["Shortining_Service"] = _safe(shortening_service, 0, url)
    features["having_At_Symbol"] = _safe(having_at_symbol, 0, url)
    features["double_slash_redirecting"] = _safe(double_slash_redirecting, 0, url)
    features["Prefix_Suffix"] = _safe(prefix_suffix, 0, url)
    features["having_Sub_Domain"] = _safe(having_sub_domain, 0, url)
    features["HTTPS_token"] = _safe(https_token, 0, url)
    features["port"] = _safe(port, 0, url)

    # HTML fetch
    doc = None
    redirect_hops = 0
    try:
        resp = requests.get(
            url,
            timeout=_HTTP_TIMEOUT,
            headers={"User-Agent": _USER_AGENT},
            allow_redirects=True,
        )
        doc = BeautifulSoup(resp.text, "lxml")
        redirect_hops = len(resp.history)
    except Exception as e:
        log.debug("HTTP fetch failed for %s: %s", url, e)

    if doc is not None:
        features["Favicon"] = _safe(favicon, 0, doc, host)
        features["Request_URL"] = _safe(request_url, 0, doc, host)
        features["URL_of_Anchor"] = _safe(url_of_anchor, 0, doc, host)
        features["Links_in_tags"] = _safe(links_in_tags, 0, doc, host)
        features["SFH"] = _safe(sfh, 0, doc, host)
        features["Submitting_to_email"] = _safe(submitting_to_email, 0, doc)
        features["Iframe"] = _safe(iframe, 0, doc)
        features["RightClick"] = _safe(right_click, 0, doc)
        features["popUpWidnow"] = _safe(popup_window, 0, doc)
        features["on_mouseover"] = _safe(on_mouseover, 0, doc)
    else:
        for name in (
            "Favicon", "Request_URL", "URL_of_Anchor", "Links_in_tags",
            "SFH", "Submitting_to_email", "Iframe", "RightClick",
            "popUpWidnow", "on_mouseover",
        ):
            features[name] = 0

    features["Redirect"] = _redirect_score(redirect_hops)

    # SSL
    features["SSLfinal_State"] = _safe(ssl_final_state, 0, url)

    # WHOIS
    rec = fetch_whois(host) if host else None
    features["Domain_registeration_length"] = _safe(domain_registration_length, 0, rec)
    features["age_of_domain"] = _safe(age_of_domain, 0, rec)
    features["Abnormal_URL"] = _safe(abnormal_url, 0, rec, host)

    # DNS
    features["DNSRecord"] = _safe(dns_record, 0, host) if host else 0

    # Approximated
    for name in APPROXIMATED_FEATURES:
        features[name] = 0

    # Safety net: ensure every feature name is present
    for name in FEATURE_NAMES:
        features.setdefault(name, 0)

    diagnostics = {"reachable": doc is not None}
    return features, diagnostics
