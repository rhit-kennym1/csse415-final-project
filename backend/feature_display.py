"""Map the raw -1/0/1 feature values to human-readable display rows.

Produces the eight items shown on the demo's "Extracted Features" panel,
matching the project slide ("What Makes a URL Look Phishy?"). Each row is
{"label", "value", "signal"} where signal is one of ok / neutral / suspicious
and drives the colour of the indicator in the UI.
"""
from __future__ import annotations


def _url_length(v: int) -> tuple[str, str]:
    return {
        1:  ("Short (under 54 chars)", "ok"),
        0:  ("Medium (54-75 chars)", "neutral"),
        -1: ("Long (over 75 chars)", "suspicious"),
    }.get(v, ("Unknown", "neutral"))


def _ssl(v: int) -> tuple[str, str]:
    return {
        1:  ("Valid certificate (1+ year old)", "ok"),
        0:  ("HTTPS, but a recent/short-lived certificate", "neutral"),
        -1: ("No HTTPS or an untrusted certificate", "suspicious"),
    }.get(v, ("Unknown", "neutral"))


def _suspicious_symbols(f: dict) -> tuple[str, str]:
    detected = []
    if f.get("having_At_Symbol") == -1:
        detected.append("@ symbol")
    if f.get("double_slash_redirecting") == -1:
        detected.append("// redirect")
    if f.get("Prefix_Suffix") == -1:
        detected.append("hyphen in domain")
    if f.get("having_IPhaving_IP_Address") == -1:
        detected.append("IP address")
    if not detected:
        return ("None detected", "ok")
    return (", ".join(detected), "suspicious")


def _url_anchor(v: int) -> tuple[str, str]:
    return {
        1:  ("Mostly internal links", "ok"),
        0:  ("Mix of internal and external links", "neutral"),
        -1: ("Mostly external/empty links", "suspicious"),
    }.get(v, ("Unknown", "neutral"))


def _domain_age(v: int) -> tuple[str, str]:
    return {
        1:  ("Established (6+ months)", "ok"),
        -1: ("New or unknown (under 6 months)", "suspicious"),
    }.get(v, ("Unknown", "neutral"))


def _redirects(v: int) -> tuple[str, str]:
    return {
        1:  ("0-1 redirects", "ok"),
        0:  ("2-3 redirects", "neutral"),
        -1: ("4 or more redirects", "suspicious"),
    }.get(v, ("Unknown", "neutral"))


def _iframes_popups(f: dict) -> tuple[str, str]:
    detected = []
    if f.get("Iframe") == -1:
        detected.append("iframe present")
    if f.get("popUpWidnow") == -1:
        detected.append("pop-up window")
    if not detected:
        return ("None detected", "ok")
    return (", ".join(detected), "suspicious")


def _web_traffic(v: int) -> tuple[str, str]:
    # Always approximated — the Alexa traffic-rank data source was deprecated.
    return ("Not available (data source deprecated)", "neutral")


def build_display(features: dict) -> list[dict]:
    """Build the ordered list of display rows for the UI."""
    url_len_value, url_len_signal = _url_length(features.get("URLURL_Length", 0))
    ssl_value, ssl_signal = _ssl(features.get("SSLfinal_State", 0))
    sym_value, sym_signal = _suspicious_symbols(features)
    anchor_value, anchor_signal = _url_anchor(features.get("URL_of_Anchor", 0))
    age_value, age_signal = _domain_age(features.get("age_of_domain", 0))
    redir_value, redir_signal = _redirects(features.get("Redirect", 0))
    frame_value, frame_signal = _iframes_popups(features)
    traffic_value, traffic_signal = _web_traffic(features.get("web_traffic", 0))

    return [
        {"label": "URL length",         "value": url_len_value, "signal": url_len_signal},
        {"label": "HTTPS / SSL state",  "value": ssl_value,     "signal": ssl_signal},
        {"label": "Suspicious symbols", "value": sym_value,     "signal": sym_signal},
        {"label": "URL Anchor",         "value": anchor_value,  "signal": anchor_signal},
        {"label": "Domain age",         "value": age_value,     "signal": age_signal},
        {"label": "Redirects",          "value": redir_value,   "signal": redir_signal},
        {"label": "iframes & pop-ups",  "value": frame_value,   "signal": frame_signal},
        {"label": "Web traffic rank",   "value": traffic_value, "signal": traffic_signal},
    ]
