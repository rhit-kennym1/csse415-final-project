"""HTML-document feature extractors. Each takes a parsed BeautifulSoup doc."""
from __future__ import annotations

from urllib.parse import urlparse

from bs4 import BeautifulSoup


def _is_same_domain(url: str, host: str) -> bool:
    """True if url is relative or points to the same host."""
    if not url:
        return True  # treat empty as relative/internal
    parsed = urlparse(url)
    if not parsed.netloc:
        return True  # relative URL
    return parsed.netloc.lower().endswith(host.lower())


def _bucketize(pct: float, low: float, high: float) -> int:
    """Map a percentage to {1, 0, -1} via two thresholds."""
    if pct < low:
        return 1
    if pct <= high:
        return 0
    return -1


def favicon(doc: BeautifulSoup, host: str) -> int:
    link = doc.find("link", rel=lambda r: r and "icon" in r.lower())
    if not link or not link.get("href"):
        return 1
    return 1 if _is_same_domain(link["href"], host) else -1


def request_url(doc: BeautifulSoup, host: str) -> int:
    """Percentage of img/audio/video/embed/iframe resources from external domains."""
    tags = doc.find_all(["img", "audio", "video", "embed", "iframe"])
    sources = [t.get("src", "") for t in tags if t.get("src")]
    if not sources:
        return 1
    external = sum(1 for s in sources if not _is_same_domain(s, host))
    pct = (external / len(sources)) * 100
    return _bucketize(pct, low=22, high=61)


def url_of_anchor(doc: BeautifulSoup, host: str) -> int:
    """Percentage of <a> tags that point outside the domain or have no real target."""
    anchors = doc.find_all("a")
    if not anchors:
        return 1
    bad = 0
    for a in anchors:
        href = (a.get("href") or "").strip()
        if not href or href in {"#", "#content", "javascript:void(0)"}:
            bad += 1
        elif not _is_same_domain(href, host):
            bad += 1
    pct = (bad / len(anchors)) * 100
    return _bucketize(pct, low=31, high=67)


def links_in_tags(doc: BeautifulSoup, host: str) -> int:
    """Percentage of <meta>/<script>/<link> tags whose URLs are external."""
    tags = doc.find_all(["meta", "script", "link"])
    sources = []
    for t in tags:
        sources.append(t.get("href", "") or t.get("src", ""))
    sources = [s for s in sources if s]
    if not sources:
        return 1
    external = sum(1 for s in sources if not _is_same_domain(s, host))
    pct = (external / len(sources)) * 100
    return _bucketize(pct, low=17, high=81)


def sfh(doc: BeautifulSoup, host: str) -> int:
    """Server Form Handler — checks the action target of forms."""
    forms = doc.find_all("form")
    if not forms:
        return 1
    # If any form has a suspicious action, downgrade the whole page.
    worst = 1
    for form in forms:
        action = (form.get("action") or "").strip()
        if not action or action == "about:blank":
            return -1
        if not _is_same_domain(action, host):
            worst = min(worst, 0)
    return worst


def submitting_to_email(doc: BeautifulSoup) -> int:
    forms = doc.find_all("form")
    for form in forms:
        action = (form.get("action") or "").lower()
        if action.startswith("mailto:"):
            return -1
    # Also check raw HTML for "mail(" PHP-style submissions
    raw = str(doc).lower()
    if "mailto:" in raw and "<form" in raw:
        return -1
    return 1


def iframe(doc: BeautifulSoup) -> int:
    return -1 if doc.find("iframe") else 1


def right_click(doc: BeautifulSoup) -> int:
    raw = str(doc).lower()
    if "contextmenu" in raw and "preventdefault" in raw:
        return -1
    if "event.button" in raw and "2" in raw:
        return -1
    return 1


def popup_window(doc: BeautifulSoup) -> int:
    raw = str(doc).lower()
    return -1 if "window.open(" in raw else 1


def on_mouseover(doc: BeautifulSoup) -> int:
    for tag in doc.find_all(attrs={"onmouseover": True}):
        handler = (tag["onmouseover"] or "").lower()
        if "window.status" in handler:
            return -1
    return 1
