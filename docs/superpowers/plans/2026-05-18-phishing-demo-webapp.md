# Phishing Detection Demo Web App — Implementation Plan

> **For agentic workers:** REQUIRED SUB-SKILL: Use superpowers:subagent-driven-development (recommended) or superpowers:executing-plans to implement this plan task-by-task. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Build an interactive web app that takes a URL, extracts dataset-1 features live, runs predictions through the four best-performing models from the project, and compares them with Google Safe Browsing — deployed on Firebase Hosting (frontend) + Cloud Run (Python backend).

**Architecture:** Static HTML/JS frontend served by Firebase Hosting calls a single `POST /api/predict` endpoint on a Cloud Run FastAPI service. The backend extracts ~26 features live (5 are defaulted to neutral 0), loads four pre-trained sklearn pipelines from joblib, and runs the URL through them in parallel with a Google Safe Browsing v4 lookup.

**Tech Stack:** Python 3.11, FastAPI, scikit-learn, xgboost, joblib, requests, BeautifulSoup4, python-whois, dnspython, vanilla HTML/JS/CSS, Docker, Google Cloud Run, Firebase Hosting.

---

## Reference: Dataset Feature Encoding

The 30 features (target column `Result` is `-1` legitimate / `1` phishing) follow this canonical order, matching `dataset1.csv` columns with whitespace stripped:

```
having_IPhaving_IP_Address, URLURL_Length, Shortining_Service, having_At_Symbol,
double_slash_redirecting, Prefix_Suffix, having_Sub_Domain, SSLfinal_State,
Domain_registeration_length, Favicon, port, HTTPS_token, Request_URL,
URL_of_Anchor, Links_in_tags, SFH, Submitting_to_email, Abnormal_URL, Redirect,
on_mouseover, RightClick, popUpWidnow, Iframe, age_of_domain, DNSRecord,
web_traffic, Page_Rank, Google_Index, Links_pointing_to_page, Statistical_report
```

Each feature value is one of `{-1, 0, 1}`.

---

## Reference: Per-Model Preprocessing (from notebooks)

| Model | Preprocessing | Hyperparameters |
|---|---|---|
| Gradient Boosting (`GradientBoosting.ipynb`) | `StandardScaler` → `PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)` | `learning_rate=0.1, max_depth=5, n_estimators=200, random_state=42` |
| Random Forest (`RandForest.ipynb`) | `PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)` | `max_depth=30, max_features=10, n_estimators=125, random_state=42, n_jobs=-1` |
| XGBoost (`415XGBoost.ipynb`) | `StandardScaler` → `PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)` | `learning_rate=0.1, max_depth=5, n_estimators=200, random_state=42, eval_metric='logloss'` |
| SVM (`SVM.ipynb`) | `StandardScaler` → `PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)` → `StandardScaler` | `C=10, gamma=0.1, kernel='rbf', probability=True, random_state=42` |

SVM needs `probability=True` so we can call `predict_proba` at inference time.

---

## Task 1: Project Scaffolding

**Files:**
- Create: `backend/requirements.txt`
- Create: `backend/requirements-dev.txt`
- Create: `backend/.gitignore`
- Create: `backend/models/.gitkeep`
- Create: `backend/tests/__init__.py`
- Create: `frontend/.gitkeep`
- Modify: `.gitignore` (project root — add `backend/models/*.joblib`)

- [ ] **Step 1: Create backend layout and add Python deps**

`backend/requirements.txt`:
```
fastapi==0.115.0
uvicorn[standard]==0.30.6
requests==2.32.3
beautifulsoup4==4.12.3
lxml==5.3.0
python-whois==0.9.4
dnspython==2.6.1
scikit-learn==1.5.2
xgboost==2.1.1
joblib==1.4.2
numpy==2.1.1
pandas==2.2.3
pydantic==2.9.2
```

`backend/requirements-dev.txt`:
```
-r requirements.txt
pytest==8.3.3
pytest-asyncio==0.24.0
httpx==0.27.2
responses==0.25.3
```

`backend/.gitignore`:
```
__pycache__/
*.pyc
models/*.joblib
.venv/
.pytest_cache/
```

`backend/models/.gitkeep`: (empty file)

`backend/tests/__init__.py`: (empty file)

`frontend/.gitkeep`: (empty file)

Append to project root `.gitignore`:
```
# Trained models — large binaries, regenerated locally before deploy
backend/models/*.joblib
```

- [ ] **Step 2: Verify Python deps install**

```bash
cd backend && python -m venv .venv && .venv/Scripts/activate && pip install -r requirements-dev.txt
```

Expected: all packages install without conflicts. Confirm `python -c "import fastapi, sklearn, xgboost, whois, dns.resolver, bs4; print('ok')"` prints `ok`.

- [ ] **Step 3: Commit**

```bash
git add backend/requirements.txt backend/requirements-dev.txt backend/.gitignore backend/models/.gitkeep backend/tests/__init__.py frontend/.gitkeep .gitignore
git commit -m "chore: scaffold backend and frontend directories"
```

---

## Task 2: Feature Constants Module

**Files:**
- Create: `backend/features.py`
- Create: `backend/tests/test_features_module.py`

- [ ] **Step 1: Write the failing test**

`backend/tests/test_features_module.py`:
```python
from backend.features import FEATURE_NAMES, APPROXIMATED_FEATURES, to_vector


def test_feature_names_count_and_order():
    assert len(FEATURE_NAMES) == 30
    assert FEATURE_NAMES[0] == "having_IPhaving_IP_Address"
    assert FEATURE_NAMES[-1] == "Statistical_report"


def test_approximated_features_are_subset():
    assert set(APPROXIMATED_FEATURES) <= set(FEATURE_NAMES)
    assert set(APPROXIMATED_FEATURES) == {
        "web_traffic", "Page_Rank", "Google_Index",
        "Links_pointing_to_page", "Statistical_report",
    }


def test_to_vector_preserves_order():
    sample = {name: 0 for name in FEATURE_NAMES}
    sample["having_IPhaving_IP_Address"] = -1
    sample["Statistical_report"] = 1
    vec = to_vector(sample)
    assert vec[0] == -1
    assert vec[-1] == 1
    assert len(vec) == 30
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_features_module.py -v`
Expected: FAIL with `ModuleNotFoundError: No module named 'backend.features'`

- [ ] **Step 3: Write the module**

`backend/features.py`:
```python
"""Canonical feature names and helpers for the phishing model input vector."""
from __future__ import annotations

FEATURE_NAMES: list[str] = [
    "having_IPhaving_IP_Address",
    "URLURL_Length",
    "Shortining_Service",
    "having_At_Symbol",
    "double_slash_redirecting",
    "Prefix_Suffix",
    "having_Sub_Domain",
    "SSLfinal_State",
    "Domain_registeration_length",
    "Favicon",
    "port",
    "HTTPS_token",
    "Request_URL",
    "URL_of_Anchor",
    "Links_in_tags",
    "SFH",
    "Submitting_to_email",
    "Abnormal_URL",
    "Redirect",
    "on_mouseover",
    "RightClick",
    "popUpWidnow",
    "Iframe",
    "age_of_domain",
    "DNSRecord",
    "web_traffic",
    "Page_Rank",
    "Google_Index",
    "Links_pointing_to_page",
    "Statistical_report",
]

APPROXIMATED_FEATURES: list[str] = [
    "web_traffic",
    "Page_Rank",
    "Google_Index",
    "Links_pointing_to_page",
    "Statistical_report",
]


def to_vector(features: dict[str, int]) -> list[int]:
    """Convert a feature dict to a list in canonical order."""
    return [features[name] for name in FEATURE_NAMES]
```

Also create `backend/__init__.py` (empty file) so `from backend.features import …` works.

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/test_features_module.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/__init__.py backend/features.py backend/tests/test_features_module.py
git commit -m "feat(backend): add feature name constants and vector helper"
```

---

## Task 3: URL-Only Feature Extractors

**Files:**
- Create: `backend/extractors/__init__.py`
- Create: `backend/extractors/url_only.py`
- Create: `backend/tests/test_url_only.py`

These 9 features need no network calls — they're pure functions of the URL string. Encoding rules follow the UCI Phishing Websites Dataset.

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_url_only.py`:
```python
from backend.extractors.url_only import (
    having_ip_address,
    url_length,
    shortening_service,
    having_at_symbol,
    double_slash_redirecting,
    prefix_suffix,
    having_sub_domain,
    https_token,
    port,
)


def test_having_ip_address_with_ip():
    assert having_ip_address("http://125.98.3.123/login") == -1


def test_having_ip_address_with_domain():
    assert having_ip_address("https://example.com/login") == 1


def test_url_length_short():
    assert url_length("https://a.com") == 1


def test_url_length_medium():
    assert url_length("https://" + "a" * 60 + ".com") == 0


def test_url_length_long():
    assert url_length("https://" + "a" * 100 + ".com") == -1


def test_shortening_service_bitly():
    assert shortening_service("https://bit.ly/abc") == -1


def test_shortening_service_normal():
    assert shortening_service("https://example.com/long/path") == 1


def test_having_at_symbol_present():
    assert having_at_symbol("https://example.com/@evil") == -1


def test_having_at_symbol_absent():
    assert having_at_symbol("https://example.com/path") == 1


def test_double_slash_redirecting_late():
    # // appears after position 7 (the scheme's //), so this is suspicious
    assert double_slash_redirecting("http://example.com//evil") == -1


def test_double_slash_redirecting_only_scheme():
    assert double_slash_redirecting("https://example.com/path") == 1


def test_prefix_suffix_with_dash():
    assert prefix_suffix("https://my-bank.com") == -1


def test_prefix_suffix_without_dash():
    assert prefix_suffix("https://mybank.com") == 1


def test_having_sub_domain_one_dot():
    assert having_sub_domain("https://example.com") == 1


def test_having_sub_domain_two_dots():
    assert having_sub_domain("https://www.example.com") == 1  # www stripped


def test_having_sub_domain_many_subdomains():
    assert having_sub_domain("https://a.b.c.example.com") == -1


def test_https_token_in_path_ok():
    assert https_token("https://example.com/https/path") == 1


def test_https_token_in_domain_suspicious():
    assert https_token("http://https-example.com/path") == -1


def test_port_standard():
    assert port("https://example.com") == 1


def test_port_nonstandard():
    assert port("http://example.com:8080/path") == -1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && pytest tests/test_url_only.py -v`
Expected: collection error or all fail with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the module**

`backend/extractors/__init__.py`: (empty file)

`backend/extractors/url_only.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && pytest tests/test_url_only.py -v`
Expected: 20 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/extractors/__init__.py backend/extractors/url_only.py backend/tests/test_url_only.py
git commit -m "feat(backend): add URL-only feature extractors"
```

---

## Task 4: HTML-Based Feature Extractors

**Files:**
- Create: `backend/extractors/html_based.py`
- Create: `backend/tests/test_html_based.py`

These 11 features come from parsing the fetched HTML. The fetch happens once and the parsed document is shared across extractors.

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_html_based.py`:
```python
from urllib.parse import urlparse
from bs4 import BeautifulSoup

from backend.extractors.html_based import (
    favicon,
    request_url,
    url_of_anchor,
    links_in_tags,
    sfh,
    submitting_to_email,
    iframe,
    right_click,
    popup_window,
    on_mouseover,
)


def _parse(html: str):
    return BeautifulSoup(html, "lxml")


def test_favicon_same_domain():
    doc = _parse('<link rel="icon" href="/favicon.ico">')
    assert favicon(doc, "example.com") == 1


def test_favicon_external_domain():
    doc = _parse('<link rel="icon" href="https://evil.com/favicon.ico">')
    assert favicon(doc, "example.com") == -1


def test_favicon_missing():
    doc = _parse("<html></html>")
    assert favicon(doc, "example.com") == 1  # absent = neutral=ok per UCI


def test_request_url_all_internal():
    doc = _parse('<img src="/a.png"><img src="https://example.com/b.png">')
    assert request_url(doc, "example.com") == 1


def test_request_url_mostly_external():
    doc = _parse(
        '<img src="https://cdn1.com/a.png">'
        '<img src="https://cdn2.com/b.png">'
        '<img src="https://cdn3.com/c.png">'
    )
    assert request_url(doc, "example.com") == -1


def test_url_of_anchor_all_internal():
    doc = _parse('<a href="/page">x</a><a href="/page2">y</a>')
    assert url_of_anchor(doc, "example.com") == 1


def test_url_of_anchor_all_external():
    doc = _parse('<a href="https://evil.com/a">x</a><a href="#">y</a>')
    assert url_of_anchor(doc, "example.com") == -1


def test_links_in_tags_internal():
    doc = _parse('<link href="/style.css"><script src="/app.js"></script>')
    assert links_in_tags(doc, "example.com") == 1


def test_sfh_empty():
    doc = _parse('<form action=""></form>')
    assert sfh(doc, "example.com") == -1


def test_sfh_about_blank():
    doc = _parse('<form action="about:blank"></form>')
    assert sfh(doc, "example.com") == -1


def test_sfh_external():
    doc = _parse('<form action="https://evil.com/submit"></form>')
    assert sfh(doc, "example.com") == 0


def test_sfh_same_domain():
    doc = _parse('<form action="/submit"></form>')
    assert sfh(doc, "example.com") == 1


def test_submitting_to_email_mailto():
    doc = _parse('<form action="mailto:foo@bar.com"></form>')
    assert submitting_to_email(doc) == -1


def test_submitting_to_email_none():
    doc = _parse('<form action="/submit"></form>')
    assert submitting_to_email(doc) == 1


def test_iframe_present():
    doc = _parse('<iframe src="https://evil.com"></iframe>')
    assert iframe(doc) == -1


def test_iframe_absent():
    doc = _parse("<div>hi</div>")
    assert iframe(doc) == 1


def test_right_click_disabled():
    doc = _parse(
        "<script>document.addEventListener('contextmenu', e => e.preventDefault())</script>"
    )
    assert right_click(doc) == -1


def test_right_click_normal():
    doc = _parse("<script>console.log('hi')</script>")
    assert right_click(doc) == 1


def test_popup_window_present():
    doc = _parse("<script>window.open('http://evil.com', 'popup')</script>")
    assert popup_window(doc) == -1


def test_popup_window_absent():
    doc = _parse("<script>console.log('hi')</script>")
    assert popup_window(doc) == 1


def test_on_mouseover_changes_status():
    doc = _parse('<a onmouseover="window.status=\'bank.com\'">x</a>')
    assert on_mouseover(doc) == -1


def test_on_mouseover_normal():
    doc = _parse('<a onmouseover="highlight(this)">x</a>')
    assert on_mouseover(doc) == 1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && pytest tests/test_html_based.py -v`
Expected: all fail (`ModuleNotFoundError`).

- [ ] **Step 3: Implement the module**

`backend/extractors/html_based.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && pytest tests/test_html_based.py -v`
Expected: 22 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/extractors/html_based.py backend/tests/test_html_based.py
git commit -m "feat(backend): add HTML-document feature extractors"
```

---

## Task 5: SSL Feature Extractor

**Files:**
- Create: `backend/extractors/ssl_probe.py`
- Create: `backend/tests/test_ssl_probe.py`

`SSLfinal_State`: `1` if HTTPS with valid trusted cert AND cert age ≥ 1 year, `0` if HTTPS with valid cert but < 1 year, `-1` otherwise.

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_ssl_probe.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && pytest tests/test_ssl_probe.py -v`
Expected: all fail (`ModuleNotFoundError`).

- [ ] **Step 3: Implement the module**

`backend/extractors/ssl_probe.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && pytest tests/test_ssl_probe.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/extractors/ssl_probe.py backend/tests/test_ssl_probe.py
git commit -m "feat(backend): add SSL certificate probe extractor"
```

---

## Task 6: WHOIS Feature Extractors

**Files:**
- Create: `backend/extractors/whois_lookup.py`
- Create: `backend/tests/test_whois_lookup.py`

Three features: `Domain_registeration_length`, `age_of_domain`, `Abnormal_URL`. All three share one WHOIS lookup.

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_whois_lookup.py`:
```python
from datetime import datetime, timedelta
from types import SimpleNamespace
from unittest.mock import patch

from backend.extractors.whois_lookup import (
    domain_registration_length,
    age_of_domain,
    abnormal_url,
)


def _whois(
    creation=None, expiration=None, domain_name="example.com"
):
    return SimpleNamespace(
        creation_date=creation,
        expiration_date=expiration,
        domain_name=domain_name,
    )


def test_registration_length_long():
    rec = _whois(expiration=datetime.utcnow() + timedelta(days=400))
    assert domain_registration_length(rec) == 1


def test_registration_length_short():
    rec = _whois(expiration=datetime.utcnow() + timedelta(days=100))
    assert domain_registration_length(rec) == -1


def test_registration_length_missing():
    rec = _whois(expiration=None)
    assert domain_registration_length(rec) == -1


def test_registration_length_list_takes_first():
    """python-whois sometimes returns a list of dates."""
    rec = _whois(
        expiration=[
            datetime.utcnow() + timedelta(days=400),
            datetime.utcnow() + timedelta(days=100),
        ]
    )
    assert domain_registration_length(rec) == 1


def test_age_of_domain_old():
    rec = _whois(creation=datetime.utcnow() - timedelta(days=200))
    assert age_of_domain(rec) == 1


def test_age_of_domain_new():
    rec = _whois(creation=datetime.utcnow() - timedelta(days=30))
    assert age_of_domain(rec) == -1


def test_age_of_domain_missing():
    rec = _whois(creation=None)
    assert age_of_domain(rec) == -1


def test_abnormal_url_host_in_whois():
    rec = _whois(domain_name="example.com")
    assert abnormal_url(rec, "example.com") == 1


def test_abnormal_url_host_not_in_whois():
    rec = _whois(domain_name="other.com")
    assert abnormal_url(rec, "example.com") == -1


def test_abnormal_url_whois_missing():
    rec = _whois(domain_name=None)
    assert abnormal_url(rec, "example.com") == -1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && pytest tests/test_whois_lookup.py -v`
Expected: all fail (`ModuleNotFoundError`).

- [ ] **Step 3: Implement the module**

`backend/extractors/whois_lookup.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && pytest tests/test_whois_lookup.py -v`
Expected: 10 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/extractors/whois_lookup.py backend/tests/test_whois_lookup.py
git commit -m "feat(backend): add WHOIS-based feature extractors"
```

---

## Task 7: DNS Feature Extractor

**Files:**
- Create: `backend/extractors/dns_lookup.py`
- Create: `backend/tests/test_dns_lookup.py`

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_dns_lookup.py`:
```python
from unittest.mock import patch

from backend.extractors.dns_lookup import dns_record


def test_dns_record_resolves():
    with patch("backend.extractors.dns_lookup.dns.resolver.resolve") as m:
        m.return_value = ["192.0.2.1"]
        assert dns_record("example.com") == 1


def test_dns_record_no_answer():
    import dns.resolver
    with patch("backend.extractors.dns_lookup.dns.resolver.resolve") as m:
        m.side_effect = dns.resolver.NXDOMAIN
        assert dns_record("nonexistent.invalid") == -1


def test_dns_record_timeout():
    import dns.exception
    with patch("backend.extractors.dns_lookup.dns.resolver.resolve") as m:
        m.side_effect = dns.exception.Timeout
        assert dns_record("slow.example.com") == -1
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && pytest tests/test_dns_lookup.py -v`
Expected: all fail.

- [ ] **Step 3: Implement the module**

`backend/extractors/dns_lookup.py`:
```python
"""DNS record presence feature extractor."""
from __future__ import annotations

import dns.resolver
import dns.exception

_TIMEOUT_SECONDS = 5


def dns_record(host: str) -> int:
    resolver = dns.resolver.Resolver()
    resolver.timeout = _TIMEOUT_SECONDS
    resolver.lifetime = _TIMEOUT_SECONDS
    try:
        answers = dns.resolver.resolve(host, "A", lifetime=_TIMEOUT_SECONDS)
        return 1 if list(answers) else -1
    except (
        dns.resolver.NXDOMAIN,
        dns.resolver.NoAnswer,
        dns.resolver.NoNameservers,
        dns.exception.Timeout,
        OSError,
    ):
        return -1
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && pytest tests/test_dns_lookup.py -v`
Expected: 3 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/extractors/dns_lookup.py backend/tests/test_dns_lookup.py
git commit -m "feat(backend): add DNS record presence extractor"
```

---

## Task 8: Feature Extractor Orchestrator

**Files:**
- Create: `backend/feature_extractor.py`
- Create: `backend/tests/test_feature_extractor.py`

Brings together all the extractors, performs the one HTTP fetch and one WHOIS call, and returns a complete feature dict.

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_feature_extractor.py`:
```python
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && pytest tests/test_feature_extractor.py -v`
Expected: fail with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the orchestrator**

`backend/feature_extractor.py`:
```python
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


def extract_features(url: str) -> dict[str, int]:
    """Run all extractors. Any single failure falls back to neutral 0."""
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

    return features
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && pytest tests/test_feature_extractor.py -v`
Expected: 4 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/feature_extractor.py backend/tests/test_feature_extractor.py
git commit -m "feat(backend): add feature extractor orchestrator"
```

---

## Task 9: Model Training Script

**Files:**
- Create: `backend/train_models.py`
- Create: `backend/tests/test_train_smoke.py`

Trains and saves the four pipelines locally before deploy. Not run on Cloud Run.

- [ ] **Step 1: Write the smoke test**

`backend/tests/test_train_smoke.py`:
```python
"""Smoke test for the training script — runs on a tiny synthetic dataset."""
import numpy as np
import pandas as pd
import pytest

from backend.train_models import build_pipelines


def test_pipelines_train_and_predict(tmp_path):
    rng = np.random.default_rng(0)
    n = 200
    X = pd.DataFrame(rng.choice([-1, 0, 1], size=(n, 30)))
    X.columns = [f"f{i}" for i in range(30)]
    y = rng.choice([-1, 1], size=n)

    pipelines = build_pipelines()
    assert set(pipelines.keys()) == {"gb", "rf", "xgb", "svm"}

    for name, pipe in pipelines.items():
        pipe.fit(X.values, y)
        proba = pipe.predict_proba(X.values[:3])
        assert proba.shape == (3, 2), f"{name} returned {proba.shape}"
        assert np.allclose(proba.sum(axis=1), 1.0)
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_train_smoke.py -v`
Expected: fail with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the script**

`backend/train_models.py`:
```python
"""Train and save the four best models. Run locally before deploying:

    python -m backend.train_models

Writes joblib files to backend/models/{gb,rf,xgb,svm}.joblib.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier, RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import PolynomialFeatures, StandardScaler
from sklearn.svm import SVC
from xgboost import XGBClassifier

from backend.features import FEATURE_NAMES

DEFAULT_DATA_CSV = Path(__file__).resolve().parent.parent / "dataset1.csv"
MODEL_DIR = Path(__file__).resolve().parent / "models"


def build_pipelines() -> dict[str, Pipeline]:
    """Per-model pipelines mirroring the project notebooks exactly."""
    return {
        "gb": Pipeline([
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
            ("clf", GradientBoostingClassifier(
                learning_rate=0.1, max_depth=5, n_estimators=200, random_state=42,
            )),
        ]),
        "rf": Pipeline([
            ("poly", PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)),
            ("clf", RandomForestClassifier(
                max_depth=30, max_features=10, n_estimators=125,
                random_state=42, n_jobs=-1,
            )),
        ]),
        "xgb": Pipeline([
            ("scaler", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2, interaction_only=True, include_bias=False)),
            ("clf", XGBClassifier(
                learning_rate=0.1, max_depth=5, n_estimators=200,
                random_state=42, eval_metric="logloss",
            )),
        ]),
        "svm": Pipeline([
            ("scaler1", StandardScaler()),
            ("poly", PolynomialFeatures(degree=2, interaction_only=False, include_bias=False)),
            ("scaler2", StandardScaler()),
            ("clf", SVC(C=10, gamma=0.1, kernel="rbf", probability=True, random_state=42)),
        ]),
    }


def load_dataset(csv_path: Path) -> tuple[np.ndarray, np.ndarray]:
    """Read dataset1.csv, strip whitespace from column names, return X, y."""
    df = pd.read_csv(csv_path)
    df.columns = [c.strip() for c in df.columns]
    # XGBoost only accepts {0, 1} for binary labels; remap -1 → 0
    y_raw = df["Result"].values
    y = np.where(y_raw == 1, 1, 0)
    X = df[FEATURE_NAMES].values
    return X, y


def main(csv_path: Path = DEFAULT_DATA_CSV, model_dir: Path = MODEL_DIR) -> None:
    X, y = load_dataset(csv_path)
    print(f"Loaded {X.shape[0]} rows, {X.shape[1]} features.")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, stratify=y, random_state=42,
    )
    model_dir.mkdir(parents=True, exist_ok=True)
    for name, pipe in build_pipelines().items():
        print(f"Training {name}…", flush=True)
        pipe.fit(X_train, y_train)
        acc = pipe.score(X_test, y_test)
        out_path = model_dir / f"{name}.joblib"
        joblib.dump(pipe, out_path)
        print(f"  {name}: test accuracy = {acc:.4f}, saved to {out_path}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv", type=Path, default=DEFAULT_DATA_CSV)
    parser.add_argument("--out", type=Path, default=MODEL_DIR)
    args = parser.parse_args()
    main(args.csv, args.out)
```

- [ ] **Step 4: Run the smoke test**

Run: `cd backend && pytest tests/test_train_smoke.py -v`
Expected: 1 passed (this takes ~10-30 seconds because it actually fits all four models on the synthetic dataset).

- [ ] **Step 5: Run the full training pipeline against the real dataset**

Run: `cd backend && python -m backend.train_models`
Expected: prints per-model accuracies (~0.95-0.98 each) and creates `backend/models/{gb,rf,xgb,svm}.joblib`. Random Forest training may take 1-3 minutes due to the 495-feature poly expansion.

- [ ] **Step 6: Commit**

```bash
git add backend/train_models.py backend/tests/test_train_smoke.py
git commit -m "feat(backend): add model training script for top 4 models"
```

Note: `backend/models/*.joblib` files are gitignored — they get regenerated locally and live in the Cloud Run container image only.

---

## Task 10: Safe Browsing Client

**Files:**
- Create: `backend/safe_browsing.py`
- Create: `backend/tests/test_safe_browsing.py`

Calls Google Safe Browsing v4 `threatMatches:find`. Returns a dict with `verdict` (`"safe"`, `"phishing"`, `"malware"`, `"unwanted"`, `"unknown"`) and `threat_types` list.

- [ ] **Step 1: Write the failing tests**

`backend/tests/test_safe_browsing.py`:
```python
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


def test_missing_api_key_returns_unknown():
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
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && pytest tests/test_safe_browsing.py -v`
Expected: fail with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the client**

`backend/safe_browsing.py`:
```python
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
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && pytest tests/test_safe_browsing.py -v`
Expected: 5 passed.

- [ ] **Step 5: Commit**

```bash
git add backend/safe_browsing.py backend/tests/test_safe_browsing.py
git commit -m "feat(backend): add Google Safe Browsing v4 client"
```

---

## Task 11: FastAPI Health Endpoint

**Files:**
- Create: `backend/main.py`
- Create: `backend/tests/test_main_health.py`

- [ ] **Step 1: Write the failing test**

`backend/tests/test_main_health.py`:
```python
from fastapi.testclient import TestClient

from backend.main import app


def test_health_returns_ok():
    client = TestClient(app)
    resp = client.get("/api/health")
    assert resp.status_code == 200
    assert resp.json() == {"status": "ok"}
```

- [ ] **Step 2: Run test to verify it fails**

Run: `cd backend && pytest tests/test_main_health.py -v`
Expected: fail with `ModuleNotFoundError`.

- [ ] **Step 3: Implement the minimal app**

`backend/main.py`:
```python
"""FastAPI app entry point."""
from __future__ import annotations

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

app = FastAPI(title="Phishing Detection Demo")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:5000", "http://localhost:8000"],
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}
```

(CORS allowlist for Firebase Hosting domain is added in Task 12 when we know the deploy target; for now localhost dev origins are sufficient.)

- [ ] **Step 4: Run test to verify it passes**

Run: `cd backend && pytest tests/test_main_health.py -v`
Expected: 1 passed.

- [ ] **Step 5: Verify locally with uvicorn**

Run: `cd backend && uvicorn backend.main:app --port 8000` (run from project root, not inside `backend/`)
In another terminal: `curl http://localhost:8000/api/health`
Expected: `{"status":"ok"}`

Then stop the server with Ctrl-C.

- [ ] **Step 6: Commit**

```bash
git add backend/main.py backend/tests/test_main_health.py
git commit -m "feat(backend): add FastAPI app with health endpoint"
```

---

## Task 12: FastAPI Predict Endpoint

**Files:**
- Modify: `backend/main.py`
- Create: `backend/tests/test_main_predict.py`

Adds `POST /api/predict`, model loading at startup, and concurrent feature extraction + Safe Browsing call.

- [ ] **Step 1: Write the failing test**

`backend/tests/test_main_predict.py`:
```python
from unittest.mock import patch

from fastapi.testclient import TestClient

import backend.main as main_mod
from backend.main import app


def test_predict_returns_full_response_shape():
    fake_features = {name: 0 for name in main_mod.FEATURE_NAMES}
    fake_safe = {"verdict": "safe", "threat_types": []}

    with patch.object(main_mod, "extract_features", return_value=fake_features), \
         patch.object(main_mod, "check_url", return_value=fake_safe), \
         patch.object(main_mod, "_predict_with_models", return_value=[
             {"model": "Gradient Boosting", "verdict": "phishing", "probability": 0.92},
             {"model": "Random Forest",     "verdict": "phishing", "probability": 0.88},
             {"model": "XGBoost",           "verdict": "legitimate", "probability": 0.21},
             {"model": "SVM",               "verdict": "phishing", "probability": 0.77},
         ]):
        client = TestClient(app)
        resp = client.post("/api/predict", json={"url": "https://example.com"})

    assert resp.status_code == 200
    body = resp.json()
    assert body["url"] == "https://example.com"
    assert len(body["predictions"]) == 4
    assert {p["model"] for p in body["predictions"]} == {
        "Gradient Boosting", "Random Forest", "XGBoost", "SVM",
    }
    assert body["safe_browsing"] == fake_safe
    assert "approximated" in body["features_meta"]


def test_predict_rejects_malformed_url():
    client = TestClient(app)
    resp = client.post("/api/predict", json={"url": "not-a-url"})
    assert resp.status_code == 400
    assert resp.json()["error"] == "invalid_url"


def test_predict_safe_browsing_failure_still_returns_predictions():
    fake_features = {name: 0 for name in main_mod.FEATURE_NAMES}
    fake_safe = {"verdict": "unknown", "threat_types": [], "error": "no_api_key"}

    with patch.object(main_mod, "extract_features", return_value=fake_features), \
         patch.object(main_mod, "check_url", return_value=fake_safe), \
         patch.object(main_mod, "_predict_with_models", return_value=[
             {"model": "Gradient Boosting", "verdict": "legitimate", "probability": 0.1},
         ] * 4):
        client = TestClient(app)
        resp = client.post("/api/predict", json={"url": "https://example.com"})

    assert resp.status_code == 200
    assert resp.json()["safe_browsing"]["verdict"] == "unknown"
```

- [ ] **Step 2: Run tests to verify they fail**

Run: `cd backend && pytest tests/test_main_predict.py -v`
Expected: fail (endpoint not implemented).

- [ ] **Step 3: Replace `backend/main.py` with the full implementation**

`backend/main.py`:
```python
"""FastAPI app entry point: /api/health and /api/predict."""
from __future__ import annotations

import asyncio
import logging
import os
from pathlib import Path
from urllib.parse import urlparse

import joblib
from fastapi import FastAPI, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field

from backend.feature_extractor import extract_features
from backend.features import FEATURE_NAMES, APPROXIMATED_FEATURES, to_vector
from backend.safe_browsing import check_url

log = logging.getLogger("phishing_demo")
logging.basicConfig(level=logging.INFO)

MODEL_DIR = Path(__file__).resolve().parent / "models"

MODEL_FILES = {
    "Gradient Boosting": "gb.joblib",
    "Random Forest":     "rf.joblib",
    "XGBoost":           "xgb.joblib",
    "SVM":               "svm.joblib",
}

_MODELS: dict[str, object] = {}


def _load_models() -> None:
    for display_name, filename in MODEL_FILES.items():
        path = MODEL_DIR / filename
        if not path.exists():
            log.warning("Missing model file %s — endpoint will skip this model.", path)
            continue
        _MODELS[display_name] = joblib.load(path)
        log.info("Loaded model: %s", display_name)


def _predict_with_models(vector: list[int]) -> list[dict]:
    """Run all loaded models against the feature vector."""
    results = []
    for name, pipe in _MODELS.items():
        # All four pipelines were trained with labels remapped to {0, 1}
        # where 1 = phishing. predict_proba columns follow pipe.classes_.
        proba = pipe.predict_proba([vector])[0]
        phishing_idx = list(pipe.classes_).index(1)
        prob_phishing = float(proba[phishing_idx])
        verdict = "phishing" if prob_phishing >= 0.5 else "legitimate"
        results.append({
            "model": name,
            "verdict": verdict,
            "probability": round(prob_phishing, 4),
        })
    return results


# --- FastAPI app -------------------------------------------------------------

app = FastAPI(title="Phishing Detection Demo")

allowed_origins = [
    "http://localhost:5000",
    "http://localhost:8000",
]
# Firebase Hosting domain comes from env var so it can be set at deploy time.
extra_origin = os.environ.get("FRONTEND_ORIGIN")
if extra_origin:
    allowed_origins.append(extra_origin)

app.add_middleware(
    CORSMiddleware,
    allow_origins=allowed_origins,
    allow_methods=["GET", "POST"],
    allow_headers=["*"],
)


@app.on_event("startup")
def _on_startup() -> None:
    _load_models()


class PredictRequest(BaseModel):
    url: str = Field(min_length=1, max_length=2048)


def _validate_url(url: str) -> str | None:
    """Return None if valid, else an error message."""
    try:
        parsed = urlparse(url)
    except ValueError:
        return "could not parse url"
    if parsed.scheme not in {"http", "https"}:
        return "url must use http or https"
    if not parsed.hostname:
        return "url has no hostname"
    return None


@app.get("/api/health")
def health() -> dict:
    return {"status": "ok"}


@app.post("/api/predict")
async def predict(req: PredictRequest) -> JSONResponse:
    url = req.url.strip()
    err = _validate_url(url)
    if err:
        return JSONResponse(
            status_code=400,
            content={"error": "invalid_url", "message": err},
        )

    loop = asyncio.get_running_loop()
    extract_task = loop.run_in_executor(None, extract_features, url)
    safe_task = loop.run_in_executor(None, check_url, url, None)

    features, safe_browsing = await asyncio.gather(extract_task, safe_task)
    vector = to_vector(features)
    predictions = _predict_with_models(vector)

    return JSONResponse(content={
        "url": url,
        "predictions": predictions,
        "safe_browsing": safe_browsing,
        "features_meta": {
            "approximated": APPROXIMATED_FEATURES,
        },
    })
```

- [ ] **Step 4: Run tests to verify they pass**

Run: `cd backend && pytest tests/test_main_predict.py -v tests/test_main_health.py -v`
Expected: 4 passed.

- [ ] **Step 5: Run the full test suite**

Run: `cd backend && pytest -v`
Expected: all tests from Tasks 2-12 pass (≈45 tests total).

- [ ] **Step 6: Manual smoke test against a real URL**

Run: `cd backend && uvicorn backend.main:app --port 8000` (from project root)
In another terminal:
```bash
curl -s -X POST http://localhost:8000/api/predict \
  -H "Content-Type: application/json" \
  -d '{"url": "https://github.com"}' | python -m json.tool
```
Expected: a JSON response with 4 predictions (probabilities between 0 and 1), `safe_browsing.verdict == "safe"` (or `"unknown"` without an API key), and no errors. Stop the server.

- [ ] **Step 7: Commit**

```bash
git add backend/main.py backend/tests/test_main_predict.py
git commit -m "feat(backend): add /api/predict endpoint with concurrent extraction and safe-browsing"
```

---

## Task 13: Frontend HTML

**Files:**
- Create: `frontend/index.html`

- [ ] **Step 1: Write the page**

`frontend/index.html`:
```html
<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>Phishing Detector — CSSE415 Demo</title>
  <link rel="stylesheet" href="style.css" />
</head>
<body>
  <header>
    <h1>Phishing Website Detector</h1>
    <p class="subtitle">
      CSSE415 final project — predicts whether a URL is a phishing site using
      four trained models, and compares with Google Safe Browsing.
    </p>
  </header>

  <main>
    <section id="input-section">
      <form id="scan-form">
        <label for="url-input">URL to scan</label>
        <input
          id="url-input"
          type="url"
          placeholder="https://example.com"
          required
          autocomplete="off"
        />
        <button type="submit" id="scan-button">Scan</button>
      </form>
      <p class="disclaimer">
        Class demo only. Not a production security tool. Some dataset features
        (PageRank, Alexa traffic, etc.) are approximated because the underlying
        APIs were deprecated.
      </p>
    </section>

    <section id="loading-section" hidden>
      <div class="spinner" aria-label="Loading"></div>
      <p>Extracting features and querying models…</p>
    </section>

    <section id="results-section" hidden>
      <h2 id="scanned-url"></h2>

      <div class="results-grid">
        <div class="panel">
          <h3>Our Models</h3>
          <ul id="model-predictions"></ul>
        </div>
        <div class="panel">
          <h3>Google Safe Browsing</h3>
          <div id="safe-browsing-result"></div>
        </div>
      </div>

      <p id="agreement-line"></p>
      <p id="approximated-footnote" class="footnote"></p>

      <button id="reset-button">Scan another URL</button>
    </section>

    <section id="error-section" hidden>
      <h2>Something went wrong</h2>
      <p id="error-message"></p>
      <button id="error-reset-button">Try again</button>
    </section>
  </main>

  <footer>
    <p>
      Ervin Perkowski, Mark J. Kenny, Sanil Maheshwari ·
      <a href="https://github.com/rhit-kennym1/csse415-final-project">source</a>
    </p>
  </footer>

  <script src="app.js"></script>
</body>
</html>
```

- [ ] **Step 2: Commit**

```bash
git add frontend/index.html
git commit -m "feat(frontend): add page skeleton"
```

---

## Task 14: Frontend JavaScript

**Files:**
- Create: `frontend/app.js`

Handles form submit, state swapping, and rendering of predictions + Safe Browsing.

- [ ] **Step 1: Write the script**

`frontend/app.js`:
```javascript
const API_BASE = window.location.origin; // Firebase Hosting rewrites /api/** to Cloud Run

const sections = {
  input:    document.getElementById("input-section"),
  loading:  document.getElementById("loading-section"),
  results:  document.getElementById("results-section"),
  error:    document.getElementById("error-section"),
};

function showOnly(name) {
  for (const [key, el] of Object.entries(sections)) {
    el.hidden = key !== name;
  }
}

function verdictBadge(verdict) {
  const span = document.createElement("span");
  span.className = `badge badge-${verdict}`;
  span.textContent = verdict;
  return span;
}

function renderModelPredictions(predictions) {
  const ul = document.getElementById("model-predictions");
  ul.innerHTML = "";
  for (const p of predictions) {
    const li = document.createElement("li");
    const name = document.createElement("strong");
    name.textContent = p.model;
    const bar = document.createElement("div");
    bar.className = "prob-bar";
    const fill = document.createElement("div");
    fill.className = `prob-fill prob-${p.verdict}`;
    fill.style.width = `${Math.round(p.probability * 100)}%`;
    fill.textContent = `${Math.round(p.probability * 100)}% phishing`;
    bar.appendChild(fill);

    li.appendChild(name);
    li.appendChild(verdictBadge(p.verdict));
    li.appendChild(bar);
    ul.appendChild(li);
  }
}

function renderSafeBrowsing(sb) {
  const container = document.getElementById("safe-browsing-result");
  container.innerHTML = "";
  container.appendChild(verdictBadge(sb.verdict));
  if (sb.threat_types && sb.threat_types.length) {
    const list = document.createElement("ul");
    for (const t of sb.threat_types) {
      const li = document.createElement("li");
      li.textContent = t.replace(/_/g, " ").toLowerCase();
      list.appendChild(li);
    }
    container.appendChild(list);
  }
  if (sb.error) {
    const note = document.createElement("p");
    note.className = "footnote";
    note.textContent = `Safe Browsing unavailable: ${sb.error}`;
    container.appendChild(note);
  }
}

function renderAgreement(predictions, sb) {
  const line = document.getElementById("agreement-line");
  if (sb.verdict === "safe" || sb.verdict === "phishing" ||
      sb.verdict === "malware" || sb.verdict === "unwanted") {
    const sbSaysPhish = sb.verdict !== "safe";
    const agreeing = predictions.filter(p =>
      (p.verdict === "phishing") === sbSaysPhish
    ).length;
    line.textContent =
      `${agreeing} of ${predictions.length} models agree with Google Safe Browsing.`;
  } else {
    line.textContent = "Google Safe Browsing did not return a verdict.";
  }
}

function renderApproximated(meta) {
  const el = document.getElementById("approximated-footnote");
  if (!meta || !meta.approximated || !meta.approximated.length) {
    el.textContent = "";
    return;
  }
  el.textContent =
    `${meta.approximated.length} features approximated (defaulted to neutral): ` +
    meta.approximated.join(", ") + ".";
}

async function scan(url) {
  showOnly("loading");
  let resp;
  try {
    resp = await fetch(`${API_BASE}/api/predict`, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ url }),
    });
  } catch (e) {
    showError(`Network error: ${e.message}`);
    return;
  }
  const data = await resp.json().catch(() => ({}));
  if (!resp.ok) {
    showError(data.message || data.error || `HTTP ${resp.status}`);
    return;
  }
  document.getElementById("scanned-url").textContent = data.url;
  renderModelPredictions(data.predictions);
  renderSafeBrowsing(data.safe_browsing);
  renderAgreement(data.predictions, data.safe_browsing);
  renderApproximated(data.features_meta);
  showOnly("results");
}

function showError(msg) {
  document.getElementById("error-message").textContent = msg;
  showOnly("error");
}

function reset() {
  document.getElementById("url-input").value = "";
  showOnly("input");
}

// Wake up Cloud Run as soon as the page loads, so the first scan is faster.
fetch(`${API_BASE}/api/health`).catch(() => {});

document.getElementById("scan-form").addEventListener("submit", (e) => {
  e.preventDefault();
  const url = document.getElementById("url-input").value.trim();
  if (url) scan(url);
});
document.getElementById("reset-button").addEventListener("click", reset);
document.getElementById("error-reset-button").addEventListener("click", reset);
```

- [ ] **Step 2: Commit**

```bash
git add frontend/app.js
git commit -m "feat(frontend): add scan flow and result rendering"
```

---

## Task 15: Frontend CSS

**Files:**
- Create: `frontend/style.css`

- [ ] **Step 1: Write the stylesheet**

`frontend/style.css`:
```css
:root {
  --green: #1d8a4a;
  --red: #c0392b;
  --gray: #6b7280;
  --bg: #f7f7f9;
  --panel: #ffffff;
  --border: #e3e4e8;
  --text: #1f2933;
}

* { box-sizing: border-box; }

body {
  margin: 0;
  font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, sans-serif;
  background: var(--bg);
  color: var(--text);
  line-height: 1.5;
}

header, main, footer {
  max-width: 960px;
  margin: 0 auto;
  padding: 1.5rem;
}

header h1 { margin: 0 0 0.25rem; font-size: 1.75rem; }
.subtitle { margin: 0; color: var(--gray); }

form {
  display: flex;
  gap: 0.5rem;
  flex-wrap: wrap;
  align-items: end;
}
form label { display: block; font-weight: 600; margin-bottom: 0.25rem; }
form input[type="url"] {
  flex: 1 1 320px;
  padding: 0.6rem 0.75rem;
  border: 1px solid var(--border);
  border-radius: 6px;
  font-size: 1rem;
}
button {
  padding: 0.6rem 1.2rem;
  border: none;
  border-radius: 6px;
  background: var(--text);
  color: white;
  font-size: 1rem;
  cursor: pointer;
}
button:hover { opacity: 0.9; }

.disclaimer, .footnote {
  margin-top: 0.5rem;
  font-size: 0.85rem;
  color: var(--gray);
}

.spinner {
  width: 40px; height: 40px;
  border: 4px solid var(--border);
  border-top-color: var(--text);
  border-radius: 50%;
  margin: 1rem auto;
  animation: spin 0.8s linear infinite;
}
@keyframes spin { to { transform: rotate(360deg); } }

#results-section h2 { word-break: break-all; font-size: 1.1rem; color: var(--gray); }

.results-grid {
  display: grid;
  grid-template-columns: repeat(auto-fit, minmax(280px, 1fr));
  gap: 1rem;
  margin: 1rem 0;
}
.panel {
  background: var(--panel);
  border: 1px solid var(--border);
  border-radius: 8px;
  padding: 1rem;
}
.panel h3 { margin: 0 0 0.75rem; font-size: 1rem; }

#model-predictions { list-style: none; padding: 0; margin: 0; }
#model-predictions li {
  display: grid;
  grid-template-columns: 1fr auto;
  grid-template-rows: auto auto;
  gap: 0.25rem 0.5rem;
  padding: 0.5rem 0;
  border-bottom: 1px solid var(--border);
}
#model-predictions li:last-child { border-bottom: none; }
.prob-bar {
  grid-column: 1 / -1;
  background: var(--border);
  border-radius: 4px;
  height: 1.4rem;
  overflow: hidden;
}
.prob-fill {
  height: 100%;
  color: white;
  font-size: 0.8rem;
  display: flex;
  align-items: center;
  padding: 0 0.5rem;
  white-space: nowrap;
}
.prob-phishing { background: var(--red); }
.prob-legitimate { background: var(--green); }

.badge {
  display: inline-block;
  padding: 0.15rem 0.5rem;
  border-radius: 999px;
  font-size: 0.8rem;
  text-transform: uppercase;
  color: white;
  font-weight: 600;
}
.badge-safe, .badge-legitimate { background: var(--green); }
.badge-phishing, .badge-malware, .badge-unwanted { background: var(--red); }
.badge-unknown { background: var(--gray); }

#agreement-line { font-weight: 600; }

footer {
  text-align: center;
  font-size: 0.85rem;
  color: var(--gray);
  padding-top: 0;
}
```

- [ ] **Step 2: Manually verify the UI locally**

Start the backend in one terminal: `cd backend && uvicorn backend.main:app --port 8000` (from project root).
Start a static server for the frontend in another: `cd frontend && python -m http.server 5000`.

For local development, the frontend's `API_BASE` (set to `window.location.origin`) will be `http://localhost:5000`, but the backend is at `http://localhost:8000`. To bridge this for local dev: open `frontend/app.js` and temporarily change the first line to `const API_BASE = "http://localhost:8000";`. Revert before committing — in production, Firebase Hosting handles the rewrite.

Then open `http://localhost:5000` in a browser, enter `https://github.com`, click Scan, and confirm:
- Loading section appears
- After ~5-10s, results section appears with 4 model predictions and one Safe Browsing verdict
- "Scan another URL" returns to the input view

- [ ] **Step 3: Commit**

```bash
git add frontend/style.css
git commit -m "feat(frontend): add stylesheet"
```

---

## Task 16: Dockerfile

**Files:**
- Create: `backend/Dockerfile`
- Create: `backend/.dockerignore`

- [ ] **Step 1: Write the Dockerfile**

`backend/Dockerfile`:
```dockerfile
FROM python:3.11-slim

# Build deps for lxml, xgboost, scikit-learn
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    libxml2-dev \
    libxslt-dev \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy backend package (sibling files set the working module path)
COPY . ./backend/
# Trained models live under backend/models/ — must exist locally before docker build
RUN test -f backend/models/gb.joblib || (echo "Missing models — run python -m backend.train_models first" && exit 1)

ENV PORT=8080
EXPOSE 8080

# Cloud Run sets $PORT — uvicorn binds to it.
CMD ["sh", "-c", "uvicorn backend.main:app --host 0.0.0.0 --port ${PORT}"]
```

`backend/.dockerignore`:
```
__pycache__/
*.pyc
.pytest_cache/
.venv/
tests/
```

- [ ] **Step 2: Build the image locally**

Run: `cd backend && docker build -t phishing-demo:local .`
Expected: image builds without errors. (Requires Docker Desktop on Windows.)

- [ ] **Step 3: Run the container locally and smoke-test**

Run: `docker run --rm -p 8080:8080 phishing-demo:local`
In another terminal: `curl http://localhost:8080/api/health`
Expected: `{"status":"ok"}`. Stop the container with Ctrl-C.

- [ ] **Step 4: Commit**

```bash
git add backend/Dockerfile backend/.dockerignore
git commit -m "feat(deploy): add Dockerfile for Cloud Run"
```

---

## Task 17: Firebase Hosting Configuration

**Files:**
- Create: `firebase.json`
- Create: `.firebaserc`

- [ ] **Step 1: Write the configs**

`firebase.json`:
```json
{
  "hosting": {
    "public": "frontend",
    "ignore": [
      "firebase.json",
      "**/.*",
      "**/node_modules/**"
    ],
    "rewrites": [
      {
        "source": "/api/**",
        "run": {
          "serviceId": "phishing-demo",
          "region": "us-central1"
        }
      }
    ]
  }
}
```

`.firebaserc`:
```json
{
  "projects": {
    "default": "REPLACE_WITH_YOUR_FIREBASE_PROJECT_ID"
  }
}
```

After `firebase init`, the `default` field will be populated automatically. The literal `REPLACE_WITH_YOUR_FIREBASE_PROJECT_ID` is a placeholder for the implementer.

- [ ] **Step 2: Commit**

```bash
git add firebase.json .firebaserc
git commit -m "feat(deploy): add Firebase Hosting config with Cloud Run rewrite"
```

---

## Task 18: Deployment Documentation

**Files:**
- Create: `DEPLOY.md`

- [ ] **Step 1: Write the deploy guide**

`DEPLOY.md`:
````markdown
# Deploying the Phishing Detection Demo

This is a one-time setup followed by repeatable deploy commands. The app has
two pieces: a Cloud Run backend (Python ML service) and a Firebase Hosting
frontend that proxies API calls to it.

## Prerequisites

- Google Cloud account with billing enabled (Cloud Run is free-tier friendly)
- Firebase project linked to the same GCP project
- `gcloud` CLI and `firebase` CLI installed and authenticated
- Docker Desktop (for building the backend image)
- A Google Safe Browsing API key (free) —
  https://developers.google.com/safe-browsing/v4/get-started

## One-Time Setup

1. **Set the project IDs:**
   ```bash
   gcloud config set project <YOUR_GCP_PROJECT_ID>
   ```
   In `.firebaserc`, replace `REPLACE_WITH_YOUR_FIREBASE_PROJECT_ID` with your
   Firebase project ID (same as the GCP one).

2. **Enable required Google Cloud APIs:**
   ```bash
   gcloud services enable run.googleapis.com cloudbuild.googleapis.com \
     artifactregistry.googleapis.com safebrowsing.googleapis.com
   ```

3. **Train the models locally** (only needed once, or whenever you retrain):
   ```bash
   cd backend
   python -m venv .venv && .venv\Scripts\activate     # Windows
   pip install -r requirements-dev.txt
   python -m backend.train_models
   ```
   This writes `backend/models/{gb,rf,xgb,svm}.joblib`. These files get baked
   into the Docker image and are not committed to git.

## Deploy the Backend (Cloud Run)

```bash
cd backend
gcloud builds submit --tag gcr.io/<YOUR_GCP_PROJECT_ID>/phishing-demo
gcloud run deploy phishing-demo \
  --image gcr.io/<YOUR_GCP_PROJECT_ID>/phishing-demo \
  --region us-central1 \
  --platform managed \
  --allow-unauthenticated \
  --memory 1Gi \
  --cpu 1 \
  --timeout 60 \
  --set-env-vars SAFE_BROWSING_API_KEY=<YOUR_API_KEY>,FRONTEND_ORIGIN=https://<YOUR_FIREBASE_PROJECT_ID>.web.app
```

After deploy, `gcloud` prints the Cloud Run URL. Verify it's healthy:

```bash
curl https://phishing-demo-XXXX.run.app/api/health
# {"status":"ok"}
```

## Deploy the Frontend (Firebase Hosting)

```bash
firebase deploy --only hosting
```

The CLI prints the public URL (e.g. `https://<project-id>.web.app`). Open it
in a browser and verify that:

- The page loads with the URL input form
- Submitting `https://github.com` returns 4 model predictions and a Safe
  Browsing verdict within ~10 seconds
- `/api/health` is reachable through Firebase (it should be proxied to
  Cloud Run via the rewrite in `firebase.json`)

## Updating

To redeploy after code changes:
- Backend changes → re-run `gcloud builds submit` and `gcloud run deploy`
- Frontend changes → re-run `firebase deploy --only hosting`
- Model changes → re-run `python -m backend.train_models`, then redeploy
  the backend

## Troubleshooting

| Symptom | Likely cause |
|---|---|
| First scan times out at ~60s | Cloud Run cold start. The frontend hits `/api/health` on page load to warm it; if you skip that, the first scan is slow. |
| `safe_browsing.verdict == "unknown"` | `SAFE_BROWSING_API_KEY` env var missing or invalid on Cloud Run. |
| CORS errors in browser console | `FRONTEND_ORIGIN` env var doesn't match the Firebase Hosting URL. Update with `gcloud run services update phishing-demo --update-env-vars FRONTEND_ORIGIN=…` |
| Docker build fails on `Missing models` | Run `python -m backend.train_models` before building the image. |
````

- [ ] **Step 2: Commit**

```bash
git add DEPLOY.md
git commit -m "docs: add deployment guide"
```

---

## Final Verification

- [ ] **Step 1: Run the full test suite**

```bash
cd backend && pytest -v
```

Expected: all tests pass (≈50 tests across all modules).

- [ ] **Step 2: Manual end-to-end smoke test (local)**

1. Train models: `cd backend && python -m backend.train_models`
2. Start backend: `cd backend && uvicorn backend.main:app --port 8000` (run from project root)
3. In `frontend/app.js`, temporarily set `API_BASE = "http://localhost:8000"`
4. Serve frontend: `cd frontend && python -m http.server 5000`
5. Open `http://localhost:5000`
6. Scan three URLs of different character: `https://github.com` (expected legitimate), a known shortener like `https://bit.ly/3xyz` (expected phishing-leaning), and a fresh domain
7. Confirm: predictions render, probability bars display, Safe Browsing verdict shows, "Scan another URL" works
8. Revert `API_BASE` to `window.location.origin` before committing

- [ ] **Step 3: Final commit if anything was tweaked during the smoke test**

```bash
git status
git add -A   # only if there are intentional last-minute fixes
git commit -m "fix: address issues found during end-to-end smoke test"
```
