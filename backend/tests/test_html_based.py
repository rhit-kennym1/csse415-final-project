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
