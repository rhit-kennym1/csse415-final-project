from backend.feature_display import build_display
from backend.features import FEATURE_NAMES


def _base(**overrides):
    """A neutral feature dict (all 0) with optional overrides."""
    f = {name: 0 for name in FEATURE_NAMES}
    f.update(overrides)
    return f


def _by_label(display, label):
    return next(item for item in display if item["label"] == label)


def test_display_has_eight_slide_items():
    display = build_display(_base())
    labels = [item["label"] for item in display]
    assert labels == [
        "URL length",
        "HTTPS / SSL state",
        "Suspicious symbols",
        "URL Anchor",
        "Domain age",
        "Redirects",
        "iframes & pop-ups",
        "Web traffic rank",
    ]


def test_url_length_long_is_suspicious():
    item = _by_label(build_display(_base(URLURL_Length=-1)), "URL length")
    assert item["signal"] == "suspicious"
    assert "Long" in item["value"]


def test_url_length_short_is_ok():
    item = _by_label(build_display(_base(URLURL_Length=1)), "URL length")
    assert item["signal"] == "ok"


def test_ssl_no_https_is_suspicious():
    item = _by_label(build_display(_base(SSLfinal_State=-1)), "HTTPS / SSL state")
    assert item["signal"] == "suspicious"


def test_suspicious_symbols_none_detected():
    item = _by_label(
        build_display(_base(
            having_At_Symbol=1, double_slash_redirecting=1,
            Prefix_Suffix=1, having_IPhaving_IP_Address=1,
        )),
        "Suspicious symbols",
    )
    assert item["signal"] == "ok"
    assert "None" in item["value"]


def test_suspicious_symbols_lists_detected_ones():
    item = _by_label(
        build_display(_base(
            having_At_Symbol=-1, having_IPhaving_IP_Address=-1,
            double_slash_redirecting=1, Prefix_Suffix=1,
        )),
        "Suspicious symbols",
    )
    assert item["signal"] == "suspicious"
    assert "@" in item["value"]
    assert "IP address" in item["value"]


def test_iframes_and_popups_detected():
    item = _by_label(build_display(_base(Iframe=-1, popUpWidnow=1)), "iframes & pop-ups")
    assert item["signal"] == "suspicious"
    assert "iframe" in item["value"].lower()


def test_iframes_and_popups_none():
    item = _by_label(build_display(_base(Iframe=1, popUpWidnow=1)), "iframes & pop-ups")
    assert item["signal"] == "ok"


def test_web_traffic_always_not_available():
    item = _by_label(build_display(_base(web_traffic=0)), "Web traffic rank")
    assert item["signal"] == "neutral"
    assert "available" in item["value"].lower()


def test_domain_age_established_is_ok():
    item = _by_label(build_display(_base(age_of_domain=1)), "Domain age")
    assert item["signal"] == "ok"


def test_every_item_has_required_fields():
    for item in build_display(_base()):
        assert set(item.keys()) == {"label", "value", "signal"}
        assert item["signal"] in {"ok", "neutral", "suspicious"}
