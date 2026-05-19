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
