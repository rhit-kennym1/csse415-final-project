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
