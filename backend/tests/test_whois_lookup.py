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
