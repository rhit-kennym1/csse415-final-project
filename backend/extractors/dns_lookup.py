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
