from __future__ import annotations

from urllib.parse import urlsplit, urlunsplit

from django.conf import settings
from django.http import HttpRequest, HttpResponsePermanentRedirect


class CanonicalHostRedirectMiddleware:
    """Redirect configured alias hosts to the canonical public host."""

    def __init__(self, get_response):
        self.get_response = get_response

    def __call__(self, request: HttpRequest):
        canonical_host = str(getattr(settings, "CANONICAL_HOST", "") or "").strip().lower()
        redirect_hosts = {
            str(host or "").strip().lower()
            for host in getattr(settings, "CANONICAL_HOST_REDIRECTS", [])
            if str(host or "").strip()
        }

        request_host = request.get_host().split(":", 1)[0].strip().lower()
        if canonical_host and request_host in redirect_hosts and request_host != canonical_host:
            parsed = urlsplit(request.get_full_path())
            target = urlunsplit(
                (
                    "https",
                    canonical_host,
                    parsed.path,
                    parsed.query,
                    parsed.fragment,
                )
            )
            return HttpResponsePermanentRedirect(target)

        return self.get_response(request)
