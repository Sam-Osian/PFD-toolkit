from __future__ import annotations

from django.conf import settings


def _site_base_url() -> str:
    return str(getattr(settings, "WORKBENCH_BASE_URL", "https://pfdtoolkit.org") or "").rstrip("/")


def canonical_path(path: str) -> str:
    clean_path = str(path or "/").split("?", 1)[0] or "/"
    if not clean_path.startswith("/"):
        clean_path = f"/{clean_path}"
    if clean_path == "/workspaces/":
        return "/workbooks/"
    if clean_path.startswith("/workspaces/"):
        return f"/workbooks/{clean_path.removeprefix('/workspaces/')}"
    return clean_path


def absolute_url(path: str) -> str:
    return f"{_site_base_url()}{canonical_path(path)}"


def seo_defaults(request):
    canonical = absolute_url(getattr(request, "path", "/"))
    home_url = absolute_url("/")
    return {
        "seo_site_name": "PFD Toolkit",
        "seo_default_title": "PFD Toolkit",
        "seo_default_description": (
            "Search, filter, and analyse Prevention of Future Death reports with PFD Toolkit."
        ),
        "seo_canonical_url": canonical,
        "seo_home_url": home_url,
        "seo_site_base_url": _site_base_url(),
    }
