from __future__ import annotations

from datetime import date
from xml.sax.saxutils import escape

from django.http import HttpRequest, HttpResponse
from django.urls import reverse
from django.views.decorators.http import require_GET

from .seo import absolute_url


SITEMAP_ROUTES: tuple[tuple[str, str, str], ...] = (
    ("landing", "daily", "1.0"),
    ("explore", "daily", "0.9"),
    ("collection-list", "daily", "0.8"),
    ("about", "monthly", "0.7"),
    ("research", "monthly", "0.7"),
    ("workbook-public-list", "weekly", "0.5"),
)

SITEMAP_COLLECTION_SLUGS: tuple[str, ...] = (
    "wales",
    "nhs",
    "gov_department",
    "prisons",
    "health_regulators",
    "local_gov",
)


@require_GET
def robots_txt(request: HttpRequest) -> HttpResponse:
    sitemap_url = absolute_url(reverse("sitemap-xml"))
    body = "\n".join(
        [
            "User-agent: *",
            "Allow: /$",
            "Allow: /about/",
            "Allow: /research/",
            "Allow: /explore/",
            "Allow: /collections/",
            "Allow: /workbooks/public/",
            "Disallow: /admin/",
            "Disallow: /auth/",
            "Disallow: /settings/",
            "Disallow: /workspaces/",
            "Disallow: /s/",
            "",
            f"Sitemap: {sitemap_url}",
            "",
        ]
    )
    return HttpResponse(body, content_type="text/plain; charset=utf-8")


@require_GET
def sitemap_xml(request: HttpRequest) -> HttpResponse:
    today = date.today().isoformat()
    urls: list[tuple[str, str, str, str]] = []
    for route_name, changefreq, priority in SITEMAP_ROUTES:
        urls.append((absolute_url(reverse(route_name)), today, changefreq, priority))
    for slug in SITEMAP_COLLECTION_SLUGS:
        urls.append(
            (
                absolute_url(reverse("collection-detail", kwargs={"collection_slug": slug})),
                today,
                "weekly",
                "0.7",
            )
        )

    url_entries = "\n".join(
        (
            "  <url>\n"
            f"    <loc>{escape(location)}</loc>\n"
            f"    <lastmod>{lastmod}</lastmod>\n"
            f"    <changefreq>{changefreq}</changefreq>\n"
            f"    <priority>{priority}</priority>\n"
            "  </url>"
        )
        for location, lastmod, changefreq, priority in urls
    )
    body = (
        '<?xml version="1.0" encoding="UTF-8"?>\n'
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">\n'
        f"{url_entries}\n"
        "</urlset>\n"
    )
    return HttpResponse(body, content_type="application/xml; charset=utf-8")
