from django.contrib.auth import get_user_model
from django.core import mail
from django.test import TestCase, override_settings
from django.urls import reverse
from unittest.mock import patch

import pandas as pd

from accounts.views import _explore_report_rows


User = get_user_model()


class AccountsModelTests(TestCase):
    def test_user_model_uses_email_as_username_field(self):
        self.assertEqual(User.USERNAME_FIELD, "email")

    def test_user_creation_with_email(self):
        user = User.objects.create_user(
            email="person@example.com",
            password="example-pass-123",
            first_name="Person",
            last_name="Example",
        )
        self.assertEqual(user.email, "person@example.com")
        self.assertTrue(user.check_password("example-pass-123"))


class AccountsViewTests(TestCase):
    def test_landing_page_loads(self):
        response = self.client.get(reverse("landing"))
        self.assertEqual(response.status_code, 200)

    def test_services_page_loads_and_is_linked_from_navigation(self):
        response = self.client.get(reverse("services"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Custom PFD analysis")
        self.assertContains(response, "Dedicated dashboard")
        self.assertContains(response, 'method="post"')

        landing = self.client.get(reverse("landing"))
        self.assertContains(landing, reverse("services"))

    @override_settings(
        EMAIL_BACKEND="django.core.mail.backends.locmem.EmailBackend",
        DEFAULT_FROM_EMAIL="noreply@oreliandata.co.uk",
        SERVICES_ENQUIRY_EMAIL="sam.osian@oreliandata.co.uk",
    )
    def test_services_enquiry_sends_email(self):
        response = self.client.post(
            reverse("services"),
            {
                "name": "Alex Researcher",
                "organisation": "Example NHS Trust",
                "email": "alex@example.org",
                "work_type": "Dedicated dashboard",
                "project_summary": (
                    "We want to understand repeated medication safety concerns in "
                    "Prevention of Future Death reports."
                ),
                "website": "",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Enquiry sent.")
        self.assertEqual(len(mail.outbox), 1)
        message = mail.outbox[0]
        self.assertEqual(message.to, ["sam.osian@oreliandata.co.uk"])
        self.assertEqual(message.reply_to, ["alex@example.org"])
        self.assertIn("Dedicated dashboard", message.subject)
        self.assertIn("Example NHS Trust", message.body)
        self.assertIn("medication safety concerns", message.body)

    @override_settings(EMAIL_BACKEND="django.core.mail.backends.locmem.EmailBackend")
    def test_services_enquiry_validation_blocks_invalid_submission(self):
        response = self.client.post(
            reverse("services"),
            {
                "name": "",
                "organisation": "",
                "email": "not-an-email",
                "work_type": "Dedicated dashboard",
                "project_summary": "Too short",
                "website": "",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "This field is required")
        self.assertEqual(len(mail.outbox), 0)

    @override_settings(EMAIL_BACKEND="django.core.mail.backends.locmem.EmailBackend")
    def test_services_enquiry_honeypot_does_not_send_email(self):
        response = self.client.post(
            reverse("services"),
            {
                "name": "Spam Bot",
                "organisation": "Noise",
                "email": "bot@example.org",
                "work_type": "Rapid scoping review",
                "project_summary": "This appears valid but the hidden field is filled in.",
                "website": "https://spam.example",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Enquiry sent.")
        self.assertEqual(len(mail.outbox), 0)

    def test_legal_pages_load_and_are_linked_from_navigation(self):
        privacy = self.client.get(reverse("privacy-policy"))
        self.assertEqual(privacy.status_code, 200)
        self.assertContains(privacy, "Privacy Policy")
        self.assertContains(privacy, reverse("cookie-policy"))

        cookies = self.client.get(reverse("cookie-policy"))
        self.assertEqual(cookies.status_code, 200)
        self.assertContains(cookies, "Cookie Policy")
        self.assertContains(cookies, reverse("privacy-policy"))

        landing = self.client.get(reverse("landing"))
        self.assertContains(landing, reverse("privacy-policy"))
        self.assertContains(landing, reverse("cookie-policy"))
        self.assertNotContains(landing, "Toolkit ready")
        self.assertNotContains(landing, "pfd_toolkit adapters connected")

    def test_signed_in_sidebar_account_panel_still_renders(self):
        user = User.objects.create_user(
            email="signed-in@example.com",
            password="example-pass-123",
            first_name="Signed",
            last_name="In",
        )
        self.client.force_login(user)

        response = self.client.get(reverse("landing"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Signed In")
        self.assertContains(response, "signed-in@example.com")
        self.assertContains(response, reverse("privacy-policy"))
        self.assertContains(response, reverse("cookie-policy"))
        self.assertNotContains(response, "Toolkit ready")
        self.assertNotContains(response, "pfd_toolkit adapters connected")

    def test_login_redirects_to_auth0_when_configured(self):
        response = self.client.get(reverse("accounts-login"))
        self.assertEqual(response.status_code, 302)
        self.assertIn("authorize?", response.url)

    def test_admin_login_proxy_redirects_to_auth_login(self):
        response = self.client.get("/admin/login/")
        self.assertEqual(response.status_code, 302)
        self.assertIn("/auth/login/", response.url)

    @patch("accounts.views.load_collections_dataset")
    @patch("accounts.views.reports_for_collection")
    def test_explore_renders_live_scope_counts(self, mock_reports_for_collection, mock_load_collections):
        reports_df = pd.DataFrame(
            [
                {"date": "2025-01-01", "area": "Area A", "receiver": "Receiver X"},
                {"date": "2025-01-02", "area": "Area B", "receiver": "Receiver Y"},
            ]
        )
        mock_load_collections.return_value = reports_df
        mock_reports_for_collection.return_value = reports_df

        response = self.client.get(reverse("explore"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Explore · 2 reports")

    @patch("accounts.views.load_collections_dataset")
    def test_explore_handles_dataset_failure(self, mock_load_collections):
        mock_load_collections.side_effect = RuntimeError("dataset down")

        response = self.client.get(reverse("explore"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Dataset temporarily unavailable")

    @patch("accounts.views.load_collections_dataset")
    def test_explore_reports_panel_applies_active_filters(self, mock_load_collections):
        reports_df = pd.DataFrame(
            [
                {
                    "id": "2025-0001",
                    "date": "2025-01-10",
                    "coroner": "Coroner A",
                    "area": "North",
                    "receiver": "NHS Trust",
                    "title": "North report",
                    "url": "https://example.com/north",
                },
                {
                    "id": "2025-0002",
                    "date": "2025-01-11",
                    "coroner": "Coroner B",
                    "area": "South",
                    "receiver": "Local Authority",
                    "title": "South report",
                    "url": "https://example.com/south",
                },
            ]
        )
        mock_load_collections.return_value = reports_df

        response = self.client.get(
            reverse("explore-reports-panel"),
            {
                "ai_filter": "custom",
                "area": "North",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "North report")
        self.assertNotContains(response, "South report")
        self.assertContains(response, "Showing 1-1 of 1")
        body = response.content.decode("utf-8")
        self.assertLess(body.find("<th>ID</th>"), body.find("<th>URL</th>"))
        self.assertLess(body.find("<th>URL</th>"), body.find("<th>Date</th>"))

    @patch("accounts.views.load_collections_dataset")
    def test_explore_reports_panel_paginates(self, mock_load_collections):
        rows = [
            {
                "id": f"2025-{index:04d}",
                "date": f"2025-01-{(index % 28) + 1:02d}",
                "coroner": f"Coroner {index}",
                "area": "North",
                "receiver": "NHS Trust",
                "title": f"Report {index}",
                "url": f"https://example.com/report-{index}",
            }
            for index in range(1, 61)
        ]
        reports_df = pd.DataFrame(rows)
        mock_load_collections.return_value = reports_df

        response = self.client.get(
            reverse("explore-reports-panel"),
            {
                "ai_filter": "custom",
                "page": 2,
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Page 2 of 4")
        self.assertContains(response, "Showing 16-30 of 60")

    @patch("accounts.views.load_collections_dataset")
    def test_explore_reports_panel_renders_date_columns_without_time(self, mock_load_collections):
        reports_df = pd.DataFrame(
            [
                {
                    "id": "2026-0001",
                    "date": "2026-04-14 00:00:00",
                    "response_date": "2026-04-20 00:00:00",
                    "coroner": "Coroner A",
                    "area": "North",
                    "receiver": "NHS Trust",
                    "title": "Report A",
                    "url": "https://example.com/a",
                },
            ]
        )
        mock_load_collections.return_value = reports_df

        response = self.client.get(reverse("explore-reports-panel"), {"ai_filter": "custom"})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "2026-04-14")
        self.assertNotContains(response, "2026-04-14 00:00:00")
        self.assertContains(response, "2026-04-20")
        self.assertNotContains(response, "2026-04-20 00:00:00")

    def test_explore_report_rows_marks_theme_booleans_for_checkbox_rendering(self):
        reports_df = pd.DataFrame(
            [
                {
                    "id": "2026-0001",
                    "date": "2026-04-14 00:00:00",
                    "coroner": "Coroner A",
                    "area": "North",
                    "receiver": "NHS Trust",
                    "title": "Report A",
                    "url": "https://example.com/a",
                    "theme_care_coordination": True,
                    "theme_discharge_failures": False,
                },
            ]
        )
        rows = _explore_report_rows(
            reports_df=reports_df,
            ordered_columns=["id", "theme_care_coordination", "theme_discharge_failures"],
        )
        self.assertEqual(len(rows), 1)
        cells = rows[0]["cells"]
        self.assertEqual(cells[0]["value"], "2026-0001")
        self.assertFalse(cells[0]["is_theme_boolean"])
        self.assertTrue(cells[1]["is_theme_boolean"])
        self.assertTrue(cells[1]["bool_value"])
        self.assertTrue(cells[2]["is_theme_boolean"])
        self.assertFalse(cells[2]["bool_value"])

    @patch("accounts.views.load_collections_dataset")
    def test_explore_export_csv_downloads_filtered_scope(self, mock_load_collections):
        reports_df = pd.DataFrame(
            [
                {
                    "id": "2026-0001",
                    "date": "2026-04-14 00:00:00",
                    "coroner": "Coroner A",
                    "area": "North",
                    "receiver": "NHS Trust",
                    "title": "North report",
                    "url": "https://example.com/north",
                },
                {
                    "id": "2026-0002",
                    "date": "2026-04-15 00:00:00",
                    "coroner": "Coroner B",
                    "area": "South",
                    "receiver": "Local Authority",
                    "title": "South report",
                    "url": "https://example.com/south",
                },
            ]
        )
        mock_load_collections.return_value = reports_df

        response = self.client.get(reverse("explore-export-csv"), {"ai_filter": "custom", "area": "North"})
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "text/csv; charset=utf-8")
        self.assertIn("attachment; filename=", response["Content-Disposition"])

    @patch("accounts.views.load_collections_dataset")
    @patch("accounts.views.reports_for_collection")
    def test_explore_persists_filters_across_navigation(self, mock_reports_for_collection, mock_load_collections):
        reports_df = pd.DataFrame(
            [
                {"date": "2025-01-01", "area": "North", "receiver": "NHS Trust"},
                {"date": "2025-01-02", "area": "South", "receiver": "Local Authority"},
            ]
        )
        mock_load_collections.return_value = reports_df
        mock_reports_for_collection.return_value = reports_df

        first = self.client.get(
            reverse("explore"),
            {
                "ai_filter": "custom",
                "area": ["North"],
                "receiver": ["NHS Trust"],
                "date_start": "2025-01-01",
                "date_end": "2025-01-31",
            },
        )
        self.assertEqual(first.status_code, 200)

        second = self.client.get(reverse("explore"))
        self.assertEqual(second.status_code, 200)
        explore = second.context["explore"]
        self.assertEqual(explore["selected_ai_filter"], "custom")
        self.assertEqual(explore["selected_areas"], ["North"])
        self.assertEqual(explore["selected_receivers"], ["NHS Trust"])
        self.assertEqual(explore["date_start"], "2025-01-01")
        self.assertEqual(explore["date_end"], "2025-01-31")

    @patch("accounts.views.load_collections_dataset")
    @patch("accounts.views.reports_for_collection")
    def test_explore_reset_clears_persisted_filters(self, mock_reports_for_collection, mock_load_collections):
        reports_df = pd.DataFrame(
            [
                {"date": "2025-01-01", "area": "North", "receiver": "NHS Trust"},
                {"date": "2025-01-02", "area": "South", "receiver": "Local Authority"},
            ]
        )
        mock_load_collections.return_value = reports_df
        mock_reports_for_collection.return_value = reports_df

        self.client.get(
            reverse("explore"),
            {
                "ai_filter": "custom",
                "area": ["North"],
                "receiver": ["NHS Trust"],
                "date_start": "2025-01-01",
                "date_end": "2025-01-31",
            },
        )

        response = self.client.get(f"{reverse('explore')}?reset=1", follow=True)
        self.assertEqual(response.status_code, 200)
        explore = response.context["explore"]
        self.assertEqual(explore["selected_areas"], [])
        self.assertEqual(explore["selected_receivers"], [])
        self.assertEqual(explore["date_start"], "")
        self.assertEqual(explore["date_end"], "")


@override_settings(WORKBENCH_BASE_URL="https://pfdtoolkit.org")
class SeoViewTests(TestCase):
    def test_public_pages_render_core_seo_metadata_without_version_titles(self):
        for route_name in ("landing", "about", "research", "services", "privacy-policy", "cookie-policy"):
            response = self.client.get(reverse(route_name))
            self.assertEqual(response.status_code, 200)
            self.assertContains(response, '<meta name="description"', html=False)
            self.assertContains(response, '<meta name="robots" content="index, follow">', html=False)
            self.assertContains(response, '<link rel="canonical" href="https://pfdtoolkit.org/', html=False)
            self.assertContains(response, '<meta property="og:title"', html=False)
            self.assertContains(response, '<meta name="twitter:card" content="summary">', html=False)
            self.assertNotContains(response, "v0.2")
            self.assertNotContains(response, "V0.2")

    @patch("accounts.views.load_collections_dataset")
    @patch("accounts.views.reports_for_collection")
    def test_explore_page_has_indexable_search_metadata(self, mock_reports_for_collection, mock_load_collections):
        reports_df = pd.DataFrame(
            [{"date": "2025-01-01", "area": "Area A", "receiver": "Receiver X", "title": "Report A"}]
        )
        mock_load_collections.return_value = reports_df
        mock_reports_for_collection.return_value = reports_df

        response = self.client.get(reverse("explore"), {"q": "medication"})
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Explore Prevention of Future Death Reports · PFD Toolkit")
        self.assertContains(response, '<meta name="robots" content="index, follow">', html=False)
        self.assertContains(response, '<link rel="canonical" href="https://pfdtoolkit.org/explore/">', html=False)
        self.assertContains(response, "SearchResultsPage")
        self.assertNotContains(response, "v0.2")

    def test_robots_and_sitemap_are_available(self):
        robots = self.client.get(reverse("robots-txt"))
        self.assertEqual(robots.status_code, 200)
        self.assertEqual(robots["Content-Type"], "text/plain; charset=utf-8")
        robots_body = robots.content.decode("utf-8")
        self.assertIn("Sitemap: https://pfdtoolkit.org/sitemap.xml", robots_body)
        self.assertIn("Disallow: /admin/", robots_body)
        self.assertIn("Allow: /collections/", robots_body)
        self.assertIn("Allow: /services/", robots_body)
        self.assertIn("Allow: /privacy/", robots_body)
        self.assertIn("Allow: /cookies/", robots_body)

        sitemap = self.client.get(reverse("sitemap-xml"))
        self.assertEqual(sitemap.status_code, 200)
        self.assertEqual(sitemap["Content-Type"], "application/xml; charset=utf-8")
        sitemap_body = sitemap.content.decode("utf-8")
        self.assertIn("<loc>https://pfdtoolkit.org/</loc>", sitemap_body)
        self.assertIn("<loc>https://pfdtoolkit.org/explore/</loc>", sitemap_body)
        self.assertIn("<loc>https://pfdtoolkit.org/services/</loc>", sitemap_body)
        self.assertIn("<loc>https://pfdtoolkit.org/privacy/</loc>", sitemap_body)
        self.assertIn("<loc>https://pfdtoolkit.org/cookies/</loc>", sitemap_body)
        self.assertIn("<loc>https://pfdtoolkit.org/collections/wales/</loc>", sitemap_body)
