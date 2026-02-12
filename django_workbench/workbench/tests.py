import pandas as pd
from django.test import TestCase
from django.urls import reverse

from .models import Workbook
from .state import dataframe_to_payload


class WorkbenchViewTests(TestCase):
    """Smoke tests for the Workbench page."""

    def test_index_renders(self) -> None:
        response = self.client.get(reverse("workbench:index"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "PFD Toolkit Workbench")

    def test_explore_dashboard_payload_splits_receivers(self) -> None:
        reports_df = pd.DataFrame(
            [
                {
                    "date": "2024-01-10",
                    "coroner": "A. Example",
                    "area": "London Inner South",
                    "receiver": "NHS England; Department of Health ; NHS England",
                },
                {
                    "date": "2024-02-20",
                    "coroner": "B. Example",
                    "area": "London Inner South",
                    "receiver": "",
                },
            ]
        )

        session = self.client.session
        session["reports_df"] = dataframe_to_payload(reports_df)
        session["reports_df_initial"] = dataframe_to_payload(reports_df)
        session["theme_summary_table"] = dataframe_to_payload(
            pd.DataFrame([{"Theme": "Communication", "Count": 2, "%": 100.0}])
        )
        session.save()

        response = self.client.get(reverse("workbench:explore"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Interactive dashboard")

        payload = response.context["explore_dashboard_payload"]
        self.assertEqual(payload["rows"][0]["receivers"], ["NHS England", "Department of Health"])
        self.assertIn("NHS England", payload["options"]["receivers"])
        self.assertIn("Department of Health", payload["options"]["receivers"])

    def test_workbook_create_save_update_and_public_view(self) -> None:
        reports_df = pd.DataFrame(
            [
                {
                    "date": "2024-01-10",
                    "coroner": "A. Example",
                    "area": "London",
                    "receiver": "NHS England",
                },
                {
                    "date": "2024-02-20",
                    "coroner": "B. Example",
                    "area": "Merseyside",
                    "receiver": "Department of Health",
                },
            ]
        )
        session = self.client.session
        session["reports_df"] = dataframe_to_payload(reports_df)
        session["reports_df_initial"] = dataframe_to_payload(reports_df)
        session.save()

        create_response = self.client.post(
            reverse("workbench:workbook_create"),
            data={
                "title": "Emergency Response Workbook",
                "filters": {"coroner": ["A. Example"], "area": [], "receiver": []},
            },
            content_type="application/json",
        )
        self.assertEqual(create_response.status_code, 200)
        create_payload = create_response.json()
        self.assertTrue(create_payload["ok"])
        self.assertIn("/workbooks/", create_payload["share_url"])
        self.assertIn("-emergency-response-workbook/", create_payload["share_url"])
        self.assertEqual(Workbook.objects.count(), 1)
        workbook = Workbook.objects.get()
        self.assertEqual(workbook.title, "Emergency Response Workbook")
        self.assertEqual(
            workbook.snapshot["dashboard_payload"]["selected"]["coroner"],
            ["A. Example"],
        )

        save_response = self.client.post(
            reverse("workbench:workbook_save", kwargs={"public_id": workbook.public_id}),
            data={
                "title": "Emergency Response Workbook v2",
                "edit_token": str(workbook.edit_token),
                "filters": {"coroner": [], "area": ["Merseyside"], "receiver": []},
            },
            content_type="application/json",
        )
        self.assertEqual(save_response.status_code, 200)
        save_payload = save_response.json()
        self.assertTrue(save_payload["ok"])
        workbook.refresh_from_db()
        self.assertEqual(workbook.title, "Emergency Response Workbook v2")
        self.assertEqual(workbook.snapshot["dashboard_payload"]["selected"]["area"], ["Merseyside"])
        workbook.snapshot["theme_summary_df"] = pd.DataFrame(
            [{"Theme": "Communication", "Count": 1, "%": 100.0}]
        ).to_json(orient="split")
        workbook.save(update_fields=["snapshot", "updated_at"])

        public_response = self.client.get(
            reverse(
                "workbench:workbook_public",
                kwargs={
                    "share_number": workbook.share_number,
                    "title_slug": "emergency-response-workbook-v2",
                },
            )
        )
        self.assertEqual(public_response.status_code, 200)
        self.assertContains(public_response, "Read-only")
        self.assertContains(public_response, "Emergency Response Workbook v2")
        self.assertContains(public_response, "Download reports")
        self.assertContains(public_response, "Show individual reports")
        self.assertContains(public_response, "Earliest / latest report")
        self.assertContains(public_response, "Thematic snapshot")

    def test_workbook_title_validation_rejects_invalid_characters(self) -> None:
        reports_df = pd.DataFrame([{"date": "2024-01-10", "coroner": "A", "area": "B", "receiver": "C"}])
        session = self.client.session
        session["reports_df"] = dataframe_to_payload(reports_df)
        session["reports_df_initial"] = dataframe_to_payload(reports_df)
        session.save()

        response = self.client.post(
            reverse("workbench:workbook_create"),
            data={"title": "Invalid/Name?", "filters": {"coroner": [], "area": [], "receiver": []}},
            content_type="application/json",
        )
        self.assertEqual(response.status_code, 400)
        payload = response.json()
        self.assertFalse(payload["ok"])
        self.assertIn("letters, numbers, spaces, and hyphens", payload["error"])

    def test_workbook_public_download_endpoint_returns_csv(self) -> None:
        reports_df = pd.DataFrame([{"date": "2024-01-10", "coroner": "A", "area": "B", "receiver": "C"}])
        session = self.client.session
        session["reports_df"] = dataframe_to_payload(reports_df)
        session["reports_df_initial"] = dataframe_to_payload(reports_df)
        session.save()

        create_response = self.client.post(
            reverse("workbench:workbook_create"),
            data={"title": "Download Sheet", "filters": {"coroner": [], "area": [], "receiver": []}},
            content_type="application/json",
        )
        self.assertEqual(create_response.status_code, 200)
        workbook = Workbook.objects.get(title="Download Sheet")
        response = self.client.post(
            reverse(
                "workbench:workbook_download",
                kwargs={"share_number": workbook.share_number, "title_slug": "download-sheet"},
            ),
            data={"download_include_dataset": "on"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "text/csv; charset=utf-8")
