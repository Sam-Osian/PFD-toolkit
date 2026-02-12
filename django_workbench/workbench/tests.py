import pandas as pd
from django.test import TestCase
from django.urls import reverse

from .models import Workbook
from .state import dataframe_from_payload, dataframe_to_payload


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

    def test_exclude_and_restore_report_updates_working_dataset(self) -> None:
        reports_df = pd.DataFrame(
            [
                {"date": "2024-01-10", "coroner": "A", "area": "B", "receiver": "C"},
                {"date": "2024-01-11", "coroner": "X", "area": "Y", "receiver": "Z"},
            ]
        )
        session = self.client.session
        session["reports_df"] = dataframe_to_payload(reports_df)
        session["reports_df_initial"] = dataframe_to_payload(reports_df)
        session.save()

        self.client.get(reverse("workbench:explore"))
        hydrated_session = self.client.session
        hydrated_reports = dataframe_from_payload(hydrated_session["reports_df"])
        row_id = str(hydrated_reports.iloc[0]["_workbench_row_id"])

        exclude_response = self.client.post(
            f"{reverse('workbench:explore')}?page=1",
            data={
                "action": "exclude_report",
                "report_row_id": row_id,
                "exclusion_reason": "Out of scope",
            },
        )
        self.assertEqual(exclude_response.status_code, 302)
        self.assertIn("?page=1", exclude_response["Location"])

        excluded_session = self.client.session
        active_reports = dataframe_from_payload(excluded_session["reports_df"])
        excluded_reports = dataframe_from_payload(excluded_session["excluded_reports_df"])
        self.assertEqual(len(active_reports), 1)
        self.assertEqual(len(excluded_reports), 1)
        self.assertEqual(str(excluded_reports.iloc[0]["_workbench_row_id"]), row_id)
        self.assertEqual(excluded_reports.iloc[0]["_exclusion_reason"], "Out of scope")

        restore_response = self.client.post(
            f"{reverse('workbench:explore')}?page=1",
            data={
                "action": "restore_excluded_report",
                "report_row_id": row_id,
            },
        )
        self.assertEqual(restore_response.status_code, 302)
        self.assertIn("?page=1", restore_response["Location"])

        restored_session = self.client.session
        restored_reports = dataframe_from_payload(restored_session["reports_df"])
        remaining_excluded = dataframe_from_payload(restored_session["excluded_reports_df"])
        self.assertEqual(len(restored_reports), 2)
        self.assertEqual(len(remaining_excluded), 0)

    def test_explore_download_option_for_excluded_reports_only_shows_when_present(self) -> None:
        reports_df = pd.DataFrame(
            [
                {"date": "2024-01-10", "coroner": "A", "area": "B", "receiver": "C"},
                {"date": "2024-01-11", "coroner": "X", "area": "Y", "receiver": "Z"},
            ]
        )
        session = self.client.session
        session["reports_df"] = dataframe_to_payload(reports_df)
        session["reports_df_initial"] = dataframe_to_payload(reports_df)
        session.save()

        response_before = self.client.get(reverse("workbench:explore"))
        self.assertEqual(response_before.status_code, 200)
        self.assertNotContains(response_before, "download_include_excluded")
        self.assertNotContains(response_before, "Show excluded reports")

        hydrated = dataframe_from_payload(self.client.session["reports_df"])
        row_id = str(hydrated.iloc[0]["_workbench_row_id"])
        self.client.post(
            reverse("workbench:explore"),
            data={"action": "exclude_report", "report_row_id": row_id, "exclusion_reason": "Out of scope"},
        )

        response_after = self.client.get(reverse("workbench:explore"))
        self.assertEqual(response_after.status_code, 200)
        self.assertContains(response_after, "download_include_excluded")
        self.assertContains(response_after, "Show excluded reports (1)")

    def test_excluded_reports_are_persisted_into_shared_view(self) -> None:
        reports_df = pd.DataFrame(
            [
                {"date": "2024-01-10", "coroner": "A", "area": "B", "receiver": "C"},
                {"date": "2024-01-11", "coroner": "X", "area": "Y", "receiver": "Z"},
            ]
        )
        session = self.client.session
        session["reports_df"] = dataframe_to_payload(reports_df)
        session["reports_df_initial"] = dataframe_to_payload(reports_df)
        session.save()

        self.client.get(reverse("workbench:explore"))
        hydrated_session = self.client.session
        hydrated_reports = dataframe_from_payload(hydrated_session["reports_df"])
        row_id = str(hydrated_reports.iloc[0]["_workbench_row_id"])
        self.client.post(
            reverse("workbench:explore"),
            data={
                "action": "exclude_report",
                "report_row_id": row_id,
                "exclusion_reason": "Duplicate source",
            },
        )

        create_response = self.client.post(
            reverse("workbench:workbook_create"),
            data={"title": "Shared Exclusions", "filters": {"coroner": [], "area": [], "receiver": []}},
            content_type="application/json",
        )
        self.assertEqual(create_response.status_code, 200)
        workbook = Workbook.objects.get(title="Shared Exclusions")
        self.assertIn("excluded_reports_df", workbook.snapshot)

        public_response = self.client.get(
            reverse(
                "workbench:workbook_public",
                kwargs={"share_number": workbook.share_number, "title_slug": "shared-exclusions"},
            )
        )
        self.assertEqual(public_response.status_code, 200)
        self.assertContains(public_response, "Show excluded reports (1)")
        self.assertContains(public_response, "Duplicate source")
        self.assertContains(public_response, "download_include_excluded")
        self.assertContains(public_response, "Read-only view: exclusions and reasons are shown for transparency.")
        self.assertNotContains(public_response, "dataset-restore-btn")

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
        self.assertContains(public_response, "Make editable copy")
        self.assertContains(public_response, "Show individual reports")
        self.assertNotContains(public_response, "Show excluded reports")
        self.assertNotContains(public_response, "download_include_excluded")
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

    def test_workbook_public_download_supports_excluded_reports_csv(self) -> None:
        reports_df = pd.DataFrame(
            [
                {"date": "2024-01-10", "coroner": "A", "area": "B", "receiver": "C"},
                {"date": "2024-01-11", "coroner": "X", "area": "Y", "receiver": "Z"},
            ]
        )
        session = self.client.session
        session["reports_df"] = dataframe_to_payload(reports_df)
        session["reports_df_initial"] = dataframe_to_payload(reports_df)
        session.save()

        self.client.get(reverse("workbench:explore"))
        hydrated = dataframe_from_payload(self.client.session["reports_df"])
        row_id = str(hydrated.iloc[0]["_workbench_row_id"])
        self.client.post(
            reverse("workbench:explore"),
            data={"action": "exclude_report", "report_row_id": row_id, "exclusion_reason": "Duplicate"},
        )

        create_response = self.client.post(
            reverse("workbench:workbook_create"),
            data={"title": "Download Excluded", "filters": {"coroner": [], "area": [], "receiver": []}},
            content_type="application/json",
        )
        self.assertEqual(create_response.status_code, 200)
        workbook = Workbook.objects.get(title="Download Excluded")

        response = self.client.post(
            reverse(
                "workbench:workbook_download",
                kwargs={"share_number": workbook.share_number, "title_slug": "download-excluded"},
            ),
            data={
                "download_include_dataset": "0",
                "download_include_excluded": "on",
            },
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "text/csv; charset=utf-8")
        self.assertIn("pfd_excluded_reports.csv", response["Content-Disposition"])
        self.assertIn("reason_for_exclusion", response.content.decode("utf-8"))

    def test_workbook_clone_creates_editable_personal_copy(self) -> None:
        reports_df = pd.DataFrame(
            [
                {"date": "2024-01-10", "coroner": "A", "area": "B", "receiver": "C"},
                {"date": "2024-01-11", "coroner": "X", "area": "Y", "receiver": "Z"},
            ]
        )
        session = self.client.session
        session["reports_df"] = dataframe_to_payload(reports_df)
        session["reports_df_initial"] = dataframe_to_payload(reports_df)
        session["theme_summary_table"] = dataframe_to_payload(
            pd.DataFrame([{"Theme": "Communication", "Count": 2, "%": 100.0}])
        )
        session.save()

        create_response = self.client.post(
            reverse("workbench:workbook_create"),
            data={"title": "Shared Sheet", "filters": {"coroner": ["A"], "area": [], "receiver": []}},
            content_type="application/json",
        )
        self.assertEqual(create_response.status_code, 200)
        original_workbook = Workbook.objects.get(title="Shared Sheet")

        clone_response = self.client.post(
            reverse(
                "workbench:workbook_clone",
                kwargs={"share_number": original_workbook.share_number, "title_slug": "shared-sheet"},
            ),
        )
        self.assertEqual(clone_response.status_code, 302)
        self.assertEqual(Workbook.objects.count(), 2)

        cloned_workbook = Workbook.objects.exclude(public_id=original_workbook.public_id).get()
        self.assertEqual(cloned_workbook.title, "")
        self.assertEqual(cloned_workbook.snapshot, original_workbook.snapshot)

        location = clone_response["Location"]
        self.assertIn("/explore-pfds/?", location)
        self.assertIn(f"workbook={cloned_workbook.public_id}", location)
        self.assertIn(f"edit={cloned_workbook.edit_token}", location)
        self.assertIn("coroner=A", location)

        cloned_session = self.client.session
        cloned_reports_df = dataframe_from_payload(cloned_session["reports_df"])
        self.assertEqual(len(cloned_reports_df), 1)
        self.assertEqual(cloned_reports_df.iloc[0]["coroner"], "A")
