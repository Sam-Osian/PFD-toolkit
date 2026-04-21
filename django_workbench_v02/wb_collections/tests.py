from unittest.mock import patch

import pandas as pd
from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse
from django.utils import timezone

from wb_investigations.models import Investigation
from wb_collections.services import refresh_collection_cards_snapshot
from wb_workspaces.models import Workspace

User = get_user_model()


class CollectionViewTests(TestCase):
    def setUp(self):
        self.user = User.objects.create_user(email="collections@example.com", password="x")
        self.dataset = pd.DataFrame(
            [
                {
                    "date": "2024-01-01",
                    "coroner": "A",
                    "area": "Wales",
                    "receiver": "NHS Trust",
                    "investigation": "Investigation text",
                    "circumstances": "Circumstances text",
                    "concerns": "Medication incident",
                    "url": "https://example.com/r1",
                    "collection_wales": True,
                    "collection_nhs": True,
                    "theme_medication_safety": True,
                    "theme_unapproved_x": True,
                },
                {
                    "date": "2024-01-02",
                    "coroner": "B",
                    "area": "London",
                    "receiver": "Local Authority",
                    "investigation": "Another investigation",
                    "circumstances": "Another circumstances",
                    "concerns": "Road safety issue",
                    "url": "https://example.com/r2",
                    "collection_wales": False,
                    "collection_nhs": False,
                    "theme_medication_safety": False,
                    "theme_unapproved_x": False,
                },
            ]
        )

    @patch("wb_collections.services.load_reports")
    def test_collection_list_renders(self, mock_load_reports):
        mock_load_reports.return_value = self.dataset.copy()
        refresh_collection_cards_snapshot(force_refresh_dataset=False)
        response = self.client.get(reverse("collection-list"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Wales")
        self.assertContains(response, "NHS Bodies")
        self.assertNotContains(response, "All reports")
        self.assertNotContains(response, "Custom collection")

    @patch("wb_collections.services.load_reports")
    def test_collection_detail_custom_search_filters_rows(self, mock_load_reports):
        mock_load_reports.return_value = self.dataset.copy()
        response = self.client.get(
            reverse("collection-detail", kwargs={"collection_slug": "custom-search"}),
            {"q": "medication"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Collection ·")
        self.assertContains(response, "Filter")
        self.assertContains(response, "Interactive report list")
        self.assertNotContains(response, "AI filter:")

    @patch("wb_collections.services.load_reports")
    def test_collection_reports_panel_uses_collection_scope(self, mock_load_reports):
        mock_load_reports.return_value = self.dataset.copy()
        response = self.client.get(
            reverse("collection-reports-panel", kwargs={"collection_slug": "custom-search"}),
            {"q": "medication", "show_reports": "1"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Showing 1-1 of 1")
        self.assertContains(response, "Medication incident")
        self.assertNotContains(response, "Road safety issue")

    @patch("wb_collections.services.load_reports")
    def test_collection_reports_panel_uses_selected_collection_slug(self, mock_load_reports):
        mock_load_reports.return_value = self.dataset.copy()
        response = self.client.get(
            reverse("collection-reports-panel", kwargs={"collection_slug": "nhs"}),
            {"show_reports": "1"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Showing 1-1 of 1")
        self.assertContains(response, "NHS Trust")
        self.assertNotContains(response, "Local Authority")

    @patch("wb_collections.services.load_reports")
    def test_collection_export_csv_uses_collection_scope(self, mock_load_reports):
        mock_load_reports.return_value = self.dataset.copy()
        response = self.client.get(
            reverse("collection-export-csv", kwargs={"collection_slug": "custom-search"}),
            {"q": "medication"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertEqual(response["Content-Type"], "text/csv; charset=utf-8")
        csv_text = response.content.decode("utf-8")
        self.assertIn("Medication incident", csv_text)
        self.assertNotIn("Road safety issue", csv_text)

    @patch("wb_collections.services.load_reports")
    def test_collection_copy_creates_workspace_and_investigation_scope(self, mock_load_reports):
        mock_load_reports.return_value = self.dataset.copy()
        self.client.force_login(self.user)
        response = self.client.post(
            reverse("collection-copy", kwargs={"collection_slug": "custom-search"}),
            data={
                "q": "medication",
                "coroner": ["A"],
                "area": ["Wales"],
                "receiver": ["NHS Trust"],
                "workbook_title": "Medication Copy",
            },
        )
        self.assertEqual(response.status_code, 302)
        workspace = Workspace.objects.filter(created_by=self.user).latest("created_at")
        self.assertEqual(workspace.title, "Medication Copy")
        investigation = Investigation.objects.get(workspace=workspace)
        self.assertEqual(investigation.scope_json.get("collection_slug"), "custom-search")
        self.assertEqual(investigation.scope_json.get("collection_query"), "medication")
        self.assertEqual(
            investigation.scope_json.get("selected_filters"),
            {
                "coroner": ["A"],
                "area": ["Wales"],
                "receiver": ["NHS Trust"],
            },
        )
        allowlist = investigation.scope_json.get("report_identity_allowlist") or []
        self.assertEqual(len(allowlist), 1)

    @patch("wb_collections.services.load_reports")
    def test_collection_list_hides_only_all_and_custom_cards(self, mock_load_reports):
        mock_load_reports.return_value = self.dataset.copy()
        refresh_collection_cards_snapshot(force_refresh_dataset=False)
        response = self.client.get(reverse("collection-list"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Wales")
        self.assertContains(response, "NHS Bodies")
        self.assertContains(response, "Government departments")
        self.assertContains(response, "Prisons")
        self.assertNotContains(response, "All reports")
        self.assertNotContains(response, "Custom collection")
        self.assertContains(response, "Medication Safety")

    @patch("wb_collections.services._approved_theme_columns_with_status")
    @patch("wb_collections.services.load_reports")
    def test_collection_list_respects_approved_theme_schema(
        self,
        mock_load_reports,
        mock_approved_columns,
    ):
        mock_load_reports.return_value = self.dataset.copy()
        mock_approved_columns.return_value = (["theme_medication_safety"], True)
        refresh_collection_cards_snapshot(force_refresh_dataset=False)
        response = self.client.get(reverse("collection-list"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Wales")
        self.assertContains(response, "Medication Safety")
        self.assertNotContains(response, "Unapproved X")

    @patch("wb_collections.services._approved_theme_columns_with_status")
    @patch("wb_collections.services.load_reports")
    def test_collection_list_falls_back_to_discovered_when_theme_schema_invalid(
        self,
        mock_load_reports,
        mock_approved_columns,
    ):
        mock_load_reports.return_value = self.dataset.copy()
        mock_approved_columns.return_value = ([], False)
        refresh_collection_cards_snapshot(force_refresh_dataset=False)
        response = self.client.get(reverse("collection-list"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Wales")
        self.assertContains(response, "Medication Safety")
        self.assertContains(response, "Unapproved X")

    @patch("wb_collections.views.refresh_collection_cards_snapshot")
    @patch("wb_collections.views.get_collection_cards_for_list")
    def test_collection_list_auto_refreshes_snapshot_when_missing(
        self,
        mock_get_collection_cards_for_list,
        mock_refresh_snapshot,
    ):
        mock_get_collection_cards_for_list.side_effect = [
            ([], None),
            (
                [
                    {
                        "slug": "custom",
                        "title": "All reports",
                        "description": "Open the full PFD archive.",
                        "count": 2,
                    },
                    {
                        "slug": "wales",
                        "title": "Wales",
                        "description": "Wales collection.",
                        "count": 1,
                    },
                ],
                timezone.now(),
            ),
        ]
        response = self.client.get(reverse("collection-list"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Wales")
        mock_refresh_snapshot.assert_called_once_with(force_refresh_dataset=False)

    @patch("wb_collections.views.collection_cards")
    @patch("wb_collections.views.load_collections_dataset")
    @patch("wb_collections.views.refresh_collection_cards_snapshot")
    @patch("wb_collections.views.get_collection_cards_for_list")
    def test_collection_list_falls_back_to_live_cards_when_snapshot_unavailable(
        self,
        mock_get_collection_cards_for_list,
        mock_refresh_snapshot,
        mock_load_collections_dataset,
        mock_collection_cards,
    ):
        mock_get_collection_cards_for_list.return_value = ([], None)
        mock_refresh_snapshot.side_effect = RuntimeError("snapshot refresh failed")
        mock_load_collections_dataset.return_value = self.dataset.copy()
        mock_collection_cards.return_value = [
            {
                "slug": "custom",
                "title": "All reports",
                "description": "Open the full PFD archive.",
                "count": 2,
            },
            {
                "slug": "wales",
                "title": "Wales",
                "description": "Wales collection.",
                "count": 1,
            },
        ]

        response = self.client.get(reverse("collection-list"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Wales")
        self.assertNotContains(response, "All reports")
        mock_load_collections_dataset.assert_called_once_with(force_refresh=False)
        mock_collection_cards.assert_called_once()

    @patch("wb_collections.services.load_reports")
    def test_multiple_collection_copies_are_retained(self, mock_load_reports):
        mock_load_reports.return_value = self.dataset.copy()
        self.client.force_login(self.user)

        first = self.client.post(
            reverse("collection-copy", kwargs={"collection_slug": "custom-search"}),
            data={"q": "medication", "workbook_title": "Medication Copy"},
        )
        self.assertEqual(first.status_code, 302)
        second = self.client.post(
            reverse("collection-copy", kwargs={"collection_slug": "custom"}),
            data={"workbook_title": "All Reports Copy"},
        )
        self.assertEqual(second.status_code, 302)

        self.assertEqual(self.user.created_workspaces.count(), 2)
