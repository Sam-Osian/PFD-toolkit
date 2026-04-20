from unittest.mock import patch

import pandas as pd
from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

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
        self.assertContains(response, "All reports")
        self.assertContains(response, "Custom collection")

    @patch("wb_collections.services.load_reports")
    def test_collection_detail_custom_search_filters_rows(self, mock_load_reports):
        mock_load_reports.return_value = self.dataset.copy()
        response = self.client.get(
            reverse("collection-detail", kwargs={"collection_slug": "custom-search"}),
            {"q": "medication"},
        )
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Matching reports: 1")
        self.assertContains(response, "Investigation")
        self.assertContains(response, "Circumstances")

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
    def test_collection_list_includes_theme_collection_cards(self, mock_load_reports):
        mock_load_reports.return_value = self.dataset.copy()
        refresh_collection_cards_snapshot(force_refresh_dataset=False)
        response = self.client.get(reverse("collection-list"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Medication Safety")

    @patch("wb_collections.services._approved_theme_columns_with_status")
    @patch("wb_collections.services.load_reports")
    def test_collection_list_locks_to_approved_theme_schema(self, mock_load_reports, mock_approved_columns):
        mock_load_reports.return_value = self.dataset.copy()
        mock_approved_columns.return_value = (["theme_medication_safety"], True)
        refresh_collection_cards_snapshot(force_refresh_dataset=False)
        response = self.client.get(reverse("collection-list"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Medication Safety")
        self.assertNotContains(response, "Unapproved X")

    @patch("wb_collections.services._approved_theme_columns_with_status")
    @patch("wb_collections.services.load_reports")
    def test_collection_list_falls_back_to_discovered_when_schema_invalid(
        self,
        mock_load_reports,
        mock_approved_columns,
    ):
        mock_load_reports.return_value = self.dataset.copy()
        mock_approved_columns.return_value = ([], False)
        refresh_collection_cards_snapshot(force_refresh_dataset=False)
        response = self.client.get(reverse("collection-list"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Medication Safety")
        self.assertContains(response, "Unapproved X")

    @patch("wb_collections.views.load_collections_dataset")
    def test_collection_list_does_not_load_dataset_on_request(self, mock_load_dataset):
        response = self.client.get(reverse("collection-list"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Counter snapshot not available yet")
        mock_load_dataset.assert_not_called()

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
