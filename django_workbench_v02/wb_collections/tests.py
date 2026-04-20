from unittest.mock import patch

import pandas as pd
from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse

from wb_investigations.models import Investigation

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
                    "concerns": "Medication incident",
                    "url": "https://example.com/r1",
                    "collection_wales": True,
                    "collection_nhs": True,
                },
                {
                    "date": "2024-01-02",
                    "coroner": "B",
                    "area": "London",
                    "receiver": "Local Authority",
                    "concerns": "Road safety issue",
                    "url": "https://example.com/r2",
                    "collection_wales": False,
                    "collection_nhs": False,
                },
            ]
        )

    @patch("wb_collections.services.load_reports")
    def test_collection_list_renders(self, mock_load_reports):
        mock_load_reports.return_value = self.dataset.copy()
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

    @patch("wb_collections.services.load_reports")
    def test_collection_copy_creates_workspace_and_investigation(self, mock_load_reports):
        mock_load_reports.return_value = self.dataset.copy()
        self.client.force_login(self.user)
        response = self.client.post(
            reverse("collection-copy", kwargs={"collection_slug": "custom-search"}),
            data={"q": "medication"},
        )
        self.assertEqual(response.status_code, 302)
        investigation = Investigation.objects.filter(created_by=self.user).latest("created_at")
        self.assertEqual(investigation.scope_json.get("collection_slug"), "custom-search")
        self.assertEqual(investigation.scope_json.get("collection_query"), "medication")
        self.assertEqual(len(investigation.scope_json.get("report_identity_allowlist", [])), 1)
