import pandas as pd
from django.test import TestCase
from django.urls import reverse

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
        session.save()

        response = self.client.get(reverse("workbench:explore"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "Interactive dashboard")

        payload = response.context["explore_dashboard_payload"]
        self.assertEqual(payload["rows"][0]["receivers"], ["NHS England", "Department of Health"])
        self.assertIn("NHS England", payload["options"]["receivers"])
        self.assertIn("Department of Health", payload["options"]["receivers"])
