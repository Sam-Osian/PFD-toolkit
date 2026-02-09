from django.test import TestCase
from django.urls import reverse


class WorkbenchViewTests(TestCase):
    """Smoke tests for the Workbench page."""

    def test_index_renders(self) -> None:
        response = self.client.get(reverse("workbench:index"))
        self.assertEqual(response.status_code, 200)
        self.assertContains(response, "PFD Toolkit Workbench")
