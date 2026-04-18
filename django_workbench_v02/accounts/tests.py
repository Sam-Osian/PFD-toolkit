from django.contrib.auth import get_user_model
from django.test import TestCase
from django.urls import reverse


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

    def test_login_redirects_to_auth0_when_configured(self):
        response = self.client.get(reverse("accounts-login"))
        self.assertEqual(response.status_code, 302)
        self.assertIn("authorize?", response.url)

    def test_admin_login_proxy_redirects_to_auth_login(self):
        response = self.client.get("/admin/login/")
        self.assertEqual(response.status_code, 302)
        self.assertIn("/auth/login/", response.url)
