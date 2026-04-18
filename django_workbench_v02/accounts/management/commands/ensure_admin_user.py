from django.contrib.auth import get_user_model
from django.core.management.base import BaseCommand, CommandError


class Command(BaseCommand):
    help = "Ensure a single admin user exists with superuser/staff permissions."

    def add_arguments(self, parser):
        parser.add_argument("--email", required=True)
        parser.add_argument("--first-name", default="Sam")
        parser.add_argument("--last-name", default="Osian")
        parser.add_argument("--password", default=None)

    def handle(self, *args, **options):
        email = options["email"].strip().lower()
        if not email:
            raise CommandError("A valid --email is required.")

        user_model = get_user_model()

        defaults = {
            "first_name": options["first_name"].strip(),
            "last_name": options["last_name"].strip(),
            "is_active": True,
            "is_staff": True,
            "is_superuser": True,
        }
        user, created = user_model.objects.get_or_create(email=email, defaults=defaults)

        changed = False
        for field, value in defaults.items():
            if getattr(user, field) != value:
                setattr(user, field, value)
                changed = True

        password = options["password"]
        if password:
            user.set_password(password)
            changed = True
        elif created:
            user.set_unusable_password()
            changed = True

        if changed:
            user.save()

        status = "created" if created else "updated"
        self.stdout.write(self.style.SUCCESS(f"Admin user {status}: {user.email}"))
        if not password:
            self.stdout.write(
                self.style.WARNING(
                    "No password was supplied. Use managed Auth0 login or set a local "
                    "password with `manage.py changepassword` if needed."
                )
            )
