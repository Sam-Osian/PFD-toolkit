from __future__ import annotations

from django.conf import settings
from django.core.management.base import BaseCommand

from wb_workspaces.lifecycle import run_lifecycle_maintenance


class Command(BaseCommand):
    help = "Run inactivity keepalive/expiry maintenance for artifacts and workspaces."

    def add_arguments(self, parser):
        parser.add_argument(
            "--inactivity-days",
            type=int,
            default=int(getattr(settings, "LIFECYCLE_INACTIVITY_DAYS", 365)),
            help="Sliding inactivity window in days (default from settings).",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Compute lifecycle actions without writing changes.",
        )
        parser.add_argument(
            "--skip-workspace-archive",
            action="store_true",
            help="Do not archive stale workspaces in this run.",
        )

    def handle(self, *args, **options):
        inactivity_days = options["inactivity_days"]
        dry_run = options["dry_run"]
        skip_workspace_archive = options["skip_workspace_archive"]

        result = run_lifecycle_maintenance(
            inactivity_days=inactivity_days,
            dry_run=dry_run,
            archive_workspaces=not skip_workspace_archive,
        )

        mode = "DRY RUN" if dry_run else "APPLIED"
        self.stdout.write(f"[{mode}] inactivity_days={result.inactivity_days}")
        self.stdout.write(f"Artifacts scanned: {result.artifacts_scanned}")
        self.stdout.write(f"Artifacts expired: {result.artifacts_expired}")
        self.stdout.write(
            f"Artifact expiry refreshed: {result.artifacts_expiry_refreshed}"
        )
        if skip_workspace_archive:
            self.stdout.write("Workspaces archived: skipped")
        else:
            self.stdout.write(f"Workspaces scanned: {result.workspaces_scanned}")
            self.stdout.write(f"Workspaces archived: {result.workspaces_archived}")
