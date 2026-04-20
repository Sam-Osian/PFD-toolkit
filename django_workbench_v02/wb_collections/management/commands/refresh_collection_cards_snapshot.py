from __future__ import annotations

from django.core.management.base import BaseCommand

from wb_collections.services import refresh_collection_cards_snapshot


class Command(BaseCommand):
    help = "Refresh persisted collection-card counters snapshot."

    def add_arguments(self, parser):
        parser.add_argument(
            "--no-force-refresh",
            action="store_true",
            help="Reuse current in-process dataset cache instead of forcing source refresh.",
        )

    def handle(self, *args, **options):
        no_force_refresh = bool(options["no_force_refresh"])
        snapshot = refresh_collection_cards_snapshot(
            force_refresh_dataset=not no_force_refresh
        )
        self.stdout.write(
            self.style.SUCCESS(
                f"snapshot_key={snapshot.key} cards={len(snapshot.cards_json)} "
                f"source_rows={snapshot.source_row_count} generated_at={snapshot.generated_at.isoformat()}"
            )
        )
