from __future__ import annotations

from django.core.management.base import BaseCommand

from wb_notifications.services import dispatch_pending_notifications, run_notification_dispatch_loop


class Command(BaseCommand):
    help = "Dispatch pending completion notifications for terminal runs."

    def add_arguments(self, parser):
        parser.add_argument(
            "--once",
            action="store_true",
            help="Process one dispatch cycle and exit.",
        )
        parser.add_argument(
            "--max-items",
            type=int,
            default=50,
            help="Maximum pending notifications to process per cycle.",
        )
        parser.add_argument(
            "--poll-seconds",
            type=float,
            default=5.0,
            help="Poll interval when running in loop mode.",
        )
        parser.add_argument(
            "--max-cycles",
            type=int,
            default=None,
            help="Optional cap on loop cycles (for controlled runs).",
        )

    def handle(self, *args, **options):
        once = options["once"]
        max_items = options["max_items"]
        poll_seconds = options["poll_seconds"]
        max_cycles = options["max_cycles"]

        if once:
            result = dispatch_pending_notifications(max_items=max_items)
            self.stdout.write(
                self.style.SUCCESS(
                    f"scanned={result.scanned} sent={result.sent} "
                    f"failed={result.failed} cancelled={result.cancelled}"
                )
            )
            return

        total = run_notification_dispatch_loop(
            poll_seconds=poll_seconds,
            max_items_per_cycle=max_items,
            max_cycles=max_cycles,
        )
        self.stdout.write(
            self.style.SUCCESS(
                f"total_scanned={total.scanned} total_sent={total.sent} "
                f"total_failed={total.failed} total_cancelled={total.cancelled}"
            )
        )
