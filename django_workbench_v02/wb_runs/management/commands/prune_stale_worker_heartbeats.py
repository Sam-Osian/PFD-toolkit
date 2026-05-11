from __future__ import annotations

from datetime import timedelta

from django.conf import settings
from django.core.management.base import BaseCommand
from django.utils import timezone

from wb_runs.models import InvestigationRun, RunStatus, RunWorkerHeartbeat


IDLE_LIKE_STATES = {
    "",
    "idle",
    "polling",
    "processed",
    "error",
    "pipeline_queue_error",
}

ACTIVE_RUN_STATUSES = {
    RunStatus.STARTING,
    RunStatus.RUNNING,
    RunStatus.CANCELLING,
}


class Command(BaseCommand):
    help = (
        "Prune stale worker heartbeat rows while protecting active and explicitly"
        " protected workers (for example DGX local workers)."
    )

    def add_arguments(self, parser):
        parser.add_argument(
            "--stale-seconds",
            type=int,
            default=int(getattr(settings, "WORKER_HEARTBEAT_PRUNE_STALE_SECONDS", 172800)),
            help="Delete rows older than this many seconds (default: settings value or 172800).",
        )
        parser.add_argument(
            "--protected-worker-id",
            action="append",
            default=[],
            help="Worker ID to always protect. Can be passed multiple times.",
        )
        parser.add_argument(
            "--include-non-idle",
            action="store_true",
            help="Also prune stale rows that are not in an idle-like state.",
        )
        parser.add_argument(
            "--dry-run",
            action="store_true",
            help="Show what would be deleted without deleting records.",
        )

    def handle(self, *args, **options):
        stale_seconds = max(1, int(options.get("stale_seconds") or 1))
        include_non_idle = bool(options.get("include_non_idle"))
        dry_run = bool(options.get("dry_run"))

        protected_ids = set(getattr(settings, "WORKER_HEARTBEAT_PROTECTED_IDS", ()) or ())
        for worker_id in options.get("protected_worker_id") or []:
            value = str(worker_id or "").strip()
            if value:
                protected_ids.add(value)

        active_worker_ids = set(
            InvestigationRun.objects.filter(status__in=ACTIVE_RUN_STATUSES)
            .exclude(worker_id="")
            .values_list("worker_id", flat=True)
        )

        threshold = timezone.now() - timedelta(seconds=stale_seconds)
        queryset = RunWorkerHeartbeat.objects.filter(last_seen_at__lt=threshold).order_by("last_seen_at")
        if protected_ids:
            queryset = queryset.exclude(worker_id__in=protected_ids)
        if active_worker_ids:
            queryset = queryset.exclude(worker_id__in=active_worker_ids)

        candidates = []
        skipped_non_idle = 0
        for heartbeat in queryset:
            state = str(heartbeat.state or "").strip().lower()
            if not include_non_idle and state not in IDLE_LIKE_STATES:
                skipped_non_idle += 1
                continue
            candidates.append(heartbeat)

        candidate_ids = [heartbeat.pk for heartbeat in candidates]
        if dry_run:
            self.stdout.write(
                self.style.WARNING(
                    f"Dry run: {len(candidate_ids)} stale heartbeat row(s) would be deleted."
                )
            )
        else:
            deleted_count, _ = RunWorkerHeartbeat.objects.filter(pk__in=candidate_ids).delete()
            self.stdout.write(
                self.style.SUCCESS(
                    f"Deleted {deleted_count} stale heartbeat row(s)."
                )
            )

        self.stdout.write(
            "Protected workers: "
            + (", ".join(sorted(protected_ids)) if protected_ids else "(none)")
        )
        self.stdout.write(f"Active worker IDs protected automatically: {len(active_worker_ids)}")
        if skipped_non_idle and not include_non_idle:
            self.stdout.write(
                f"Skipped {skipped_non_idle} stale non-idle row(s). Use --include-non-idle to prune them."
            )
        if candidates:
            preview = "\n".join(
                f"- {hb.worker_id} state={hb.state or '-'} last_seen_at={hb.last_seen_at.isoformat()}"
                for hb in candidates[:40]
            )
            self.stdout.write("Rows matched:\n" + preview)
            if len(candidates) > 40:
                self.stdout.write(f"... and {len(candidates) - 40} more.")
