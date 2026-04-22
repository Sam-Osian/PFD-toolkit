from datetime import timedelta

from django.conf import settings
from django.core.management.base import BaseCommand, CommandError
from django.utils import timezone

from wb_runs.models import RunWorkerHeartbeat


class Command(BaseCommand):
    help = "Check whether run-worker heartbeat is fresh enough for launch readiness."

    def add_arguments(self, parser):
        parser.add_argument(
            "--worker-id",
            default="",
            help="Optional specific worker id to validate.",
        )
        parser.add_argument(
            "--stale-seconds",
            type=int,
            default=None,
            help="Override stale threshold in seconds (default uses WORKER_HEARTBEAT_STALE_SECONDS).",
        )

    def handle(self, *args, **options):
        worker_id = str(options.get("worker_id") or "").strip()
        stale_seconds = options.get("stale_seconds")
        if stale_seconds is None:
            stale_seconds = int(getattr(settings, "WORKER_HEARTBEAT_STALE_SECONDS", 120))
        stale_seconds = max(1, int(stale_seconds))

        threshold = timezone.now() - timedelta(seconds=stale_seconds)
        queryset = RunWorkerHeartbeat.objects.all()
        if worker_id:
            queryset = queryset.filter(worker_id=worker_id)

        heartbeat = queryset.order_by("-last_seen_at").first()
        if heartbeat is None:
            scope = f"worker '{worker_id}'" if worker_id else "any run worker"
            raise CommandError(f"No heartbeat found for {scope}.")

        if heartbeat.last_seen_at < threshold:
            raise CommandError(
                "Run worker heartbeat is stale "
                f"(worker={heartbeat.worker_id}, last_seen_at={heartbeat.last_seen_at.isoformat()})."
            )

        self.stdout.write(
            self.style.SUCCESS(
                "Run worker heartbeat healthy: "
                f"worker={heartbeat.worker_id} state={heartbeat.state or 'unknown'} "
                f"last_seen_at={heartbeat.last_seen_at.isoformat()}"
            )
        )
