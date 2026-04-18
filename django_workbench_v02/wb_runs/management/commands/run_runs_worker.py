import uuid

from django.core.management.base import BaseCommand

from wb_runs.worker import (
    finalize_stuck_cancellations,
    process_single_available_run,
    run_worker_loop,
)


class Command(BaseCommand):
    help = "Run the investigation-run worker loop."

    def add_arguments(self, parser):
        parser.add_argument("--worker-id", default="")
        parser.add_argument("--poll-seconds", type=float, default=5.0)
        parser.add_argument("--max-runs", type=int, default=None)
        parser.add_argument(
            "--once",
            action="store_true",
            help="Process at most one available run and exit.",
        )
        parser.add_argument(
            "--sleep-between-stages-seconds",
            type=float,
            default=0.0,
            help="Optional delay between stage transitions (useful for demos).",
        )
        parser.add_argument(
            "--finalize-cancelling-only",
            action="store_true",
            help="Only reconcile cancelling runs into cancelled terminal state.",
        )

    def handle(self, *args, **options):
        worker_id = options["worker_id"] or f"worker-{uuid.uuid4()}"
        poll_seconds = options["poll_seconds"]
        max_runs = options["max_runs"]
        once = options["once"]
        sleep_between_stages_seconds = options["sleep_between_stages_seconds"]
        finalize_cancelling_only = options["finalize_cancelling_only"]

        if finalize_cancelling_only:
            count = finalize_stuck_cancellations(worker_id=worker_id)
            self.stdout.write(
                self.style.SUCCESS(
                    f"[{worker_id}] Finalized {count} cancelling run(s)."
                )
            )
            return

        if once:
            run = process_single_available_run(
                worker_id=worker_id,
                sleep_between_stages_seconds=sleep_between_stages_seconds,
            )
            if run is None:
                self.stdout.write(f"[{worker_id}] No queued runs found.")
            else:
                self.stdout.write(
                    self.style.SUCCESS(
                        f"[{worker_id}] Processed run {run.id} -> {run.status}."
                    )
                )
            return

        processed = run_worker_loop(
            worker_id=worker_id,
            poll_seconds=poll_seconds,
            max_runs=max_runs,
            sleep_between_stages_seconds=sleep_between_stages_seconds,
        )
        self.stdout.write(
            self.style.SUCCESS(
                f"[{worker_id}] Worker loop exited after processing {processed} run(s)."
            )
        )
