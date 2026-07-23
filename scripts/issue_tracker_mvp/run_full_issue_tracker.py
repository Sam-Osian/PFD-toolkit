#!/usr/bin/env python3
"""Run the frozen production issue-tracker workflow with resumable stages."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd


SCRIPT_DIR = Path(__file__).resolve().parent
FROZEN_INDEX_CONFIG = {
    "embedding_mode": "dual",
    "original_view_weight": 0.44,
    "top_k": 40,
    "edge_similarity": 0.84,
    "split_similarity": 0.872,
    "min_centroid_similarity": 0.83,
    "min_recurring_reports": 3,
}
STAGE_ORDER = ("extract", "normalize", "index", "registry", "audit", "risk")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input-csv", type=Path, default=Path("all_reports.csv"))
    parser.add_argument("--output-dir", type=Path, default=Path("artifacts/issue_index_v3_full"))
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=None,
        help="Existing run to resume; otherwise a timestamped directory is created.",
    )
    parser.add_argument(
        "--stage",
        choices=["all", *STAGE_ORDER],
        default="all",
        help="Run the complete workflow or one independently resumable stage.",
    )
    parser.add_argument("--model", default="gemma4:26b")
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--extraction-workers", type=int, default=1)
    parser.add_argument("--normalization-workers", type=int, default=2)
    parser.add_argument("--normalization-batch-size", type=int, default=12)
    parser.add_argument("--request-timeout", type=int, default=360)
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--previous-registry-dir", type=Path, default=None)
    parser.add_argument("--max-extraction-failures", type=int, default=0)
    parser.add_argument("--minimum-evidence-valid-rate", type=float, default=0.98)
    parser.add_argument("--maximum-normalization-fallback-rate", type=float, default=0.02)
    parser.add_argument(
        "--prepare-adjudication-risk",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Build the deterministic risk queue; never runs model adjudication.",
    )
    return parser.parse_args()


def resolve_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir:
        run_dir = args.run_dir.expanduser().resolve()
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        run_dir = (args.output_dir / f"run_{timestamp}").expanduser().resolve()
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def read_json(path: Path) -> dict[str, Any]:
    if not path.exists():
        raise FileNotFoundError(f"Required stage metric is missing: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def build_input_coverage(input_csv: Path, run_dir: Path) -> dict[str, Any]:
    frame = pd.read_csv(input_csv).fillna("")
    required = {"id", "url", "concerns", "circumstances"}
    if missing := required - set(frame.columns):
        raise ValueError(f"Input CSV is missing columns: {sorted(missing)}")
    if "investigation" not in frame.columns:
        frame["investigation"] = ""
    for column in ("id", "url", "concerns", "circumstances", "investigation"):
        frame[column] = frame[column].astype(str).str.strip()
    has_source = frame[
        ["concerns", "circumstances", "investigation"]
    ].apply(lambda column: column.str.len().gt(0)).any(axis=1)
    duplicate_url = frame["url"].duplicated(keep="first")
    usable = has_source & ~duplicate_url
    excluded = frame.loc[
        ~usable, ["id", "url", "concerns", "circumstances", "investigation"]
    ].copy()
    excluded["exclusion_reason"] = [
        "duplicate_url" if duplicate else "missing_source_text"
        for duplicate in duplicate_url[~usable]
    ]
    excluded.to_csv(run_dir / "00_excluded_reports.csv", index=False)
    coverage = {
        "input_rows": int(len(frame)),
        "unique_urls": int(frame["url"].nunique()),
        "usable_reports": int(usable.sum()),
        "excluded_reports": int((~usable).sum()),
        "excluded_missing_source_text": int((~has_source).sum()),
        "excluded_duplicate_url": int(duplicate_url.sum()),
    }
    (run_dir / "00_input_coverage.json").write_text(
        json.dumps(coverage, indent=2), encoding="utf-8"
    )
    return coverage


def write_workflow_manifest(
    path: Path, *, run_dir: Path, args: argparse.Namespace, stages: dict[str, Any]
) -> None:
    payload = {
        "workflow_version": "full-issue-tracker-v1",
        "run_dir": str(run_dir),
        "input_csv": str(args.input_csv.expanduser().resolve()),
        "model": args.model,
        "embedding_model": args.embedding_model,
        "frozen_index_config": FROZEN_INDEX_CONFIG,
        "quality_gates": {
            "max_extraction_failures": args.max_extraction_failures,
            "minimum_evidence_valid_rate": args.minimum_evidence_valid_rate,
            "maximum_normalization_fallback_rate": (
                args.maximum_normalization_fallback_rate
            ),
        },
        "stages": stages,
        "updated_at": datetime.now(timezone.utc).isoformat(),
    }
    temporary = path.with_suffix(".tmp")
    temporary.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    temporary.replace(path)


def run_command(command: list[str]) -> None:
    print("\nRunning:", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def validate_extraction(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    metrics = read_json(run_dir / "01_extraction_metrics.json")
    coverage = read_json(run_dir / "00_input_coverage.json")
    failures = int(metrics.get("reports_failed", 0))
    valid = int(metrics.get("valid_evidence_quotes", 0))
    invalid = int(metrics.get("invalid_evidence_quotes", 0))
    evidence_rate = valid / max(1, valid + invalid)
    if failures > args.max_extraction_failures:
        raise RuntimeError(
            f"Extraction gate failed: {failures} reports failed; "
            f"maximum is {args.max_extraction_failures}"
        )
    if evidence_rate < args.minimum_evidence_valid_rate:
        raise RuntimeError(
            f"Extraction gate failed: evidence validity {evidence_rate:.2%}; "
            f"minimum is {args.minimum_evidence_valid_rate:.2%}"
        )
    reports_in_scope = int(metrics.get("reports_in_scope", 0))
    if reports_in_scope != int(coverage["usable_reports"]):
        raise RuntimeError(
            f"Extraction coverage mismatch: {reports_in_scope} reports were in scope, "
            f"but preflight found {coverage['usable_reports']} usable reports"
        )
    return {
        "status": "complete",
        "reports_succeeded": int(metrics.get("reports_succeeded", 0)),
        "reports_failed": failures,
        "evidence_valid_rate": round(evidence_rate, 6),
    }


def validate_normalization(run_dir: Path, args: argparse.Namespace) -> dict[str, Any]:
    metrics = read_json(
        run_dir / "01_issue_occurrences_normalized.metrics.json"
    )
    selected = int(metrics.get("issues_selected", 0))
    fallback = int(
        metrics.get("issues_fallback_original", metrics.get("issues_failed", 0))
    )
    fallback_rate = fallback / max(1, selected)
    if fallback_rate > args.maximum_normalization_fallback_rate:
        raise RuntimeError(
            f"Normalization gate failed: fallback rate {fallback_rate:.2%}; "
            f"maximum is {args.maximum_normalization_fallback_rate:.2%}"
        )
    normalized_path = run_dir / "01_issue_occurrences_normalized.csv"
    if not normalized_path.exists():
        raise FileNotFoundError(f"Normalized occurrence file is missing: {normalized_path}")
    return {
        "status": "complete",
        "issues_selected": selected,
        "issues_normalized": int(metrics.get("issues_normalized", 0)),
        "issues_fallback_original": fallback,
        "fallback_rate": round(fallback_rate, 6),
    }


def main() -> None:
    args = parse_args()
    if args.extraction_workers < 1 or args.normalization_workers < 1:
        raise ValueError("Worker counts must be at least 1")
    if not 1 <= args.normalization_batch_size <= 16:
        raise ValueError("--normalization-batch-size must be between 1 and 16")
    if not 0.0 <= args.minimum_evidence_valid_rate <= 1.0:
        raise ValueError("--minimum-evidence-valid-rate must be between 0 and 1")
    if not 0.0 <= args.maximum_normalization_fallback_rate <= 1.0:
        raise ValueError("--maximum-normalization-fallback-rate must be between 0 and 1")
    run_dir = resolve_run_dir(args)
    requested = STAGE_ORDER if args.stage == "all" else (args.stage,)
    manifest_path = run_dir / "workflow_manifest.json"
    existing = read_json(manifest_path) if manifest_path.exists() else {}
    stages: dict[str, Any] = dict(existing.get("stages") or {})
    if "extract" in requested:
        stages["input_coverage"] = {
            "status": "complete",
            **build_input_coverage(args.input_csv, run_dir),
        }
    write_workflow_manifest(
        manifest_path, run_dir=run_dir, args=args, stages=stages
    )
    print(f"Production run directory: {run_dir}", flush=True)
    python = sys.executable

    if "extract" in requested:
        command = [
            python,
            str(SCRIPT_DIR / "build_issue_index.py"),
            "--stage",
            "extract",
            "--run-dir",
            str(run_dir),
            "--input-csv",
            str(args.input_csv),
            "--subset-size",
            "0",
            "--model",
            args.model,
            "--embedding-model",
            args.embedding_model,
            "--ollama-host",
            args.ollama_host,
            "--extraction-workers",
            str(args.extraction_workers),
            "--request-timeout",
            str(args.request_timeout),
            "--no-label-subissues",
        ]
        run_command(command)
        stages["extract"] = validate_extraction(run_dir, args)
        write_workflow_manifest(
            manifest_path, run_dir=run_dir, args=args, stages=stages
        )

    if "normalize" in requested:
        run_command(
            [
                python,
                str(SCRIPT_DIR / "normalize_issue_occurrences.py"),
                "--input-csv",
                str(run_dir / "01_issue_occurrences.csv"),
                "--output-csv",
                str(run_dir / "01_issue_occurrences_normalized.csv"),
                "--model",
                args.model,
                "--ollama-host",
                args.ollama_host,
                "--batch-size",
                str(args.normalization_batch_size),
                "--workers",
                str(args.normalization_workers),
                "--request-timeout",
                str(args.request_timeout),
            ]
        )
        stages["normalize"] = validate_normalization(run_dir, args)
        write_workflow_manifest(
            manifest_path, run_dir=run_dir, args=args, stages=stages
        )

    if "index" in requested:
        command = [
            python,
            str(SCRIPT_DIR / "build_issue_index.py"),
            "--stage",
            "index",
            "--run-dir",
            str(run_dir),
            "--input-csv",
            str(args.input_csv),
            "--occurrences-csv",
            str(run_dir / "01_issue_occurrences_normalized.csv"),
            "--embedding-mode",
            "dual",
            "--original-view-weight",
            str(FROZEN_INDEX_CONFIG["original_view_weight"]),
            "--embedding-model",
            args.embedding_model,
            "--ollama-host",
            args.ollama_host,
            "--model",
            args.model,
            "--top-k",
            str(FROZEN_INDEX_CONFIG["top_k"]),
            "--edge-similarity",
            str(FROZEN_INDEX_CONFIG["edge_similarity"]),
            "--split-similarity",
            str(FROZEN_INDEX_CONFIG["split_similarity"]),
            "--min-centroid-similarity",
            str(FROZEN_INDEX_CONFIG["min_centroid_similarity"]),
            "--min-recurring-reports",
            str(FROZEN_INDEX_CONFIG["min_recurring_reports"]),
            "--no-label-subissues",
        ]
        if args.allow_model_download:
            command.append("--allow-model-download")
        run_command(command)
        metrics = read_json(run_dir / "04_index_metrics.json")
        stages["index"] = {
            "status": "complete",
            "total_issue_occurrences": int(
                metrics.get("total_issue_occurrences", 0)
            ),
            "recurring_subissues": int(metrics.get("recurring_subissues", 0)),
            "embedding_mode": metrics.get("embedding_mode"),
            "original_view_weight": metrics.get("original_view_weight"),
        }
        write_workflow_manifest(
            manifest_path, run_dir=run_dir, args=args, stages=stages
        )

    if "registry" in requested:
        command = [
            python,
            str(SCRIPT_DIR / "build_issue_registry.py"),
            "--run-dir",
            str(run_dir),
        ]
        if args.previous_registry_dir:
            command.extend(
                ["--previous-registry-dir", str(args.previous_registry_dir)]
            )
        run_command(command)
        metrics = read_json(run_dir / "05_issue_registry" / "registry_metrics.json")
        stages["registry"] = {"status": "complete", **metrics}
        write_workflow_manifest(
            manifest_path, run_dir=run_dir, args=args, stages=stages
        )

    if "audit" in requested:
        stratified_dir = run_dir / "07_quality_audit_stratified"
        random_dir = run_dir / "07_quality_audit_random"
        run_command(
            [
                python,
                str(SCRIPT_DIR / "build_recurring_issue_audit.py"),
                "--run-dir",
                str(run_dir),
                "--output-dir",
                str(stratified_dir),
                "--group-review-size",
                "100",
                "--large-sample-size",
                "25",
                "--boundary-sample-size",
                "25",
                "--facet-risk-sample-size",
                "25",
                "--missed-link-review-size",
                "70",
            ]
        )
        run_command(
            [
                python,
                str(SCRIPT_DIR / "build_recurring_issue_audit.py"),
                "--run-dir",
                str(run_dir),
                "--output-dir",
                str(random_dir),
                "--group-review-size",
                "60",
                "--large-sample-size",
                "0",
                "--boundary-sample-size",
                "0",
                "--facet-risk-sample-size",
                "0",
                "--missed-link-review-size",
                "0",
                "--seed",
                "20260724",
            ]
        )
        stratified_metrics = read_json(stratified_dir / "audit_manifest.json")
        random_metrics = read_json(random_dir / "audit_manifest.json")
        stages["audit"] = {
            "status": "complete",
            "stratified_groups_selected": int(
                stratified_metrics.get("groups_selected", 0)
            ),
            "random_groups_selected": int(random_metrics.get("groups_selected", 0)),
            "missed_link_pairs_selected": int(
                stratified_metrics.get("missed_link_pairs_selected", 0)
            ),
            "stratified_audit_dir": str(stratified_dir),
            "random_audit_dir": str(random_dir),
        }
        write_workflow_manifest(
            manifest_path, run_dir=run_dir, args=args, stages=stages
        )

    if "risk" in requested and args.prepare_adjudication_risk:
        run_command(
            [
                python,
                str(SCRIPT_DIR / "adjudicate_recurring_issue_groups.py"),
                "--run-dir",
                str(run_dir),
                "--stage",
                "risk",
            ]
        )
        risk_path = run_dir / "08_targeted_adjudication" / "01_group_risk_scores.csv"
        stages["risk"] = {
            "status": "complete",
            "risk_queue_csv": str(risk_path),
            "model_adjudication_run": False,
        }
        write_workflow_manifest(
            manifest_path, run_dir=run_dir, args=args, stages=stages
        )

    print(f"\nWorkflow stage complete. Resume with --run-dir {run_dir}")


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted. Rerun the same command with --run-dir.", file=sys.stderr)
        raise SystemExit(130)
