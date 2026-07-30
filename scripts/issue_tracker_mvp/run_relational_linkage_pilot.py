#!/usr/bin/env python3
"""Run the reviewed-case relational-linkage pilot end to end."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize the reviewed cases with v2 and compare linkage thresholds."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--model", default="gemma4:26b")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--request-timeout", type=int, default=360)
    parser.add_argument(
        "--embedding-model",
        default="Qwen/Qwen3-Embedding-8B",
    )
    parser.add_argument("--allow-model-download", action="store_true")
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    root = args.run_dir / "10_relational_linkage_pilot"
    selection = root / "pilot_selection.csv"
    normalized = root / "01_relational_occurrences_v2.csv"
    cache = root / "embedding_cache"
    run(
        [
            sys.executable,
            str(SCRIPT_DIR / "prepare_relational_linkage_pilot.py"),
            "--run-dir",
            str(args.run_dir),
            "--output-csv",
            str(selection),
        ]
    )
    run(
        [
            sys.executable,
            str(SCRIPT_DIR / "normalize_issue_occurrences.py"),
            "--input-csv",
            str(args.run_dir / "01_issue_occurrences_normalized.csv"),
            "--selection-csv",
            str(selection),
            "--output-csv",
            str(normalized),
            "--ollama-host",
            args.ollama_host,
            "--model",
            args.model,
            "--workers",
            str(args.workers),
            "--batch-size",
            str(args.batch_size),
            "--request-timeout",
            str(args.request_timeout),
        ]
    )
    configurations = {
        "prototype_precision": {
            "edge": 0.85,
            "group": 0.85,
            "mode": "prototype",
            "density": 1.0,
        },
        "density_strict": {
            "edge": 0.85,
            "group": 0.85,
            "mode": "constrained_density",
            "density": 0.60,
        },
        "density_guarded": {
            "edge": 0.85,
            "group": 0.83,
            "mode": "constrained_density",
            "density": 0.70,
        },
        "density_exploratory": {
            "edge": 0.85,
            "group": 0.83,
            "mode": "constrained_density",
            "density": 0.60,
        },
    }
    comparison: list[dict[str, object]] = []
    for name, configuration in configurations.items():
        edge_score = configuration["edge"]
        group_score = configuration["group"]
        output = root / name
        command = [
            sys.executable,
            str(SCRIPT_DIR / "run_relational_linkage_experiment.py"),
            "--input-csv",
            str(normalized),
            "--output-dir",
            str(output),
            "--embedding-cache-dir",
            str(cache),
            "--embedding-model",
            args.embedding_model,
            "--minimum-edge-score",
            str(edge_score),
            "--minimum-group-score",
            str(group_score),
            "--grouping-mode",
            str(configuration["mode"]),
            "--minimum-edge-density",
            str(configuration["density"]),
            "--audit-run-dir",
            str(args.run_dir),
        ]
        if args.allow_model_download:
            command.append("--allow-model-download")
        run(command)
        metrics = json.loads((output / "metrics.json").read_text(encoding="utf-8"))
        comparison.append(
            {
                "configuration": name,
                "minimum_edge_score": edge_score,
                "minimum_group_score": group_score,
                "grouping_mode": configuration["mode"],
                "minimum_edge_density": configuration["density"],
                "eligible_occurrences": metrics["occurrences_eligible"],
                "recurring_groups": metrics["recurring_groups"],
                "recurring_occurrences": metrics["recurring_occurrences"],
                "audit_pass_rate": metrics.get("manual_audit", {}).get("pass_rate"),
                "group_audit_pass_rate": (
                    metrics.get("manual_audit", {})
                    .get("by_type", {})
                    .get("group", {})
                    .get("pass_rate")
                ),
                "pair_audit_pass_rate": (
                    metrics.get("manual_audit", {})
                    .get("by_type", {})
                    .get("pair", {})
                    .get("pass_rate")
                ),
                "expected_same_pair_recall": (
                    metrics.get("manual_audit", {})
                    .get("relational_pairwise", {})
                    .get("expected_same_recall")
                ),
                "expected_different_pair_specificity": (
                    metrics.get("manual_audit", {})
                    .get("relational_pairwise", {})
                    .get("expected_different_specificity")
                ),
            }
        )
    (root / "threshold_comparison.json").write_text(
        json.dumps(comparison, indent=2), encoding="utf-8"
    )
    print(json.dumps({"threshold_comparison": comparison}, indent=2))


if __name__ == "__main__":
    main()
