#!/usr/bin/env python3
"""Run the frozen precision pipeline on untouched stratified holdout groups."""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


SCRIPT_DIR = Path(__file__).resolve().parent


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--group-count", type=int, default=20)
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--model", default="gemma4:26b")
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--batch-size", type=int, default=12)
    return parser.parse_args()


def run(command: list[str]) -> None:
    print("+", " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()
    root = args.run_dir / "11_relational_holdout"
    selection = root / "holdout_selection.csv"
    normalized = root / "01_relational_occurrences_v2.csv"
    experiment = root / "precision"
    run(
        [
            sys.executable,
            str(SCRIPT_DIR / "prepare_relational_holdout.py"),
            "--run-dir",
            str(args.run_dir),
            "--output-csv",
            str(selection),
            "--group-count",
            str(args.group_count),
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
        ]
    )
    run(
        [
            sys.executable,
            str(SCRIPT_DIR / "run_relational_linkage_experiment.py"),
            "--input-csv",
            str(normalized),
            "--output-dir",
            str(experiment),
            "--embedding-cache-dir",
            str(root / "embedding_cache"),
            "--minimum-edge-score",
            "0.85",
            "--minimum-group-score",
            "0.85",
            "--grouping-mode",
            "prototype",
        ]
    )
    run(
        [
            sys.executable,
            str(SCRIPT_DIR / "build_relational_review_packet.py"),
            "--normalized-csv",
            str(normalized),
            "--assignments-csv",
            str(experiment / "04_group_assignments.csv"),
            "--groups-csv",
            str(experiment / "05_relational_groups.csv"),
            "--output-dir",
            str(root / "review"),
        ]
    )


if __name__ == "__main__":
    main()
