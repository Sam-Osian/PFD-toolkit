#!/usr/bin/env python3
"""Evaluate a soft blend of original and actor/object-normalized issue embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import tune_issue_index as tuning


CONFIGS = (
    tuning.TuningConfig("strict", 12, 0.88, 0.90, 0.86),
    tuning.TuningConfig("moderate", 40, 0.86, 0.89, 0.85),
    tuning.TuningConfig("recall", 40, 0.84, 0.87, 0.83),
    tuning.TuningConfig("higher_recall", 40, 0.82, 0.86, 0.82),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Blend cached original and normalized embeddings and evaluate grouping."
    )
    parser.add_argument("--original-run-dir", required=True, type=Path)
    parser.add_argument("--normalized-run-dir", required=True, type=Path)
    parser.add_argument("--occurrences-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--original-weight", type=float, default=0.5)
    parser.add_argument("--min-recurring-reports", type=int, default=3)
    return parser.parse_args()


def load_aligned_embeddings(
    original_dir: Path, normalized_dir: Path, occurrences: pd.DataFrame
) -> tuple[np.ndarray, np.ndarray]:
    original_occurrences = pd.read_csv(original_dir / "01_issue_occurrences.csv").fillna("")
    normalized_occurrences = pd.read_csv(normalized_dir / "01_issue_occurrences.csv").fillna("")
    expected_ids = occurrences["issue_id"].tolist()
    if original_occurrences["issue_id"].tolist() != expected_ids:
        raise ValueError("Original embeddings are not aligned to the supplied occurrences")
    if normalized_occurrences["issue_id"].tolist() != expected_ids:
        raise ValueError("Normalized embeddings are not aligned to the supplied occurrences")
    original = np.load(original_dir / "02_issue_embeddings.npy")
    normalized = np.load(normalized_dir / "02_issue_embeddings.npy")
    if original.shape != normalized.shape or len(original) != len(occurrences):
        raise ValueError("Embedding arrays have incompatible shapes")
    return original, normalized


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.original_weight <= 1.0:
        raise ValueError("--original-weight must be between 0 and 1")
    occurrences = pd.read_csv(args.occurrences_csv).fillna("")
    original, normalized = load_aligned_embeddings(
        args.original_run_dir, args.normalized_run_dir, occurrences
    )
    blended = (
        args.original_weight * original
        + (1.0 - args.original_weight) * normalized
    )
    blended /= np.maximum(np.linalg.norm(blended, axis=1, keepdims=True), 1e-12)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    np.save(args.output_dir / "blended_embeddings.npy", blended)

    summaries: list[dict[str, object]] = []
    review_rows: list[pd.DataFrame] = []
    for config in CONFIGS:
        summary, groups, assignments, _ = tuning.evaluate_config(
            occurrences,
            blended,
            config,
            min_recurring_reports=args.min_recurring_reports,
        )
        summaries.append(summary)
        groups.to_csv(args.output_dir / f"{config.name}_groups.csv", index=False)
        assignments.to_csv(
            args.output_dir / f"{config.name}_assignments.csv", index=False
        )
        review_groups = groups[
            groups["recurrence_status"].isin(["recurring", "emerging"])
        ]
        if review_groups.empty:
            continue
        members = assignments.merge(
            occurrences[
                [
                    "issue_id",
                    "canonical_issue",
                    "canonical_issue_original",
                    "responsible_actor_role",
                    "issue_object",
                ]
            ],
            on="issue_id",
            how="left",
        )
        member_summary = (
            members.groupby("candidate_group_id", as_index=False)
            .agg(
                member_issue_ids=("issue_id", " | ".join),
                member_issues=("canonical_issue", " | ".join),
                member_original_issues=("canonical_issue_original", " | ".join),
                member_actors=("responsible_actor_role", " | ".join),
                member_objects=("issue_object", " | ".join),
            )
        )
        review = review_groups.merge(member_summary, on="candidate_group_id", how="left")
        review["manual_coherent"] = ""
        review["manual_over_merged"] = ""
        review["manual_notes"] = ""
        review_rows.append(review)

    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(args.output_dir / "summary.csv", index=False)
    if review_rows:
        pd.concat(review_rows, ignore_index=True).to_csv(
            args.output_dir / "recurring_emerging_review.csv", index=False
        )
    manifest = {
        "original_run_dir": str(args.original_run_dir),
        "normalized_run_dir": str(args.normalized_run_dir),
        "occurrences_csv": str(args.occurrences_csv),
        "original_weight": args.original_weight,
        "normalized_weight": 1.0 - args.original_weight,
        "min_recurring_reports": args.min_recurring_reports,
        "configs": [config.__dict__ for config in CONFIGS],
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(summary_frame.to_string(index=False))
    print(f"\nComparison artefacts: {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
