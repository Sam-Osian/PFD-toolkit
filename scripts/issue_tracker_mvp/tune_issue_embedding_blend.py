#!/usr/bin/env python3
"""Tune dual-view embedding weights and split thresholds against a reviewed audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

import build_issue_index as pipeline
import score_issue_tuning_against_audit as audit_scoring
import tune_issue_index as tuning


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument(
        "--original-weight",
        action="append",
        type=float,
        dest="original_weights",
    )
    parser.add_argument(
        "--split-similarity",
        action="append",
        type=float,
        dest="split_similarities",
    )
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--edge-similarity", type=float, default=0.84)
    parser.add_argument("--min-centroid-similarity", type=float, default=0.83)
    parser.add_argument("--min-recurring-reports", type=int, default=3)
    parser.add_argument(
        "--save-config",
        action="append",
        default=[],
        help="Configuration name whose group and assignment CSVs should be retained.",
    )
    return parser.parse_args()


def config_name(weight: float, split: float) -> str:
    return f"w{round(1000 * weight):03d}_s{round(1000 * split):03d}"


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    audit_dir = args.audit_dir.expanduser().resolve()
    output_dir = args.output_dir.expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    weights = args.original_weights or [0.25, 0.35, 0.45, 0.50, 0.55, 0.65, 0.75]
    splits = args.split_similarities or [0.870, 0.872, 0.874, 0.876]
    for value in [
        *weights,
        *splits,
        args.edge_similarity,
        args.min_centroid_similarity,
    ]:
        if not 0.0 <= value <= 1.0:
            raise ValueError(
                "Embedding weights and similarity parameters must be in [0, 1]"
            )
    occurrences = pd.read_csv(run_dir / "02_embedding_occurrences.csv").fillna("")
    original = np.load(run_dir / "02_issue_embeddings_original.npy")
    normalized = np.load(run_dir / "02_issue_embeddings_normalized.npy")
    reviewed = pd.read_csv(audit_dir / "group_reviewed.csv").fillna("")
    members = pd.read_csv(audit_dir / "group_review_members.csv").fillna("")
    summaries: list[dict[str, object]] = []
    details: list[pd.DataFrame] = []
    for weight in weights:
        embeddings = pipeline.blend_embedding_views(original, normalized, weight)
        for split in splits:
            name = config_name(weight, split)
            print(f"Evaluating {name}...", flush=True)
            config = tuning.TuningConfig(
                name,
                args.top_k,
                args.edge_similarity,
                split,
                args.min_centroid_similarity,
            )
            index_summary, groups, assignments, _ = tuning.evaluate_config(
                occurrences,
                embeddings,
                config,
                min_recurring_reports=args.min_recurring_reports,
            )
            if name in args.save_config:
                groups.to_csv(output_dir / f"{name}_groups.csv", index=False)
                assignments.to_csv(output_dir / f"{name}_assignments.csv", index=False)
            audit_summary, detail = audit_scoring.score_config(
                name,
                reviewed,
                members,
                assignments,
                pd.Series(index_summary),
            )
            summaries.append(
                {
                    "original_view_weight": weight,
                    "split_similarity": split,
                    **audit_summary,
                }
            )
            detail.insert(1, "original_view_weight", weight)
            detail.insert(2, "split_similarity", split)
            details.append(detail)
    result = pd.DataFrame(summaries).sort_values(
        ["valid_group_preservation_percent", "precision_priority_score"],
        ascending=[False, False],
    )
    result.to_csv(output_dir / "blend_audit_summary.csv", index=False)
    pd.concat(details, ignore_index=True).to_csv(
        output_dir / "blend_audit_group_outcomes.csv", index=False
    )
    (output_dir / "blend_audit_summary.json").write_text(
        json.dumps(result.to_dict("records"), indent=2), encoding="utf-8"
    )
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
