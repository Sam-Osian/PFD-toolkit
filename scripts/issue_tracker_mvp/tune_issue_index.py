#!/usr/bin/env python3
"""Compare deterministic issue-index grouping configurations.

This script reuses an existing occurrence CSV and embedding cache. It never
calls the extraction or labelling models and does not overwrite index outputs.
"""

from __future__ import annotations

import argparse
import json
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import build_issue_index as pipeline


@dataclass(frozen=True)
class TuningConfig:
    name: str
    top_k: int
    edge_similarity: float
    split_similarity: float
    min_centroid_similarity: float


DEFAULT_CONFIGS = (
    TuningConfig("baseline", 12, 0.88, 0.90, 0.86),
    TuningConfig("recommended", 40, 0.84, 0.87, 0.83),
    TuningConfig("high_recall", 40, 0.82, 0.86, 0.82),
)


def parse_config(value: str) -> TuningConfig:
    try:
        name, top_k, edge, split, centroid = value.split(":")
        config = TuningConfig(name, int(top_k), float(edge), float(split), float(centroid))
    except ValueError as exc:
        raise argparse.ArgumentTypeError(
            "config must be NAME:TOP_K:EDGE_SIMILARITY:SPLIT_SIMILARITY:CENTROID_SIMILARITY"
        ) from exc
    if not name.strip() or config.top_k < 1:
        raise argparse.ArgumentTypeError("config name must be set and TOP_K must be positive")
    for field in ("edge_similarity", "split_similarity", "min_centroid_similarity"):
        if not 0.0 <= getattr(config, field) <= 1.0:
            raise argparse.ArgumentTypeError(f"{field} must be between 0 and 1")
    return config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare grouping parameters using an existing issue-index embedding cache."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="Defaults to <run-dir>/05_tuning.",
    )
    parser.add_argument("--min-recurring-reports", type=int, default=3)
    parser.add_argument("--review-size", type=int, default=60)
    parser.add_argument("--near-duplicate-similarity", type=float, default=0.90)
    parser.add_argument(
        "--config",
        action="append",
        type=parse_config,
        help=(
            "Repeatable NAME:TOP_K:EDGE:SPLIT:CENTROID configuration. "
            "When omitted, baseline, recommended, and high-recall configurations are used."
        ),
    )
    return parser.parse_args()


def _normalised_centroid(embeddings: np.ndarray, members: list[int]) -> np.ndarray:
    centroid = embeddings[members].mean(axis=0)
    return centroid / max(float(np.linalg.norm(centroid)), 1e-12)


def evaluate_config(
    occurrences: pd.DataFrame,
    embeddings: np.ndarray,
    config: TuningConfig,
    *,
    min_recurring_reports: int,
) -> tuple[dict[str, Any], pd.DataFrame, pd.DataFrame, dict[str, np.ndarray]]:
    components, edges = pipeline.mutual_neighbor_components(
        embeddings,
        occurrences["report_key"].map(pipeline.clean_text).tolist(),
        top_k=config.top_k,
        threshold=config.edge_similarity,
    )
    split_candidates: list[list[int]] = []
    for component in components:
        split_candidates.extend(
            pipeline.split_component(component, embeddings, config.split_similarity)
        )

    candidates: list[tuple[list[int], dict[int, float]]] = []
    for members in split_candidates:
        retained, scores = pipeline.apply_centroid_filter(
            members, embeddings, config.min_centroid_similarity
        )
        if retained:
            candidates.append((retained, scores))
    candidates.sort(
        key=lambda item: (
            -occurrences.iloc[item[0]]["report_key"].nunique(),
            -len(item[0]),
            min(occurrences.iloc[item[0]]["issue_id"]),
        )
    )

    group_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    centroids: dict[str, np.ndarray] = {}
    recurring_occurrence_count = 0
    status_counts = {"recurring": 0, "emerging": 0, "isolated": 0}
    for members, scores in candidates:
        subset = occurrences.iloc[members]
        report_count = int(subset["report_key"].nunique())
        status = pipeline.recurrence_status(report_count, min_recurring_reports)
        status_counts[status] += 1
        member_ids = sorted(subset["issue_id"].map(pipeline.clean_text))
        group_id = f"candidate_{pipeline.stable_hash(config.name, *member_ids)}"
        centroid = _normalised_centroid(embeddings, members)
        centroids[group_id] = centroid
        representative_index = max(members, key=lambda index: scores[index])
        if status == "recurring":
            recurring_occurrence_count += len(members)
        group_rows.append(
            {
                "config": config.name,
                "candidate_group_id": group_id,
                "recurrence_status": status,
                "report_count": report_count,
                "issue_count": len(members),
                "median_centroid_similarity": float(
                    np.median([scores[index] for index in members])
                ),
                "representative_issue": pipeline.clean_text(
                    occurrences.iloc[representative_index]["canonical_issue"]
                ),
                "subject_domain": pipeline.dominant_array_value(
                    subset, "issue_themes"
                ),
                "failure_mode": pipeline.dominant_value(subset, "failure_state"),
                "sample_issues": " | ".join(
                    subset["canonical_issue"].map(pipeline.clean_text).drop_duplicates().head(5)
                ),
            }
        )
        for index in members:
            assignment_rows.append(
                {
                    "config": config.name,
                    "candidate_group_id": group_id,
                    "recurrence_status": status,
                    "issue_id": occurrences.iloc[index]["issue_id"],
                    "report_key": occurrences.iloc[index]["report_key"],
                    "assignment_similarity": scores[index],
                }
            )

    assigned_count = sum(len(members) for members, _ in candidates)
    total = len(occurrences)
    summary = {
        **asdict(config),
        "total_issue_occurrences": total,
        "mutual_neighbour_edges": len(edges),
        "initial_components": len(components),
        "candidate_subissues": len(candidates),
        "recurring_subissues": status_counts["recurring"],
        "emerging_subissues": status_counts["emerging"],
        "isolated_subissues": status_counts["isolated"],
        "assigned_issue_occurrences": assigned_count,
        "assigned_percent": round(100.0 * assigned_count / total, 2) if total else 0.0,
        "recurring_issue_occurrences": recurring_occurrence_count,
        "recurring_percent": round(100.0 * recurring_occurrence_count / total, 2)
        if total
        else 0.0,
        "ungrouped_issue_occurrences": total - assigned_count,
    }
    return summary, pd.DataFrame(group_rows), pd.DataFrame(assignment_rows), centroids


def build_near_duplicate_rows(
    recurring_groups: pd.DataFrame,
    centroids: dict[str, np.ndarray],
    *,
    minimum_similarity: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    records = recurring_groups.to_dict("records")
    for left_position, left in enumerate(records):
        for right in records[left_position + 1 :]:
            similarity = float(
                centroids[left["candidate_group_id"]] @ centroids[right["candidate_group_id"]]
            )
            if similarity < minimum_similarity:
                continue
            rows.append(
                {
                    "config": left["config"],
                    "left_group_id": left["candidate_group_id"],
                    "right_group_id": right["candidate_group_id"],
                    "centroid_similarity": similarity,
                    "left_representative": left["representative_issue"],
                    "right_representative": right["representative_issue"],
                }
            )
    return pd.DataFrame(rows).sort_values(
        "centroid_similarity", ascending=False, ignore_index=True
    ) if rows else pd.DataFrame(
        columns=[
            "config", "left_group_id", "right_group_id", "centroid_similarity",
            "left_representative", "right_representative",
        ]
    )


def build_review_queue(recurring_groups: pd.DataFrame, review_size: int) -> pd.DataFrame:
    if recurring_groups.empty or review_size <= 0:
        return recurring_groups.iloc[0:0].copy()
    per_stratum = max(1, review_size // 3)
    selections = (
        (
            "largest",
            recurring_groups.sort_values(
                ["report_count", "issue_count", "candidate_group_id"],
                ascending=[False, False, True],
            ).head(per_stratum),
        ),
        (
            "lowest_cohesion",
            recurring_groups.sort_values(
                ["median_centroid_similarity", "candidate_group_id"]
            ).head(per_stratum),
        ),
        (
            "recurrence_boundary",
            recurring_groups[
                recurring_groups["report_count"] == recurring_groups["report_count"].min()
            ].sort_values("candidate_group_id").head(per_stratum),
        ),
    )
    reasons: dict[str, set[str]] = {}
    selected_rows: dict[str, dict[str, Any]] = {}
    for reason, frame in selections:
        for row in frame.to_dict("records"):
            group_id = row["candidate_group_id"]
            selected_rows[group_id] = row
            reasons.setdefault(group_id, set()).add(reason)
    if len(selected_rows) < review_size:
        for row in recurring_groups.sort_values("candidate_group_id").to_dict("records"):
            group_id = row["candidate_group_id"]
            if group_id not in selected_rows:
                selected_rows[group_id] = row
                reasons[group_id] = {"fill"}
            if len(selected_rows) >= review_size:
                break
    queue = pd.DataFrame(selected_rows.values()).head(review_size)
    queue.insert(
        3,
        "review_reason",
        queue["candidate_group_id"].map(lambda value: " | ".join(sorted(reasons[value]))),
    )
    queue["coherent"] = ""
    queue["over_merged"] = ""
    queue["near_duplicate_of"] = ""
    queue["review_notes"] = ""
    return queue


def run_tuning(
    run_dir: Path,
    output_dir: Path,
    configs: list[TuningConfig],
    *,
    min_recurring_reports: int,
    review_size: int,
    near_duplicate_similarity: float,
) -> pd.DataFrame:
    occurrence_path = run_dir / "02_embedding_occurrences.csv"
    if not occurrence_path.exists():
        occurrence_path = run_dir / "01_issue_occurrences.csv"
    embeddings_path = run_dir / "02_issue_embeddings.npy"
    if not occurrence_path.exists() or not embeddings_path.exists():
        raise FileNotFoundError(
            "run directory must contain an occurrence snapshot and 02_issue_embeddings.npy"
        )
    occurrences = pd.read_csv(occurrence_path).fillna("").reset_index(drop=True)
    embeddings = np.load(embeddings_path)
    if len(occurrences) != len(embeddings):
        raise ValueError("Occurrence and embedding row counts do not match")
    output_dir.mkdir(parents=True, exist_ok=True)

    summaries: list[dict[str, Any]] = []
    for config in configs:
        print(f"Evaluating {config.name}...", flush=True)
        summary, groups, assignments, centroids = evaluate_config(
            occurrences,
            embeddings,
            config,
            min_recurring_reports=min_recurring_reports,
        )
        summaries.append(summary)
        recurring = groups[groups["recurrence_status"] == "recurring"].copy()
        groups.to_csv(output_dir / f"{config.name}_groups.csv", index=False)
        assignments.to_csv(output_dir / f"{config.name}_assignments.csv", index=False)
        recurring.to_csv(output_dir / f"{config.name}_recurring_groups.csv", index=False)
        build_near_duplicate_rows(
            recurring,
            centroids,
            minimum_similarity=near_duplicate_similarity,
        ).to_csv(output_dir / f"{config.name}_near_duplicates.csv", index=False)
        build_review_queue(recurring, review_size).to_csv(
            output_dir / f"{config.name}_review_queue.csv", index=False
        )

    summary_frame = pd.DataFrame(summaries)
    summary_frame.to_csv(output_dir / "tuning_summary.csv", index=False)
    (output_dir / "tuning_summary.json").write_text(
        json.dumps(summaries, indent=2), encoding="utf-8"
    )
    (output_dir / "run_config.json").write_text(
        json.dumps(
            {
                "source_run_dir": str(run_dir.resolve()),
                "min_recurring_reports": min_recurring_reports,
                "review_size": review_size,
                "near_duplicate_similarity": near_duplicate_similarity,
                "configs": [asdict(config) for config in configs],
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return summary_frame


def main() -> None:
    args = parse_args()
    if args.min_recurring_reports < 2:
        raise ValueError("--min-recurring-reports must be at least 2")
    if args.review_size < 0:
        raise ValueError("--review-size cannot be negative")
    if not 0.0 <= args.near_duplicate_similarity <= 1.0:
        raise ValueError("--near-duplicate-similarity must be between 0 and 1")
    run_dir = args.run_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else run_dir / "05_tuning"
    )
    configs = args.config or list(DEFAULT_CONFIGS)
    summary = run_tuning(
        run_dir,
        output_dir,
        configs,
        min_recurring_reports=args.min_recurring_reports,
        review_size=args.review_size,
        near_duplicate_similarity=args.near_duplicate_similarity,
    )
    print(f"\nTuning artefacts: {output_dir}")
    print(summary.to_string(index=False))


if __name__ == "__main__":
    main()
