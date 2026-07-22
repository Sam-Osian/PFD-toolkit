#!/usr/bin/env python3
"""Apply deterministic human-reviewed merge and split constraints to issue groups."""

from __future__ import annotations

import argparse
import json
from collections import defaultdict
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import build_issue_index as pipeline


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Repair a reviewed recurring-group candidate without model calls."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument(
        "--tuning-dir", type=Path, default=None, help="Defaults to <run-dir>/05_tuning."
    )
    parser.add_argument("--config-name", default="recommended")
    parser.add_argument("--review-csv", type=Path, default=None)
    parser.add_argument(
        "--split-constraints-json",
        type=Path,
        default=None,
        help="Optional reviewed mapping from over-merged group IDs to issue-ID partitions.",
    )
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--split-similarity", type=float, default=0.92)
    parser.add_argument("--min-recurring-reports", type=int, default=3)
    return parser.parse_args()


class UnionFind:
    def __init__(self, values: set[str]) -> None:
        self.parent = {value: value for value in values}

    def find(self, value: str) -> str:
        parent = self.parent[value]
        if parent != value:
            self.parent[value] = self.find(parent)
        return self.parent[value]

    def union(self, left: str, right: str) -> None:
        left_root = self.find(left)
        right_root = self.find(right)
        if left_root != right_root:
            self.parent[max(left_root, right_root)] = min(left_root, right_root)


def duplicate_targets(value: Any) -> list[str]:
    return [item.strip() for item in pipeline.clean_text(value).split("|") if item.strip()]


def reviewed_constraints(
    reviews: pd.DataFrame, group_ids: set[str]
) -> tuple[set[str], set[tuple[str, str]]]:
    reviews = reviews.fillna("")
    reviewed_ids = set(reviews["candidate_group_id"])
    if reviewed_ids != group_ids:
        raise ValueError(
            "Review CSV must cover every recurring group exactly once; "
            f"missing={sorted(group_ids - reviewed_ids)}, extra={sorted(reviewed_ids - group_ids)}"
        )
    if not reviews["coherent"].isin(["yes", "no"]).all():
        raise ValueError("Every reviewed group must have coherent=yes or coherent=no")
    if not reviews["over_merged"].isin(["yes", "no"]).all():
        raise ValueError("Every reviewed group must have over_merged=yes or over_merged=no")
    over_merged = set(reviews.loc[reviews["over_merged"] == "yes", "candidate_group_id"])
    merge_pairs: set[tuple[str, str]] = set()
    for row in reviews.itertuples(index=False):
        for target in duplicate_targets(row.near_duplicate_of):
            if target not in group_ids:
                raise ValueError(f"Unknown duplicate target {target} for {row.candidate_group_id}")
            if row.candidate_group_id in over_merged or target in over_merged:
                continue
            merge_pairs.add(tuple(sorted((row.candidate_group_id, target))))
    return over_merged, merge_pairs


def _centroid_scores(
    members: list[int], embeddings: np.ndarray
) -> tuple[dict[int, float], int, float]:
    local = embeddings[members]
    centroid = local.mean(axis=0)
    centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
    similarities = local @ centroid
    scores = {
        member: float(score)
        for member, score in zip(members, similarities, strict=False)
    }
    representative = members[int(np.argmax(similarities))]
    return scores, representative, float(np.median(similarities))


def repair_groups(
    occurrences: pd.DataFrame,
    embeddings: np.ndarray,
    groups: pd.DataFrame,
    assignments: pd.DataFrame,
    reviews: pd.DataFrame,
    *,
    split_similarity: float,
    min_recurring_reports: int,
    split_constraints: dict[str, list[list[str]]] | None = None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    recurring = groups[groups["recurrence_status"] == "recurring"].copy()
    group_ids = set(recurring["candidate_group_id"])
    over_merged, merge_pairs = reviewed_constraints(reviews, group_ids)
    issue_index = {
        issue_id: position for position, issue_id in enumerate(occurrences["issue_id"])
    }
    members_by_group: dict[str, list[int]] = {}
    for group_id in group_ids:
        issue_ids = assignments.loc[
            assignments["candidate_group_id"] == group_id, "issue_id"
        ]
        members_by_group[group_id] = [issue_index[issue_id] for issue_id in issue_ids]

    union_find = UnionFind(group_ids - over_merged)
    for left, right in merge_pairs:
        union_find.union(left, right)
    merge_components: dict[str, list[str]] = defaultdict(list)
    for group_id in group_ids - over_merged:
        merge_components[union_find.find(group_id)].append(group_id)

    repair_units: list[tuple[str, list[str], list[int]]] = []
    for source_groups in merge_components.values():
        source_groups.sort()
        members = sorted(
            {member for group_id in source_groups for member in members_by_group[group_id]}
        )
        action = "merged" if len(source_groups) > 1 else "unchanged"
        repair_units.append((action, source_groups, members))
    split_output_count = 0
    for group_id in sorted(over_merged):
        if split_constraints and group_id in split_constraints:
            partitions = split_constraints[group_id]
            source_issue_ids = {
                occurrences.iloc[index]["issue_id"] for index in members_by_group[group_id]
            }
            constrained_issue_ids = [issue_id for part in partitions for issue_id in part]
            if len(constrained_issue_ids) != len(set(constrained_issue_ids)):
                raise ValueError(f"Split constraint for {group_id} contains duplicate issue IDs")
            if set(constrained_issue_ids) != source_issue_ids:
                raise ValueError(
                    f"Split constraint for {group_id} must partition all source issue IDs"
                )
            split_members = [
                [issue_index[issue_id] for issue_id in partition] for partition in partitions
            ]
        else:
            split_members = pipeline.split_component(
                members_by_group[group_id], embeddings, split_similarity
            )
        split_output_count += len(split_members)
        for members in split_members:
            repair_units.append(("split", [group_id], members))

    group_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    for action, source_groups, members in repair_units:
        subset = occurrences.iloc[members]
        report_count = int(subset["report_key"].nunique())
        status = pipeline.recurrence_status(report_count, min_recurring_reports)
        member_ids = sorted(subset["issue_id"].map(pipeline.clean_text))
        repaired_id = f"repair_{pipeline.stable_hash(*member_ids)}"
        scores, representative, cohesion = _centroid_scores(members, embeddings)
        group_rows.append(
            {
                "repaired_group_id": repaired_id,
                "repair_action": action,
                "source_group_ids": " | ".join(source_groups),
                "recurrence_status": status,
                "report_count": report_count,
                "issue_count": len(members),
                "median_centroid_similarity": cohesion,
                "representative_issue": pipeline.clean_text(
                    occurrences.iloc[representative]["canonical_issue"]
                ),
                "subject_domain": pipeline.dominant_array_value(
                    subset, "issue_themes"
                ),
                "failure_mode": pipeline.dominant_value(subset, "failure_state"),
                "sample_issues": " | ".join(
                    subset["canonical_issue"].map(pipeline.clean_text).drop_duplicates().head(8)
                ),
            }
        )
        for member in members:
            assignment_rows.append(
                {
                    "repaired_group_id": repaired_id,
                    "repair_action": action,
                    "source_group_ids": " | ".join(source_groups),
                    "recurrence_status": status,
                    "issue_id": occurrences.iloc[member]["issue_id"],
                    "report_key": occurrences.iloc[member]["report_key"],
                    "assignment_similarity": scores[member],
                }
            )
    repaired_groups = pd.DataFrame(group_rows).sort_values(
        ["report_count", "issue_count", "repaired_group_id"],
        ascending=[False, False, True],
        ignore_index=True,
    )
    repaired_assignments = pd.DataFrame(assignment_rows)
    repaired_by_issue = repaired_assignments.set_index("issue_id")["repaired_group_id"].to_dict()
    merge_constraints_satisfied = sum(
        len(
            {
                repaired_by_issue[occurrences.iloc[index]["issue_id"]]
                for source_group in pair
                for index in members_by_group[source_group]
            }
        )
        == 1
        for pair in merge_pairs
    )
    split_constraints_satisfied = 0
    if split_constraints:
        for group_id, partitions in split_constraints.items():
            output_ids = [
                {repaired_by_issue[issue_id] for issue_id in partition}
                for partition in partitions
            ]
            if all(len(values) == 1 for values in output_ids) and len(
                {next(iter(values)) for values in output_ids}
            ) == len(partitions):
                split_constraints_satisfied += 1
    changed = repaired_groups[repaired_groups["repair_action"] != "unchanged"].copy()
    changed.insert(3, "review_reason", changed["repair_action"])
    changed["coherent"] = ""
    changed["over_merged"] = ""
    changed["over_split"] = ""
    changed["repair_appropriate"] = ""
    changed["review_notes"] = ""
    metrics = {
        "source_recurring_groups": len(group_ids),
        "reviewed_over_merged_groups": len(over_merged),
        "explicit_split_constraint_groups": len(split_constraints or {}),
        "confirmed_merge_pairs": len(merge_pairs),
        "merge_constraints_satisfied": merge_constraints_satisfied,
        "split_constraints_satisfied": split_constraints_satisfied,
        "stable_unchanged_groups": int(
            (repaired_groups["repair_action"] == "unchanged").sum()
        ),
        "merged_output_groups": int((repaired_groups["repair_action"] == "merged").sum()),
        "split_output_groups": split_output_count,
        "repaired_groups_all_statuses": len(repaired_groups),
        "repaired_recurring_groups": int(
            (repaired_groups["recurrence_status"] == "recurring").sum()
        ),
        "repaired_emerging_groups": int(
            (repaired_groups["recurrence_status"] == "emerging").sum()
        ),
        "repaired_isolated_groups": int(
            (repaired_groups["recurrence_status"] == "isolated").sum()
        ),
        "changed_groups_for_review": len(changed),
    }
    return repaired_groups, repaired_assignments, changed, metrics


def run_repair(
    run_dir: Path,
    tuning_dir: Path,
    output_dir: Path,
    review_csv: Path,
    *,
    config_name: str,
    split_similarity: float,
    min_recurring_reports: int,
    split_constraints_json: Path | None = None,
) -> dict[str, Any]:
    occurrence_path = run_dir / "02_embedding_occurrences.csv"
    if not occurrence_path.exists():
        occurrence_path = run_dir / "01_issue_occurrences.csv"
    occurrences = pd.read_csv(occurrence_path).fillna("")
    embeddings = np.load(run_dir / "02_issue_embeddings.npy")
    groups = pd.read_csv(tuning_dir / f"{config_name}_groups.csv").fillna("")
    assignments = pd.read_csv(tuning_dir / f"{config_name}_assignments.csv").fillna("")
    reviews = pd.read_csv(review_csv).fillna("")
    split_constraints = (
        json.loads(split_constraints_json.read_text(encoding="utf-8"))
        if split_constraints_json and split_constraints_json.exists()
        else None
    )
    repaired_groups, repaired_assignments, changed, metrics = repair_groups(
        occurrences,
        embeddings,
        groups,
        assignments,
        reviews,
        split_similarity=split_similarity,
        min_recurring_reports=min_recurring_reports,
        split_constraints=split_constraints,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    repaired_groups.to_csv(output_dir / "repaired_groups.csv", index=False)
    repaired_assignments.to_csv(output_dir / "repaired_assignments.csv", index=False)
    changed.to_csv(output_dir / "changed_groups_review_queue.csv", index=False)
    (output_dir / "repair_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (output_dir / "repair_config.json").write_text(
        json.dumps(
            {
                "source_run_dir": str(run_dir.resolve()),
                "source_tuning_dir": str(tuning_dir.resolve()),
                "source_review_csv": str(review_csv.resolve()),
                "split_constraints_json": str(split_constraints_json.resolve())
                if split_constraints_json
                else None,
                "config_name": config_name,
                "split_similarity": split_similarity,
                "min_recurring_reports": min_recurring_reports,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.split_similarity <= 1.0:
        raise ValueError("--split-similarity must be between 0 and 1")
    run_dir = args.run_dir.expanduser().resolve()
    tuning_dir = (
        args.tuning_dir.expanduser().resolve()
        if args.tuning_dir
        else run_dir / "05_tuning"
    )
    review_csv = (
        args.review_csv.expanduser().resolve()
        if args.review_csv
        else tuning_dir / f"{args.config_name}_full_review.csv"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else run_dir / "06_review_repair"
    )
    split_constraints_json = (
        args.split_constraints_json.expanduser().resolve()
        if args.split_constraints_json
        else tuning_dir / f"{args.config_name}_split_constraints.json"
    )
    metrics = run_repair(
        run_dir,
        tuning_dir,
        output_dir,
        review_csv,
        config_name=args.config_name,
        split_similarity=args.split_similarity,
        min_recurring_reports=args.min_recurring_reports,
        split_constraints_json=split_constraints_json,
    )
    print(f"Repair artefacts: {output_dir}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
