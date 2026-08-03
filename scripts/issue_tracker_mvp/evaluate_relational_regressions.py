#!/usr/bin/env python3
"""Evaluate stable issue memberships against relational regression cases."""

from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
from typing import Any

import pandas as pd


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path).fillna("")


def dominant_group(assignments: pd.DataFrame, issue_ids: set[str]) -> tuple[str, int]:
    counts = Counter(
        assignments.loc[
            assignments["issue_id"].isin(issue_ids), "relational_group_id"
        ].astype(str)
    )
    return counts.most_common(1)[0] if counts else ("", 0)


def evaluate_cases(
    cases: dict[str, Any],
    baseline_assignments: pd.DataFrame,
    baseline_groups: pd.DataFrame,
    candidate_assignments: pd.DataFrame,
) -> pd.DataFrame:
    baseline_members = {
        group_id: set(local["issue_id"].astype(str))
        for group_id, local in baseline_assignments.groupby("relational_group_id")
    }
    baseline_prototypes = baseline_groups.set_index("relational_group_id")[
        "prototype_issue_id"
    ].astype(str).to_dict()
    candidate_map = candidate_assignments.set_index("issue_id")[
        "relational_group_id"
    ].astype(str).to_dict()
    rows: list[dict[str, Any]] = []
    for case in cases["cases"]:
        group_ids = case["group_ids"]
        missing_groups = [gid for gid in group_ids if gid not in baseline_members]
        if missing_groups:
            raise ValueError(
                f"Regression case {case['case_id']} has unknown baseline groups: "
                f"{missing_groups}"
            )
        expectation = case["expectation"]
        passed: bool | None
        details: dict[str, Any] = {}
        if expectation == "merge":
            left, right = (baseline_members[group_id] for group_id in group_ids)
            candidate_groups = set(candidate_map.get(issue_id, "") for issue_id in left)
            candidate_groups &= set(candidate_map.get(issue_id, "") for issue_id in right)
            coverage_rows = []
            for candidate_group in candidate_groups - {""}:
                left_coverage = sum(
                    candidate_map.get(issue_id) == candidate_group for issue_id in left
                ) / len(left)
                right_coverage = sum(
                    candidate_map.get(issue_id) == candidate_group for issue_id in right
                ) / len(right)
                coverage_rows.append(
                    (min(left_coverage, right_coverage), candidate_group, left_coverage, right_coverage)
                )
            best = max(coverage_rows, default=(0.0, "", 0.0, 0.0))
            threshold = float(case.get("minimum_member_coverage_each", 1.0))
            passed = best[0] >= threshold
            details = {
                "common_candidate_group": best[1],
                "left_member_coverage": best[2],
                "right_member_coverage": best[3],
                "required_coverage": threshold,
            }
        elif expectation == "separate":
            prototypes = [baseline_prototypes[group_id] for group_id in group_ids]
            candidate_groups = [candidate_map.get(issue_id, "") for issue_id in prototypes]
            passed = len(set(candidate_groups)) == len(candidate_groups) and all(
                candidate_groups
            )
            details = {"candidate_prototype_groups": candidate_groups}
        elif expectation == "preserve_group":
            issue_ids = baseline_members[group_ids[0]]
            group_id, retained = dominant_group(candidate_assignments, issue_ids)
            coverage = retained / len(issue_ids)
            threshold = float(case.get("minimum_member_coverage", 1.0))
            passed = coverage >= threshold
            details = {
                "candidate_group": group_id,
                "member_coverage": coverage,
                "required_coverage": threshold,
            }
        elif expectation == "do_not_expand":
            issue_ids = baseline_members[group_ids[0]]
            group_id, retained = dominant_group(candidate_assignments, issue_ids)
            candidate_size = int(
                candidate_assignments["relational_group_id"].eq(group_id).sum()
            )
            passed = retained == len(issue_ids) and candidate_size == len(issue_ids)
            details = {
                "candidate_group": group_id,
                "baseline_members_retained": retained,
                "baseline_size": len(issue_ids),
                "candidate_size": candidate_size,
            }
        elif expectation == "observe_only":
            issue_ids = baseline_members[group_ids[0]]
            group_id, retained = dominant_group(candidate_assignments, issue_ids)
            passed = None
            details = {
                "candidate_group": group_id,
                "member_coverage": retained / len(issue_ids),
                "candidate_size": int(
                    candidate_assignments["relational_group_id"].eq(group_id).sum()
                ),
            }
        else:
            raise ValueError(f"Unknown expectation: {expectation}")
        rows.append(
            {
                "case_id": case["case_id"],
                "expectation": expectation,
                "review_status": case["review_status"],
                "passed": passed,
                "details": json.dumps(details, sort_keys=True),
            }
        )
    return pd.DataFrame(rows)


def cases_from_duplicate_spot_check(frame: pd.DataFrame) -> list[dict[str, Any]]:
    """Convert analyst duplicate decisions into stable-membership checks."""
    cases: list[dict[str, Any]] = []
    for row in frame.itertuples(index=False):
        if row.decision == "same_whole_group":
            expectation = "merge"
            extra = {"minimum_member_coverage_each": 0.75}
        elif row.decision in {
            "separate",
            "related_but_distinct",
            "contaminated_not_mergeable",
        }:
            expectation = "separate"
            extra = {}
        else:
            # A shared core plus contaminated membership cannot be expressed as
            # a whole-group merge/separate expectation. Keep it in the evidence
            # file until member-level repair decisions exist.
            continue
        cases.append(
            {
                "case_id": f"spot_{int(row.rank):02d}_{row.group_a}_{row.group_b}",
                "expectation": expectation,
                "review_status": "analyst_spot_check",
                "group_ids": [row.group_a, row.group_b],
                **extra,
            }
        )
    return cases


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cases-json", required=True, type=Path)
    parser.add_argument("--baseline-dir", required=True, type=Path)
    parser.add_argument("--candidate-dir", required=True, type=Path)
    parser.add_argument("--spot-check-csv", type=Path)
    parser.add_argument("--output-csv", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    cases = json.loads(args.cases_json.read_text(encoding="utf-8"))
    if args.spot_check_csv:
        cases["cases"].extend(
            cases_from_duplicate_spot_check(load_csv(args.spot_check_csv))
        )
    result = evaluate_cases(
        cases,
        load_csv(args.baseline_dir / "04_group_assignments.csv"),
        load_csv(args.baseline_dir / "05_relational_groups.csv"),
        load_csv(args.candidate_dir / "04_group_assignments.csv"),
    )
    if args.output_csv:
        args.output_csv.parent.mkdir(parents=True, exist_ok=True)
        result.to_csv(args.output_csv, index=False)
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
