#!/usr/bin/env python3
"""Trace an issue or relational group through persisted linkage artifacts."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


TRACE_VERSION = "relational-case-trace-v1"


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path).fillna("")


def records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    return json.loads(frame.to_json(orient="records"))


def build_trace(
    occurrences: pd.DataFrame,
    assignments: pd.DataFrame,
    groups: pd.DataFrame,
    candidates: pd.DataFrame,
    accepted_edges: pd.DataFrame,
    *,
    issue_ids: list[str] | None = None,
    group_ids: list[str] | None = None,
    reports: pd.DataFrame | None = None,
) -> dict[str, Any]:
    issue_ids = issue_ids or []
    group_ids = group_ids or []
    selected_assignments = assignments[
        assignments["issue_id"].isin(issue_ids)
        | assignments["relational_group_id"].isin(group_ids)
    ].copy()
    selected_issue_ids = set(selected_assignments["issue_id"].astype(str)) | set(
        issue_ids
    )
    selected_group_ids = set(
        selected_assignments["relational_group_id"].astype(str)
    ) | set(group_ids)
    selected_occurrences = occurrences[
        occurrences["issue_id"].isin(selected_issue_ids)
    ].copy()
    if reports is not None and not reports.empty:
        report_columns = [
            column
            for column in ["report_key", "report_id", "report_url", "report_date"]
            if column in reports
        ]
        selected_occurrences = selected_occurrences.merge(
            reports[report_columns], on="report_key", how="left"
        )

    related_candidates = candidates[
        candidates["left_issue_id"].isin(selected_issue_ids)
        | candidates["right_issue_id"].isin(selected_issue_ids)
    ].copy()
    related_accepted = accepted_edges[
        accepted_edges["left_issue_id"].isin(selected_issue_ids)
        | accepted_edges["right_issue_id"].isin(selected_issue_ids)
    ].copy()
    issue_to_group = dict(
        zip(assignments["issue_id"].astype(str), assignments["relational_group_id"])
    )
    for frame in (related_candidates, related_accepted):
        frame["left_relational_group_id"] = (
            frame["left_issue_id"].map(issue_to_group).fillna("")
        )
        frame["right_relational_group_id"] = (
            frame["right_issue_id"].map(issue_to_group).fillna("")
        )
        frame["crosses_final_group_boundary"] = frame[
            "left_relational_group_id"
        ].ne(frame["right_relational_group_id"])

    missing_issues = sorted(set(issue_ids) - set(occurrences["issue_id"].astype(str)))
    missing_groups = sorted(
        set(group_ids) - set(groups["relational_group_id"].astype(str))
    )
    return {
        "trace_version": TRACE_VERSION,
        "requested_issue_ids": issue_ids,
        "requested_group_ids": group_ids,
        "missing_issue_ids": missing_issues,
        "missing_group_ids": missing_groups,
        "groups": records(
            groups[groups["relational_group_id"].isin(selected_group_ids)]
        ),
        "assignments": records(selected_assignments),
        "occurrences": records(selected_occurrences),
        "candidate_pairs": records(
            related_candidates.sort_values("adjusted_similarity", ascending=False)
        ),
        "accepted_edges": records(
            related_accepted.sort_values("adjusted_similarity", ascending=False)
        ),
        "interpretation": {
            "candidate_but_not_accepted": (
                "The pair was retrieved and scored but failed its veto, score, or mutual-top-k acceptance condition."
            ),
            "accepted_but_different_final_groups": (
                "The pair passed edge acceptance but its groups were not consolidated. Inspect diagnostic duplicate-group blocker evidence."
            ),
            "not_a_candidate": (
                "The historical artifact does not persist directed retrieval ranks, so absence can only be localized to retrieval/candidate generation, not assigned a more exact reason."
            ),
        },
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--reports-csv", type=Path)
    parser.add_argument("--issue-id", action="append", default=[])
    parser.add_argument("--group-id", action="append", default=[])
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.issue_id and not args.group_id:
        raise SystemExit("Provide at least one --issue-id or --group-id")
    base = args.artifact_dir
    payload = build_trace(
        load_csv(base / "01_linkage_quality_gate.csv"),
        load_csv(base / "04_group_assignments.csv"),
        load_csv(base / "05_relational_groups.csv"),
        load_csv(base / "02_candidate_pairs.csv"),
        load_csv(base / "03_accepted_edges.csv"),
        issue_ids=args.issue_id,
        group_ids=args.group_id,
        reports=load_csv(args.reports_csv) if args.reports_csv else None,
    )
    rendered = json.dumps(payload, indent=2)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(rendered, encoding="utf-8")
    else:
        print(rendered)


if __name__ == "__main__":
    main()
