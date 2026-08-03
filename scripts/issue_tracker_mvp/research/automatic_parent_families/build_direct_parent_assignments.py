#!/usr/bin/env python3
"""Archived: merge tight-core and direct-retrieval parent assignments."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


ASSIGNMENT_VERSION = "direct-parent-assignment-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family-summary-csv", required=True, type=Path)
    parser.add_argument("--memberships-csv", required=True, type=Path)
    parser.add_argument("--group-assignments-csv", required=True, type=Path)
    parser.add_argument("--candidate-adjudication-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def build_assignments(
    summaries: pd.DataFrame,
    memberships: pd.DataFrame,
    group_assignments: pd.DataFrame,
    adjudication: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    accepted = adjudication[adjudication["parent_assignment_status"].eq("auto_accept")].copy()
    selected_families = set(adjudication["family_id"].astype(str))
    core_memberships = memberships[
        memberships["family_id"].astype(str).isin(selected_families)
        & memberships["membership_type"].eq("core")
    ][["family_id", "child_group_id"]].drop_duplicates()
    core = core_memberships.merge(
        group_assignments[["report_key", "issue_id", "relational_group_id"]],
        left_on="child_group_id",
        right_on="relational_group_id",
        how="inner",
        validate="one_to_many",
    )
    core_rows = (
        core.sort_values(["family_id", "report_key", "issue_id"])
        .drop_duplicates(["family_id", "report_key"])
        .assign(
            parent_membership_source="tight_core",
            parent_evidence_issue_id=lambda frame: frame["issue_id"],
            model_confidence="",
            model_reason="",
        )
    )[
        [
            "family_id", "report_key", "parent_membership_source",
            "parent_evidence_issue_id", "child_group_id",
            "model_confidence", "model_reason",
        ]
    ]
    direct_rows = accepted.assign(
        parent_membership_source="direct_retrieval",
        parent_evidence_issue_id=lambda frame: frame["issue_id"],
        child_group_id=lambda frame: frame.get("representative_group_id", ""),
        model_confidence=lambda frame: frame["confidence"],
        model_reason=lambda frame: frame["reason"],
    )[
        [
            "family_id", "report_key", "parent_membership_source",
            "parent_evidence_issue_id", "child_group_id",
            "model_confidence", "model_reason",
        ]
    ]
    overlap = direct_rows.merge(
        core_rows[["family_id", "report_key"]],
        on=["family_id", "report_key"], how="inner",
    )
    if not overlap.empty:
        raise ValueError("Direct candidates must exclude reports already in the tight core")
    output = pd.concat([core_rows, direct_rows], ignore_index=True).sort_values(
        ["family_id", "parent_membership_source", "report_key"]
    )
    if output.duplicated(["family_id", "report_key"]).any():
        raise ValueError("Parent/report assignments must be unique")

    summary_lookup = summaries.set_index("family_id").to_dict("index")
    rows: list[dict[str, Any]] = []
    for family_id, local in output.groupby("family_id"):
        core_count = int(local["parent_membership_source"].eq("tight_core").sum())
        direct_count = int(local["parent_membership_source"].eq("direct_retrieval").sum())
        metadata = summary_lookup[str(family_id)]
        rows.append(
            {
                "family_id": family_id,
                "family_label_hint": metadata.get("family_label_hint", ""),
                "family_prototype": metadata.get("family_prototype", ""),
                "tight_core_report_count": core_count,
                "direct_retrieval_report_count": direct_count,
                "total_report_count": core_count + direct_count,
                "report_count_increase": (
                    direct_count / core_count if core_count else None
                ),
            }
        )
    summary = pd.DataFrame(rows).sort_values("total_report_count", ascending=False)
    metrics = {
        "assignment_version": ASSIGNMENT_VERSION,
        "families": len(summary),
        "parent_report_assignments": len(output),
        "tight_core_assignments": int(output["parent_membership_source"].eq("tight_core").sum()),
        "direct_retrieval_assignments": int(output["parent_membership_source"].eq("direct_retrieval").sum()),
    }
    return output, summary, metrics


def main() -> None:
    args = parse_args()
    summaries = pd.read_csv(args.family_summary_csv).fillna("")
    memberships = pd.read_csv(args.memberships_csv).fillna("")
    group_assignments = pd.read_csv(args.group_assignments_csv).fillna("")
    adjudication = pd.read_csv(args.candidate_adjudication_csv).fillna("")
    output, summary, metrics = build_assignments(
        summaries, memberships, group_assignments, adjudication
    )
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_dir / "01_parent_report_assignments.csv", index=False)
    summary.to_csv(args.output_dir / "02_parent_summary.csv", index=False)
    (args.output_dir / "manifest.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
