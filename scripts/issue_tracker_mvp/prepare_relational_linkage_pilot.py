#!/usr/bin/env python3
"""Build a stable pilot selection from manually reviewed recurrence cases."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Select all occurrences needed for the relational-linkage pilot."
    )
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    return parser.parse_args()


def build_selection(run_dir: Path) -> pd.DataFrame:
    member_path = run_dir / "07_quality_audit_random" / "group_review_members.csv"
    group_audit_path = run_dir / "09_validation" / "random_group_audit.csv"
    link_audit_path = run_dir / "09_validation" / "missed_link_audit.csv"
    for path in (member_path, group_audit_path, link_audit_path):
        if not path.exists():
            raise FileNotFoundError(f"Required reviewed pilot input not found: {path}")
    members = pd.read_csv(member_path).fillna("")
    decisions = pd.read_csv(group_audit_path).fillna("")
    reviewed_groups = set(decisions["subissue_id"].astype(str))
    selected_members = members[members["subissue_id"].astype(str).isin(reviewed_groups)][
        ["issue_id"]
    ].copy()
    selected_members["pilot_source"] = "reviewed_group"
    links = pd.read_csv(link_audit_path).fillna("")
    link_rows = pd.concat(
        [
            links[["left_issue_id"]].rename(columns={"left_issue_id": "issue_id"}),
            links[["right_issue_id"]].rename(columns={"right_issue_id": "issue_id"}),
        ],
        ignore_index=True,
    )
    link_rows["pilot_source"] = "reviewed_pair"
    selection = pd.concat([selected_members, link_rows], ignore_index=True)
    return (
        selection.groupby("issue_id", as_index=False)["pilot_source"]
        .agg(lambda values: "|".join(sorted(set(values))))
        .sort_values("issue_id")
        .reset_index(drop=True)
    )


def main() -> None:
    args = parse_args()
    selection = build_selection(args.run_dir)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    selection.to_csv(args.output_csv, index=False)
    print(
        json.dumps(
            {
                "run_dir": str(args.run_dir),
                "output_csv": str(args.output_csv),
                "pilot_issue_count": len(selection),
                "source_counts": selection["pilot_source"].value_counts().to_dict(),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
