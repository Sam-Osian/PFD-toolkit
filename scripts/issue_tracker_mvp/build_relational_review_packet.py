#!/usr/bin/env python3
"""Build a human-review packet for newly formed relational groups."""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--normalized-csv", required=True, type=Path)
    parser.add_argument("--assignments-csv", required=True, type=Path)
    parser.add_argument("--groups-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    normalized = pd.read_csv(args.normalized_csv).fillna("")
    assignments = pd.read_csv(args.assignments_csv).fillna("")
    groups = pd.read_csv(args.groups_csv).fillna("")
    recurring = groups[groups["recurrence_status"] == "recurring"].copy()
    members = assignments[
        assignments["relational_group_id"].isin(recurring["relational_group_id"])
    ].merge(normalized, on=["issue_id", "report_key"], how="left")
    review_rows: list[dict[str, object]] = []
    markdown = ["# Relational holdout group review", ""]
    for _, group in recurring.sort_values(
        ["report_count", "relational_group_id"], ascending=[False, True]
    ).iterrows():
        group_id = group["relational_group_id"]
        local = members[members["relational_group_id"] == group_id]
        review_rows.append(
            {
                "relational_group_id": group_id,
                "report_count": group["report_count"],
                "occurrence_count": group["occurrence_count"],
                "prototype_canonical_issue": group["prototype_canonical_issue"],
                "audit_decision": "",
                "incorrect_member_issue_ids": "",
                "audit_notes": "",
            }
        )
        markdown.extend(
            [
                f"## {group_id}",
                "",
                f"Prototype: {group['prototype_canonical_issue']}",
                "",
            ]
        )
        for _, member in local.iterrows():
            markdown.extend(
                [
                    f"- `{member['issue_id']}` — {member['canonical_issue']}",
                    f"  - Relation: actor={member.get('responsible_actor_role', '')}; "
                    f"action={member.get('failed_action', '')}; "
                    f"object={member.get('issue_object', '')}; "
                    f"counterparty={member.get('counterparty_role', '')}; "
                    f"direction={member.get('communication_direction', '')}",
                    f"  - Evidence: {member.get('evidence_quote', '')}",
                ]
            )
        markdown.append("")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(review_rows).to_csv(
        args.output_dir / "group_review_queue.csv", index=False
    )
    members.to_csv(args.output_dir / "group_review_members.csv", index=False)
    (args.output_dir / "group_review_packet.md").write_text(
        "\n".join(markdown), encoding="utf-8"
    )
    print(
        f"Wrote {len(review_rows)} recurring groups and {len(members)} members "
        f"to {args.output_dir}"
    )


if __name__ == "__main__":
    main()
