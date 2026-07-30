#!/usr/bin/env python3
"""Build untouched random and diagnostic audits for relational issue groups."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


AUDIT_VERSION = "relational-full-audit-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalized-csv", required=True, type=Path)
    parser.add_argument("--assignments-csv", required=True, type=Path)
    parser.add_argument("--groups-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--random-size", type=int, default=60)
    parser.add_argument("--boundary-size", type=int, default=20)
    parser.add_argument("--high-frequency-size", type=int, default=20)
    parser.add_argument("--low-cohesion-size", type=int, default=20)
    parser.add_argument("--seed", type=int, default=20260730)
    parser.add_argument(
        "--exclude-members-csv",
        type=Path,
        action="append",
        default=[],
        help="Optional prior review-member CSV; groups containing those issue IDs are excluded.",
    )
    return parser.parse_args()


def load_excluded_issue_ids(paths: list[Path]) -> set[str]:
    excluded: set[str] = set()
    for path in paths:
        frame = pd.read_csv(path).fillna("")
        if "issue_id" not in frame:
            raise ValueError(f"Exclusion file lacks issue_id: {path}")
        excluded.update(frame["issue_id"].astype(str))
    return excluded


def sample_rows(frame: pd.DataFrame, size: int, seed: int) -> pd.DataFrame:
    if size <= 0 or frame.empty:
        return frame.iloc[0:0].copy()
    return frame.sample(n=min(size, len(frame)), random_state=seed)


def review_frames(
    groups: pd.DataFrame,
    members: pd.DataFrame,
    *,
    stratum: str,
) -> tuple[pd.DataFrame, pd.DataFrame, str]:
    groups = groups.sort_values(
        ["report_count", "relational_group_id"], ascending=[False, True]
    )
    selected_members = members[
        members["relational_group_id"].isin(groups["relational_group_id"])
    ].copy()
    queue_rows: list[dict[str, Any]] = []
    markdown = [f"# Relational group audit: {stratum}", ""]
    for _, group in groups.iterrows():
        group_id = group["relational_group_id"]
        local = selected_members[
            selected_members["relational_group_id"].eq(group_id)
        ]
        queue_rows.append(
            {
                "relational_group_id": group_id,
                "audit_stratum": stratum,
                "report_count": group["report_count"],
                "occurrence_count": group["occurrence_count"],
                "prototype_canonical_issue": group["prototype_canonical_issue"],
                "minimum_pair_similarity": group["minimum_pair_similarity"],
                "median_pair_similarity": group["median_pair_similarity"],
                "audit_decision": "",
                "incorrect_member_issue_ids": "",
                "split_description": "",
                "audit_notes": "",
            }
        )
        markdown.extend(
            [
                f"## {group_id}",
                "",
                f"Prototype: {group['prototype_canonical_issue']}",
                "",
                (
                    f"Reports: {group['report_count']}; "
                    f"minimum pair similarity: {group['minimum_pair_similarity']:.3f}; "
                    f"median pair similarity: {group['median_pair_similarity']:.3f}"
                ),
                "",
            ]
        )
        for _, member in local.iterrows():
            markdown.extend(
                [
                    f"- `{member['issue_id']}` — {member['canonical_issue']}",
                    (
                        "  - Relation: "
                        f"actor={member.get('responsible_actor_role', '')}; "
                        f"action={member.get('failed_action', '')}; "
                        f"object={member.get('issue_object', '')}; "
                        f"counterparty={member.get('counterparty_role', '')}; "
                        f"failure={member.get('failure_state', '')}; "
                        f"direction={member.get('communication_direction', '')}"
                    ),
                    f"  - Evidence: {member.get('evidence_quote', '')}",
                ]
            )
        markdown.append("")
    return pd.DataFrame(queue_rows), selected_members, "\n".join(markdown)


def write_review(
    output_dir: Path,
    groups: pd.DataFrame,
    members: pd.DataFrame,
    *,
    stratum: str,
) -> dict[str, int]:
    queue, selected_members, markdown = review_frames(
        groups, members, stratum=stratum
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    queue.to_csv(output_dir / "group_review_queue.csv", index=False)
    selected_members.to_csv(output_dir / "group_review_members.csv", index=False)
    (output_dir / "group_review_packet.md").write_text(markdown, encoding="utf-8")
    return {"groups": len(queue), "members": len(selected_members)}


def main() -> None:
    args = parse_args()
    normalized = pd.read_csv(args.normalized_csv).fillna("")
    assignments = pd.read_csv(args.assignments_csv).fillna("")
    groups = pd.read_csv(args.groups_csv).fillna("")
    recurring = groups[groups["recurrence_status"].eq("recurring")].copy()
    members = assignments.merge(
        normalized, on=["issue_id", "report_key"], how="left"
    )
    excluded_issue_ids = load_excluded_issue_ids(args.exclude_members_csv)
    if excluded_issue_ids:
        excluded_groups = set(
            members.loc[
                members["issue_id"].astype(str).isin(excluded_issue_ids),
                "relational_group_id",
            ]
        )
        recurring = recurring[
            ~recurring["relational_group_id"].isin(excluded_groups)
        ].copy()

    random_groups = sample_rows(recurring, args.random_size, args.seed)
    used = set(random_groups["relational_group_id"])
    boundary = sample_rows(
        recurring[
            recurring["report_count"].between(3, 4)
            & ~recurring["relational_group_id"].isin(used)
        ],
        args.boundary_size,
        args.seed + 1,
    )
    used.update(boundary["relational_group_id"])
    high_frequency = sample_rows(
        recurring[
            recurring["report_count"].ge(10)
            & ~recurring["relational_group_id"].isin(used)
        ],
        args.high_frequency_size,
        args.seed + 2,
    )
    used.update(high_frequency["relational_group_id"])
    low_cohesion = (
        recurring[~recurring["relational_group_id"].isin(used)]
        .sort_values(
            ["minimum_pair_similarity", "relational_group_id"],
            ascending=[True, True],
        )
        .head(args.low_cohesion_size)
    )
    diagnostic = pd.concat(
        [
            boundary.assign(audit_stratum="boundary"),
            high_frequency.assign(audit_stratum="high_frequency"),
            low_cohesion.assign(audit_stratum="low_cohesion"),
        ],
        ignore_index=True,
    )

    args.output_dir.mkdir(parents=True, exist_ok=True)
    random_metrics = write_review(
        args.output_dir / "random",
        random_groups,
        members,
        stratum="random",
    )
    diagnostic_metrics: dict[str, dict[str, int]] = {}
    for stratum, local in diagnostic.groupby("audit_stratum"):
        diagnostic_metrics[stratum] = write_review(
            args.output_dir / "diagnostic" / stratum,
            local,
            members,
            stratum=stratum,
        )
    manifest = {
        "audit_version": AUDIT_VERSION,
        "normalized_csv": str(args.normalized_csv),
        "assignments_csv": str(args.assignments_csv),
        "groups_csv": str(args.groups_csv),
        "seed": args.seed,
        "eligible_recurring_groups": len(recurring),
        "excluded_prior_issue_ids": len(excluded_issue_ids),
        "random": random_metrics,
        "diagnostic": diagnostic_metrics,
        "interpretation": {
            "random": "Use to estimate full-output group coherence.",
            "diagnostic": "Use to diagnose failure modes; do not treat as prevalence weighted.",
        },
    }
    (args.output_dir / "audit_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
