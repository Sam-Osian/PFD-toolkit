#!/usr/bin/env python3
"""Validate manual issue-index audit decisions and calculate audit metrics."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


GROUP_DECISIONS = {"accept", "accept_with_exclusions", "split", "reject", "uncertain"}
PAIR_DECISIONS = {"yes", "no", "uncertain"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    return parser.parse_args()


def pair_key(left: object, right: object) -> str:
    return "|".join(sorted((str(left), str(right))))


def load_json_rows(path: Path) -> pd.DataFrame:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, list):
        raise ValueError(f"{path.name} must contain a JSON list")
    return pd.DataFrame(payload).fillna("")


def validate_exact_coverage(
    expected: set[str], actual: set[str], *, label: str
) -> None:
    if expected != actual:
        raise ValueError(
            f"{label} decision coverage mismatch: "
            f"missing={sorted(expected - actual)}, extra={sorted(actual - expected)}"
        )


def score_group_review(audit_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    queue = pd.read_csv(audit_dir / "group_review_queue.csv").fillna("")
    decisions = load_json_rows(audit_dir / "group_decisions.json")
    if decisions["subissue_id"].duplicated().any():
        raise ValueError("Group decisions contain duplicate subissue IDs")
    validate_exact_coverage(
        set(queue["subissue_id"]), set(decisions["subissue_id"]), label="Group"
    )
    if not decisions["audit_decision"].isin(GROUP_DECISIONS).all():
        raise ValueError("Group decisions contain an unsupported audit_decision")
    reviewed = queue.drop(
        columns=[
            "audit_decision",
            "incorrect_member_issue_ids",
            "proposed_split",
            "audit_notes",
        ]
    ).merge(decisions, on="subissue_id", how="left", validate="one_to_one")
    if "incorrect_member_issue_ids" not in reviewed:
        reviewed["incorrect_member_issue_ids"] = ""
    reviewed["incorrect_member_issue_ids"] = reviewed["incorrect_member_issue_ids"].map(
        lambda value: " | ".join(value) if isinstance(value, list) else value
    )
    for column in ("proposed_split", "audit_notes"):
        if column not in reviewed:
            reviewed[column] = ""
    counts = reviewed["audit_decision"].value_counts().to_dict()
    total = len(reviewed)
    metrics = {
        "groups_reviewed": total,
        "decision_counts": counts,
        "coherent_as_is_percent": round(100.0 * counts.get("accept", 0) / total, 2),
        "acceptable_after_exclusions_percent": round(
            100.0
            * (counts.get("accept", 0) + counts.get("accept_with_exclusions", 0))
            / total,
            2,
        ),
        "overmerge_split_percent": round(100.0 * counts.get("split", 0) / total, 2),
        "false_recurrence_reject_percent": round(
            100.0 * counts.get("reject", 0) / total, 2
        ),
        "decision_counts_by_stratum": {
            reason: frame["audit_decision"].value_counts().to_dict()
            for reason, frame in reviewed.groupby("review_reason")
        },
        "issue_members_by_decision": reviewed.groupby("audit_decision")["issue_count"]
        .sum()
        .astype(int)
        .to_dict(),
    }
    return reviewed, metrics


def score_missed_links(audit_dir: Path) -> tuple[pd.DataFrame, dict[str, Any]]:
    try:
        queue = pd.read_csv(audit_dir / "missed_link_review_queue.csv").fillna("")
    except pd.errors.EmptyDataError:
        queue = pd.DataFrame()
    decisions = load_json_rows(audit_dir / "missed_link_decisions.json")
    if queue.empty:
        if not decisions.empty:
            raise ValueError("Missed-link decisions were supplied for an empty queue")
        metrics = {
            "pairs_reviewed": 0,
            "pair_decision_counts": {},
            "preferred_action_counts": {},
            "above_threshold_pairs_reviewed": 0,
            "above_threshold_already_grouped": 0,
            "above_threshold_pair_agreement_percent": 0.0,
            "distinct_pairs_reviewed": 0,
            "distinct_pair_same_issue_percent": 0.0,
            "missed_recurring_candidates": 0,
            "missed_emerging_links": 0,
            "manual_follow_up": 0,
            "decisions_by_stratum": {},
        }
        return queue, metrics
    queue["pair_key"] = queue.apply(
        lambda row: pair_key(row["left_issue_id"], row["right_issue_id"]), axis=1
    )
    decisions["pair_key"] = decisions.apply(
        lambda row: pair_key(row["left_issue_id"], row["right_issue_id"]), axis=1
    )
    if decisions["pair_key"].duplicated().any():
        raise ValueError("Missed-link decisions contain duplicate issue pairs")
    validate_exact_coverage(
        set(queue["pair_key"]), set(decisions["pair_key"]), label="Missed-link"
    )
    if not decisions["same_recurring_issue"].isin(PAIR_DECISIONS).all():
        raise ValueError("Missed-link decisions contain an unsupported pair decision")
    reviewed = queue.drop(
        columns=["same_recurring_issue", "preferred_action", "audit_notes"]
    ).merge(
        decisions.drop(columns=["row", "left_issue_id", "right_issue_id"]),
        on="pair_key",
        how="left",
        validate="one_to_one",
    )
    if "audit_notes" not in reviewed:
        reviewed["audit_notes"] = ""
    reviewed["already_same_group"] = reviewed["left_subissue_id"].ne("") & reviewed[
        "left_subissue_id"
    ].eq(reviewed["right_subissue_id"])
    distinct = reviewed[~reviewed["already_same_group"]]
    high = reviewed[reviewed["review_reason"].eq("above_edge_threshold")]
    action_counts = reviewed["preferred_action"].value_counts().to_dict()
    metrics = {
        "pairs_reviewed": len(reviewed),
        "pair_decision_counts": reviewed["same_recurring_issue"]
        .value_counts()
        .to_dict(),
        "preferred_action_counts": action_counts,
        "above_threshold_pairs_reviewed": len(high),
        "above_threshold_already_grouped": int(high["already_same_group"].sum()),
        "above_threshold_pair_agreement_percent": round(
            100.0 * high["same_recurring_issue"].eq("yes").sum() / max(len(high), 1), 2
        ),
        "distinct_pairs_reviewed": len(distinct),
        "distinct_pair_same_issue_percent": round(
            100.0
            * distinct["same_recurring_issue"].eq("yes").sum()
            / max(len(distinct), 1),
            2,
        ),
        "missed_recurring_candidates": action_counts.get(
            "missed_recurring_candidate", 0
        ),
        "missed_emerging_links": action_counts.get("missed_emerging_link", 0),
        "manual_follow_up": action_counts.get("manual_follow_up", 0),
        "decisions_by_stratum": {
            reason: frame["same_recurring_issue"].value_counts().to_dict()
            for reason, frame in reviewed.groupby("review_reason")
        },
    }
    return reviewed, metrics


def write_summary(path: Path, metrics: dict[str, Any]) -> None:
    groups = metrics["group_review"]
    links = metrics["missed_link_review"]
    lines = [
        "# Issue-index quality audit results",
        "",
        "## Recurring-group audit",
        "",
        f"- Groups reviewed: {groups['groups_reviewed']}",
        f"- Coherent as-is: {groups['coherent_as_is_percent']}%",
        (
            "- Acceptable after removing explicit incorrect members: "
            f"{groups['acceptable_after_exclusions_percent']}%"
        ),
        f"- Require splitting: {groups['overmerge_split_percent']}%",
        f"- Reject as false recurrence: {groups['false_recurrence_reject_percent']}%",
        "",
        "## Missed-link audit",
        "",
        f"- Pairs reviewed: {links['pairs_reviewed']}",
        (
            "- Above-threshold pairs already grouped: "
            f"{links['above_threshold_already_grouped']}/"
            f"{links['above_threshold_pairs_reviewed']}"
        ),
        (
            "- Above-threshold semantic agreement: "
            f"{links['above_threshold_pair_agreement_percent']}%"
        ),
        f"- Missed recurring candidates in sample: {links['missed_recurring_candidates']}",
        f"- Missed two-report emerging links in sample: {links['missed_emerging_links']}",
        f"- Manual follow-up pairs: {links['manual_follow_up']}",
        "",
        "These are stratified audit results, not population-weighted estimates.",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    audit_dir = args.audit_dir.expanduser().resolve()
    group_review, group_metrics = score_group_review(audit_dir)
    missed_review, missed_metrics = score_missed_links(audit_dir)
    group_review.to_csv(audit_dir / "group_reviewed.csv", index=False)
    missed_review.to_csv(audit_dir / "missed_link_reviewed.csv", index=False)
    metrics = {
        "group_review": group_metrics,
        "missed_link_review": missed_metrics,
    }
    (audit_dir / "audit_results.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    write_summary(audit_dir / "audit_results.md", metrics)
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
