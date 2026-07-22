#!/usr/bin/env python3
"""Score deterministic issue-index configurations against reviewed audit groups."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--audit-dir", type=Path, required=True)
    parser.add_argument("--tuning-dir", type=Path, required=True)
    return parser.parse_args()


def member_ids(value: object) -> set[str]:
    if isinstance(value, list):
        return {str(item) for item in value}
    text = str(value).strip()
    return {item.strip() for item in text.split("|") if item.strip()}


def percent(numerator: float, denominator: float) -> float:
    return round(100.0 * numerator / denominator, 2) if denominator else 0.0


def group_outcome(
    source: pd.DataFrame,
    decision: pd.Series,
    assignments: pd.DataFrame,
) -> dict[str, Any]:
    source_ids = set(source["issue_id"])
    exclusions = member_ids(decision.get("incorrect_member_issue_ids", ""))
    core_ids = source_ids - exclusions
    scoped = assignments[assignments["issue_id"].isin(source_ids)].copy()
    core = scoped[scoped["issue_id"].isin(core_ids)]
    group_counts = core["candidate_group_id"].value_counts()
    dominant_group = str(group_counts.index[0]) if len(group_counts) else ""
    dominant_core_count = int(group_counts.iloc[0]) if len(group_counts) else 0
    dominant_core_fraction = dominant_core_count / max(len(core_ids), 1)
    dominant_status = ""
    if dominant_group:
        dominant_status = str(
            core.loc[
                core["candidate_group_id"].eq(dominant_group), "recurrence_status"
            ].iloc[0]
        )
    dominant_core_reports = int(
        source[
            source["issue_id"].isin(
                set(core.loc[core["candidate_group_id"].eq(dominant_group), "issue_id"])
            )
        ]["report_key"].nunique()
    )
    source_recurring_survives = False
    for candidate_group_id, frame in scoped.groupby("candidate_group_id"):
        if (
            not candidate_group_id
            or not frame["recurrence_status"].eq("recurring").any()
        ):
            continue
        reports = source[source["issue_id"].isin(frame["issue_id"])][
            "report_key"
        ].nunique()
        if reports >= 3:
            source_recurring_survives = True
            break
    exclusion_detached = True
    if exclusions and dominant_group:
        exclusion_groups = set(
            scoped.loc[scoped["issue_id"].isin(exclusions), "candidate_group_id"]
        )
        exclusion_detached = dominant_group not in exclusion_groups
    source_dominant_counts = scoped["candidate_group_id"].value_counts()
    source_dominant_fraction = (
        float(source_dominant_counts.iloc[0]) / len(source_ids)
        if len(source_dominant_counts)
        else 0.0
    )
    assigned_fraction = len(scoped) / max(len(source_ids), 1)
    return {
        "subissue_id": decision["subissue_id"],
        "audit_decision": decision["audit_decision"],
        "source_issue_count": len(source_ids),
        "core_issue_count": len(core_ids),
        "assigned_fraction": assigned_fraction,
        "dominant_group": dominant_group,
        "dominant_status": dominant_status,
        "dominant_core_fraction": dominant_core_fraction,
        "dominant_core_reports": dominant_core_reports,
        "valid_core_preserved": (
            dominant_status == "recurring"
            and dominant_core_reports >= 3
            and dominant_core_fraction >= 0.8
        ),
        "exact_core_preserved": (
            dominant_status == "recurring" and dominant_core_count == len(core_ids)
        ),
        "source_recurring_survives": source_recurring_survives,
        "source_dominant_fraction": source_dominant_fraction,
        "exclusion_detached": exclusion_detached,
        "overmerge_resolved": (
            not source_recurring_survives or source_dominant_fraction <= 0.8
        ),
    }


def score_config(
    config: str,
    reviewed: pd.DataFrame,
    members: pd.DataFrame,
    assignments: pd.DataFrame,
    tuning_summary: pd.Series,
) -> tuple[dict[str, Any], pd.DataFrame]:
    outcomes = []
    for decision in reviewed.to_dict("records"):
        source = members[members["subissue_id"].eq(decision["subissue_id"])]
        outcomes.append(group_outcome(source, pd.Series(decision), assignments))
    frame = pd.DataFrame(outcomes)
    valid = frame[frame["audit_decision"].isin(["accept", "accept_with_exclusions"])]
    accepted = frame[frame["audit_decision"].eq("accept")]
    exclusions = frame[frame["audit_decision"].eq("accept_with_exclusions")]
    splits = frame[frame["audit_decision"].eq("split")]
    rejects = frame[frame["audit_decision"].eq("reject")]
    valid_preservation = percent(valid["valid_core_preserved"].sum(), len(valid))
    valid_type_survival = percent(
        valid["source_recurring_survives"].sum(), len(valid)
    )
    exact_accept_preservation = percent(
        accepted["exact_core_preserved"].sum(), len(accepted)
    )
    reject_suppression = percent(
        (~rejects["source_recurring_survives"]).sum(), len(rejects)
    )
    split_resolution = percent(splits["overmerge_resolved"].sum(), len(splits))
    exclusion_success = percent(exclusions["exclusion_detached"].sum(), len(exclusions))
    mean_valid_core = percent(valid["dominant_core_fraction"].sum(), len(valid))
    mean_assignment = percent(frame["assigned_fraction"].sum(), len(frame))
    balanced_score = round(
        0.40 * valid_preservation
        + 0.25 * reject_suppression
        + 0.25 * split_resolution
        + 0.10 * exclusion_success,
        2,
    )
    precision_score = round(
        0.25 * valid_preservation
        + 0.30 * reject_suppression
        + 0.35 * split_resolution
        + 0.10 * exclusion_success,
        2,
    )
    summary = {
        "config": config,
        "valid_type_survival_percent": valid_type_survival,
        "valid_group_preservation_percent": valid_preservation,
        "exact_accept_preservation_percent": exact_accept_preservation,
        "mean_valid_core_coassignment_percent": mean_valid_core,
        "reject_suppression_percent": reject_suppression,
        "overmerge_resolution_percent": split_resolution,
        "exclusion_detachment_percent": exclusion_success,
        "mean_audited_assignment_percent": mean_assignment,
        "balanced_quality_score": balanced_score,
        "precision_priority_score": precision_score,
        "recurring_subissues": int(tuning_summary["recurring_subissues"]),
        "recurring_issue_occurrences": int(
            tuning_summary["recurring_issue_occurrences"]
        ),
        "recurring_percent": float(tuning_summary["recurring_percent"]),
        "ungrouped_issue_occurrences": int(
            tuning_summary["ungrouped_issue_occurrences"]
        ),
    }
    frame.insert(0, "config", config)
    return summary, frame


def main() -> None:
    args = parse_args()
    audit_dir = args.audit_dir.expanduser().resolve()
    tuning_dir = args.tuning_dir.expanduser().resolve()
    reviewed = pd.read_csv(audit_dir / "group_reviewed.csv").fillna("")
    members = pd.read_csv(audit_dir / "group_review_members.csv").fillna("")
    tuning_summary = pd.read_csv(tuning_dir / "tuning_summary.csv").fillna("")
    summaries: list[dict[str, Any]] = []
    details: list[pd.DataFrame] = []
    for row in tuning_summary.to_dict("records"):
        config = str(row["name"])
        assignments = pd.read_csv(tuning_dir / f"{config}_assignments.csv").fillna("")
        summary, detail = score_config(
            config,
            reviewed,
            members,
            assignments,
            pd.Series(row),
        )
        summaries.append(summary)
        details.append(detail)
    result = pd.DataFrame(summaries).sort_values(
        ["precision_priority_score", "balanced_quality_score"],
        ascending=False,
    )
    result.to_csv(tuning_dir / "audit_scored_summary.csv", index=False)
    pd.concat(details, ignore_index=True).to_csv(
        tuning_dir / "audit_scored_group_outcomes.csv", index=False
    )
    (tuning_dir / "audit_scored_summary.json").write_text(
        json.dumps(result.to_dict("records"), indent=2), encoding="utf-8"
    )
    print(result.to_string(index=False))


if __name__ == "__main__":
    main()
