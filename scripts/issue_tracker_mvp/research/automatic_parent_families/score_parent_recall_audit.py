#!/usr/bin/env python3
"""Archived: validate and score an automatic-parent recall audit."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


DECISIONS = {"yes", "no", "uncertain"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--review-csv", required=True, type=Path)
    parser.add_argument("--assignments-csv", required=True, type=Path)
    parser.add_argument("--decisions-json", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    return parser.parse_args()


def apply_and_trace(
    queue: pd.DataFrame,
    assignments: pd.DataFrame,
    decisions_payload: list[dict[str, Any]],
) -> pd.DataFrame:
    decisions = pd.DataFrame(decisions_payload).fillna("")
    required = {"audit_id", "decision", "matched_issue_id"}
    if missing := required - set(decisions.columns):
        raise ValueError(f"Decisions lack columns: {sorted(missing)}")
    if decisions["audit_id"].astype(str).duplicated().any():
        raise ValueError("Decision audit IDs must be unique")
    expected = set(queue["audit_id"].astype(str))
    supplied = set(decisions["audit_id"].astype(str))
    if expected != supplied:
        raise ValueError(
            f"Decisions must cover queue exactly; missing={len(expected-supplied)}, extra={len(supplied-expected)}"
        )
    invalid = set(decisions["decision"].astype(str).str.casefold()) - DECISIONS
    if invalid:
        raise ValueError(f"Invalid decisions: {sorted(invalid)}")

    assignment_lookup = assignments.drop_duplicates("issue_id").set_index("issue_id").to_dict("index")
    issue_report_lookup = assignments.drop_duplicates("issue_id").set_index("issue_id")["report_key"].astype(str).to_dict()
    decision_lookup = decisions.set_index("audit_id").to_dict("index")
    rows: list[dict[str, Any]] = []
    for row in queue.to_dict("records"):
        decision = decision_lookup[str(row["audit_id"])]
        value = str(decision["decision"]).casefold()
        issue_id = str(decision.get("matched_issue_id", ""))
        if value == "yes" and issue_id:
            if issue_id not in issue_report_lookup:
                raise ValueError(f"Unknown matched issue ID: {issue_id}")
            if issue_report_lookup[issue_id] != str(row["report_key"]):
                raise ValueError(f"Matched issue belongs to another report: {issue_id}")
            assignment = assignment_lookup[issue_id]
            status = str(assignment.get("recurrence_status", ""))
            miss_stage = "other_recurring_child" if status == "recurring" else "isolated_or_pair"
            matched_group_id = str(assignment.get("relational_group_id", ""))
        elif value == "yes":
            miss_stage = "not_extracted"
            matched_group_id = ""
        else:
            miss_stage = "not_applicable"
            matched_group_id = ""
        rows.append(
            row
            | {
                "supports_parent": value,
                "matched_issue_id": issue_id,
                "matched_group_id": matched_group_id,
                "miss_stage": miss_stage,
                "review_notes": str(decision.get("notes", "")),
            }
        )
    return pd.DataFrame(rows)


def score(frame: pd.DataFrame) -> dict[str, Any]:
    result: dict[str, Any] = {"rows": len(frame), "parents": {}}
    for family_id, local in frame.groupby("family_id", sort=False):
        yes = local["supports_parent"].eq("yes")
        no = local["supports_parent"].eq("no")
        uncertain = local["supports_parent"].eq("uncertain")
        strata: dict[str, Any] = {}
        for stratum, rows in local.groupby("recall_stratum"):
            strata[str(stratum)] = {
                "rows": len(rows),
                "confirmed_misses": int(rows["supports_parent"].eq("yes").sum()),
                "rejected": int(rows["supports_parent"].eq("no").sum()),
                "uncertain": int(rows["supports_parent"].eq("uncertain").sum()),
            }
        result["parents"][str(family_id)] = {
            "review_name": str(local.iloc[0]["parent_review_name"]),
            "reviewed": len(local),
            "confirmed_misses": int(yes.sum()),
            "rejected": int(no.sum()),
            "uncertain": int(uncertain.sum()),
            "sample_miss_yield": float(yes.mean()),
            "confirmed_misses_by_stage": local.loc[yes, "miss_stage"].value_counts().to_dict(),
            "confirmed_miss_groups": int(local.loc[yes & local["matched_group_id"].astype(str).ne(""), "matched_group_id"].nunique()),
            "strata": strata,
        }
    result["interpretation"] = (
        "The sample is deliberately enriched for candidate misses. Miss yield is diagnostic and must not be extrapolated to corpus prevalence."
    )
    return result


def main() -> None:
    args = parse_args()
    queue = pd.read_csv(args.review_csv).fillna("")
    assignments = pd.read_csv(args.assignments_csv).fillna("")
    payload = json.loads(args.decisions_json.read_text(encoding="utf-8"))
    output = apply_and_trace(queue, assignments, payload)
    metrics = score(output)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_dir / "02_reviewed_parent_recall.csv", index=False)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
