#!/usr/bin/env python3
"""Archived: validate and score an overlapping-family attachment audit."""

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
    parser.add_argument(
        "--decisions-json",
        type=Path,
        help="Optional JSON array with audit_id, decision, reason, and notes.",
    )
    parser.add_argument("--output-json", type=Path)
    return parser.parse_args()


def apply_decisions(
    frame: pd.DataFrame, decisions_payload: list[dict[str, Any]]
) -> pd.DataFrame:
    decisions = pd.DataFrame(decisions_payload)
    required = {"audit_id", "decision"}
    if missing := required - set(decisions.columns):
        raise ValueError(f"Decision payload lacks columns: {sorted(missing)}")
    if decisions["audit_id"].astype(str).duplicated().any():
        raise ValueError("Decision payload contains duplicate audit IDs")
    expected = set(frame["audit_id"].astype(str))
    supplied = set(decisions["audit_id"].astype(str))
    if expected != supplied:
        raise ValueError(
            "Decision payload must cover the audit exactly; "
            f"missing={len(expected - supplied)}, extra={len(supplied - expected)}"
        )
    decision_lookup = decisions.set_index("audit_id").to_dict("index")
    output = frame.copy()
    output["supports_secondary_family"] = output["audit_id"].map(
        lambda audit_id: decision_lookup[str(audit_id)]["decision"]
    )
    output["decision_reason"] = output["audit_id"].map(
        lambda audit_id: decision_lookup[str(audit_id)].get("reason", "")
    )
    output["review_notes"] = output["audit_id"].map(
        lambda audit_id: decision_lookup[str(audit_id)].get("notes", "")
    )
    return output


def score(frame: pd.DataFrame) -> dict[str, Any]:
    if frame["audit_id"].astype(str).duplicated().any():
        raise ValueError("Audit IDs must be unique")
    decisions = frame["supports_secondary_family"].astype(str).str.casefold()
    invalid = sorted(set(decisions) - DECISIONS)
    if invalid:
        raise ValueError(
            "Every audit row requires yes/no/uncertain; invalid values: "
            f"{invalid}"
        )
    yes_all = decisions.eq("yes")
    no_all = decisions.eq("no")
    decided_all = yes_all | no_all
    total_decided_weight = int(
        frame.loc[decided_all, "child_report_count"].sum()
    )
    result: dict[str, Any] = {
        "rows": len(frame),
        "overall": {
            "yes": int(yes_all.sum()),
            "no": int(no_all.sum()),
            "uncertain": int(decisions.eq("uncertain").sum()),
            "attachment_precision_among_decided": (
                float(yes_all.sum() / decided_all.sum())
                if decided_all.sum()
                else None
            ),
            "report_weighted_precision_among_decided": (
                int(frame.loc[yes_all, "child_report_count"].sum())
                / total_decided_weight
                if total_decided_weight
                else None
            ),
            "rejection_reasons": frame.get(
                "decision_reason", pd.Series("", index=frame.index)
            ).loc[no_all]
            .astype(str)
            .value_counts()
            .to_dict(),
        },
        "strata": {},
    }
    for stratum, local in frame.assign(_decision=decisions).groupby(
        "audit_stratum"
    ):
        decided = local[local["_decision"].isin(["yes", "no"])]
        yes = int(local["_decision"].eq("yes").sum())
        no = int(local["_decision"].eq("no").sum())
        uncertain = int(local["_decision"].eq("uncertain").sum())
        report_weight_yes = int(
            local.loc[
                local["_decision"].eq("yes"), "child_report_count"
            ].sum()
        )
        report_weight_decided = int(decided["child_report_count"].sum())
        result["strata"][str(stratum)] = {
            "rows": len(local),
            "yes": yes,
            "no": no,
            "uncertain": uncertain,
            "attachment_precision_among_decided": (
                yes / (yes + no) if yes + no else None
            ),
            "report_weighted_precision_among_decided": (
                report_weight_yes / report_weight_decided
                if report_weight_decided
                else None
            ),
            "families": local["target_family_id"].nunique(),
        }
    result["interpretation"] = (
        "The sample is family-balanced, not population-weighted. Precision "
        "describes reviewed attachment proposals, not extraction recall."
    )
    return result


def main() -> None:
    args = parse_args()
    frame = pd.read_csv(args.review_csv).fillna("")
    if args.decisions_json:
        payload = json.loads(args.decisions_json.read_text(encoding="utf-8"))
        if not isinstance(payload, list):
            raise ValueError("Decision JSON must contain an array")
        frame = apply_decisions(frame, payload)
    metrics = score(frame)
    if args.output_json:
        args.output_json.parent.mkdir(parents=True, exist_ok=True)
        args.output_json.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
