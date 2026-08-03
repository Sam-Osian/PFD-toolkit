#!/usr/bin/env python3
"""Archived: build calibrated, non-exclusive parent-family assignments.

Precise relational groups remain the operational child layer. Broad parent
families are additional many-to-many labels, never replacements for child
group identity.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from build_issue_family_recall_pilot import FAMILIES, family_masks


HIERARCHY_VERSION = "issue-family-hierarchy-v1"
CHANNELS = ("strict_lexical", "schema_rule", "semantic_only")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidate-pool-csv", required=True, type=Path)
    parser.add_argument("--adjudication-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--minimum-precision", type=float, default=0.90)
    parser.add_argument("--minimum-calibration-rows", type=int, default=8)
    parser.add_argument("--calibration-fraction", type=float, default=0.70)
    parser.add_argument("--seed", type=int, default=20260730)
    return parser.parse_args()


def as_bool(value: Any) -> bool:
    if isinstance(value, bool):
        return value
    return str(value).strip().casefold() in {"1", "true", "yes"}


def split_value(family_id: str, issue_id: str, seed: int) -> float:
    digest = hashlib.sha256(
        f"{seed}:{family_id}:{issue_id}".encode("utf-8")
    ).digest()
    return int.from_bytes(digest[:8], "big") / float(2**64)


def add_selection_channel(frame: pd.DataFrame) -> pd.DataFrame:
    output = frame.copy()
    output["strict_selected"] = False
    family_lookup = {family.family_id: family for family in FAMILIES}
    for family_id, indices in output.groupby("family_id").groups.items():
        family = family_lookup[str(family_id)]
        strict, _ = family_masks(output.loc[indices], family)
        output.loc[indices, "strict_selected"] = strict.to_numpy()
    output["rule_selected"] = output["rule_selected"].map(as_bool)
    output["semantic_only"] = output["semantic_only"].map(as_bool)
    output["selection_channel"] = np.select(
        [
            output["strict_selected"],
            output["rule_selected"],
            output["semantic_only"],
        ],
        list(CHANNELS),
        default="semantic_only",
    )
    return output


def choose_threshold(
    frame: pd.DataFrame,
    *,
    minimum_precision: float,
    minimum_rows: int,
) -> dict[str, Any]:
    if len(frame) < minimum_rows:
        return {
            "threshold": None,
            "calibration_rows": len(frame),
            "selected_rows": 0,
            "precision": None,
            "recall": 0.0,
            "reason": "insufficient_calibration_rows",
        }
    ordered = frame.sort_values("family_similarity", ascending=False)
    positives = int(ordered["supports_family"].sum())
    best: dict[str, Any] | None = None
    for threshold in sorted(
        ordered["family_similarity"].astype(float).unique()
    ):
        selected = ordered[
            ordered["family_similarity"].astype(float).ge(float(threshold))
        ]
        if len(selected) < minimum_rows:
            continue
        precision = float(selected["supports_family"].mean())
        if precision < minimum_precision:
            continue
        true_positives = int(selected["supports_family"].sum())
        candidate = {
            "threshold": float(threshold),
            "calibration_rows": len(frame),
            "selected_rows": len(selected),
            "precision": precision,
            "recall": true_positives / positives if positives else 0.0,
            "reason": "precision_target_met",
        }
        if best is None or (
            candidate["selected_rows"],
            candidate["recall"],
            candidate["precision"],
        ) > (
            best["selected_rows"],
            best["recall"],
            best["precision"],
        ):
            best = candidate
    if best is not None:
        return best
    return {
        "threshold": None,
        "calibration_rows": len(frame),
        "selected_rows": 0,
        "precision": None,
        "recall": 0.0,
        "reason": "precision_target_not_met",
    }


def predict_membership(
    frame: pd.DataFrame,
    thresholds: dict[str, dict[str, dict[str, Any]]],
) -> pd.Series:
    result = pd.Series(False, index=frame.index)
    for (family_id, channel), indices in frame.groupby(
        ["family_id", "selection_channel"]
    ).groups.items():
        threshold = thresholds[str(family_id)][str(channel)]["threshold"]
        if threshold is None:
            continue
        result.loc[indices] = frame.loc[indices, "family_similarity"].astype(
            float
        ).ge(float(threshold))
    return result


def evaluation_metrics(frame: pd.DataFrame, predicted: pd.Series) -> dict[str, Any]:
    if frame.empty:
        return {
            "rows": 0,
            "precision": None,
            "recall": None,
            "report_recall": None,
        }
    truth = frame["supports_family"].astype(bool)
    true_positive = int((truth & predicted).sum())
    false_positive = int((~truth & predicted).sum())
    false_negative = int((truth & ~predicted).sum())
    supported_reports = set(frame.loc[truth, "report_key"].astype(str))
    captured_reports = set(frame.loc[truth & predicted, "report_key"].astype(str))
    return {
        "rows": len(frame),
        "supported": int(truth.sum()),
        "predicted_members": int(predicted.sum()),
        "true_positive": true_positive,
        "false_positive": false_positive,
        "false_negative": false_negative,
        "precision": (
            true_positive / (true_positive + false_positive)
            if true_positive + false_positive
            else None
        ),
        "recall": (
            true_positive / (true_positive + false_negative)
            if true_positive + false_negative
            else None
        ),
        "report_recall": (
            len(captured_reports) / len(supported_reports)
            if supported_reports
            else None
        ),
    }


def build_hierarchy(
    pool: pd.DataFrame,
    adjudication: pd.DataFrame,
    *,
    minimum_precision: float,
    minimum_calibration_rows: int,
    calibration_fraction: float,
    seed: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    if not 0.0 < calibration_fraction < 1.0:
        raise ValueError("calibration_fraction must be between zero and one")
    if not 0.0 < minimum_precision <= 1.0:
        raise ValueError("minimum_precision must be in (0, 1]")
    keyed = ["family_id", "issue_id"]
    if pool.duplicated(keyed).any():
        raise ValueError("Candidate pool contains duplicate family/issue rows")
    decisions = adjudication[
        [*keyed, "supports_family", "confidence", "reason"]
    ].copy()
    decisions = decisions[decisions["supports_family"].astype(str).ne("")]
    decisions["supports_family"] = decisions["supports_family"].map(as_bool)
    if decisions.duplicated(keyed).any():
        raise ValueError("Adjudication contains duplicate family/issue rows")
    frame = add_selection_channel(
        pool.merge(decisions, on=keyed, how="left", validate="one_to_one")
    )
    evaluated = frame["supports_family"].notna()
    frame["calibration_split"] = ""
    frame.loc[evaluated, "calibration_split"] = [
        (
            "calibration"
            if split_value(str(family), str(issue), seed) < calibration_fraction
            else "evaluation"
        )
        for family, issue in frame.loc[evaluated, keyed].itertuples(
            index=False, name=None
        )
    ]

    thresholds: dict[str, dict[str, dict[str, Any]]] = {}
    for family in FAMILIES:
        thresholds[family.family_id] = {}
        for channel in CHANNELS:
            local = frame[
                frame["family_id"].eq(family.family_id)
                & frame["selection_channel"].eq(channel)
                & frame["calibration_split"].eq("calibration")
            ].copy()
            thresholds[family.family_id][channel] = choose_threshold(
                local,
                minimum_precision=minimum_precision,
                minimum_rows=minimum_calibration_rows,
            )

    predicted = predict_membership(frame, thresholds)
    frame["calibrated_prediction"] = predicted
    frame["parent_membership_status"] = np.where(
        predicted, "calibrated_member", "review_candidate"
    )
    frame["is_parent_member"] = predicted
    supported = evaluated & frame["supports_family"].astype(bool)
    rejected = evaluated & ~frame["supports_family"].astype(bool)
    frame.loc[supported, "parent_membership_status"] = "validated_member"
    frame.loc[supported, "is_parent_member"] = True
    frame.loc[rejected, "parent_membership_status"] = "validated_rejection"
    frame.loc[rejected, "is_parent_member"] = False

    parent_members = frame[frame["is_parent_member"]].copy()
    summary_rows: list[dict[str, Any]] = []
    child_rows: list[dict[str, Any]] = []
    family_metrics: dict[str, Any] = {}
    for family in FAMILIES:
        local = frame[frame["family_id"].eq(family.family_id)]
        members = parent_members[
            parent_members["family_id"].eq(family.family_id)
        ]
        recurring = members[
            members["refinement_status"].eq("refined_recurring")
            & members["refined_group_id"].astype(str).ne("")
        ]
        summary_rows.append(
            {
                "family_id": family.family_id,
                "family_label": family.label,
                "candidate_occurrences": len(local),
                "candidate_reports": local["report_key"].nunique(),
                "assigned_occurrences": len(members),
                "assigned_reports": members["report_key"].nunique(),
                "operational_child_groups": recurring[
                    "refined_group_id"
                ].nunique(),
                "assigned_in_review_groups": int(
                    members["refinement_status"].eq("review_required").sum()
                ),
                "assigned_isolated_or_unassigned": int(
                    (
                        ~members["refinement_status"].isin(
                            ["refined_recurring", "review_required"]
                        )
                    ).sum()
                ),
            }
        )
        for child_group_id, child in recurring.groupby("refined_group_id"):
            child_rows.append(
                {
                    "family_id": family.family_id,
                    "family_label": family.label,
                    "refined_group_id": child_group_id,
                    "assigned_occurrences": len(child),
                    "assigned_reports": child["report_key"].nunique(),
                    "validated_occurrences": int(
                        child["parent_membership_status"].eq(
                            "validated_member"
                        ).sum()
                    ),
                }
            )
        holdout = local[local["calibration_split"].eq("evaluation")]
        family_metrics[family.family_id] = {
            "evaluation": evaluation_metrics(
                holdout, predicted.loc[holdout.index]
            ),
            "thresholds": thresholds[family.family_id],
        }

    issue_parents = (
        parent_members.groupby(["issue_id", "report_key"], as_index=False)
        .agg(
            parent_family_ids=(
                "family_id",
                lambda values: "|".join(sorted(set(map(str, values)))),
            ),
            parent_family_labels=(
                "family_label",
                lambda values: "|".join(sorted(set(map(str, values)))),
            ),
            parent_family_count=("family_id", "nunique"),
        )
    )
    metrics = {
        "hierarchy_version": HIERARCHY_VERSION,
        "minimum_precision": minimum_precision,
        "minimum_calibration_rows": minimum_calibration_rows,
        "calibration_fraction": calibration_fraction,
        "seed": seed,
        "candidate_rows": len(frame),
        "adjudicated_rows": int(evaluated.sum()),
        "assigned_rows": int(frame["is_parent_member"].sum()),
        "assigned_issue_ids": int(
            frame.loc[frame["is_parent_member"], "issue_id"].nunique()
        ),
        "multi_family_issue_ids": int(
            (issue_parents["parent_family_count"] > 1).sum()
        ),
        "families": family_metrics,
        "interpretation": (
            "Thresholds are calibrated on a stratified, group-balanced sample. "
            "Evaluation metrics are a held-out diagnostic, not corpus prevalence. "
            "Review candidates remain unconfirmed and precise child groups are unchanged."
        ),
    }
    return (
        frame,
        pd.DataFrame(summary_rows),
        pd.DataFrame(child_rows),
        issue_parents,
        metrics,
    )


def main() -> None:
    args = parse_args()
    pool = pd.read_csv(args.candidate_pool_csv).fillna("")
    adjudication = pd.read_csv(args.adjudication_csv).fillna("")
    outputs = build_hierarchy(
        pool,
        adjudication,
        minimum_precision=args.minimum_precision,
        minimum_calibration_rows=args.minimum_calibration_rows,
        calibration_fraction=args.calibration_fraction,
        seed=args.seed,
    )
    assignments, summary, child_groups, issue_parents, metrics = outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(
        args.output_dir / "01_parent_family_assignments.csv", index=False
    )
    summary.to_csv(args.output_dir / "02_parent_family_summary.csv", index=False)
    child_groups.to_csv(
        args.output_dir / "03_parent_child_groups.csv", index=False
    )
    issue_parents.to_csv(
        args.output_dir / "04_issue_parent_families.csv", index=False
    )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
