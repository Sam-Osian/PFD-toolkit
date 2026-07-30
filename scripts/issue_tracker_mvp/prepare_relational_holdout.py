#!/usr/bin/env python3
"""Select untouched stratified groups for relational-linkage holdout review."""

from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd


HOLDOUT_VERSION = "relational-holdout-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True, type=Path)
    parser.add_argument("--output-csv", required=True, type=Path)
    parser.add_argument("--group-count", type=int, default=20)
    return parser.parse_args()


def stable_order(value: str) -> str:
    return hashlib.sha256(f"{HOLDOUT_VERSION}|{value}".encode()).hexdigest()


def build_holdout(run_dir: Path, group_count: int) -> tuple[pd.DataFrame, list[str]]:
    stratified = run_dir / "07_quality_audit_stratified"
    queue = pd.read_csv(stratified / "group_review_queue.csv").fillna("")
    members = pd.read_csv(stratified / "group_review_members.csv").fillna("")
    reviewed = pd.read_csv(
        run_dir / "09_validation" / "random_group_audit.csv"
    ).fillna("")
    excluded = set(reviewed["subissue_id"].astype(str))
    candidates = [
        value
        for value in queue["subissue_id"].astype(str).unique()
        if value and value not in excluded
    ]
    selected_groups = sorted(candidates, key=stable_order)[:group_count]
    selection = members[members["subissue_id"].isin(selected_groups)][
        ["issue_id", "subissue_id"]
    ].copy()
    selection = selection.rename(columns={"subissue_id": "holdout_source_group"})
    selection["pilot_source"] = "untouched_stratified_holdout"
    return selection.sort_values("issue_id").reset_index(drop=True), selected_groups


def main() -> None:
    args = parse_args()
    selection, groups = build_holdout(args.run_dir, args.group_count)
    args.output_csv.parent.mkdir(parents=True, exist_ok=True)
    selection.to_csv(args.output_csv, index=False)
    manifest = {
        "holdout_version": HOLDOUT_VERSION,
        "run_dir": str(args.run_dir),
        "group_count": len(groups),
        "issue_count": len(selection),
        "source_group_ids": groups,
        "selection_csv": str(args.output_csv),
    }
    args.output_csv.with_suffix(".manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
