#!/usr/bin/env python3
"""One-off receiver normalization for Department of/for variants."""

from __future__ import annotations

from pathlib import Path

import pandas as pd


TARGET_FILES = (
    Path("scripts/data/all_reports.csv"),
    Path("scripts/data/all_reports_with_collections.csv"),
)

RECEIVER_COLUMN = "receiver"
HEALTH_EXCEPTION = "Department of Health and Social Care"


def _normalise_segment(segment: str) -> str:
    stripped = segment.strip()
    if not stripped:
        return stripped
    if stripped == HEALTH_EXCEPTION:
        return stripped
    if stripped.startswith("Department of "):
        return stripped.replace("Department of ", "Department for ", 1)
    return stripped


def _normalise_receiver(receiver: str) -> str:
    if not isinstance(receiver, str):
        return receiver
    segments = [segment.strip() for segment in receiver.split(";")]
    normalised_segments = [_normalise_segment(segment) for segment in segments if segment.strip()]
    return "; ".join(normalised_segments)


def main() -> None:
    for path in TARGET_FILES:
        if not path.exists():
            print(f"Skipping missing file: {path}")
            continue

        reports = pd.read_csv(path)
        if RECEIVER_COLUMN not in reports.columns:
            print(f"Skipping {path}: no '{RECEIVER_COLUMN}' column.")
            continue

        original = reports[RECEIVER_COLUMN].copy()
        reports[RECEIVER_COLUMN] = reports[RECEIVER_COLUMN].apply(_normalise_receiver)
        changed_rows = int((original.fillna("") != reports[RECEIVER_COLUMN].fillna("")).sum())
        reports.to_csv(path, index=False)
        print(f"{path}: updated {changed_rows} receiver row(s).")


if __name__ == "__main__":
    main()
