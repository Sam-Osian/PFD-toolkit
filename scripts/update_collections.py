#!/usr/bin/env python3
"""Update rule-based collection columns in the current all_reports.csv dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from pfd_toolkit.collections import COLLECTION_COLUMNS, apply_collection_columns


DATA_PATH = Path("all_reports.csv")


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"{DATA_PATH} not found.")

    reports = pd.read_csv(DATA_PATH)
    collection_columns = list(COLLECTION_COLUMNS.values())

    missing_columns = [column for column in collection_columns if column not in reports.columns]
    if missing_columns:
        row_mask = pd.Series(True, index=reports.index)
    else:
        row_mask = reports[collection_columns].isna().any(axis=1)

    rows_to_update = int(row_mask.sum())
    if rows_to_update == 0:
        print("No collection updates required.")
        return

    apply_collection_columns(reports, row_mask=row_mask)
    reports.to_csv(DATA_PATH, index=False)

    print(f"Updated collection columns for {rows_to_update} row(s).")
    for column in collection_columns:
        print(f"{column}: {int(reports[column].fillna(False).sum())}")

    with open(".github/workflows/update_summary.txt", "a") as summary_file:
        summary_file.write("\n")
        summary_file.write(f"Collection columns updated for {rows_to_update} row(s).\n")


if __name__ == "__main__":
    main()
