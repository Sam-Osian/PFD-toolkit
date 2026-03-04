#!/usr/bin/env python3
"""Generate rule-based receiver collection columns for a PFD dataset."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
from pfd_toolkit.collections import apply_collection_columns


INPUT_CSV = Path("all_reports.csv")
OUTPUT_CSV = Path("all_reports.csv")
OVERWRITE_INPUT = True


def main() -> None:
    input_path = INPUT_CSV
    output_path = input_path if OVERWRITE_INPUT else OUTPUT_CSV

    reports = pd.read_csv(input_path)
    apply_collection_columns(reports)

    reports.to_csv(output_path, index=False)

    theme_columns = [column for column in reports.columns if column.startswith("theme_")]
    print(f"Rows processed: {len(reports)}")
    print(f"Wrote collection dataset to: {output_path}")
    for column in theme_columns:
        print(f"{column}: {int(reports[column].fillna(False).sum())}")


if __name__ == "__main__":
    main()
