#!/usr/bin/env python3
"""
Helper script for retrieving the number of reports contained
within the dataset bundled with PFD Toolkit.
"""

import pandas as pd

csv_path = "src/pfd_toolkit/data/all_reports.csv"
md_path = "docs/index.md"

# Count number of rows/reports
df = pd.read_csv(csv_path)
count = len(df)

# Replace {{NUM_REPORTS}} in docs/index.md
with open(md_path, "r", encoding="utf-8") as f:
    text = f.read()

text = text.replace("{{NUM_REPORTS}}", str(count))

with open(md_path, "w", encoding="utf-8") as f:
    f.write(text)
