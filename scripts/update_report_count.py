#!/usr/bin/env python3
"""
Helper script for retrieving the number of reports contained
within the dataset bundled with PFD Toolkit and updating 
index.md to reflect the current count.
"""

import re
import pandas as pd

csv_path = "src/pfd_toolkit/data/all_reports.csv"
md_path = "docs/index.md"

# Count number of rows/reports
df = pd.read_csv(csv_path)
count = len(df)

# Read the markdown file
with open(md_path, "r", encoding="utf-8") as f:
    text = f.read()

# Replace the old 4-digit number inside ** with the current count.
# It matches lines like: includes **5720** PFD reports
pattern = r"(includes \*\*)\d{4}(\*\* PFD reports)"
replacement = rf"\g<1>{count}\g<2>"
text = re.sub(pattern, replacement, text)

# Write the updated markdown back to disk
with open(md_path, "w", encoding="utf-8") as f:
    f.write(text)
