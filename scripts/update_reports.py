#!/usr/bin/env python3
"""
Helper script for continuously updating the CSV containing all 
Prevention of Future Death (PFD) reports which is bundled with
the repo (`../pfd_toolkit/data`).
"""

import pandas as pd
from pfd_toolkit import PFDScraper
from pathlib import Path

DATA_PATH = Path("/src/pfd_toolkit/data/all_reports.csv")

# -- INITIALISE SCRAPER ENGINE --
scraper = PFDScraper()

# -- LOAD EXISTING REPORTS -- 
if DATA_PATH.exists():
    old_df = pd.read_csv(DATA_PATH)
    old_count = len(old_df)
else:
    old_df = None
    old_count = 0

# -- GET LATEST REPORT DATE -- 
# Come back to this later. I fear that an rrroneous in at least one
# PFD report could throw this off. Hardcode start date as as 2025-01-01 for now

# -- TOP UP REPORTS -- 
new_df = scraper.top_up(old_reports=old_df, date_from="2025-01-01")

if new_df is not None:
    new_count = len(new_df)
    added_count = new_count - old_count

    if added_count > 0:
        new_df.to_csv(DATA_PATH, index=False)
        print(f"✅ CSV refreshed - {added_count} new report(s) added. Total reports: {new_count}.")
        # Write counts to a file for the workflow summary
        with open(".github/workflows/update_summary.txt", "w") as f:
            f.write(f"{added_count} new reports added. Total reports: {new_count}.\n")
    else:
        print("No new reports found - nothing to commit.")
else:
    print("No new reports found - nothing to commit.")