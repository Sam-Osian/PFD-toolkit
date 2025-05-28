#!/usr/bin/env python3
"""
Helper script for continuously updating the CSV containing all
Prevention of Future Death (PFD) reports which is bundled with
the repo (`../pfd_toolkit/data`).
"""

import pandas as pd
from pfd_toolkit import PFDScraper, LLM
from pathlib import Path
import os

DATA_PATH = Path("./src/pfd_toolkit/data/all_reports.csv")

# -- INITIALISE LLM CLIENT --
llm_client = LLM(api_key=os.environ["OPENAI_API_KEY"], max_workers=30)

# -- INITIALISE SCRAPER ENGINE --
scraper = PFDScraper(llm=llm_client, llm_fallback=True)

# -- LOAD EXISTING REPORTS --
if DATA_PATH.exists():
    old_df = pd.read_csv(DATA_PATH)
    old_count = len(old_df)
    print(
        f"DEBUG: Successfully read {DATA_PATH}. Initial old_count (DataFrame rows): {old_count}"
    )
else:
    old_df = None
    old_count = 0
    print(f"DEBUG: {DATA_PATH} not found.")

# -- GET LATEST REPORT DATE --
# Come back to this later. I fear that an rrroneous in at least one
# PFD report could throw this off. Hardcode start date as as 2025-01-01 for now

# -- TOP UP REPORTS --
new_df = scraper.top_up(old_reports=old_df, start_date="2025-05-01")

if new_df is not None:
    new_count = len(new_df)
    print(f"DEBUG: new_df generated. new_count (DataFrame rows): {new_count}")
    added_count = new_count - old_count
    print(f"DEBUG: calculated added_count: {added_count}")

    # If new report(s) were found
    if added_count > 0:
        new_df.to_csv(DATA_PATH, index=False)
        print(
            f"✅ CSV refreshed - {added_count} new report(s) added. Total reports: {new_count}."
        )
        # Write counts to a file for the workflow summary
        with open(".github/workflows/update_summary.txt", "w") as f:
            f.write(f"{added_count} new reports added. Total reports: {new_count}.\n")

    # If no new report(s) were found
    else:
        print("No new reports found - nothing to commit.")
        # Note in the workflow summary that no reports were found
        with open(".github/workflows/update_summary.txt", "w") as f:
            f.write("No new reports were identified.")
else:
    print("No new reports found - nothing to commit.")
