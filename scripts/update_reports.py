#!/usr/bin/env python3
"""
Helper script for continuously updating the CSV containing all 
Prevention of Future Death (PFD) reports which is bundled with
the repo (`..src/pfd_toolkit/data`). 
"""

DATA_PATH = "../pfd_toolkit/data/all_reports.csv" # Where the complete set of PFD reports lives

from pfd_toolkit import PFDScraper
import pandas as pd

# -- INITALISE SCRAPER ENGINE --
scraper = PFDScraper()


# -- UPDATE REPORTS -- 
# Read the existing set of reports
old_df = pd.read_csv(DATA_PATH)

# 'Top up' with new ones
new_df = scraper.top_up(old_reports=old_df, date_from="2025-01-01")


# -- PROTECTION LOGIC --

if new_df is not None and (old_df is None or len(new_df) > len(old_df)):
    new_df.to_csv(DATA_PATH, index=False)
    print("✅  CSV refreshed – will be committed by the workflow.")
else:
    print("ℹ️  No new reports found – nothing to commit.")