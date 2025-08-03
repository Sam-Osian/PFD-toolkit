#!/usr/bin/env python3
"""
Helper script for continuously updating the CSV containing all
Prevention of Future Deaths (PFD) reports.
"""

import os
from pathlib import Path

import pandas as pd
import requests
from pfd_toolkit import LLM, Scraper

DATA_URL = (
    "https://github.com/Sam-Osian/PFD-toolkit/releases/download/"
    "dataset-latest/all_reports.csv"
)
DATA_PATH = Path("all_reports.csv")

# Initialise LLM client
llm_client = LLM(api_key=os.environ["OPENAI_API_KEY"], max_workers=18, model="gpt-4.1")

# Initialise scraper
scraper = Scraper(llm=llm_client, scraping_strategy=[2,-1,1])

# Retrieve the latest dataset from the GitHub release
try:
    resp = requests.get(DATA_URL, timeout=30)
    resp.raise_for_status()
    DATA_PATH.write_bytes(resp.content)
    print(f"DEBUG: Downloaded dataset from {DATA_URL}")
except Exception:
    print(f"DEBUG: Failed to download dataset from {DATA_URL}")

# Load existing reports
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


# Top up reports
new_df = scraper.top_up(old_reports=old_df, start_date="2025-05-01",
                        clean=True)

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
