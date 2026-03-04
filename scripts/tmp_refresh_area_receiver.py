#!/usr/bin/env python3
"""Utility for refreshing ``area`` and ``receiver`` in all_reports.csv."""

from __future__ import annotations

import os
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from pfd_toolkit import Cleaner, LLM, Scraper


# Temporary script settings.
# This is a scaffold for manual use while the refresh workflow is still being
# refined; keep the constants editable at the top of the file.
INPUT_CSV = Path("scripts/data/all_reports.csv")
OUTPUT_CSV = Path("scripts/data/all_reports_refreshed.csv")
ROW_LIMIT: int | None = None
OVERWRITE_INPUT = False
MODEL = "gpt-4.1"
REFRESH_FIELDS = ["area", "receiver"]
# Match the staged extraction order used by scripts/update_reports.py:
# LLM first, then HTML, with PDF disabled.
SCRAPING_STRATEGY = [2, -1, 1]
MAX_WORKERS = 20
MAX_REQUESTS = 10
CHUNK_SIZE = 250
CHECKPOINT_EVERY_CHUNK = True
DEBUG_AREA_OTHERS = True
DEBUG_AREA_OTHERS_LIMIT = 25
DEBUG_RECEIVER_CHANGES = True
DEBUG_RECEIVER_CHANGES_LIMIT = 25
ANALYSE_EXISTING_OUTPUT = False
RESUME_FROM_OUTPUT = True
ONLY_RESCRAPE_AREA_OTHER = False


project_root = Path(__file__).resolve().parents[1]
load_dotenv(project_root / "api.env")
api_key = os.getenv("OPENAI_API_KEY")
if not api_key:
    raise RuntimeError("OPENAI_API_KEY not found in project-root api.env.")

input_path = INPUT_CSV
output_path = input_path if OVERWRITE_INPUT else OUTPUT_CSV

reports = pd.read_csv(input_path)
if ONLY_RESCRAPE_AREA_OTHER:
    if "area" not in REFRESH_FIELDS:
        raise ValueError(
            "ONLY_RESCRAPE_AREA_OTHER=True requires 'area' to be included in REFRESH_FIELDS."
        )
    source_path = output_path if output_path.exists() else input_path
    reports = pd.read_csv(source_path)
    reports = reports.loc[reports["area"] == "Other"].copy()
    print(f"Targeting current 'Other' area rows from: {source_path}")

if ROW_LIMIT is not None:
    reports = reports.head(ROW_LIMIT).copy()

before_area = reports["area"].copy() if "area" in reports.columns else pd.Series(dtype="object")
before_receiver = (
    reports["receiver"].copy() if "receiver" in reports.columns else pd.Series(dtype="object")
)

if ANALYSE_EXISTING_OUTPUT:
    if not output_path.exists():
        raise FileNotFoundError(
            f"Refreshed output not found at {output_path}. Run the refresh first or change OUTPUT_CSV."
        )

    updated = pd.read_csv(output_path)
    comparison = reports.merge(
        updated[[ "url", "area", "receiver" ]],
        on="url",
        how="inner",
        suffixes=("_old", "_new"),
    )
    area_changed = int((comparison["area_old"].fillna("") != comparison["area_new"].fillna("")).sum())
    receiver_changed = int(
        (comparison["receiver_old"].fillna("") != comparison["receiver_new"].fillna("")).sum()
    )

    print(f"Input rows: {len(reports)}")
    print(f"Compared rows: {len(comparison)}")
    print(f"Rows with updated area: {area_changed}")
    print(f"Rows with updated receiver: {receiver_changed}")
    print(f"Old 'Other' count: {int((comparison['area_old'] == 'Other').sum())}")
    print(f"New 'Other' count: {int((comparison['area_new'] == 'Other').sum())}")

    area_debug_path = output_path.with_name(f"{output_path.stem}_area_other_debug.csv")
    if area_debug_path.exists():
        debug_df = pd.read_csv(area_debug_path)
        print(f"Wrote area='Other' debug CSV already exists at: {area_debug_path}")
        print(debug_df.head(DEBUG_AREA_OTHERS_LIMIT).to_string(index=False))
    else:
        other_df = comparison.loc[
            comparison["area_new"] == "Other",
            ["url", "area_old", "area_new"],
        ].copy()
        print(
            "No raw-area debug CSV found, so showing current Other rows from the refreshed file only."
        )
        if not other_df.empty:
            print(other_df.head(DEBUG_AREA_OTHERS_LIMIT).to_string(index=False))

    receiver_debug_path = output_path.with_name(
        f"{output_path.stem}_receiver_change_debug.csv"
    )
    if receiver_debug_path.exists():
        receiver_debug_df = pd.read_csv(receiver_debug_path)
        print(f"Wrote receiver change debug CSV already exists at: {receiver_debug_path}")
        print(receiver_debug_df.head(DEBUG_RECEIVER_CHANGES_LIMIT).to_string(index=False))
    else:
        changed_receiver_df = comparison.loc[
            comparison["receiver_old"].fillna("") != comparison["receiver_new"].fillna(""),
            ["url", "receiver_old", "receiver_new"],
        ].copy()
        print(
            "No receiver debug CSV found, so showing changed receiver rows from the refreshed file only."
        )
        if not changed_receiver_df.empty:
            print(changed_receiver_df.head(DEBUG_RECEIVER_CHANGES_LIMIT).to_string(index=False))
    raise SystemExit(0)

llm = LLM(api_key=api_key, model=MODEL, max_workers=MAX_WORKERS, seed=123)
scraper = Scraper(
    llm=llm,
    scraping_strategy=SCRAPING_STRATEGY,
    max_workers=MAX_WORKERS,
    max_requests=MAX_REQUESTS,
)

chunks = []
area_debug_rows = []
receiver_debug_rows = []
total_rows = len(reports)
num_chunks = (total_rows + CHUNK_SIZE - 1) // CHUNK_SIZE
resume_row_count = 0

if RESUME_FROM_OUTPUT and output_path.exists() and not ONLY_RESCRAPE_AREA_OTHER:
    existing_output = pd.read_csv(output_path)
    expected_prefix = reports.head(len(existing_output)).copy()
    if set(existing_output.columns) != set(reports.columns):
        raise ValueError(
            "Existing output columns do not match the input dataset. "
            "Delete the output file or disable RESUME_FROM_OUTPUT."
        )
    existing_output = existing_output[reports.columns.tolist()]
    if len(existing_output) > len(reports):
        raise ValueError(
            "Existing output has more rows than the current input slice. "
            "Delete the output file or disable RESUME_FROM_OUTPUT."
        )
    if not existing_output.empty:
        if existing_output["url"].tolist() != expected_prefix["url"].tolist():
            raise ValueError(
                "Existing output does not match the leading rows of the current input dataset. "
                "Delete the output file or disable RESUME_FROM_OUTPUT."
            )
        chunks.append(existing_output)
        resume_row_count = len(existing_output)
        print(f"Resuming from existing output: skipping first {resume_row_count} rows.")

for chunk_index, start in enumerate(range(resume_row_count, total_rows, CHUNK_SIZE), start=(resume_row_count // CHUNK_SIZE) + 1):
    stop = min(start + CHUNK_SIZE, total_rows)
    chunk = reports.iloc[start:stop].copy()
    print(f"Refreshing rows {start + 1}-{stop} of {total_rows} (chunk {chunk_index}/{num_chunks})")

    raw_chunk = scraper.rescrape_fields(
        reports_df=chunk,
        fields=REFRESH_FIELDS,
        clean=False,
    )

    cleaner = Cleaner(
        reports=raw_chunk.copy(),
        llm=llm,
        include_coroner=False,
        include_receiver="receiver" in REFRESH_FIELDS,
        include_area="area" in REFRESH_FIELDS,
        include_investigation=False,
        include_circumstances=False,
        include_concerns=False,
    )
    refreshed_chunk = cleaner.clean_reports()

    refreshed_chunk = chunk.drop(columns=REFRESH_FIELDS, errors="ignore").merge(
        refreshed_chunk[[scraper.COL_URL, *REFRESH_FIELDS]],
        on=scraper.COL_URL,
        how="left",
    )

    # Keep original values when the refresh pipeline fails to recover a value
    # for a stale or unavailable URL.
    for field in REFRESH_FIELDS:
        if field in chunk.columns and field in refreshed_chunk.columns:
            refreshed_chunk[field] = refreshed_chunk[field].where(
                refreshed_chunk[field].notna(),
                chunk[field],
            )

    if DEBUG_AREA_OTHERS:
        merged_area_debug = (
            chunk[[scraper.COL_URL, "area"]]
            .rename(columns={"area": "area_old"})
            .merge(
                raw_chunk[[scraper.COL_URL, "area"]].rename(columns={"area": "area_raw"}),
                on=scraper.COL_URL,
                how="left",
            )
            .merge(
                refreshed_chunk[[scraper.COL_URL, "area"]].rename(columns={"area": "area_new"}),
                on=scraper.COL_URL,
                how="left",
            )
        )
        chunk_other_rows = merged_area_debug[merged_area_debug["area_new"] == "Other"].copy()
        if not chunk_other_rows.empty:
            area_debug_rows.append(chunk_other_rows)

    if DEBUG_RECEIVER_CHANGES:
        merged_receiver_debug = (
            chunk[[scraper.COL_URL, "receiver"]]
            .rename(columns={"receiver": "receiver_old"})
            .merge(
                raw_chunk[[scraper.COL_URL, "receiver"]].rename(
                    columns={"receiver": "receiver_raw"}
                ),
                on=scraper.COL_URL,
                how="left",
            )
            .merge(
                refreshed_chunk[[scraper.COL_URL, "receiver"]].rename(
                    columns={"receiver": "receiver_new"}
                ),
                on=scraper.COL_URL,
                how="left",
            )
        )
        chunk_receiver_changes = merged_receiver_debug[
            merged_receiver_debug["receiver_old"].fillna("")
            != merged_receiver_debug["receiver_new"].fillna("")
        ].copy()
        if not chunk_receiver_changes.empty:
            receiver_debug_rows.append(chunk_receiver_changes)

    chunks.append(refreshed_chunk)

    if CHECKPOINT_EVERY_CHUNK:
        checkpoint_df = pd.concat(chunks, ignore_index=True)
        checkpoint_df.to_csv(output_path, index=False)
        print(f"Checkpoint written to: {output_path}")

updated = pd.concat(chunks, ignore_index=True)
updated.to_csv(output_path, index=False)

area_changed = int((updated["area"].fillna("") != before_area.fillna("")).sum())
receiver_changed = int((updated["receiver"].fillna("") != before_receiver.fillna("")).sum())

print(f"Input rows: {len(reports)}")
print(f"Updated rows: {len(updated)}")
print(f"Rows with updated area: {area_changed}")
print(f"Rows with updated receiver: {receiver_changed}")
print(f"Wrote refreshed CSV to: {output_path}")

if DEBUG_AREA_OTHERS and area_debug_rows:
    debug_df = pd.concat(area_debug_rows, ignore_index=True)
    debug_path = output_path.with_name(f"{output_path.stem}_area_other_debug.csv")
    debug_df.to_csv(debug_path, index=False)
    print(f"Wrote area='Other' debug CSV to: {debug_path}")
    print(debug_df.head(DEBUG_AREA_OTHERS_LIMIT).to_string(index=False))

if DEBUG_RECEIVER_CHANGES and receiver_debug_rows:
    receiver_debug_df = pd.concat(receiver_debug_rows, ignore_index=True)
    receiver_debug_path = output_path.with_name(
        f"{output_path.stem}_receiver_change_debug.csv"
    )
    receiver_debug_df.to_csv(receiver_debug_path, index=False)
    print(f"Wrote receiver change debug CSV to: {receiver_debug_path}")
    print(receiver_debug_df.head(DEBUG_RECEIVER_CHANGES_LIMIT).to_string(index=False))
