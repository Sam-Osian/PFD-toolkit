"""Evaluate Screener performance against ONS consensus labels.

This script reads report sections and ground-truth labels directly from
``ons_replication/ONS_master_spreadsheet.xlsx`` and measures the accuracy,
sensitivity, specificity, and elapsed time for a single LLM model. Results are
written to ``screener_performance.txt``.
"""

from __future__ import annotations

import os
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv

from pfd_toolkit import LLM, Screener
from pfd_toolkit.config import GeneralConfig

# ---------------------------------------------------------------------
# 1. Load ONS ground-truth data
# ---------------------------------------------------------------------
ons_path = (
    Path(__file__).resolve().parent.parent
    / "ons_replication"
    / "ONS_master_spreadsheet.xlsx"
)
ons_df = pd.read_excel(ons_path, sheet_name=0)

ons_df = ons_df.rename(
    columns={
        "Investigation section": GeneralConfig.COL_INVESTIGATION,
        "Circumstances of death section": GeneralConfig.COL_CIRCUMSTANCES,
        "Matters of concern section": GeneralConfig.COL_CONCERNS,
        "Consensus": "consensus",
        "Ref": GeneralConfig.COL_ID,
    }
)

keep_cols = [
    GeneralConfig.COL_ID,
    GeneralConfig.COL_INVESTIGATION,
    GeneralConfig.COL_CIRCUMSTANCES,
    GeneralConfig.COL_CONCERNS,
    "consensus",
]
reports = ons_df[keep_cols].copy()

reports["consensus"] = (
    reports["consensus"]
    .astype(str)
    .str.strip()
    .str.lower()
    .map({"yes": True, "no": False, "1": True, "0": False})
)

print(f"Loaded {len(reports)} reports from ONS spreadsheet.")

# ---------------------------------------------------------------------
# 2. Initialise LLM and Screener
# ---------------------------------------------------------------------
load_dotenv(Path(__file__).resolve().parent.parent / "api.env")
llm_client = LLM(
    api_key=os.getenv("OPENAI_API_KEY"),
    max_workers=17,
    model="gpt-4.1",
    seed=12345,
    temperature=0,
    timeout=20,
)

user_query = """
Where the deceased is 18 or younger *at the time of death* AND the death was due to suicide.
Age may not be explicitly noted, but could be implied through recent use of child services (e.g. CAMHS),
mention of being "in Year 10", etc.
"""

screener = Screener(
    llm=llm_client,
    reports=reports,
    include_investigation=True,
    include_circumstances=True,
    include_concerns=True,
)

# ---------------------------------------------------------------------
# 3. Run screener and compute metrics
# ---------------------------------------------------------------------
start = time.perf_counter()
classified = screener.screen_reports(
    search_query=user_query,
    filter_df=False,
    result_col_name="model_pred",
)
elapsed = time.perf_counter() - start

pred = classified["model_pred"].astype(bool)
truth = classified["consensus"].astype(bool)

tp = ((pred == True) & (truth == True)).sum()
tn = ((pred == False) & (truth == False)).sum()
fp = ((pred == True) & (truth == False)).sum()
fn = ((pred == False) & (truth == True)).sum()

total = tp + tn + fp + fn
accuracy = (tp + tn) / total if total else float("nan")
sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
specificity = tn / (tn + fp) if (tn + fp) else float("nan")

# ---------------------------------------------------------------------
# 4. Save results
# ---------------------------------------------------------------------
out_path = Path(__file__).resolve().parent / "screener_performance.txt"
with out_path.open("w", encoding="utf-8") as fh:
    fh.write(f"Model: {llm_client.model}\n")
    fh.write(f"Accuracy:    {accuracy:.3f}\n")
    fh.write(f"Sensitivity: {sensitivity:.3f}\n")
    fh.write(f"Specificity: {specificity:.3f}\n")
    fh.write(f"Elapsed (s): {elapsed:.2f}\n")
