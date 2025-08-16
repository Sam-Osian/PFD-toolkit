"""Compare Screener performance across multiple LLM models against ONS consensus labels.

This script reads report sections and ground-truth labels directly from
``ons_replication/ONS_master_spreadsheet.xlsx`` and measures the accuracy,
sensitivity, specificity, and elapsed time for several LLM models. Results are
written to ``model_comparison.txt``.
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
# 2. Initialise models and query
# ---------------------------------------------------------------------
load_dotenv(Path(__file__).resolve().parent.parent / "api.env")

# List of models and the temperature values to use for each.
# GPT-5 models do not support a custom temperature parameter, so while a
# default value is recorded here for completeness, it is not passed to the
# client when initialising those models.
models = [
    {"name": "gpt-4.1", "temperature": 0},
    {"name": "gpt-4.1-mini", "temperature": 0},
    {"name": "gpt-4.1-nano", "temperature": 0},
    {"name": "gpt-5", "temperature": 1},
    {"name": "gpt-5-mini", "temperature": 1},
    {"name": "gpt-5-nano", "temperature": 1},
]

user_query = """
Where the deceased is 18 or younger *at the time of death* AND the death was due to suicide.
Age may not be explicitly noted, but could be implied through recent use of child services (e.g. CAMHS),
mention of being "in Year 10", etc.
"""

# ---------------------------------------------------------------------
# 3. Run screener for each model and compute metrics
# ---------------------------------------------------------------------
out_path = Path(__file__).resolve().parent / "model_comparison.txt"

# Determine which models have already been tested
existing_models: set[str] = set()
if out_path.exists():
    with out_path.open("r", encoding="utf-8") as fh:
        for line in fh:
            if line.startswith("Model:"):
                existing_models.add(line.split(":", 1)[1].strip())

# Filter models to only those not yet evaluated
models_to_run = [spec for spec in models if spec["name"] not in existing_models]
if not models_to_run:
    print("All models already tested. Nothing to do.")
    raise SystemExit

with out_path.open("a", encoding="utf-8") as fh:
    for spec in models_to_run:
        model = spec["name"]
        temp = spec["temperature"]
        print(f"Testing model: {model}")

        llm_kwargs = {
            "api_key": os.getenv("OPENAI_API_KEY"),
            "max_workers": 10,
            "model": model,
            "seed": 12345,
            "timeout": 20,
            "temperature": 1 if model.startswith("gpt-5") else temp,
        }

        llm_client = LLM(**llm_kwargs)

        screener = Screener(
            llm=llm_client,
            reports=reports,
            include_investigation=True,
            include_circumstances=True,
            include_concerns=True,
        )

        start = time.perf_counter()
        classified = screener.screen_reports(
            search_query=user_query,
            filter_df=False,
            result_col_name="model_pred",
        )
        elapsed = time.perf_counter() - start

        pred = classified["model_pred"].astype(bool)
        truth = classified["consensus"].astype(bool)

        tp = (pred & truth).sum()
        tn = ((~pred) & (~truth)).sum()
        fp = (pred & ~truth).sum()
        fn = ((~pred) & truth).sum()

        total = tp + tn + fp + fn
        accuracy = (tp + tn) / total if total else float("nan")
        sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
        specificity = tn / (tn + fp) if (tn + fp) else float("nan")

        fh.write(f"Model: {model}\n")
        fh.write(f"Temperature: {temp}\n")
        fh.write(f"Accuracy:    {accuracy:.3f}\n")
        fh.write(f"Sensitivity: {sensitivity:.3f}\n")
        fh.write(f"Specificity: {specificity:.3f}\n")
        fh.write(f"Elapsed (s): {elapsed:.2f}\n")
        fh.write("\n")

