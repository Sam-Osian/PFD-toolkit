"""Compare Screener performance across multiple LLM models against ONS consensus labels.

This script reads report sections and ground-truth labels directly from
``ons_replication/ONS_master_spreadsheet.xlsx`` and measures the accuracy,
sensitivity, and specificity for several LLM models. Results are written to
``model_comparison.csv``.
"""

from __future__ import annotations

import os
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

# Settings for OpenRouter models
OPENROUTER_KEY = os.getenv("OPENROUTER_API_KEY")
OPENROUTER_URL = "https://openrouter.ai/api/v1"

# List of models and the temperature values to use for each.
# GPT-5 models do not support a custom temperature parameter, so while a
# default value is recorded here for completeness, it is not passed to the
# client when initialising those models.
models = [
    # OPENAI API MODELS
    {"name": "gpt-4.1", "temperature": 0},
    {"name": "gpt-4o", "temperature": 0},
    {"name": "gpt-4.1-mini", "temperature": 0},
    {"name": "gpt-4.1-nano", "temperature": 0},
    {"name": "gpt-5", "temperature": 1},
    {"name": "gpt-5-mini", "temperature": 1},
    {"name": "gpt-5-nano", "temperature": 1},
    {"name": "o4-mini", "temperature": 1},
    {"name": "o3", "temperature": 1},

    # OPENROUTER MISTRAL MODELS
    {
        "name": "mistralai/devstral-small",
        "temperature": 0,
        "base_url": OPENROUTER_URL,
        "api_key": OPENROUTER_KEY,
    },
    {
        "name": "mistralai/devstral-medium",
        "temperature": 0,
        "base_url": OPENROUTER_URL,
        "api_key": OPENROUTER_KEY,
    },
    {
        "name": "mistralai/mistral-medium-3.1",
        "temperature": 0,
        "base_url": OPENROUTER_URL,
        "api_key": OPENROUTER_KEY,
    },
    {
        "name": "mistralai/mistral-large-2411",
        "temperature": 0,
        "base_url": OPENROUTER_URL,
        "api_key": OPENROUTER_KEY,
    },
    {
        "name": "mistralai/codestral-2508",
        "temperature": 0,
        "base_url": OPENROUTER_URL,
        "api_key": OPENROUTER_KEY,
    },

    # OPENROUTER OTHER MODELS
    {
        "name": "google/gemma-3-4b-it",
        "temperature": 0,
        "base_url": OPENROUTER_URL,
        "api_key": OPENROUTER_KEY,
    },
    {
        "name": "google/gemini-2.5-flash",
        "temperature": 0,
        "base_url": OPENROUTER_URL,
        "api_key": OPENROUTER_KEY,
    },
    {
        "name": "google/gemini-2.0-flash-001",
        "temperature": 0,
        "base_url": OPENROUTER_URL,
        "api_key": OPENROUTER_KEY,
    },
    {
        "name": "deepseek/deepseek-chat-v3-0324",
        "temperature": 0,
        "base_url": OPENROUTER_URL,
        "api_key": OPENROUTER_KEY,
    },
    {
        "name": "moonshotai/kimi-k2",
        "temperature": 0,
        "base_url": OPENROUTER_URL,
        "api_key": OPENROUTER_KEY,
    },
    {
        "name": "qwen/qwen3-235b-a22b-2507",
        "temperature": 0,
        "base_url": OPENROUTER_URL,
        "api_key": OPENROUTER_KEY,
    },
    {
        "name": "meta-llama/llama-4-maverick",
        "temperature": 0,
        "base_url": OPENROUTER_URL,
        "api_key": OPENROUTER_KEY,
    },
    {
        "name": "x-ai/grok-3-mini",
        "temperature": 0,
        "base_url": OPENROUTER_URL,
        "api_key": OPENROUTER_KEY,
    },
    {
        "name": "x-ai/grok-4",
        "temperature": 0,
        "base_url": OPENROUTER_URL,
        "api_key": OPENROUTER_KEY,
    },
    {
        "name": "x-ai/grok-3",
        "temperature": 0,
        "base_url": OPENROUTER_URL,
        "api_key": OPENROUTER_KEY,
    },
    {
        "name": "google/gemini-2.5-pro-preview",
        "temperature": 0,
        "base_url": OPENROUTER_URL,
        "api_key": OPENROUTER_KEY,
    },
    {
        "name": "cohere/command-a",
        "temperature": 0,
        "base_url": OPENROUTER_URL,
        "api_key": OPENROUTER_KEY,
    },



    # Mistral models
    {
        "name": "mistral-nemo:12b",
        "temperature": 0,
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "timeout": 10**9,
     },
    {
        "name": "mistral-small:22b",
        "temperature": 0,
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "timeout": 10**9,
     },
    {
        "name": "mistral-small:24b",
        "temperature": 0,
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "timeout": 10**9,
     },

    # Local models from other providers
    {
        "name": "gemma3:12b",
        "temperature": 0,
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "timeout": 10**9,
     },
    {
        "name": "gemma3:27b",
        "temperature": 0,
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "timeout": 10**9,
     },
    {
        "name": "gemma2:27b",
        "temperature": 0,
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "timeout": 10**9,
     },
    {
        "name": "qwen3:32b",
        "temperature": 0,
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "timeout": 10**9,
     },
    {
        "name": "qwen3:30b",
        "temperature": 0,
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "timeout": 10**9,
     },
    {
        "name": "qwen2.5:72b",
        "temperature": 0,
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "timeout": 10**9,
     },
    {
        "name": "qwen2.5:32b",
        "temperature": 0,
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "timeout": 10**9,
     },
    {
        "name": "llava:34b",
        "temperature": 0,
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "timeout": 10**9,
     },
    {
        "name": "phi4:14b",
        "temperature": 0,
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "timeout": 10**9,
     },
    {
        "name": "llama3:70b",
        "temperature": 0,
        "base_url": "http://localhost:11434/v1",
        "api_key": "ollama",
        "timeout": 10**9,
     },
]

user_query = """
Where the deceased is 18 or younger *at the time of death* AND the death was due to suicide.
Age may not be explicitly noted, but could be implied through recent use of child services (e.g. CAMHS),
mention of being "in Year 10", etc.
"""

# ---------------------------------------------------------------------
# 3. Prepare output file and determine models to run
# ---------------------------------------------------------------------
out_path = Path(__file__).resolve().parent / "model_comparison.csv"

# Determine which models have already been tested
if out_path.exists():
    results_df = pd.read_csv(out_path)
else:
    results_df = pd.DataFrame(columns=["model", "accuracy", "sensitivity", "specificity"])

existing_models: set[str] = set(results_df["model"].astype(str))

# Filter models to only those not yet evaluated
models_to_run = [spec for spec in models if spec["name"] not in existing_models]
if not models_to_run:
    print("All models already tested. Nothing to do.")
    raise SystemExit

for spec in models_to_run:
    model = spec["name"]
    temp = spec["temperature"]
    print(f"Testing model: {model}")

    llm_kwargs = {
        "api_key": spec.get("api_key", os.getenv("OPENAI_API_KEY")),
        "max_workers": 8,
        "model": model,
        "seed": 12345,
        "timeout": spec.get("timeout", 20),
        "temperature": 1 if model.startswith("gpt-5") else temp,
    }

    if "base_url" in spec:
        llm_kwargs["base_url"] = spec["base_url"]

    llm_client = LLM(**llm_kwargs)

    screener = Screener(
        llm=llm_client,
        reports=reports,
        include_investigation=True,
        include_circumstances=True,
        include_concerns=True,
    )

    classified = screener.screen_reports(
        search_query=user_query,
        filter_df=False,
        result_col_name="model_pred",
    )

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

    results_df = pd.concat(
        [
            results_df,
            pd.DataFrame(
                [
                    {
                        "model": model,
                        "accuracy": accuracy,
                        "sensitivity": sensitivity,
                        "specificity": specificity,
                    }
                ]
            ),
        ],
        ignore_index=True,
    )

    results_df.to_csv(out_path, index=False)
    with out_path.open("r+") as fh:
        os.fsync(fh.fileno())
