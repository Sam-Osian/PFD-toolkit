"""Compare screener performance across local and proprietary LLMs."""

from pathlib import Path
import os

import pandas as pd
from dotenv import load_dotenv

from pfd_toolkit import LLM, Screener
from pfd_toolkit.config import GeneralConfig

DATA_PATH = Path("../../ons_replication/PFD Toolkit--Consensus Comparison.xlsx")
RESULTS_PATH = Path("model_comparison.csv")
SHEET_NAME = "Consensus annotations"

MODEL_SPECS = [
    # OpenAI API models
    {"name": "gpt-4.1", "temperature": 0},
    {"name": "gpt-4o", "temperature": 0},
    {"name": "gpt-4.1-mini", "temperature": 0},
    {"name": "gpt-4.1-nano", "temperature": 0},
    {"name": "gpt-5", "temperature": 1},
    {"name": "gpt-5-mini", "temperature": 1},
    {"name": "gpt-5-nano", "temperature": 1},
    {"name": "o4-mini", "temperature": 1},
    {"name": "o3", "temperature": 1},

    # Ollama-hosted models
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


def load_reports() -> pd.DataFrame:
    df = pd.read_excel(DATA_PATH, sheet_name=SHEET_NAME)
    renamed = df.rename(
        columns={
            "Ref": GeneralConfig.COL_ID,
            "Investigation section": GeneralConfig.COL_INVESTIGATION,
            "Circumstances of death section": GeneralConfig.COL_CIRCUMSTANCES,
            "Matters of concern section": GeneralConfig.COL_CONCERNS,
            "Post-consensus verdict: Is this a child suicide case? (Yes or No)": "consensus",
        }
    )

    reports = renamed[
        [
            GeneralConfig.COL_ID,
            GeneralConfig.COL_INVESTIGATION,
            GeneralConfig.COL_CIRCUMSTANCES,
            GeneralConfig.COL_CONCERNS,
            "consensus",
        ]
    ].copy()

    reports["consensus"] = (
        reports["consensus"].astype(str).str.strip().str.lower() == "yes"
    )
    return reports


def evaluate_model(spec: dict[str, object], reports: pd.DataFrame) -> dict[str, float]:
    model_name = spec["name"]
    llm_kwargs = {
        "api_key": spec.get("api_key", os.getenv("OPENAI_API_KEY")),
        "max_workers": 8,
        "model": model_name,
        "seed": 12345,
        "timeout": spec.get("timeout", 20),
        "temperature": 1 if model_name.startswith("gpt-5") else spec["temperature"],
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

    return {
        "model": model_name,
        "accuracy": accuracy,
        "sensitivity": sensitivity,
        "specificity": specificity,
    }


def run_comparisons():
    load_dotenv("../../api.env")
    reports = load_reports()

    if RESULTS_PATH.exists():
        results_df = pd.read_csv(RESULTS_PATH)
    else:
        results_df = pd.DataFrame(
            columns=["model", "accuracy", "sensitivity", "specificity"]
        )

    completed_models = set(results_df["model"].astype(str))
    models_to_run = [spec for spec in MODEL_SPECS if spec["name"] not in completed_models]

    if not models_to_run:
        print("All models already tested.")
        return results_df

    for spec in models_to_run:
        print(f"Testing model: {spec['name']}")
        results = evaluate_model(spec, reports)
        results_df = pd.concat([results_df, pd.DataFrame([results])], ignore_index=True)
        results_df.to_csv(RESULTS_PATH, index=False)

    return results_df
