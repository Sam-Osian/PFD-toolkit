#!/usr/bin/env python3
"""
Merge analysis-stage failures from results.csv into exclusion reason counts.

This script is intentionally separate from the main runner so it can be used
without interrupting or modifying an active run.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import pandas as pd


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--results",
        default="open_llm_ons_experiment/artifacts/results.csv",
    )
    p.add_argument(
        "--counts",
        default="open_llm_ons_experiment/artifacts/discovery/exclusion_reason_counts_latest.csv",
    )
    return p.parse_args()


def normalise_failure_reason(error_reason: str) -> str:
    txt = (error_reason or "").strip().lower()
    if not txt:
        return "analysis_failed_unknown"
    if "timeout" in txt:
        return "inference_timeout"
    if "pydantic" in txt or "validation" in txt or "model_pred_contains_na" in txt:
        return "pydantic_or_schema_failure"
    if "pull" in txt or "api/pull" in txt:
        return "ollama_pull_failed"
    if "connection" in txt or "refused" in txt or "localhost:11434" in txt:
        return "ollama_connection_failed"
    if "cancel" in txt or "interrupt" in txt:
        return "interrupted"
    return "analysis_runtime_failed"


def build_analysis_counts(results_df: pd.DataFrame) -> pd.DataFrame:
    if results_df.empty or "status" not in results_df.columns:
        return pd.DataFrame(columns=["reason_code", "stage", "excluded_tag_count"])

    failures = results_df[results_df["status"].astype(str).str.lower() == "failed"].copy()
    if failures.empty:
        return pd.DataFrame(columns=["reason_code", "stage", "excluded_tag_count"])

    failures["reason_code"] = failures.get("error_reason", pd.Series(dtype=str)).fillna("").apply(normalise_failure_reason)
    counts = (
        failures.groupby("reason_code", dropna=False)
        .size()
        .reset_index(name="excluded_tag_count")
    )
    counts["stage"] = "analysis"
    return counts[["reason_code", "stage", "excluded_tag_count"]]


def main() -> None:
    args = parse_args()
    results_path = Path(args.results)
    counts_path = Path(args.counts)

    if not results_path.exists():
        raise FileNotFoundError(f"results.csv not found: {results_path}")
    if not counts_path.exists():
        raise FileNotFoundError(f"counts CSV not found: {counts_path}")

    results = pd.read_csv(results_path)
    counts = pd.read_csv(counts_path)

    for col in ("reason_code", "stage", "excluded_tag_count"):
        if col not in counts.columns:
            counts[col] = pd.NA
    counts = counts[["reason_code", "stage", "excluded_tag_count"]].copy()

    # Replace any existing analysis rows with a fresh recompute from results.csv.
    counts = counts[counts["stage"].astype(str) != "analysis"].copy()
    analysis_counts = build_analysis_counts(results)
    merged = pd.concat([counts, analysis_counts], ignore_index=True)
    merged = merged.groupby(["reason_code", "stage"], as_index=False)["excluded_tag_count"].sum()
    merged = merged.sort_values(
        by=["stage", "excluded_tag_count", "reason_code"],
        ascending=[True, False, True],
    ).reset_index(drop=True)

    merged.to_csv(counts_path, index=False)
    print(f"Analysis failures found: {len(results[results['status'].astype(str).str.lower() == 'failed'])}")
    print(f"Wrote: {counts_path}")


if __name__ == "__main__":
    main()

