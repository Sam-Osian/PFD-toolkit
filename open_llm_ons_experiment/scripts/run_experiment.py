#!/usr/bin/env python3
"""
Run ONS agreement evaluation across eligible Ollama models.

This script:
1) Loads eligible tags from discovery outputs (or frozen model_manifest.csv)
2) Pulls model if needed
3) Runs a Pydantic preflight gate
4) Runs full ONS agreement screening
5) Appends incremental results and exclusions
6) Regenerates plots after each completed model
7) Deletes models pulled by this run (never preexisting models)
8) Supports interruption-safe resume via run_state.json
"""

from __future__ import annotations

import argparse
import json
import os
from datetime import UTC, datetime
from pathlib import Path
from typing import Any

import pandas as pd
import requests
from tqdm import tqdm

from pfd_toolkit import LLM, Screener
from pfd_toolkit.config import GeneralConfig


DEFAULT_QUERY = """
Where the deceased is 18 or younger *at the time of death* AND the death was due to suicide.
Age may not be explicitly noted, but could be implied through recent use of child services (e.g. CAMHS),
mention of being "in Year 10", etc.
""".strip()


RESULT_COLUMNS = [
    "model",
    "tag",
    "family",
    "is_moe",
    "params_total_b",
    "params_active_b",
    "agreement_with_clinical_adjudication",
    "sensitivity",
    "specificity",
    "local",
    "installed_preexisting",
    "pulled_by_run",
    "started_at",
    "finished_at",
    "status",
    "error_reason",
]

EXCLUSION_COLUMNS = [
    "model",
    "tag",
    "family",
    "reason_code",
    "reason_detail",
    "occurred_at",
]


def utc_now_iso() -> str:
    return datetime.now(UTC).isoformat()


def fsync_csv(path: Path, df: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False)
    with path.open("r+") as fh:
        os.fsync(fh.fileno())


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--eligible-path",
        default="open_llm_ons_experiment/artifacts/discovery/eligible_tags_latest.csv",
    )
    parser.add_argument(
        "--manifest-path",
        default="open_llm_ons_experiment/artifacts/model_manifest.csv",
    )
    parser.add_argument(
        "--results-path",
        default="open_llm_ons_experiment/artifacts/results.csv",
    )
    parser.add_argument(
        "--exclusions-path",
        default="open_llm_ons_experiment/artifacts/exclusions.csv",
    )
    parser.add_argument(
        "--state-path",
        default="open_llm_ons_experiment/artifacts/run_state.json",
    )
    parser.add_argument(
        "--plots-dir",
        default="open_llm_ons_experiment/artifacts/plots",
    )
    parser.add_argument(
        "--ons-spreadsheet",
        default="open_llm_ons_experiment/ons_replication/PFD Toolkit--Consensus Comparison.xlsx",
    )
    parser.add_argument(
        "--ollama-base-url",
        default="http://localhost:11434",
    )
    parser.add_argument(
        "--llm-base-url",
        default="http://localhost:11434/v1",
    )
    parser.add_argument("--query", default=DEFAULT_QUERY)
    parser.add_argument("--seed", type=int, default=12345)
    parser.add_argument("--temperature", type=float, default=0.0)
    parser.add_argument("--timeout", type=float, default=600)
    parser.add_argument("--max-workers", type=int, default=1)
    parser.add_argument("--validation-attempts", type=int, default=3)
    parser.add_argument("--preflight-reports", type=int, default=5)
    parser.add_argument("--gpt41-benchmark", type=float, default=0.97)
    parser.add_argument("--delete-pulled", action="store_true", default=True)
    parser.add_argument("--retry-failed", action="store_true", default=False)
    return parser.parse_args()


def load_reports(spreadsheet_path: Path) -> pd.DataFrame:
    if not spreadsheet_path.exists():
        raise FileNotFoundError(f"Spreadsheet not found: {spreadsheet_path}")

    df = pd.read_excel(spreadsheet_path, sheet_name=0)
    df = df.rename(
        columns={
            "Investigation section": GeneralConfig.COL_INVESTIGATION,
            "Circumstances of death section": GeneralConfig.COL_CIRCUMSTANCES,
            "Matters of concern section": GeneralConfig.COL_CONCERNS,
            "Consensus": "consensus",
            "Post-consensus verdict: Is this a child suicide case? (Yes or No)": "consensus",
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
    missing = [c for c in keep_cols if c not in df.columns]
    if missing:
        raise ValueError(f"Spreadsheet missing required columns: {', '.join(missing)}")

    reports = df[keep_cols].copy()
    reports["consensus"] = (
        reports["consensus"]
        .astype(str)
        .str.strip()
        .str.lower()
        .map({"yes": True, "no": False, "1": True, "0": False})
    )
    reports = reports.dropna(subset=["consensus"]).copy()
    reports["consensus"] = reports["consensus"].astype(bool)
    return reports


def load_or_init_manifest(eligible_path: Path, manifest_path: Path) -> pd.DataFrame:
    if manifest_path.exists():
        return pd.read_csv(manifest_path)

    eligible = pd.read_csv(eligible_path)
    if "eligible" in eligible.columns:
        eligible = eligible[eligible["eligible"] == True].copy()  # noqa: E712

    cols = [
        "family",
        "tag",
        "is_moe",
        "params_total_b",
        "params_active_b",
        "updated_date_approx",
        "quantisation",
    ]
    available = [c for c in cols if c in eligible.columns]
    manifest = eligible[available].drop_duplicates(subset=["tag"]).reset_index(drop=True)
    manifest["model"] = manifest["tag"]
    manifest["manifest_created_at"] = utc_now_iso()
    fsync_csv(manifest_path, manifest)
    return manifest


def load_csv(path: Path, columns: list[str]) -> pd.DataFrame:
    if path.exists():
        df = pd.read_csv(path)
        for col in columns:
            if col not in df.columns:
                df[col] = pd.NA
        return df[columns].copy()
    return pd.DataFrame(columns=columns)


def ollama_tags(base_url: str, timeout: float) -> set[str]:
    url = base_url.rstrip("/") + "/api/tags"
    resp = requests.get(url, timeout=timeout)
    resp.raise_for_status()
    payload = resp.json()
    names: set[str] = set()
    for model in payload.get("models", []):
        name = str(model.get("name", "")).strip()
        if name:
            names.add(name)
    return names


def ollama_pull(base_url: str, tag: str, timeout: float) -> None:
    url = base_url.rstrip("/") + "/api/pull"
    resp = requests.post(url, json={"name": tag, "stream": False}, timeout=timeout)
    resp.raise_for_status()


def ollama_delete(base_url: str, tag: str, timeout: float) -> None:
    url = base_url.rstrip("/") + "/api/delete"
    resp = requests.delete(url, json={"name": tag}, timeout=timeout)
    if resp.status_code not in {200, 204, 404}:
        resp.raise_for_status()


def load_state(path: Path, manifest: pd.DataFrame, preexisting_models: set[str]) -> dict[str, Any]:
    if path.exists():
        try:
            raw = path.read_text(encoding="utf-8").strip()
            state = json.loads(raw) if raw else {}
        except Exception:
            state = {}
    else:
        state = {}

    if not state:
        state = {
            "models": {},
            "preexisting_models": sorted(preexisting_models),
            "pulled_by_run": [],
            "current_model": None,
            "phase": "idle",
            "updated_at": utc_now_iso(),
        }

    state.setdefault("models", {})
    state.setdefault("preexisting_models", sorted(preexisting_models))
    state.setdefault("pulled_by_run", [])
    state.setdefault("current_model", None)
    state.setdefault("phase", "idle")
    state.setdefault("updated_at", utc_now_iso())

    for tag in manifest["tag"].astype(str):
        state["models"].setdefault(tag, "pending")
    return state


def save_state(path: Path, state: dict[str, Any]) -> None:
    state["updated_at"] = utc_now_iso()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(state, indent=2, sort_keys=True), encoding="utf-8")


def append_exclusion(
    exclusions: pd.DataFrame,
    exclusions_path: Path,
    *,
    model: str,
    tag: str,
    family: str,
    reason_code: str,
    reason_detail: str,
) -> pd.DataFrame:
    row = pd.DataFrame(
        [
            {
                "model": model,
                "tag": tag,
                "family": family,
                "reason_code": reason_code,
                "reason_detail": reason_detail,
                "occurred_at": utc_now_iso(),
            }
        ]
    )
    exclusions = pd.concat([exclusions, row], ignore_index=True)
    fsync_csv(exclusions_path, exclusions)
    return exclusions


def upsert_result(results: pd.DataFrame, results_path: Path, row: dict[str, Any]) -> pd.DataFrame:
    if "tag" in results.columns:
        results = results[results["tag"].astype(str) != str(row["tag"])].copy()
    results = pd.concat([results, pd.DataFrame([row])], ignore_index=True)
    fsync_csv(results_path, results)
    return results


def generate_plots(results: pd.DataFrame, plots_dir: Path, gpt41_benchmark: float) -> None:
    completed = results[results["status"] == "completed"].copy()
    if completed.empty:
        return

    try:
        import plotly.express as px
        import plotly.graph_objects as go
    except Exception:
        return

    plots_dir.mkdir(parents=True, exist_ok=True)
    completed["params_total_b"] = pd.to_numeric(completed["params_total_b"], errors="coerce")
    completed["agreement_with_clinical_adjudication"] = pd.to_numeric(
        completed["agreement_with_clinical_adjudication"], errors="coerce"
    )
    completed = completed.dropna(
        subset=["params_total_b", "agreement_with_clinical_adjudication"]
    ).copy()
    if completed.empty:
        return

    fig = px.scatter(
        completed,
        x="params_total_b",
        y="agreement_with_clinical_adjudication",
        hover_data=[
            "model",
            "tag",
            "family",
            "sensitivity",
            "specificity",
            "is_moe",
            "installed_preexisting",
            "pulled_by_run",
        ],
        title="Open LLM ONS Agreement Benchmark",
    )
    fig.add_hline(
        y=gpt41_benchmark,
        line_dash="dot",
        annotation_text=f"GPT-4.1 benchmark ({gpt41_benchmark:.2%})",
    )
    fig.update_layout(
        xaxis_title="Parameters (billions)",
        yaxis_title="Agreement with clinical adjudication",
    )
    fig.write_html(str(plots_dir / "results_interactive.html"), include_plotlyjs="cdn")

    try:
        fig.write_image(str(plots_dir / "results_static.png"))
    except Exception:
        # Static image export requires optional Kaleido; HTML is still generated.
        pass


def run_preflight(
    llm: LLM,
    reports: pd.DataFrame,
    query: str,
    preflight_reports: int,
) -> tuple[bool, str]:
    sample = reports.head(preflight_reports).copy()
    screener = Screener(
        llm=llm,
        reports=sample,
        include_investigation=True,
        include_circumstances=True,
        include_concerns=True,
    )
    classified = screener.screen_reports(
        search_query=query,
        filter_df=False,
        result_col_name="preflight_pred",
    )
    if "preflight_pred" not in classified.columns:
        return False, "preflight_pred_missing"
    if classified["preflight_pred"].isna().any():
        return False, "preflight_pred_contains_na"
    return True, ""


def run_full_eval(llm: LLM, reports: pd.DataFrame, query: str) -> tuple[float, float, float]:
    screener = Screener(
        llm=llm,
        reports=reports,
        include_investigation=True,
        include_circumstances=True,
        include_concerns=True,
    )
    classified = screener.screen_reports(
        search_query=query,
        filter_df=False,
        result_col_name="model_pred",
    )
    if "model_pred" not in classified.columns:
        raise RuntimeError("model_pred_missing")
    if classified["model_pred"].isna().any():
        raise RuntimeError("model_pred_contains_na")

    pred = classified["model_pred"].astype(bool)
    truth = classified["consensus"].astype(bool)

    tp = int((pred & truth).sum())
    tn = int(((~pred) & (~truth)).sum())
    fp = int((pred & ~truth).sum())
    fn = int(((~pred) & truth).sum())

    total = tp + tn + fp + fn
    agreement = (tp + tn) / total if total else float("nan")
    sensitivity = tp / (tp + fn) if (tp + fn) else float("nan")
    specificity = tn / (tn + fp) if (tn + fp) else float("nan")
    return agreement, sensitivity, specificity


def main() -> None:
    args = parse_args()
    eligible_path = Path(args.eligible_path)
    manifest_path = Path(args.manifest_path)
    results_path = Path(args.results_path)
    exclusions_path = Path(args.exclusions_path)
    state_path = Path(args.state_path)
    plots_dir = Path(args.plots_dir)

    reports = load_reports(Path(args.ons_spreadsheet))
    manifest = load_or_init_manifest(eligible_path, manifest_path)
    results = load_csv(results_path, RESULT_COLUMNS)
    exclusions = load_csv(exclusions_path, EXCLUSION_COLUMNS)

    preexisting_models = ollama_tags(args.ollama_base_url, timeout=args.timeout)
    state = load_state(state_path, manifest, preexisting_models)

    completed_tags = set(
        results.loc[results["status"] == "completed", "tag"].astype(str).tolist()
    )
    excluded_tags = set(exclusions["tag"].astype(str).tolist())
    for tag, status in list(state["models"].items()):
        if tag in completed_tags:
            state["models"][tag] = "completed"
        elif tag in excluded_tags:
            state["models"][tag] = "excluded"
        elif status == "in_progress":
            state["models"][tag] = "pending"
    save_state(state_path, state)

    pulled_by_run = set(str(t) for t in state.get("pulled_by_run", []))
    preexisting_set = set(str(t) for t in state.get("preexisting_models", []))

    for _, row in tqdm(
        manifest.iterrows(),
        total=len(manifest),
        desc="Models",
        unit="model",
    ):
        tag = str(row["tag"])
        model = tag
        family = str(row.get("family", ""))
        model_status = state["models"].get(tag, "pending")

        if model_status in {"completed", "excluded"}:
            continue
        if model_status == "failed" and not args.retry_failed:
            continue

        started_at = utc_now_iso()
        state["models"][tag] = "in_progress"
        state["current_model"] = tag
        state["phase"] = "install"
        save_state(state_path, state)

        installed_preexisting = tag in preexisting_set
        pulled_this_model = False
        error_reason = ""

        try:
            local_tags = ollama_tags(args.ollama_base_url, timeout=args.timeout)
            if tag not in local_tags:
                ollama_pull(args.ollama_base_url, tag, timeout=args.timeout)
                pulled_this_model = True
                pulled_by_run.add(tag)
                state["pulled_by_run"] = sorted(pulled_by_run)
                save_state(state_path, state)
            else:
                pulled_this_model = tag in pulled_by_run

            llm_client = LLM(
                api_key="ollama",
                model=tag,
                base_url=args.llm_base_url,
                max_workers=args.max_workers,
                temperature=args.temperature,
                seed=args.seed,
                timeout=args.timeout,
                validation_attempts=args.validation_attempts,
            )

            state["phase"] = "preflight"
            save_state(state_path, state)
            ok, detail = run_preflight(
                llm=llm_client,
                reports=reports,
                query=args.query,
                preflight_reports=args.preflight_reports,
            )
            if not ok:
                exclusions = append_exclusion(
                    exclusions,
                    exclusions_path,
                    model=model,
                    tag=tag,
                    family=family,
                    reason_code="pydantic_gate_failed",
                    reason_detail=detail,
                )
                state["models"][tag] = "excluded"
                state["phase"] = "cleanup"
                save_state(state_path, state)
                continue

            state["phase"] = "evaluate"
            save_state(state_path, state)
            agreement, sensitivity, specificity = run_full_eval(
                llm=llm_client,
                reports=reports,
                query=args.query,
            )
            finished_at = utc_now_iso()

            results = upsert_result(
                results,
                results_path,
                {
                    "model": model,
                    "tag": tag,
                    "family": family,
                    "is_moe": row.get("is_moe"),
                    "params_total_b": row.get("params_total_b"),
                    "params_active_b": row.get("params_active_b"),
                    "agreement_with_clinical_adjudication": agreement,
                    "sensitivity": sensitivity,
                    "specificity": specificity,
                    "local": True,
                    "installed_preexisting": installed_preexisting,
                    "pulled_by_run": pulled_this_model,
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "status": "completed",
                    "error_reason": "",
                },
            )
            state["models"][tag] = "completed"
            generate_plots(results, plots_dir, args.gpt41_benchmark)

        except Exception as exc:
            finished_at = utc_now_iso()
            error_reason = f"{type(exc).__name__}: {exc}"
            results = upsert_result(
                results,
                results_path,
                {
                    "model": model,
                    "tag": tag,
                    "family": family,
                    "is_moe": row.get("is_moe"),
                    "params_total_b": row.get("params_total_b"),
                    "params_active_b": row.get("params_active_b"),
                    "agreement_with_clinical_adjudication": pd.NA,
                    "sensitivity": pd.NA,
                    "specificity": pd.NA,
                    "local": True,
                    "installed_preexisting": installed_preexisting,
                    "pulled_by_run": pulled_this_model,
                    "started_at": started_at,
                    "finished_at": finished_at,
                    "status": "failed",
                    "error_reason": error_reason,
                },
            )
            state["models"][tag] = "failed"
        finally:
            state["phase"] = "cleanup"
            if args.delete_pulled and pulled_this_model and not installed_preexisting:
                try:
                    ollama_delete(args.ollama_base_url, tag, timeout=args.timeout)
                except Exception:
                    # Keep run moving; deletion issues are captured by local model state.
                    pass
            state["current_model"] = None
            state["phase"] = "idle"
            save_state(state_path, state)

    print("Run complete.")
    print(f"Manifest:   {manifest_path}")
    print(f"Results:    {results_path}")
    print(f"Exclusions: {exclusions_path}")
    print(f"State:      {state_path}")
    print(f"Plots dir:  {plots_dir}")


if __name__ == "__main__":
    main()
