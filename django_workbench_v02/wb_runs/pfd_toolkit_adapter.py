from __future__ import annotations

import json
import os
from datetime import date
from pathlib import Path
from typing import Callable

from django.conf import settings
from django.utils import timezone

import pandas as pd
from pydantic import BaseModel, Field, create_model
from pfd_toolkit import Extractor, LLM, Screener, load_reports


OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"
LLM_REQUEST_TIMEOUT_SECONDS = 1800


class AdapterConfigurationError(RuntimeError):
    pass


class AdapterCancelledError(RuntimeError):
    pass


def _as_bool(value, default: bool) -> bool:
    if value is None:
        return default
    if isinstance(value, bool):
        return value
    if isinstance(value, (int, float)):
        return bool(value)
    if isinstance(value, str):
        lowered = value.strip().lower()
        if lowered in {"1", "true", "yes", "on"}:
            return True
        if lowered in {"0", "false", "no", "off"}:
            return False
    return default


def _normalise_parallel_workers(raw_value) -> int:
    try:
        requested = int(raw_value or 8)
    except (TypeError, ValueError):
        requested = 8
    return min(32, max(1, requested))


def _build_llm_kwargs(config: dict) -> dict:
    provider = str(config.get("provider", "openai")).strip().lower()
    model_name = (config.get("model_name") or "gpt-4.1-mini").strip() or "gpt-4.1-mini"
    timeout = int(config.get("llm_timeout_seconds") or LLM_REQUEST_TIMEOUT_SECONDS)

    if provider == "openrouter":
        api_key = (os.getenv("OPENROUTER_API_KEY") or "").strip()
        if not api_key:
            raise AdapterConfigurationError("OPENROUTER_API_KEY is required for provider=openrouter.")
        base_url = (
            (config.get("openrouter_base_url") or "").strip()
            or (os.getenv("OPENROUTER_BASE_URL") or "").strip()
            or OPENROUTER_API_BASE
        )
    else:
        api_key = (os.getenv("OPENAI_API_KEY") or "").strip()
        if not api_key:
            raise AdapterConfigurationError("OPENAI_API_KEY is required for provider=openai.")
        base_url = (
            (config.get("openai_base_url") or "").strip()
            or (os.getenv("OPENAI_BASE_URL") or "").strip()
            or None
        )

    kwargs = {
        "api_key": api_key,
        "model": model_name,
        "max_workers": _normalise_parallel_workers(config.get("max_parallel_workers")),
        "temperature": 0.0,
        "validation_attempts": 2,
        "seed": 123,
        "timeout": timeout,
    }
    if base_url:
        kwargs["base_url"] = base_url
    return kwargs


def _resolve_filter_query(*, investigation, config: dict) -> str:
    query = str(config.get("search_query") or "").strip()
    if not query:
        query = (investigation.question_text or "").strip()
    if not query:
        raise AdapterConfigurationError(
            "No filter query provided. Set input_config_json.search_query or investigation.question_text."
        )
    return query


def _resolve_date_range(*, run, config: dict) -> tuple[date, date]:
    default_start = date(2013, 1, 1)
    default_end = timezone.now().date()

    start = run.query_start_date
    end = run.query_end_date

    start_override = config.get("start_date")
    end_override = config.get("end_date")
    if start is None and start_override:
        start = date.fromisoformat(str(start_override))
    if end is None and end_override:
        end = date.fromisoformat(str(end_override))

    start = start or default_start
    end = end or default_end
    if end < start:
        raise AdapterConfigurationError("End date must be on or after start date.")
    return start, end


def _resolve_output_path(*, run, config: dict) -> Path:
    override_dir = (config.get("artifact_dir") or "").strip()
    if override_dir:
        base_dir = Path(override_dir)
    else:
        base_dir = Path(settings.BASE_DIR) / "runtime_artifacts" / "runs"
    run_dir = base_dir / str(run.id)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir / "filtered_reports.csv"


def _resolve_output_dir(*, run, config: dict) -> Path:
    override_dir = (config.get("artifact_dir") or "").strip()
    if override_dir:
        base_dir = Path(override_dir)
    else:
        base_dir = Path(settings.BASE_DIR) / "runtime_artifacts" / "runs"
    run_dir = base_dir / str(run.id)
    run_dir.mkdir(parents=True, exist_ok=True)
    return run_dir


def _patch_generate_with_progress(
    *,
    llm_client,
    progress_start: int,
    progress_end: int,
    progress_callback: Callable[[int, str], None] | None,
    cancellation_check: Callable[[], bool] | None,
    default_message: str,
):
    original_generate = llm_client.generate

    def _generate_with_progress(*args, **kwargs):
        user_callback = kwargs.pop("progress_callback", None)

        def _callback(done: int, total: int, description: str):
            if cancellation_check and cancellation_check():
                raise AdapterCancelledError("Run cancelled during LLM generation.")
            ratio = float(done) / float(total or 1)
            worker_progress = min(
                progress_end,
                max(progress_start, int(progress_start + (ratio * (progress_end - progress_start)))),
            )
            if progress_callback:
                progress_callback(worker_progress, description or default_message)
            if callable(user_callback):
                user_callback(done, total, description)

        kwargs["progress_callback"] = _callback
        return original_generate(*args, **kwargs)

    llm_client.generate = _generate_with_progress
    return llm_client


def _build_extractor(*, llm_client, reports_df):
    return Extractor(
        llm=llm_client,
        reports=reports_df,
        include_date=True,
        include_coroner=True,
        include_area=True,
        include_receiver=True,
        include_investigation=True,
        include_circumstances=True,
        include_concerns=True,
        verbose=False,
    )


def _resolve_extract_feature_model(config: dict) -> type[BaseModel]:
    rows = config.get("feature_fields")
    if not isinstance(rows, list) or not rows:
        raise AdapterConfigurationError(
            "Extract runs require input_config_json.feature_fields with at least one feature."
        )

    type_mapping = {
        "text": str,
        "free text": str,
        "string": str,
        "bool": bool,
        "boolean": bool,
        "conditional (true/false)": bool,
        "int": int,
        "integer": int,
        "whole number": int,
        "float": float,
        "decimal": float,
        "decimal number": float,
    }

    fields = {}
    for row in rows:
        if not isinstance(row, dict):
            continue
        raw_name = str(row.get("name") or row.get("field_name") or "").strip()
        description = str(row.get("description") or "").strip()
        raw_type = str(row.get("type") or "text").strip().lower()
        if not raw_name:
            continue
        python_type = type_mapping.get(raw_type)
        if python_type is None:
            raise AdapterConfigurationError(f"Unsupported feature type '{raw_type}' for '{raw_name}'.")
        if raw_name in fields:
            raise AdapterConfigurationError(f"Duplicate extract feature name '{raw_name}'.")
        fields[raw_name] = (
            python_type,
            Field(
                ...,
                description=description or raw_name,
                title=raw_name.replace("_", " ").title(),
            ),
        )

    if not fields:
        raise AdapterConfigurationError("No valid feature_fields were provided for extract workflow.")
    return create_model("RunExtractFeatures", **fields)


def _theme_summary_from_dataframe(result_df: pd.DataFrame, theme_model: type[BaseModel]) -> pd.DataFrame:
    schema = theme_model.model_json_schema()
    properties = schema.get("properties") or {}
    rows = []
    for field_name in properties.keys():
        if field_name not in result_df.columns:
            continue
        series = result_df[field_name]
        matched = int(
            series.fillna(False)
            .astype(str)
            .str.strip()
            .str.lower()
            .isin({"true", "yes", "1"})
            .sum()
        )
        rows.append({"theme": field_name, "matched_reports": matched})
    summary_df = pd.DataFrame(rows, columns=["theme", "matched_reports"])
    if not summary_df.empty:
        summary_df = summary_df.sort_values("matched_reports", ascending=False).reset_index(drop=True)
    return summary_df


def _resolve_seed_topics(config: dict):
    seed_topics = config.get("seed_topics")
    if seed_topics in {None, ""}:
        return None
    if isinstance(seed_topics, (list, dict)):
        return seed_topics
    if isinstance(seed_topics, str):
        text = seed_topics.strip()
        if not text:
            return None
        try:
            return json.loads(text)
        except json.JSONDecodeError:
            return [line.strip() for line in text.splitlines() if line.strip()]
    return None


def execute_filter_workflow(
    *,
    run,
    progress_callback: Callable[[int, str], None] | None = None,
    cancellation_check: Callable[[], bool] | None = None,
) -> dict:
    config = run.input_config_json or {}

    if cancellation_check and cancellation_check():
        raise AdapterCancelledError("Run cancelled before filter workflow started.")

    query = _resolve_filter_query(investigation=run.investigation, config=config)
    start_date, end_date = _resolve_date_range(run=run, config=config)
    n_reports = config.get("report_limit")
    if n_reports in {"", None}:
        n_reports = None
    else:
        n_reports = int(n_reports)
    refresh = _as_bool(config.get("refresh"), default=False)

    if progress_callback:
        progress_callback(8, "Loading report dataset...")

    reports_df = load_reports(
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        n_reports=n_reports,
        refresh=refresh,
    )
    total_reports = len(reports_df)

    if cancellation_check and cancellation_check():
        raise AdapterCancelledError("Run cancelled after dataset load.")

    if progress_callback:
        progress_callback(20, f"Loaded {total_reports:,} reports.")

    llm_client = LLM(**_build_llm_kwargs(config))
    llm_client = _patch_generate_with_progress(
        llm_client=llm_client,
        progress_start=20,
        progress_end=95,
        progress_callback=progress_callback,
        cancellation_check=cancellation_check,
        default_message="Processing filter prompts...",
    )

    filter_df = _as_bool(config.get("filter_df"), default=True)
    produce_spans = _as_bool(config.get("produce_spans"), default=False)
    drop_spans = _as_bool(config.get("drop_spans"), default=False)

    screener = Screener(
        llm=llm_client,
        reports=reports_df,
        verbose=False,
        include_date=True,
        include_coroner=True,
        include_area=True,
        include_receiver=True,
        include_investigation=True,
        include_circumstances=True,
        include_concerns=True,
    )
    result_df = screener.screen_reports(
        search_query=query,
        filter_df=filter_df,
        result_col_name="matches_query",
        produce_spans=produce_spans,
        drop_spans=drop_spans,
    )

    if cancellation_check and cancellation_check():
        raise AdapterCancelledError("Run cancelled after screening stage.")

    output_path = _resolve_output_path(run=run, config=config)
    result_df.to_csv(output_path, index=False)

    if filter_df:
        matched_reports = len(result_df)
    elif "matches_query" in result_df.columns:
        matched_reports = int(result_df["matches_query"].fillna(False).astype(bool).sum())
    else:
        matched_reports = len(result_df)

    return {
        "output_path": str(output_path),
        "total_reports": total_reports,
        "output_reports": len(result_df),
        "matched_reports": matched_reports,
        "search_query": query,
        "filter_df": filter_df,
        "produce_spans": produce_spans,
        "drop_spans": drop_spans,
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "report_limit": n_reports,
    }


def execute_themes_workflow(
    *,
    run,
    progress_callback: Callable[[int, str], None] | None = None,
    cancellation_check: Callable[[], bool] | None = None,
) -> dict:
    config = run.input_config_json or {}
    if cancellation_check and cancellation_check():
        raise AdapterCancelledError("Run cancelled before themes workflow started.")

    start_date, end_date = _resolve_date_range(run=run, config=config)
    n_reports = config.get("report_limit")
    if n_reports in {"", None}:
        n_reports = None
    else:
        n_reports = int(n_reports)
    refresh = _as_bool(config.get("refresh"), default=False)

    if progress_callback:
        progress_callback(8, "Loading report dataset...")
    reports_df = load_reports(
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        n_reports=n_reports,
        refresh=refresh,
    )
    total_reports = len(reports_df)
    if cancellation_check and cancellation_check():
        raise AdapterCancelledError("Run cancelled after dataset load.")
    if progress_callback:
        progress_callback(18, f"Loaded {total_reports:,} reports.")

    llm_client = LLM(**_build_llm_kwargs(config))
    llm_client = _patch_generate_with_progress(
        llm_client=llm_client,
        progress_start=18,
        progress_end=92,
        progress_callback=progress_callback,
        cancellation_check=cancellation_check,
        default_message="Processing theme discovery prompts...",
    )
    extractor = _build_extractor(llm_client=llm_client, reports_df=reports_df)

    trim_approach = str(config.get("trim_approach") or "truncate").strip().lower()
    if trim_approach not in {"truncate", "summarise"}:
        trim_approach = "truncate"

    summarise_intensity = config.get("summarise_intensity")
    if summarise_intensity is not None:
        summarise_intensity = str(summarise_intensity).strip().lower()

    max_tokens = config.get("max_tokens")
    max_words = config.get("max_words")
    max_tokens = int(max_tokens) if max_tokens not in {"", None} else None
    max_words = int(max_words) if max_words not in {"", None} else None

    extra_instructions = str(config.get("extra_theme_instructions") or "").strip() or None
    warning_threshold = int(config.get("warning_threshold") or 100000)
    error_threshold = int(config.get("error_threshold") or 500000)
    min_themes = config.get("min_themes")
    max_themes = config.get("max_themes")
    min_themes = int(min_themes) if min_themes not in {"", None} else None
    max_themes = int(max_themes) if max_themes not in {"", None} else None
    seed_topics = _resolve_seed_topics(config)

    if progress_callback:
        progress_callback(24, "Summarising reports for theme discovery...")
    summary_df = extractor.summarise(
        result_col_name="summary",
        trim_approach=trim_approach,
        summarise_intensity=summarise_intensity,
        discover_themes_extra_instructions=extra_instructions,
        max_tokens=max_tokens,
        max_words=max_words,
    )

    if cancellation_check and cancellation_check():
        raise AdapterCancelledError("Run cancelled before theme discovery stage.")
    if progress_callback:
        progress_callback(55, "Discovering theme schema...")

    theme_model = extractor.discover_themes(
        warn_exceed=warning_threshold,
        error_exceed=error_threshold,
        max_themes=max_themes,
        min_themes=min_themes,
        extra_instructions=extra_instructions,
        seed_topics=seed_topics,
        trim_approach=trim_approach,
        summarise_intensity=summarise_intensity,
        max_tokens=max_tokens,
        max_words=max_words,
    )
    if theme_model is None or not hasattr(theme_model, "model_json_schema"):
        raise AdapterConfigurationError("Theme discovery did not return a usable model schema.")

    if cancellation_check and cancellation_check():
        raise AdapterCancelledError("Run cancelled before assigning discovered themes.")
    if progress_callback:
        progress_callback(72, "Assigning discovered themes to reports...")

    themed_df = extractor.extract_features(
        feature_model=theme_model,
        force_assign=True,
        allow_multiple=True,
        skip_if_present=False,
    )
    theme_summary_df = _theme_summary_from_dataframe(themed_df, theme_model)

    output_dir = _resolve_output_dir(run=run, config=config)
    summary_output_path = output_dir / "theme_summary.csv"
    assignments_output_path = output_dir / "theme_assignments.csv"
    theme_schema_path = output_dir / "theme_schema.json"
    summaries_path = output_dir / "report_summaries.csv"

    theme_summary_df.to_csv(summary_output_path, index=False)
    themed_df.to_csv(assignments_output_path, index=False)
    summary_df.to_csv(summaries_path, index=False)
    with theme_schema_path.open("w", encoding="utf-8") as handle:
        json.dump(theme_model.model_json_schema(), handle, indent=2)

    return {
        "output_path": str(summary_output_path),
        "total_reports": total_reports,
        "discovered_themes": len(theme_summary_df),
        "theme_summary_path": str(summary_output_path),
        "theme_assignments_path": str(assignments_output_path),
        "theme_schema_path": str(theme_schema_path),
        "report_summaries_path": str(summaries_path),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "report_limit": n_reports,
    }


def execute_extract_workflow(
    *,
    run,
    progress_callback: Callable[[int, str], None] | None = None,
    cancellation_check: Callable[[], bool] | None = None,
) -> dict:
    config = run.input_config_json or {}
    if cancellation_check and cancellation_check():
        raise AdapterCancelledError("Run cancelled before extract workflow started.")

    start_date, end_date = _resolve_date_range(run=run, config=config)
    n_reports = config.get("report_limit")
    if n_reports in {"", None}:
        n_reports = None
    else:
        n_reports = int(n_reports)
    refresh = _as_bool(config.get("refresh"), default=False)

    if progress_callback:
        progress_callback(8, "Loading report dataset...")
    reports_df = load_reports(
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        n_reports=n_reports,
        refresh=refresh,
    )
    total_reports = len(reports_df)
    if cancellation_check and cancellation_check():
        raise AdapterCancelledError("Run cancelled after dataset load.")
    if progress_callback:
        progress_callback(20, f"Loaded {total_reports:,} reports.")

    llm_client = LLM(**_build_llm_kwargs(config))
    llm_client = _patch_generate_with_progress(
        llm_client=llm_client,
        progress_start=20,
        progress_end=94,
        progress_callback=progress_callback,
        cancellation_check=cancellation_check,
        default_message="Processing extraction prompts...",
    )
    extractor = _build_extractor(llm_client=llm_client, reports_df=reports_df)
    feature_model = _resolve_extract_feature_model(config)

    produce_spans = _as_bool(config.get("produce_spans"), default=False)
    drop_spans = _as_bool(config.get("drop_spans"), default=False)
    force_assign = _as_bool(config.get("force_assign"), default=False)
    allow_multiple = _as_bool(config.get("allow_multiple"), default=False)
    skip_if_present = _as_bool(config.get("skip_if_present"), default=True)
    extra_instructions = str(config.get("extra_instructions") or "").strip() or None

    if progress_callback:
        progress_callback(30, "Extracting structured fields...")
    result_df = extractor.extract_features(
        reports=reports_df,
        feature_model=feature_model,
        produce_spans=produce_spans,
        drop_spans=drop_spans,
        force_assign=force_assign,
        allow_multiple=allow_multiple,
        schema_detail="minimal",
        extra_instructions=extra_instructions,
        skip_if_present=skip_if_present,
    )

    if cancellation_check and cancellation_check():
        raise AdapterCancelledError("Run cancelled after extraction stage.")

    output_dir = _resolve_output_dir(run=run, config=config)
    output_path = output_dir / "extraction_table.csv"
    feature_schema_path = output_dir / "feature_schema.json"
    result_df.to_csv(output_path, index=False)
    with feature_schema_path.open("w", encoding="utf-8") as handle:
        json.dump(feature_model.model_json_schema(), handle, indent=2)

    return {
        "output_path": str(output_path),
        "feature_schema_path": str(feature_schema_path),
        "total_reports": total_reports,
        "output_reports": len(result_df),
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "report_limit": n_reports,
        "produce_spans": produce_spans,
        "drop_spans": drop_spans,
        "force_assign": force_assign,
        "allow_multiple": allow_multiple,
        "skip_if_present": skip_if_present,
    }
