"""Views for the standalone Django Workbench."""
from __future__ import annotations

import json
import zipfile
from datetime import date
from io import BytesIO
from urllib.parse import urlencode
from typing import Any, Optional

import pandas as pd
from django.contrib import messages
from django.http import HttpRequest, HttpResponse
from django.shortcuts import redirect, render
from django.views.decorators.http import require_http_methods

from .services import (
    build_extractor,
    build_feature_model_from_rows,
    build_llm,
    build_screener,
    load_reports_dataframe,
    parse_seed_topics,
    resolve_theme_emoji,
)
from .state import (
    build_theme_summary_table,
    clear_outputs_for_modified_dataset,
    clear_outputs_for_new_dataset,
    clear_preview_state,
    dataframe_to_payload,
    dataframe_from_payload,
    format_call,
    get_dataframe,
    get_repro_script_text,
    has_api_key,
    init_state,
    parse_optional_non_negative_int,
    push_history_snapshot,
    record_repro_action,
    redo_last_change,
    reset_repro_tracking,
    set_dataframe,
    undo_last_change,
    workspace_has_activity,
)


def _bool_from_post(request: HttpRequest, key: str, default: bool = False) -> bool:
    if key not in request.POST:
        return default
    value = request.POST.get(key)
    if value is None:
        return True
    return str(value).strip().lower() in {"1", "true", "on", "yes"}


def _parse_date(value: str, default: date) -> date:
    try:
        return date.fromisoformat((value or "").strip())
    except ValueError:
        return default


def _df_to_html(df: pd.DataFrame, table_class: str = "data-table") -> str:
    if df.empty:
        return ""
    display_df = df.copy()
    return display_df.to_html(
        index=False,
        classes=table_class,
        border=0,
        na_rep="",
        escape=False,
    )


def _normalise_dashboard_value(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, str):
        cleaned = value.strip()
    else:
        try:
            if pd.isna(value):
                return ""
        except TypeError:
            pass
        cleaned = str(value).strip()

    lowered = cleaned.lower()
    if lowered in {"nan", "nat", "none"}:
        return ""
    return cleaned


def _split_dashboard_receivers(value: Any) -> list[str]:
    raw = _normalise_dashboard_value(value)
    if not raw:
        return []

    seen: set[str] = set()
    receivers: list[str] = []
    for chunk in raw.split(";"):
        cleaned = _normalise_dashboard_value(chunk)
        if not cleaned:
            continue
        lowered = cleaned.casefold()
        if lowered in seen:
            continue
        seen.add(lowered)
        receivers.append(cleaned)
    return receivers


def _build_explore_dashboard_payload(reports_df: pd.DataFrame) -> dict[str, Any]:
    if reports_df.empty:
        return {
            "rows": [],
            "options": {
                "coroners": [],
                "areas": [],
                "receivers": [],
            },
        }

    row_count = len(reports_df)
    default_series = pd.Series([""] * row_count, index=reports_df.index, dtype=object)

    if "date" in reports_df.columns:
        date_series = pd.to_datetime(reports_df["date"], errors="coerce")
    else:
        date_series = pd.Series([pd.NaT] * row_count, index=reports_df.index, dtype="datetime64[ns]")

    coroner_series = reports_df["coroner"] if "coroner" in reports_df.columns else default_series
    area_series = reports_df["area"] if "area" in reports_df.columns else default_series
    receiver_series = reports_df["receiver"] if "receiver" in reports_df.columns else default_series

    rows: list[dict[str, Any]] = []
    coroners: set[str] = set()
    areas: set[str] = set()
    receivers: set[str] = set()

    for date_value, coroner_value, area_value, receiver_value in zip(
        date_series,
        coroner_series,
        area_series,
        receiver_series,
    ):
        date_iso = ""
        if pd.notna(date_value):
            date_iso = date_value.strftime("%Y-%m-%d")

        coroner = _normalise_dashboard_value(coroner_value)
        area = _normalise_dashboard_value(area_value)
        receiver_parts = _split_dashboard_receivers(receiver_value)

        if coroner:
            coroners.add(coroner)
        if area:
            areas.add(area)
        for receiver_name in receiver_parts:
            receivers.add(receiver_name)

        rows.append(
            {
                "date": date_iso,
                "year_month": date_iso[:7] if date_iso else "",
                "coroner": coroner,
                "area": area,
                "receivers": receiver_parts,
            }
        )

    sort_key = lambda value: value.casefold()
    return {
        "rows": rows,
        "options": {
            "coroners": sorted(coroners, key=sort_key),
            "areas": sorted(areas, key=sort_key),
            "receivers": sorted(receivers, key=sort_key),
        },
    }


def _parse_explore_filter_values(values: list[str]) -> list[str]:
    unique: dict[str, str] = {}
    for value in values:
        cleaned = _normalise_dashboard_value(value)
        if not cleaned:
            continue
        key = cleaned.casefold()
        if key not in unique:
            unique[key] = cleaned
    return list(unique.values())


def _get_explore_dashboard_filters(request: HttpRequest) -> dict[str, list[str]]:
    return {
        "coroner": _parse_explore_filter_values(request.GET.getlist("coroner")),
        "area": _parse_explore_filter_values(request.GET.getlist("area")),
        "receiver": _parse_explore_filter_values(request.GET.getlist("receiver")),
    }


def _apply_explore_dashboard_filters(
    reports_df: pd.DataFrame, filters: dict[str, list[str]]
) -> pd.DataFrame:
    if reports_df.empty:
        return reports_df

    selected_coroners = set(item.casefold() for item in filters.get("coroner", []))
    selected_areas = set(item.casefold() for item in filters.get("area", []))
    selected_receivers = set(item.casefold() for item in filters.get("receiver", []))

    if not selected_coroners and not selected_areas and not selected_receivers:
        return reports_df

    filtered_df = reports_df.copy()

    if selected_coroners:
        coroner_series = (
            filtered_df["coroner"].map(_normalise_dashboard_value)
            if "coroner" in filtered_df.columns
            else pd.Series([""] * len(filtered_df), index=filtered_df.index, dtype=object)
        )
        coroner_mask = coroner_series.map(lambda value: value.casefold() in selected_coroners)
        filtered_df = filtered_df.loc[coroner_mask]

    if selected_areas:
        area_series = (
            filtered_df["area"].map(_normalise_dashboard_value)
            if "area" in filtered_df.columns
            else pd.Series([""] * len(filtered_df), index=filtered_df.index, dtype=object)
        )
        area_mask = area_series.map(lambda value: value.casefold() in selected_areas)
        filtered_df = filtered_df.loc[area_mask]

    if selected_receivers:
        receiver_series = (
            filtered_df["receiver"]
            if "receiver" in filtered_df.columns
            else pd.Series([""] * len(filtered_df), index=filtered_df.index, dtype=object)
        )
        receiver_mask = receiver_series.map(
            lambda value: any(
                receiver_name.casefold() in selected_receivers
                for receiver_name in _split_dashboard_receivers(value)
            )
        )
        filtered_df = filtered_df.loc[receiver_mask]

    return filtered_df.reset_index(drop=True)


def _serialize_preview_state(preview_state: dict[str, Any]) -> dict[str, Any]:
    payload = dict(preview_state)
    for key in ("summary_df", "preview_df", "theme_summary"):
        if key in payload and isinstance(payload[key], pd.DataFrame):
            payload[key] = dataframe_to_payload(payload[key])
    return payload


def _deserialize_preview_state(preview_state: Any) -> Optional[dict[str, Any]]:
    if not isinstance(preview_state, dict):
        return None
    payload = dict(preview_state)
    for key in ("summary_df", "preview_df", "theme_summary"):
        payload[key] = dataframe_from_payload(preview_state.get(key))
    return payload


def _default_feature_rows() -> list[dict[str, str]]:
    return [
        {
            "Field name": "age",
            "Description": "Age of the deceased (in years), if provided.",
            "Type": "Whole number",
        },
        {
            "Field name": "is_healthcare",
            "Description": "True if the report involves a healthcare setting.",
            "Type": "Conditional (True/False)",
        },
    ]


def _get_feature_grid_df(request: HttpRequest, *, use_default: bool = True) -> pd.DataFrame:
    payload = request.session.get("feature_grid")
    if isinstance(payload, str):
        try:
            df = pd.read_json(payload, orient="split")
        except ValueError:
            df = pd.DataFrame()
    else:
        df = pd.DataFrame()

    required = ["Field name", "Description", "Type"]
    if df.empty or any(col not in df.columns for col in required):
        if not use_default:
            return pd.DataFrame(columns=required)
        return pd.DataFrame(_default_feature_rows())
    return df[required].copy()


def _set_feature_grid_df(request: HttpRequest, df: pd.DataFrame) -> None:
    request.session["feature_grid"] = dataframe_to_payload(df)


def _update_sidebar_state(request: HttpRequest) -> None:
    session = request.session

    provider_override = request.POST.get("provider_override")
    if provider_override in {"OpenAI", "OpenRouter"}:
        session["provider_override"] = provider_override

    if "openai_api_key" in request.POST:
        session["openai_api_key"] = request.POST.get("openai_api_key", "")
    if "openrouter_api_key" in request.POST:
        session["openrouter_api_key"] = request.POST.get("openrouter_api_key", "")
    if "openai_base_url" in request.POST:
        session["openai_base_url"] = request.POST.get("openai_base_url", "")
    if "openrouter_base_url" in request.POST:
        session["openrouter_base_url"] = request.POST.get("openrouter_base_url", "https://openrouter.ai/api/v1")

    model_name = request.POST.get("model_name")
    if model_name in {"gpt-4.1-mini", "gpt-4.1"}:
        session["model_name"] = model_name

    if "max_parallel_workers" in request.POST:
        try:
            workers = int(request.POST.get("max_parallel_workers", "8"))
            session["max_parallel_workers"] = min(32, max(1, workers))
        except ValueError:
            pass

    if "report_start_date" in request.POST:
        parsed_start = _parse_date(request.POST.get("report_start_date", ""), date(2013, 1, 1))
        session["report_start_date"] = parsed_start.isoformat()
    if "report_end_date" in request.POST:
        parsed_end = _parse_date(request.POST.get("report_end_date", ""), date.today())
        session["report_end_date"] = parsed_end.isoformat()
    if "n_reports" in request.POST:
        n_reports_raw = (request.POST.get("n_reports") or "").strip()
        if not n_reports_raw:
            session["report_limit"] = None
        else:
            try:
                n_reports = int(n_reports_raw)
                if n_reports > 0:
                    session["report_limit"] = n_reports
            except ValueError:
                pass


def _build_context(request: HttpRequest) -> dict[str, Any]:
    session = request.session

    reports_df_full = get_dataframe(session, "reports_df")
    dashboard_filters = _get_explore_dashboard_filters(request)
    reports_df = _apply_explore_dashboard_filters(reports_df_full, dashboard_filters)
    theme_summary_df = get_dataframe(session, "theme_summary_table")
    feature_grid_df = _get_feature_grid_df(request, use_default=True)
    feature_grid_raw = _get_feature_grid_df(request, use_default=False)
    feature_rows_ui = [
        {
            "field_name": str(row.get("Field name", "")).strip(),
            "description": str(row.get("Description", "")).strip(),
            "type": str(row.get("Type", "")).strip(),
        }
        for row in feature_grid_df.to_dict(orient="records")
    ]

    if session.get("reports_df_initial") is None and not reports_df_full.empty:
        set_dataframe(session, "reports_df_initial", reports_df_full)

    reports_count = len(reports_df)
    page_size = 100
    try:
        page = int((request.GET.get("page") or "1").strip())
    except ValueError:
        page = 1
    page = max(1, page)
    total_pages = max(1, (reports_count + page_size - 1) // page_size)
    if page > total_pages:
        page = total_pages
    page_start = (page - 1) * page_size
    page_end = page_start + page_size
    reports_page_df = reports_df.iloc[page_start:page_end].copy()
    earliest_display = "-"
    latest_display = "-"
    if not reports_df.empty and "date" in reports_df.columns:
        date_series = pd.to_datetime(reports_df["date"], errors="coerce")
        if not date_series.empty:
            earliest = date_series.min()
            latest = date_series.max()
            if pd.notna(earliest):
                earliest_display = earliest.strftime("%d %b %Y")
            if pd.notna(latest):
                latest_display = latest.strftime("%d %b %Y")

    preview_state = _deserialize_preview_state(session.get("preview_state"))
    preview_theme_summary_html = ""
    if preview_state and isinstance(preview_state.get("theme_summary"), pd.DataFrame):
        preview_theme_summary_html = _df_to_html(preview_state["theme_summary"])

    active_action = session.get("active_action")
    llm_ready = has_api_key(session)

    theme_summary_sorted = theme_summary_df
    if not theme_summary_df.empty and {"Count", "Theme"}.issubset(theme_summary_df.columns):
        theme_summary_sorted = theme_summary_df.sort_values(by=["Count", "Theme"], ascending=[False, True])

    top_theme = None
    if not theme_summary_sorted.empty:
        row = theme_summary_sorted.iloc[0]
        top_theme_name = str(row.get("Theme", "-")).strip() or "-"
        top_theme = {
            "name": top_theme_name,
            "count": int(row.get("Count", 0)),
            "pct": float(row.get("%", 0.0)),
            "emoji": resolve_theme_emoji(top_theme_name, session),
        }

    history_depth = len(session.get("history", []))
    redo_depth = len(session.get("redo_history", []))
    dashboard_filter_query = urlencode(
        [
            *[("coroner", value) for value in dashboard_filters["coroner"]],
            *[("area", value) for value in dashboard_filters["area"]],
            *[("receiver", value) for value in dashboard_filters["receiver"]],
        ],
        doseq=True,
    )

    return {
        "reports_df": reports_df,
        "reports_html": _df_to_html(reports_page_df),
        "theme_summary_df": theme_summary_sorted,
        "theme_summary_html": _df_to_html(theme_summary_sorted),
        "preview_state": preview_state,
        "preview_theme_summary_html": preview_theme_summary_html,
        "feature_rows": feature_rows_ui,
        "has_feature_grid": not feature_grid_raw.empty,
        "reports_count": reports_count,
        "reports_page": page,
        "reports_total_pages": total_pages,
        "reports_page_from": (page_start + 1) if reports_count else 0,
        "reports_page_to": min(page_end, reports_count),
        "reports_prev_page": page - 1 if page > 1 else None,
        "reports_next_page": page + 1 if page < total_pages else None,
        "earliest_display": earliest_display,
        "latest_display": latest_display,
        "active_action": active_action,
        "llm_ready": llm_ready,
        "dataset_available": not reports_df.empty,
        "has_theme_summary": not theme_summary_sorted.empty,
        "top_theme": top_theme,
        "history_depth": history_depth,
        "redo_depth": redo_depth,
        "workspace_active": workspace_has_activity(session),
        "reports_modified": bool(session.get("reports_df_modified", False)),
        "provider": session.get("provider_override", "OpenAI"),
        "openai_api_key": session.get("openai_api_key", ""),
        "openrouter_api_key": session.get("openrouter_api_key", ""),
        "openai_base_url": session.get("openai_base_url", ""),
        "openrouter_base_url": session.get("openrouter_base_url", "https://openrouter.ai/api/v1"),
        "model_name": session.get("model_name", "gpt-4.1-mini"),
        "max_parallel_workers": int(session.get("max_parallel_workers", 8) or 8),
        "report_start_date": session.get("report_start_date", date(2013, 1, 1).isoformat()),
        "report_end_date": session.get("report_end_date", date.today().isoformat()),
        "report_limit": session.get("report_limit"),
        "workspace_dataset_token": session.get("reports_df") or "",
        "dashboard_selected_coroners": dashboard_filters["coroner"],
        "dashboard_selected_areas": dashboard_filters["area"],
        "dashboard_selected_receivers": dashboard_filters["receiver"],
        "dashboard_filter_query": dashboard_filter_query,
    }


def _handle_load_reports(request: HttpRequest) -> None:
    session = request.session

    start_date = _parse_date(request.POST.get("report_start_date", ""), date(2013, 1, 1))
    end_date = _parse_date(request.POST.get("report_end_date", ""), date.today())

    if end_date < start_date:
        messages.error(request, "End date must be on or after the start date.")
        return

    n_reports_raw = (request.POST.get("n_reports") or "").strip()
    n_reports: Optional[int]
    if n_reports_raw:
        try:
            n_reports = int(n_reports_raw)
            if n_reports <= 0:
                raise ValueError
        except ValueError:
            messages.error(request, "Please enter a whole number greater than 0 for the report limit.")
            return
        session["report_limit"] = n_reports
    else:
        n_reports = session.get("report_limit")

    refresh = _bool_from_post(request, "refresh", default=False)

    try:
        df = load_reports_dataframe(
            start_date=start_date,
            end_date=end_date,
            n_reports=n_reports,
            refresh=refresh,
        )
    except Exception as exc:
        messages.error(request, f"Could not load reports: {exc}")
        return
    if df.empty:
        messages.error(request, "No reports were returned. Please adjust the date range.")
        return

    set_dataframe(session, "reports_df", df)
    set_dataframe(session, "reports_df_initial", df)
    clear_outputs_for_new_dataset(session)
    session["history"] = []
    session["redo_history"] = []
    session["explore_onboarded"] = True
    reset_repro_tracking(session)

    load_kwargs = {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "n_reports": n_reports,
        "refresh": refresh,
    }
    record_repro_action(
        session,
        "load_reports",
        "Load in reports",
        format_call("reports_df = load_reports", load_kwargs),
    )

    messages.success(request, f"Loaded {len(df):,} reports into the workspace.")


def _ensure_workspace_reports_loaded(request: HttpRequest) -> None:
    session = request.session
    reports_df = get_dataframe(session, "reports_df")
    if not reports_df.empty:
        return

    start_date = _parse_date(session.get("report_start_date", ""), date(2013, 1, 1))
    end_date = _parse_date(session.get("report_end_date", ""), date.today())
    if end_date < start_date:
        end_date = date.today()
        start_date = date(2013, 1, 1)
        session["report_start_date"] = start_date.isoformat()
        session["report_end_date"] = end_date.isoformat()

    try:
        df = load_reports_dataframe(
            start_date=start_date,
            end_date=end_date,
            n_reports=session.get("report_limit"),
            refresh=False,
        )
    except Exception as exc:
        messages.error(request, f"Could not load default reports: {exc}")
        return
    if df.empty:
        messages.error(request, "No reports were returned for the default date range.")
        return

    set_dataframe(session, "reports_df", df)
    set_dataframe(session, "reports_df_initial", df)
    clear_outputs_for_new_dataset(session)
    session["history"] = []
    session["redo_history"] = []
    reset_repro_tracking(session)

    load_kwargs = {
        "start_date": start_date.isoformat(),
        "end_date": end_date.isoformat(),
        "n_reports": session.get("report_limit"),
        "refresh": False,
    }
    record_repro_action(
        session,
        "load_reports",
        "Load in reports",
        format_call("reports_df = load_reports", load_kwargs),
    )


def _handle_set_active_action(request: HttpRequest) -> None:
    target = request.POST.get("target")
    if target in {"save", "filter", "discover", "extract"}:
        request.session["active_action"] = target


def _handle_filter_reports(request: HttpRequest) -> None:
    session = request.session
    reports_df = get_dataframe(session, "reports_df")
    if reports_df.empty:
        messages.info(request, "Load reports from the sidebar before screening.")
        return

    search_query = (request.POST.get("search_query") or "").strip()
    if not search_query:
        messages.error(request, "Describe what the Screener should look for.")
        return

    filter_df = _bool_from_post(request, "filter_df", default=True)
    produce_spans = _bool_from_post(request, "produce_spans", default=False)
    drop_spans = _bool_from_post(request, "drop_spans", default=False)

    try:
        llm_client = build_llm(session)
    except Exception as exc:
        messages.error(request, f"LLM setup failed: {exc}")
        return

    push_history_snapshot(session)
    initial_report_count = len(reports_df)

    try:
        screener = build_screener(llm_client, reports_df, session)
        match_column_name = "matches_query"
        result_df = screener.screen_reports(
            search_query=search_query,
            filter_df=filter_df,
            result_col_name=match_column_name,
            produce_spans=produce_spans,
            drop_spans=drop_spans,
        )
        screen_kwargs = {
            "search_query": search_query,
            "filter_df": filter_df,
            "result_col_name": match_column_name,
            "produce_spans": produce_spans,
            "drop_spans": drop_spans,
        }
        record_repro_action(
            session,
            "run_screener",
            "Screen the reports",
            format_call("result_df = screener.screen_reports", screen_kwargs),
        )
    except Exception as exc:
        messages.error(request, f"Screening failed: {exc}")
        return

    set_dataframe(session, "screener_result", result_df)
    set_dataframe(session, "reports_df", result_df)
    clear_outputs_for_modified_dataset(session)
    session["active_action"] = None

    if filter_df:
        matched_report_count = len(result_df)
    elif isinstance(result_df, pd.DataFrame) and "matches_query" in result_df.columns:
        matched_report_count = int(result_df["matches_query"].fillna(False).astype(bool).sum())
    else:
        matched_report_count = len(result_df)

    messages.success(
        request,
        (
            f"Screening successful! From the initial {initial_report_count:,} reports, "
            f"{matched_report_count:,} matched your search query."
        ),
    )


def _handle_discover_themes(request: HttpRequest) -> None:
    session = request.session
    reports_df = get_dataframe(session, "reports_df")
    if reports_df.empty:
        messages.info(request, "Load reports before discovering themes.")
        return

    try:
        llm_client = build_llm(session)
    except Exception as exc:
        messages.error(request, f"LLM setup failed: {exc}")
        return

    extra_theme_instructions = (request.POST.get("extra_theme_instructions") or "").strip()
    trim_approach = request.POST.get("trim_approach", "truncate")
    if trim_approach not in {"truncate", "summarise"}:
        trim_approach = "truncate"

    summarise_intensity = None
    max_tokens: Optional[int] = None
    max_words: Optional[int] = None

    if trim_approach == "truncate":
        limit_type = request.POST.get("truncation_limit_type", "tokens")
        if limit_type == "words":
            try:
                max_words = int(request.POST.get("max_words", "1500"))
            except ValueError:
                max_words = 1500
        else:
            try:
                max_tokens = int(request.POST.get("max_tokens", "3000"))
            except ValueError:
                max_tokens = 3000
    else:
        summarise_intensity = request.POST.get("summarise_intensity", "medium")
        if summarise_intensity not in {"low", "medium", "high", "very high"}:
            summarise_intensity = "medium"

    try:
        warning_threshold = int(request.POST.get("warning_threshold", "100000"))
    except ValueError:
        warning_threshold = 100000

    try:
        error_threshold = int(request.POST.get("error_threshold", "500000"))
    except ValueError:
        error_threshold = 500000

    min_themes_raw = request.POST.get("min_themes", "")
    max_themes_raw = request.POST.get("max_themes", "")

    try:
        min_themes_value = parse_optional_non_negative_int(min_themes_raw, "Minimum number of themes")
        max_themes_value = parse_optional_non_negative_int(max_themes_raw, "Maximum number of themes")
    except ValueError as exc:
        messages.error(request, str(exc))
        return

    seed_topics_text = request.POST.get("seed_topics", "")
    seed_topics = parse_seed_topics(seed_topics_text)

    try:
        extractor = build_extractor(llm_client, reports_df, session)

        summary_col_name = extractor.summary_col or "summary"
        summary_df = extractor.summarise(
            result_col_name=summary_col_name,
            trim_approach=trim_approach,
            summarise_intensity=summarise_intensity,
            discover_themes_extra_instructions=extra_theme_instructions or None,
            max_tokens=max_tokens,
            max_words=max_words,
        )
        summarise_kwargs = {
            "result_col_name": summary_col_name,
            "trim_approach": trim_approach,
            "summarise_intensity": summarise_intensity,
            "discover_themes_extra_instructions": extra_theme_instructions or None,
            "max_tokens": max_tokens,
            "max_words": max_words,
        }
        record_repro_action(
            session,
            "summarise_reports",
            "Prepare the reports for theme discovery",
            format_call("summary_df = extractor.summarise", summarise_kwargs),
        )

        ThemeModel = extractor.discover_themes(
            warn_exceed=warning_threshold,
            error_exceed=error_threshold,
            max_themes=max_themes_value,
            min_themes=min_themes_value,
            extra_instructions=extra_theme_instructions or None,
            seed_topics=seed_topics,
            trim_approach=trim_approach,
            summarise_intensity=summarise_intensity,
            max_tokens=max_tokens,
            max_words=max_words,
        )
        discover_kwargs = {
            "warn_exceed": warning_threshold,
            "error_exceed": error_threshold,
            "max_themes": max_themes_value,
            "min_themes": min_themes_value,
            "extra_instructions": extra_theme_instructions or None,
            "seed_topics": seed_topics,
            "trim_approach": trim_approach,
            "summarise_intensity": summarise_intensity,
            "max_tokens": max_tokens,
            "max_words": max_words,
        }
        record_repro_action(
            session,
            "discover_themes",
            "Discover recurring themes",
            format_call("ThemeModel = extractor.discover_themes", discover_kwargs),
        )

        if ThemeModel is None or not hasattr(ThemeModel, "model_json_schema"):
            clear_preview_state(session)
            messages.warning(request, "Theme discovery completed but did not return a schema.")
            return

        theme_schema = ThemeModel.model_json_schema()
        preview_df = extractor.extract_features(
            feature_model=ThemeModel,
            force_assign=True,
            allow_multiple=True,
            skip_if_present=False,
        )
        theme_extract_kwargs = {
            "feature_model": "ThemeModel",
            "force_assign": True,
            "allow_multiple": True,
            "skip_if_present": False,
        }
        record_repro_action(
            session,
            "assign_themes",
            "Assign discovered themes to the reports",
            format_call(
                "preview_df = extractor.extract_features",
                theme_extract_kwargs,
                raw_parameters={"feature_model"},
            ),
        )

        theme_summary_df = build_theme_summary_table(preview_df, theme_schema)
        preview_state = {
            "type": "discover",
            "summary_df": summary_df,
            "preview_df": preview_df,
            "theme_schema": theme_schema,
            "theme_summary": theme_summary_df,
            "seed_topics": seed_topics,
        }
        session["preview_state"] = _serialize_preview_state(preview_state)
        messages.success(request, "Preview ready. Review the results below and apply them when happy.")
    except Exception as exc:
        clear_preview_state(session)
        messages.error(request, f"Theme discovery failed: {exc}")


def _handle_accept_themes(request: HttpRequest) -> None:
    session = request.session
    preview_state = _deserialize_preview_state(session.get("preview_state"))
    if not preview_state or preview_state.get("type") != "discover":
        messages.error(request, "No theme preview is available to apply.")
        return

    preview_df = preview_state.get("preview_df")
    if not isinstance(preview_df, pd.DataFrame):
        messages.error(request, "No preview data available to apply.")
        return

    theme_summary_df = preview_state.get("theme_summary")
    push_history_snapshot(session)
    set_dataframe(session, "reports_df", preview_df)
    session["reports_df_modified"] = True
    set_dataframe(session, "summary_result", preview_state.get("summary_df"))
    set_dataframe(session, "extractor_result", preview_df)
    session["theme_model_schema"] = preview_state.get("theme_schema")
    set_dataframe(
        session,
        "theme_summary_table",
        theme_summary_df if isinstance(theme_summary_df, pd.DataFrame) else pd.DataFrame(),
    )
    session["seed_topics_last"] = preview_state.get("seed_topics")
    clear_preview_state(session)
    session["active_action"] = None
    messages.success(request, "Themes applied to the working dataset.")


def _handle_discard_themes(request: HttpRequest) -> None:
    clear_preview_state(request.session)
    request.session["active_action"] = None
    messages.info(request, "Theme preview discarded.")


def _parse_feature_rows_from_post(request: HttpRequest) -> pd.DataFrame:
    names = request.POST.getlist("feature_name")
    descriptions = request.POST.getlist("feature_description")
    types = request.POST.getlist("feature_type")

    rows: list[dict[str, str]] = []
    for name, description, type_label in zip(names, descriptions, types):
        row = {
            "Field name": (name or "").strip(),
            "Description": (description or "").strip(),
            "Type": (type_label or "").strip(),
        }
        if row["Field name"] or row["Description"] or row["Type"]:
            rows.append(row)

    if not rows:
        rows = _default_feature_rows()

    return pd.DataFrame(rows, columns=["Field name", "Description", "Type"])


def _handle_extract_features(request: HttpRequest) -> None:
    session = request.session
    reports_df = get_dataframe(session, "reports_df")
    if reports_df.empty:
        messages.info(request, "Load reports before extracting fields.")
        return

    try:
        llm_client = build_llm(session)
    except Exception as exc:
        messages.error(request, f"LLM setup failed: {exc}")
        return

    feature_grid = _parse_feature_rows_from_post(request)
    _set_feature_grid_df(request, feature_grid)

    produce_spans = _bool_from_post(request, "extract_produce_spans", default=False)
    drop_spans = _bool_from_post(request, "extract_drop_spans", default=False)
    force_assign = _bool_from_post(request, "extract_force_assign", default=False)
    allow_multiple = _bool_from_post(request, "extract_allow_multiple", default=False)
    skip_if_present = _bool_from_post(request, "extract_skip_if_present", default=True)
    extra_instructions = (request.POST.get("extract_extra_instructions") or "").strip()

    push_history_snapshot(session)
    try:
        extractor = build_extractor(llm_client, reports_df, session)
        feature_model = build_feature_model_from_rows(feature_grid)
        result_df = extractor.extract_features(
            reports=reports_df,
            feature_model=feature_model,
            produce_spans=produce_spans,
            drop_spans=drop_spans,
            force_assign=force_assign,
            allow_multiple=allow_multiple,
            schema_detail="minimal",
            extra_instructions=extra_instructions or None,
            skip_if_present=skip_if_present,
        )

        extract_kwargs = {
            "reports": "reports_df",
            "feature_model": "feature_model",
            "produce_spans": produce_spans,
            "drop_spans": drop_spans,
            "force_assign": force_assign,
            "allow_multiple": allow_multiple,
            "schema_detail": "minimal",
            "extra_instructions": extra_instructions or None,
            "skip_if_present": skip_if_present,
        }
        record_repro_action(
            session,
            "extract_features",
            "Pull structured information",
            format_call(
                "result_df = extractor.extract_features",
                extract_kwargs,
                raw_parameters={"reports", "feature_model"},
            ),
        )
    except Exception as exc:
        messages.error(request, f"Extraction failed: {exc}")
        return

    set_dataframe(session, "extractor_result", result_df)
    set_dataframe(session, "reports_df", result_df)
    session["reports_df_modified"] = True
    clear_preview_state(session)
    session["active_action"] = None
    messages.success(request, "Tagging complete. The working dataset has been updated.")


def _bundle_download_response(request: HttpRequest) -> HttpResponse:
    session = request.session
    reports_df = get_dataframe(session, "reports_df")
    theme_summary_df = get_dataframe(session, "theme_summary_table")
    feature_grid_df = _get_feature_grid_df(request, use_default=False)

    include_dataset = _bool_from_post(request, "download_include_dataset", default=True)
    include_theme = _bool_from_post(request, "download_include_theme", default=False)
    include_feature_grid = _bool_from_post(request, "download_include_feature_grid", default=False)
    include_script = _bool_from_post(request, "download_include_script", default=False)

    files: list[tuple[str, bytes]] = []
    if include_dataset and not reports_df.empty:
        files.append(("pfd_reports.csv", reports_df.to_csv(index=False).encode("utf-8")))
    if include_theme and not theme_summary_df.empty:
        files.append(("theme_summary.txt", theme_summary_df.to_csv(index=False, sep="\t").encode("utf-8")))
    if include_feature_grid and not feature_grid_df.empty:
        files.append(("custom_feature_grid.txt", feature_grid_df.to_csv(index=False, sep="\t").encode("utf-8")))
    if include_script:
        files.append(("reproducible_workspace.py", get_repro_script_text(session).encode("utf-8")))

    if not files:
        raise ValueError("Select at least one resource above to enable the download.")

    # If only the dataset is selected, return CSV directly instead of a ZIP archive.
    if len(files) == 1 and files[0][0] == "pfd_reports.csv":
        response = HttpResponse(files[0][1], content_type="text/csv; charset=utf-8")
        response["Content-Disposition"] = 'attachment; filename="pfd_reports.csv"'
        return response

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for filename, payload in files:
            zip_file.writestr(filename, payload)
    zip_buffer.seek(0)

    response = HttpResponse(zip_buffer.getvalue(), content_type="application/zip")
    response["Content-Disposition"] = 'attachment; filename="pfd_research_bundle.zip"'
    return response


def _handle_start_over(request: HttpRequest) -> None:
    session = request.session
    initial_df = get_dataframe(session, "reports_df_initial")
    if initial_df.empty:
        return

    push_history_snapshot(session)
    set_dataframe(session, "reports_df", initial_df)
    clear_outputs_for_new_dataset(session)
    session["history"] = []
    session["redo_history"] = []
    messages.success(request, "Workspace restored to the post-load dataset.")


def _handle_revert_reports(request: HttpRequest) -> None:
    session = request.session
    initial_df = get_dataframe(session, "reports_df_initial")
    if initial_df.empty:
        messages.info(request, "No baseline dataset is available to revert to.")
        return

    set_dataframe(session, "reports_df", initial_df)
    clear_outputs_for_new_dataset(session)
    messages.success(request, "Reverted to the initially loaded reports.")


def _handle_post_action(request: HttpRequest) -> Optional[HttpResponse]:
    _update_sidebar_state(request)
    action = request.POST.get("action", "")

    if action == "load_reports":
        _handle_load_reports(request)
    elif action == "set_active_action":
        _handle_set_active_action(request)
    elif action == "filter_reports":
        _handle_filter_reports(request)
    elif action == "discover_themes":
        _handle_discover_themes(request)
    elif action == "accept_themes":
        _handle_accept_themes(request)
    elif action == "discard_themes":
        _handle_discard_themes(request)
    elif action == "extract_features":
        _handle_extract_features(request)
    elif action == "undo":
        if undo_last_change(request.session):
            messages.success(request, "Reverted to the previous state.")
    elif action == "redo":
        if redo_last_change(request.session):
            messages.success(request, "Reapplied the next state.")
    elif action == "start_over":
        _handle_start_over(request)
    elif action == "revert_reports":
        _handle_revert_reports(request)
    elif action == "download_bundle":
        try:
            return _bundle_download_response(request)
        except Exception as exc:
            messages.info(request, str(exc))
    elif action == "save_settings":
        request.session["explore_onboarded"] = True
        messages.success(request, "Settings updated.")

    request.session.modified = True
    return None


def _set_explore_modal_flag(
    request: HttpRequest, context: dict[str, Any], current_page: str
) -> None:
    if current_page == "explore":
        context["show_config_modal"] = False
        return

    if current_page in {"filter", "themes", "extract"}:
        context["show_config_modal"] = not bool(context.get("llm_ready", False))
        return

    if (
        not request.session.get("explore_onboarded")
        and (context.get("openai_api_key") or context.get("openrouter_api_key"))
    ):
        request.session["explore_onboarded"] = True
    context["show_config_modal"] = not request.session.get("explore_onboarded", False)


@require_http_methods(["GET"])
def index(request: HttpRequest) -> HttpResponse:
    return home(request)


@require_http_methods(["GET"])
def home(request: HttpRequest) -> HttpResponse:
    init_state(request.session)
    _ensure_workspace_reports_loaded(request)
    context = _build_context(request)
    context["current_page"] = "home"
    return render(request, "workbench/home.html", context)


@require_http_methods(["GET", "POST"])
def explore(request: HttpRequest) -> HttpResponse:
    init_state(request.session)
    _ensure_workspace_reports_loaded(request)

    if request.method == "POST":
        response = _handle_post_action(request)
        if response is not None:
            return response
        return redirect("workbench:explore")

    context = _build_context(request)
    context["explore_dashboard_payload"] = _build_explore_dashboard_payload(get_dataframe(request.session, "reports_df"))
    context["explore_dashboard_payload"]["selected"] = {
        "coroner": context.get("dashboard_selected_coroners", []),
        "area": context.get("dashboard_selected_areas", []),
        "receiver": context.get("dashboard_selected_receivers", []),
    }
    _set_explore_modal_flag(request, context, "explore")
    context["current_page"] = "explore"
    return render(request, "workbench/explore.html", context)


@require_http_methods(["GET", "POST"])
def filter_page(request: HttpRequest) -> HttpResponse:
    init_state(request.session)
    _ensure_workspace_reports_loaded(request)

    if request.method == "POST":
        response = _handle_post_action(request)
        if response is not None:
            return response
        return redirect("workbench:filter")

    context = _build_context(request)
    _set_explore_modal_flag(request, context, "filter")
    context["current_page"] = "filter"
    return render(request, "workbench/filter.html", context)


@require_http_methods(["GET", "POST"])
def themes_page(request: HttpRequest) -> HttpResponse:
    init_state(request.session)
    _ensure_workspace_reports_loaded(request)

    if request.method == "POST":
        response = _handle_post_action(request)
        if response is not None:
            return response
        return redirect("workbench:themes")

    context = _build_context(request)
    _set_explore_modal_flag(request, context, "themes")
    context["current_page"] = "themes"
    return render(request, "workbench/themes.html", context)


@require_http_methods(["GET", "POST"])
def extract_page(request: HttpRequest) -> HttpResponse:
    init_state(request.session)
    _ensure_workspace_reports_loaded(request)

    if request.method == "POST":
        response = _handle_post_action(request)
        if response is not None:
            return response
        return redirect("workbench:extract")

    context = _build_context(request)
    _set_explore_modal_flag(request, context, "extract")
    context["current_page"] = "extract"
    return render(request, "workbench/extract.html", context)


@require_http_methods(["GET", "POST"])
def settings_page(request: HttpRequest) -> HttpResponse:
    init_state(request.session)
    _ensure_workspace_reports_loaded(request)

    if request.method == "POST":
        response = _handle_post_action(request)
        if response is not None:
            return response
        return redirect("workbench:settings")

    context = _build_context(request)
    context["current_page"] = "settings"
    return render(request, "workbench/settings.html", context)


@require_http_methods(["GET"])
def dataset_panel(request: HttpRequest) -> HttpResponse:
    init_state(request.session)
    _ensure_workspace_reports_loaded(request)
    context = _build_context(request)
    return render(request, "workbench/_dataset_table_section.html", context)
