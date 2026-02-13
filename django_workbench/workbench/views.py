"""Views for the standalone Django Workbench."""
from __future__ import annotations

import json
import mimetypes
import random
import re
import zipfile
import copy
import importlib
import importlib.util
import inspect
from html import escape
from datetime import date, datetime
from io import BytesIO, StringIO
from pathlib import Path
from urllib.parse import parse_qs, urlencode, urljoin, urlparse
from typing import Any, Optional
from uuid import UUID, uuid4

import markdown
import pandas as pd
import yaml
from bs4 import BeautifulSoup
from django.contrib import messages
from django.http import FileResponse, Http404, HttpRequest, HttpResponse, JsonResponse
from django.shortcuts import get_object_or_404, redirect, render
from django.urls import reverse
from django.utils.text import slugify
from django.views.decorators.http import require_http_methods

from .models import Workbook
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

SEO_PAGE_METADATA: dict[str, dict[str, str]] = {
    "home": {
        "title": "PFD Toolkit - Analyse Prevention of Future Deaths Reports with AI",
        "description": (
            "Analyse Prevention of Future Deaths reports with AI. "
            "Filter reports, discover themes, and extract structured evidence in one workflow."
        ),
        "robots": "index,follow",
    },
    "explore": {
        "title": "Explore PFD Reports | PFD Toolkit",
        "description": (
            "Explore and filter Prevention of Future Deaths reports with interactive dashboards and AI-assisted screening."
        ),
        "robots": "index,follow",
    },
    "themes": {
        "title": "Analyse Themes in PFD Reports | PFD Toolkit",
        "description": (
            "Discover recurring themes across Prevention of Future Deaths reports and preview assignments before applying."
        ),
        "robots": "index,follow",
    },
    "extract": {
        "title": "Extract Structured Data from PFD Reports | PFD Toolkit",
        "description": (
            "Define custom fields and extract structured information from Prevention of Future Deaths reports using AI."
        ),
        "robots": "index,follow",
    },
    "settings": {
        "title": "Workbench Settings | PFD Toolkit",
        "description": "Configure model providers, API keys, and workspace defaults for PFD Toolkit.",
        "robots": "noindex,nofollow",
    },
    "for_coders": {
        "title": "For Coders | PFD Toolkit",
        "description": "Technical documentation and implementation details for PFD Toolkit.",
        "robots": "index,follow",
    },
    "workbook_public": {
        "title": "Shared PFD Workbook | PFD Toolkit",
        "description": "View a shared PFD Toolkit workbook with report-level filtering and summary outputs.",
        "robots": "index,follow",
    },
}


def _bool_from_post(request: HttpRequest, key: str, default: bool = False) -> bool:
    if key not in request.POST:
        return default
    value = request.POST.get(key)
    if value is None:
        return True
    return str(value).strip().lower() in {"1", "true", "on", "yes"}


def _parse_date(value: str, default: date) -> date:
    raw = (value or "").strip()
    if not raw:
        return default

    for parser in (date.fromisoformat, lambda item: datetime.strptime(item, "%d/%m/%Y").date()):
        try:
            return parser(raw)
        except ValueError:
            continue
    return default


def _parse_date_strict(value: str) -> Optional[date]:
    raw = (value or "").strip()
    if not raw:
        return None

    if "/" in raw:
        try:
            return datetime.strptime(raw, "%d/%m/%Y").date()
        except ValueError:
            return None

    try:
        return date.fromisoformat(raw)
    except ValueError:
        return None


def _format_date_ddmmyyyy(value: str, default: date) -> str:
    parsed = _parse_date(value, default)
    return parsed.strftime("%d/%m/%Y")


def _df_to_html(df: pd.DataFrame, table_class: str = "data-table") -> str:
    if df.empty:
        return ""
    display_df = df.copy().drop(
        columns=[WORKBENCH_ROW_ID_COL, EXCLUSION_REASON_COL],
        errors="ignore",
    )
    return display_df.to_html(
        index=False,
        classes=table_class,
        border=0,
        na_rep="",
        escape=False,
    )


def _normalise_row_id(raw_value: Any) -> str:
    return str(raw_value or "").strip()


def _ensure_workbench_row_ids(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    if df.empty:
        return df.copy(), False

    result_df = df.copy()
    changed = False

    if WORKBENCH_ROW_ID_COL not in result_df.columns:
        result_df[WORKBENCH_ROW_ID_COL] = [uuid4().hex for _ in range(len(result_df))]
        return result_df, True

    seen: set[str] = set()
    normalised_ids: list[str] = []
    for value in result_df[WORKBENCH_ROW_ID_COL].tolist():
        row_id = _normalise_row_id(value)
        if not row_id or row_id in seen:
            row_id = uuid4().hex
            changed = True
        seen.add(row_id)
        normalised_ids.append(row_id)

    if list(result_df[WORKBENCH_ROW_ID_COL].astype(str)) != normalised_ids:
        changed = True
    result_df[WORKBENCH_ROW_ID_COL] = normalised_ids
    return result_df, changed


def _normalise_excluded_reports_df(df: pd.DataFrame) -> tuple[pd.DataFrame, bool]:
    if df.empty:
        return pd.DataFrame(columns=[WORKBENCH_ROW_ID_COL, EXCLUSION_REASON_COL]), False

    result_df, changed = _ensure_workbench_row_ids(df)
    if EXCLUSION_REASON_COL not in result_df.columns:
        result_df[EXCLUSION_REASON_COL] = ""
        changed = True
    else:
        reasons = result_df[EXCLUSION_REASON_COL].map(lambda value: str(value or "").strip())
        if not reasons.equals(result_df[EXCLUSION_REASON_COL]):
            changed = True
        result_df[EXCLUSION_REASON_COL] = reasons
    return result_df, changed


def _get_reports_df_with_row_ids(session: dict[str, Any]) -> pd.DataFrame:
    reports_df = get_dataframe(session, "reports_df")
    reports_df, changed = _ensure_workbench_row_ids(reports_df)
    if changed:
        set_dataframe(session, "reports_df", reports_df)
    return reports_df


def _set_reports_df_with_row_ids(session: dict[str, Any], key: str, df: pd.DataFrame) -> pd.DataFrame:
    normalised_df, _ = _ensure_workbench_row_ids(df)
    set_dataframe(session, key, normalised_df)
    return normalised_df


def _get_excluded_reports_df(session: dict[str, Any]) -> pd.DataFrame:
    excluded_df = get_dataframe(session, "excluded_reports_df")
    excluded_df, changed = _normalise_excluded_reports_df(excluded_df)
    if changed:
        set_dataframe(session, "excluded_reports_df", excluded_df)
    return excluded_df


def _set_excluded_reports_df(session: dict[str, Any], excluded_df: pd.DataFrame) -> None:
    normalised_df, _ = _normalise_excluded_reports_df(excluded_df)
    if normalised_df.empty:
        session["excluded_reports_df"] = None
        return
    set_dataframe(session, "excluded_reports_df", normalised_df)


def _format_dataset_cell_value(value: Any) -> str:
    if value is None:
        return ""
    try:
        if pd.isna(value):
            return ""
    except TypeError:
        pass
    if isinstance(value, (list, tuple, dict)):
        return json.dumps(value, ensure_ascii=False)
    return str(value)


def _dataset_table_rows(
    df: pd.DataFrame,
    *,
    include_reason: bool = False,
) -> tuple[list[str], list[dict[str, Any]]]:
    if df.empty:
        return [], []

    row_ids = (
        df[WORKBENCH_ROW_ID_COL]
        if WORKBENCH_ROW_ID_COL in df.columns
        else pd.Series([""] * len(df), index=df.index, dtype=object)
    )
    reason_series = (
        df[EXCLUSION_REASON_COL]
        if include_reason and EXCLUSION_REASON_COL in df.columns
        else pd.Series([""] * len(df), index=df.index, dtype=object)
    )

    display_df = df.drop(columns=[WORKBENCH_ROW_ID_COL, EXCLUSION_REASON_COL], errors="ignore").copy()
    columns = [str(column) for column in display_df.columns]
    if include_reason:
        columns.append(EXCLUSION_REASON_LABEL)

    rows: list[dict[str, Any]] = []
    for index, row in display_df.iterrows():
        cells = [_format_dataset_cell_value(row[column]) for column in display_df.columns]
        if include_reason:
            cells.append(_format_dataset_cell_value(reason_series.loc[index]))
        rows.append(
            {
                "row_id": _normalise_row_id(row_ids.loc[index]),
                "cells": cells,
            }
        )

    return columns, rows


def _build_excluded_reports_context(
    excluded_df: pd.DataFrame,
    *,
    allow_restore: bool,
) -> dict[str, Any]:
    columns, rows = _dataset_table_rows(excluded_df, include_reason=True)
    return {
        "excluded_reports_count": len(excluded_df),
        "excluded_reports_columns": columns,
        "excluded_reports_rows": rows,
        "excluded_reports_available": bool(rows),
        "excluded_reports_restore_enabled": allow_restore,
    }


def _excluded_reports_export_df(excluded_df: pd.DataFrame) -> pd.DataFrame:
    if excluded_df.empty:
        return pd.DataFrame()
    export_df = excluded_df.copy().drop(columns=[WORKBENCH_ROW_ID_COL], errors="ignore")
    if EXCLUSION_REASON_COL in export_df.columns:
        export_df = export_df.rename(columns={EXCLUSION_REASON_COL: "reason_for_exclusion"})
    return export_df


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


def _get_post_dashboard_filters(request: HttpRequest) -> dict[str, list[str]]:
    return {
        "coroner": _parse_explore_filter_values(request.POST.getlist("dashboard_coroner")),
        "area": _parse_explore_filter_values(request.POST.getlist("dashboard_area")),
        "receiver": _parse_explore_filter_values(request.POST.getlist("dashboard_receiver")),
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


def _resolve_reports_for_ai_action(request: HttpRequest) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    session = request.session
    reports_df = _get_reports_df_with_row_ids(session)
    post_filters = _get_post_dashboard_filters(request)
    has_post_filters = any(post_filters[field] for field in ("coroner", "area", "receiver"))
    if not has_post_filters:
        return reports_df, {"coroner": [], "area": [], "receiver": []}
    return _apply_explore_dashboard_filters(reports_df, post_filters), post_filters


def _theme_columns_from_schema(theme_schema: Any) -> list[str]:
    if not isinstance(theme_schema, dict):
        return []
    properties = theme_schema.get("properties")
    if not isinstance(properties, dict):
        return []
    return [str(column).strip() for column in properties.keys() if str(column).strip()]


def _has_existing_theme_assignments(session: dict[str, Any], reports_df: pd.DataFrame) -> bool:
    theme_schema = session.get("theme_model_schema")
    theme_columns = _theme_columns_from_schema(theme_schema)
    if theme_columns and not reports_df.empty:
        if any(column in reports_df.columns for column in theme_columns):
            return True
    theme_summary_df = get_dataframe(session, "theme_summary_table")
    return not theme_summary_df.empty


WORKBOOK_TITLE_MAX_LENGTH = 120
WORKBOOK_SNAPSHOT_MAX_BYTES = 2_500_000
WORKBOOK_TITLE_ALLOWED_PATTERN = re.compile(r"^[A-Za-z0-9 -]+$")
WORKBENCH_ROW_ID_COL = "_workbench_row_id"
EXCLUSION_REASON_COL = "_exclusion_reason"
EXCLUSION_REASON_LABEL = "Reason for exclusion"
DOCS_SOURCE_ROOT = Path(__file__).resolve().parents[2] / "docs"
DOCS_CONFIG_FILE = Path(__file__).resolve().parents[2] / "mkdocs.yml"
DOCS_LINK_HOSTS = {"127.0.0.1", "localhost", "pfdtoolkit.org", "www.pfdtoolkit.org"}
DOCS_SITE_FILE_PREFIXES = ("assets/", "stylesheets/", "search/")
DOCS_REMOVED_PATHS = {"contact"}
DOCS_MARKDOWN_EXTENSIONS = [
    "admonition",
    "attr_list",
    "codehilite",
    "fenced_code",
    "tables",
    "toc",
    "abbr",
    "md_in_html",
    "pymdownx.highlight",
    "pymdownx.inlinehilite",
    "pymdownx.superfences",
    "pymdownx.tabbed",
    "pymdownx.details",
]


def _path_within_root(path: Path, root: Path) -> bool:
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _normalise_docs_path(raw_path: Optional[str]) -> str:
    cleaned = str(raw_path or "").strip().strip("/")
    if cleaned in {"", "index", "index.html"}:
        return ""
    if cleaned.endswith("/index"):
        cleaned = cleaned[: -len("/index")]
    if cleaned.endswith(".html"):
        cleaned = cleaned[: -len(".html")]
    return cleaned.strip("/")


def _docs_markdown_file_path(doc_path: str) -> Path:
    if not doc_path:
        return DOCS_SOURCE_ROOT / "index.md"
    return DOCS_SOURCE_ROOT / f"{doc_path}.md"


def _mkdocs_path_to_doc_path(path_value: str) -> Optional[str]:
    raw_path = str(path_value or "").strip()
    if not raw_path:
        return ""

    if raw_path.startswith("/pfd-toolkit/"):
        raw_path = raw_path[len("/pfd-toolkit/") :]

    cleaned = raw_path.lstrip("/")
    if not cleaned:
        return ""
    if cleaned.startswith(DOCS_SITE_FILE_PREFIXES):
        return None
    if cleaned in {
        "CNAME",
        "favicon.ico",
        "googlef12640be708032bb.html",
        "objects.inv",
        "sitemap.xml",
        "sitemap.xml.gz",
    }:
        return None
    if cleaned.endswith(
        (
            ".json",
            ".xml",
            ".gz",
            ".png",
            ".jpg",
            ".jpeg",
            ".svg",
            ".ico",
            ".css",
            ".js",
            ".woff",
            ".woff2",
            ".ttf",
        )
    ):
        return None
    if cleaned.endswith("/"):
        cleaned = cleaned[:-1]
    if cleaned.endswith("/index.html"):
        cleaned = cleaned[: -len("/index.html")]
    if cleaned.endswith("/index.md"):
        cleaned = cleaned[: -len("/index.md")]
    if cleaned in {"index.html", "index.md"}:
        return ""
    if cleaned.endswith("/index"):
        cleaned = cleaned[: -len("/index")]
    if cleaned.endswith(".html"):
        cleaned = cleaned[: -len(".html")]
    if cleaned.endswith(".md"):
        cleaned = cleaned[: -len(".md")]
    return cleaned.strip("/")


def _docs_url_for_path(doc_path: str) -> str:
    base = reverse("workbench:for_coders")
    if not doc_path:
        return base
    return f"{base}?{urlencode({'doc': doc_path})}"


def _docs_site_file_url(file_path: str) -> str:
    cleaned = str(file_path or "").lstrip("/")
    return reverse("workbench:for_coders_site_file", kwargs={"file_path": cleaned})


def _append_query_fragment(url: str, parsed: Any) -> str:
    result = url
    if parsed.query:
        separator = "&" if "?" in result else "?"
        result = f"{result}{separator}{parsed.query}"
    if parsed.fragment:
        result = f"{result}#{parsed.fragment}"
    return result


def _rewrite_docs_url(raw_url: str, current_doc_path: str) -> str:
    value = str(raw_url or "").strip()
    if not value:
        return value
    lowered = value.lower()
    if lowered.startswith(("mailto:", "tel:", "javascript:")) or value.startswith("#"):
        return value

    parsed = urlparse(value)
    if parsed.scheme in {"http", "https"}:
        host = parsed.hostname or ""
        if host.lower() not in DOCS_LINK_HOSTS:
            return value
        doc_path = _mkdocs_path_to_doc_path(parsed.path)
        if doc_path is not None:
            return _append_query_fragment(_docs_url_for_path(doc_path), parsed)
        site_file = parsed.path.lstrip("/")
        return _append_query_fragment(_docs_site_file_url(site_file), parsed)

    base_markdown_path = f"/{current_doc_path}.md" if current_doc_path else "/index.md"
    resolved = urlparse(urljoin(f"https://pfdtoolkit.org{base_markdown_path}", value))
    doc_path = _mkdocs_path_to_doc_path(resolved.path)
    if doc_path is not None:
        return _append_query_fragment(_docs_url_for_path(doc_path), resolved)
    return _append_query_fragment(_docs_site_file_url(resolved.path.lstrip("/")), resolved)


def _normalise_title_text(value: str) -> str:
    text = str(value or "").replace("¶", " ")
    return re.sub(r"\s+", " ", text).strip()


def _rewrite_docs_dom_links(container: BeautifulSoup, current_doc_path: str) -> None:
    for tag_name, attr_name in (("a", "href"), ("img", "src"), ("source", "src")):
        for tag in container.find_all(tag_name):
            raw_value = tag.get(attr_name)
            if not raw_value:
                continue
            tag[attr_name] = _rewrite_docs_url(str(raw_value), current_doc_path)


def _extract_doc_path_from_docs_url(url: Optional[str]) -> str:
    if not url:
        return ""
    parsed = urlparse(str(url))
    doc_param = parse_qs(parsed.query).get("doc", [""])[0]
    return _normalise_docs_path(doc_param)


def _first_navigable_url(items: list[dict[str, Any]]) -> Optional[str]:
    for item in items:
        url = item.get("url")
        if url:
            return str(url)
        nested = _first_navigable_url(item.get("children", []))
        if nested:
            return nested
    return None


def _enabled_markdown_extensions() -> list[str]:
    enabled: list[str] = []
    for extension_name in DOCS_MARKDOWN_EXTENSIONS:
        if extension_name.startswith("pymdownx.") and importlib.util.find_spec(extension_name) is None:
            continue
        enabled.append(extension_name)
    return enabled


def _strip_markdown_front_matter(markdown_text: str) -> str:
    text = str(markdown_text or "")
    if not text.startswith("---"):
        return text
    match = re.match(r"^---\s*\n.*?\n---\s*\n", text, flags=re.DOTALL)
    if not match:
        return text
    return text[match.end() :]


def _resolve_object_from_import_path(import_path: str) -> Any:
    cleaned = str(import_path or "").strip()
    if not cleaned:
        raise ImportError("Missing import path.")

    parts = cleaned.split(".")
    for index in range(len(parts), 0, -1):
        module_name = ".".join(parts[:index])
        try:
            module = importlib.import_module(module_name)
        except ImportError:
            continue
        value: Any = module
        for attribute in parts[index:]:
            value = getattr(value, attribute)
        return value
    raise ImportError(f"Could not import object: {cleaned}")


def _render_mkdocstrings_stub(import_path: str) -> str:
    cleaned = str(import_path or "").strip()
    try:
        value = _resolve_object_from_import_path(cleaned)
    except Exception:
        return (
            f'!!! warning\n'
            f'    API docs for `{cleaned}` could not be generated in the Django docs view.\n'
            f'    Please check the import path and package installation.\n'
        )

    obj_name = getattr(value, "__name__", cleaned.split(".")[-1])
    markdown_lines = [f"## `{obj_name}`", ""]
    try:
        signature = str(inspect.signature(value))
        markdown_lines.extend(["```python", f"{obj_name}{signature}", "```", ""])
    except (TypeError, ValueError):
        pass

    docstring = inspect.getdoc(value) or "No docstring available."
    markdown_lines.extend([docstring, ""])
    return "\n".join(markdown_lines)


def _expand_mkdocstrings_directives(markdown_text: str) -> str:
    pattern = re.compile(r"(?m)^:::\s*([A-Za-z_][\w\.]*)\s*$")

    def _replace(match: re.Match[str]) -> str:
        return _render_mkdocstrings_stub(match.group(1))

    return pattern.sub(_replace, markdown_text)


def _build_markdown_renderer() -> markdown.Markdown:
    return markdown.Markdown(
        extensions=_enabled_markdown_extensions(),
        extension_configs={
            "toc": {"permalink": True},
            "codehilite": {"guess_lang": False, "use_pygments": True},
            "pymdownx.highlight": {
                "anchor_linenums": True,
                "line_spans": "__span",
                "pygments_lang_class": True,
            },
            "pymdownx.tabbed": {"alternate_style": True},
        },
        output_format="html5",
    )


def _extract_docs_toc_from_tokens(tokens: list[dict[str, Any]], level: int = 1) -> list[dict[str, Any]]:
    items: list[dict[str, Any]] = []
    for token in tokens:
        title = _normalise_title_text(str(token.get("name") or ""))
        anchor = str(token.get("id") or "").strip()
        if title and anchor:
            items.append({"title": title, "url": f"#{anchor}", "level": level})
        children = token.get("children", [])
        if isinstance(children, list):
            items.extend(_extract_docs_toc_from_tokens(children, level + 1))
    return items


def _docs_url_from_nav_target(raw_target: str) -> Optional[str]:
    target = str(raw_target or "").strip()
    if not target:
        return None
    parsed = urlparse(target)
    if parsed.scheme in {"http", "https"}:
        host = (parsed.hostname or "").lower()
        if host not in DOCS_LINK_HOSTS:
            return target
        doc_path = _mkdocs_path_to_doc_path(parsed.path)
        if doc_path is not None:
            return _append_query_fragment(_docs_url_for_path(doc_path), parsed)
        return _append_query_fragment(_docs_site_file_url(parsed.path.lstrip("/")), parsed)
    if parsed.scheme:
        return target
    doc_path = _mkdocs_path_to_doc_path(parsed.path or target)
    if doc_path is not None:
        return _append_query_fragment(_docs_url_for_path(doc_path), parsed)
    path_value = (parsed.path or target).lstrip("/")
    return _append_query_fragment(_docs_site_file_url(path_value), parsed)


def _build_docs_nav_items(entries: list[Any], current_doc_path: str) -> list[dict[str, Any]]:
    current_doc_url = _docs_url_for_path(current_doc_path).rstrip("/") or "/"
    nav_items: list[dict[str, Any]] = []

    for entry in entries:
        if not isinstance(entry, dict):
            continue

        for raw_title, value in entry.items():
            title = _normalise_title_text(str(raw_title or ""))
            if not title:
                continue

            children: list[dict[str, Any]] = []
            url: Optional[str] = None
            if isinstance(value, list):
                children = _build_docs_nav_items(value, current_doc_path)
                url = _first_navigable_url(children)
            elif isinstance(value, dict):
                children = _build_docs_nav_items([value], current_doc_path)
                url = _first_navigable_url(children)
            else:
                url = _docs_url_from_nav_target(str(value))

            item_doc_path = _extract_doc_path_from_docs_url(url)
            if item_doc_path in DOCS_REMOVED_PATHS:
                continue

            base_url = (str(url or "").split("#", 1)[0].rstrip("/") or "/") if url else ""
            active = bool(base_url and base_url == current_doc_url) or any(child.get("active", False) for child in children)
            nav_items.append(
                {
                    "title": title,
                    "url": url,
                    "active": active,
                    "expanded": active,
                    "children": children,
                }
            )
    return nav_items


def _load_docs_nav_from_config(current_doc_path: str) -> list[dict[str, Any]]:
    if not DOCS_CONFIG_FILE.is_file():
        return []
    config_data = yaml.load(DOCS_CONFIG_FILE.read_text(encoding="utf-8"), Loader=yaml.UnsafeLoader) or {}
    nav_entries = config_data.get("nav") or []
    if not isinstance(nav_entries, list):
        return []
    return _build_docs_nav_items(nav_entries, current_doc_path)


def _extract_docs_toc(article_tag: Any) -> list[dict[str, Any]]:
    toc_items: list[dict[str, Any]] = []
    heading_pattern = re.compile(r"^h[1-4]$")
    for heading in article_tag.find_all(heading_pattern):
        heading_id = heading.get("id")
        if not heading_id:
            continue
        title = _normalise_title_text(heading.get_text(" ", strip=True))
        if not title:
            continue
        level = int(str(heading.name)[1])
        toc_items.append(
            {
                "title": title,
                "url": f"#{heading_id}",
                "level": level,
            }
        )
    return toc_items


def _load_docs_page_payload(doc_path: str) -> dict[str, Any]:
    normalised_doc_path = _normalise_docs_path(doc_path)
    markdown_path = _docs_markdown_file_path(normalised_doc_path)
    if not markdown_path.is_file():
        raise FileNotFoundError(f"Docs page not found: {normalised_doc_path or 'index'}")

    markdown_text = markdown_path.read_text(encoding="utf-8")
    markdown_text = _strip_markdown_front_matter(markdown_text)
    markdown_text = _expand_mkdocstrings_directives(markdown_text)

    renderer = _build_markdown_renderer()
    rendered_html = renderer.convert(markdown_text)

    wrapper = BeautifulSoup(f'<article class="md-content__inner">{rendered_html}</article>', "html.parser")
    article = wrapper.select_one("article.md-content__inner")
    if article is None:
        raise ValueError(f"Docs page could not be rendered: {normalised_doc_path or 'index'}")
    _rewrite_docs_dom_links(article, normalised_doc_path)

    nav_items = _load_docs_nav_from_config(normalised_doc_path)
    toc_items = _extract_docs_toc_from_tokens(getattr(renderer, "toc_tokens", []))
    if not toc_items:
        toc_items = _extract_docs_toc(article)

    page_title = "For coders"
    heading = article.find("h1")
    if heading is not None:
        resolved_title = _normalise_title_text(heading.get_text(" ", strip=True))
        if resolved_title:
            page_title = resolved_title

    return {
        "doc_path": normalised_doc_path,
        "page_title": page_title,
        "article_html": str(article),
        "nav_items": nav_items,
        "toc_items": toc_items,
    }


def _normalise_workbook_id(raw_value: str) -> Optional[str]:
    cleaned = (raw_value or "").strip()
    if not cleaned:
        return None
    try:
        parsed = UUID(cleaned)
    except ValueError:
        return None
    return str(parsed)


def _normalise_workbook_token(raw_value: str) -> Optional[str]:
    cleaned = (raw_value or "").strip()
    if not cleaned:
        return None
    try:
        parsed = UUID(cleaned)
    except ValueError:
        return None
    return str(parsed)


def _normalise_workbook_title(raw_value: Any) -> str:
    title = str(raw_value or "").strip()
    if len(title) > WORKBOOK_TITLE_MAX_LENGTH:
        title = title[:WORKBOOK_TITLE_MAX_LENGTH].rstrip()
    return title


def _workbook_title_error(title: str) -> Optional[str]:
    if not title:
        return "Worksheet name is required."
    if not WORKBOOK_TITLE_ALLOWED_PATTERN.fullmatch(title):
        return "Worksheet name can use letters, numbers, spaces, and hyphens only."
    return None


def _workbook_title_slug(title: str) -> str:
    slug = slugify(title)
    return slug or "worksheet"


def _allocate_workbook_share_number() -> int:
    for _ in range(30):
        candidate = random.randint(100000, 999999)
        if not Workbook.objects.filter(share_number=candidate).exists():
            return candidate
    raise RuntimeError("Could not allocate a unique worksheet share number.")


def _normalise_workbook_filters(raw_filters: Any) -> dict[str, list[str]]:
    if not isinstance(raw_filters, dict):
        return {"coroner": [], "area": [], "receiver": []}
    return {
        "coroner": _parse_explore_filter_values(raw_filters.get("coroner", [])),
        "area": _parse_explore_filter_values(raw_filters.get("area", [])),
        "receiver": _parse_explore_filter_values(raw_filters.get("receiver", [])),
    }


def _snapshot_dataframe_to_payload(df: pd.DataFrame) -> str:
    if df.empty:
        return ""
    return df.to_json(orient="split", date_format="iso")


def _snapshot_dataframe_from_payload(payload: Any) -> pd.DataFrame:
    if not payload or not isinstance(payload, str):
        return pd.DataFrame()
    try:
        return pd.read_json(StringIO(payload), orient="split")
    except ValueError:
        return pd.DataFrame()


def _build_workbook_snapshot(request: HttpRequest, selected_filters: dict[str, list[str]]) -> dict[str, Any]:
    reports_df_full = _get_reports_df_with_row_ids(request.session)
    reports_df_filtered = _apply_explore_dashboard_filters(reports_df_full, selected_filters)
    theme_summary_df = get_dataframe(request.session, "theme_summary_table")
    excluded_reports_df = _get_excluded_reports_df(request.session)

    payload = _build_explore_dashboard_payload(reports_df_filtered)
    payload["selected"] = selected_filters
    return {
        "dashboard_payload": payload,
        "reports_df": _snapshot_dataframe_to_payload(reports_df_filtered),
        "excluded_reports_df": _snapshot_dataframe_to_payload(excluded_reports_df),
        "theme_summary_df": _snapshot_dataframe_to_payload(theme_summary_df),
        "saved_from_path": request.path,
    }


def _build_workbook_public_url(request: HttpRequest, workbook: Workbook) -> str:
    return request.build_absolute_uri(
        f"/workbooks/{workbook.share_number}-{_workbook_title_slug(workbook.title)}/"
    )


def _build_workbook_edit_url(request: HttpRequest, workbook: Workbook) -> str:
    return request.build_absolute_uri(
        f"/explore-pfds/?workbook={workbook.public_id}&edit={workbook.edit_token}"
    )


def _workbook_filter_query(filters: dict[str, list[str]]) -> str:
    return urlencode(
        [
            *[("coroner", value) for value in filters.get("coroner", [])],
            *[("area", value) for value in filters.get("area", [])],
            *[("receiver", value) for value in filters.get("receiver", [])],
        ],
        doseq=True,
    )


def _selected_filters_from_snapshot(snapshot: dict[str, Any]) -> dict[str, list[str]]:
    dashboard_payload = snapshot.get("dashboard_payload")
    if not isinstance(dashboard_payload, dict):
        return {"coroner": [], "area": [], "receiver": []}
    return _normalise_workbook_filters(dashboard_payload.get("selected"))


def _restore_workspace_from_snapshot(request: HttpRequest, snapshot: dict[str, Any]) -> None:
    session = request.session
    reports_df, _ = _ensure_workbench_row_ids(_snapshot_dataframe_from_payload(snapshot.get("reports_df")))
    excluded_reports_df, _ = _normalise_excluded_reports_df(
        _snapshot_dataframe_from_payload(snapshot.get("excluded_reports_df"))
    )
    theme_summary_df = _snapshot_dataframe_from_payload(snapshot.get("theme_summary_df"))

    _set_reports_df_with_row_ids(session, "reports_df", reports_df)
    _set_reports_df_with_row_ids(session, "reports_df_initial", reports_df)
    clear_outputs_for_new_dataset(session)
    _set_excluded_reports_df(session, excluded_reports_df)

    if theme_summary_df.empty:
        session["theme_summary_table"] = None
    else:
        set_dataframe(session, "theme_summary_table", theme_summary_df)

    session["history"] = []
    session["redo_history"] = []
    reset_repro_tracking(session)


def _workbook_payload_size_ok(snapshot: dict[str, Any]) -> bool:
    encoded = json.dumps(snapshot, separators=(",", ":"), ensure_ascii=False)
    return len(encoded.encode("utf-8")) <= WORKBOOK_SNAPSHOT_MAX_BYTES


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


def _resolve_active_workbook(request: HttpRequest) -> tuple[Optional[Workbook], bool]:
    workbook_id = _normalise_workbook_id(request.GET.get("workbook", ""))
    edit_token = _normalise_workbook_token(request.GET.get("edit", ""))
    if not workbook_id:
        return None, False
    workbook = Workbook.objects.filter(public_id=workbook_id).first()
    if workbook is None:
        return None, False
    is_editable = bool(edit_token and edit_token == str(workbook.edit_token))
    return workbook, is_editable


def _build_context(request: HttpRequest) -> dict[str, Any]:
    session = request.session

    reports_df_full = _get_reports_df_with_row_ids(session)
    excluded_reports_df = _get_excluded_reports_df(session)
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
        _set_reports_df_with_row_ids(session, "reports_df_initial", reports_df_full)

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
    reports_columns, reports_rows = _dataset_table_rows(reports_page_df, include_reason=False)
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
    dashboard_filter_query = _workbook_filter_query(dashboard_filters)
    report_limit = session.get("report_limit")
    if isinstance(report_limit, int) and report_limit > 0:
        report_limit_for_slider = min(7000, max(1, report_limit))
    else:
        report_limit_for_slider = min(7000, max(1, reports_count or 500))

    ai_features_used = bool(
        session.get("reports_df_modified")
        or session.get("screener_result")
        or session.get("extractor_result")
        or session.get("summary_result")
        or session.get("theme_model_schema")
        or session.get("theme_summary_table")
        or session.get("preview_state")
    )
    has_existing_themes = _has_existing_theme_assignments(session, reports_df_full)
    excluded_context = _build_excluded_reports_context(excluded_reports_df, allow_restore=True)

    return {
        "reports_df": reports_df,
        "reports_columns": reports_columns,
        "reports_rows": reports_rows,
        "dataset_editable": True,
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
        "report_start_date_display": _format_date_ddmmyyyy(
            session.get("report_start_date", ""),
            date(2013, 1, 1),
        ),
        "report_end_date_display": _format_date_ddmmyyyy(
            session.get("report_end_date", ""),
            date.today(),
        ),
        "report_limit": report_limit,
        "report_limit_for_slider": report_limit_for_slider,
        "ai_features_used": ai_features_used,
        "has_existing_themes": has_existing_themes,
        "workspace_dataset_token": session.get("reports_df") or "",
        "dashboard_selected_coroners": dashboard_filters["coroner"],
        "dashboard_selected_areas": dashboard_filters["area"],
        "dashboard_selected_receivers": dashboard_filters["receiver"],
        "dashboard_filter_query": dashboard_filter_query,
        "dataset_panel_base": "/dataset-panel/",
        "dataset_browser_base": "?page=",
        **excluded_context,
    }


def _build_seo_context(
    request: HttpRequest,
    page_key: str,
    *,
    canonical_path: str,
    title_override: Optional[str] = None,
    description_override: Optional[str] = None,
    robots_override: Optional[str] = None,
    og_type: str = "website",
) -> dict[str, str]:
    defaults = SEO_PAGE_METADATA.get(page_key, {})
    title = title_override or defaults.get("title") or "PFD Toolkit AI Workbench · Beta"
    description = (
        description_override
        or defaults.get("description")
        or "Analyse Prevention of Future Deaths reports with AI workflows."
    )
    robots_value = robots_override or defaults.get("robots") or "index,follow"
    canonical_url = request.build_absolute_uri(canonical_path)
    return {
        "seo_title": title,
        "seo_description": description,
        "seo_robots": robots_value,
        "seo_canonical_url": canonical_url,
        "seo_og_type": og_type,
    }


def _render_sitemap_xml(request: HttpRequest) -> HttpResponse:
    entries: list[tuple[str, str, str, Optional[str]]] = []
    entries.extend(
        [
            (reverse("workbench:index"), "daily", "1.0", None),
            (reverse("workbench:explore"), "daily", "0.9", None),
            (reverse("workbench:themes"), "weekly", "0.7", None),
            (reverse("workbench:extract"), "weekly", "0.7", None),
            (reverse("workbench:for_coders"), "weekly", "0.6", None),
        ]
    )

    for workbook in Workbook.objects.only("share_number", "title", "updated_at").iterator():
        path = reverse(
            "workbench:workbook_public",
            kwargs={
                "share_number": workbook.share_number,
                "title_slug": _workbook_title_slug(workbook.title),
            },
        )
        lastmod = workbook.updated_at.date().isoformat() if workbook.updated_at else None
        entries.append((path, "weekly", "0.5", lastmod))

    lines = [
        '<?xml version="1.0" encoding="UTF-8"?>',
        '<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9">',
    ]
    for path, changefreq, priority, lastmod in entries:
        location = escape(request.build_absolute_uri(path))
        lines.append("  <url>")
        lines.append(f"    <loc>{location}</loc>")
        if lastmod:
            lines.append(f"    <lastmod>{escape(lastmod)}</lastmod>")
        lines.append(f"    <changefreq>{escape(changefreq)}</changefreq>")
        lines.append(f"    <priority>{escape(priority)}</priority>")
        lines.append("  </url>")
    lines.append("</urlset>")
    return HttpResponse("\n".join(lines), content_type="application/xml; charset=utf-8")


@require_http_methods(["GET"])
def robots_txt(request: HttpRequest) -> HttpResponse:
    sitemap_url = request.build_absolute_uri(reverse("workbench:sitemap_xml"))
    body = "\n".join(
        [
            "User-agent: *",
            "Allow: /",
            f"Sitemap: {sitemap_url}",
        ]
    )
    return HttpResponse(body, content_type="text/plain; charset=utf-8")


@require_http_methods(["GET"])
def sitemap_xml(request: HttpRequest) -> HttpResponse:
    return _render_sitemap_xml(request)


def _handle_load_reports(request: HttpRequest) -> None:
    session = request.session

    start_raw = request.POST.get("report_start_date", "")
    end_raw = request.POST.get("report_end_date", "")
    start_date = _parse_date_strict(start_raw)
    end_date = _parse_date_strict(end_raw)

    if start_date is None:
        messages.error(request, "Start date is invalid. Use DD/MM/YYYY format, for example 01/01/2013.")
        return
    if end_date is None:
        messages.error(request, "End date is invalid. Use DD/MM/YYYY format, for example 11/02/2026.")
        return

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

    _set_reports_df_with_row_ids(session, "reports_df", df)
    _set_reports_df_with_row_ids(session, "reports_df_initial", df)
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

    _set_reports_df_with_row_ids(session, "reports_df", df)
    _set_reports_df_with_row_ids(session, "reports_df_initial", df)
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
    reports_df, dashboard_filters = _resolve_reports_for_ai_action(request)
    if reports_df.empty:
        if any(dashboard_filters[field] for field in ("coroner", "area", "receiver")):
            messages.error(
                request,
                "Your manual dashboard filters returned no reports. Adjust or discard them, then try again.",
            )
        else:
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
    _set_reports_df_with_row_ids(session, "reports_df", result_df)
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
    reports_df, dashboard_filters = _resolve_reports_for_ai_action(request)
    if reports_df.empty:
        if any(dashboard_filters[field] for field in ("coroner", "area", "receiver")):
            messages.error(
                request,
                "Your manual dashboard filters returned no reports. Adjust or discard them, then try again.",
            )
        else:
            messages.info(request, "Load reports before discovering themes.")
        return

    all_reports_df = get_dataframe(session, "reports_df")
    existing_theme_schema = session.get("theme_model_schema")
    existing_theme_columns = [
        column for column in _theme_columns_from_schema(existing_theme_schema)
        if column in all_reports_df.columns
    ]
    has_existing_themes = _has_existing_theme_assignments(session, all_reports_df)
    rerun_confirmed = _bool_from_post(request, "confirm_rerun_themes", default=False)

    if has_existing_themes and not rerun_confirmed:
        messages.warning(
            request,
            "Themes are already applied to this workspace. To run discovery again, first confirm that you want to discard existing themes.",
        )
        return

    if has_existing_themes:
        push_history_snapshot(session)
        cleaned_reports_df = all_reports_df.drop(columns=existing_theme_columns, errors="ignore").copy()
        _set_reports_df_with_row_ids(session, "reports_df", cleaned_reports_df)
        clear_outputs_for_modified_dataset(session)
        reports_df = (
            _apply_explore_dashboard_filters(cleaned_reports_df, dashboard_filters)
            if any(dashboard_filters[field] for field in ("coroner", "area", "receiver"))
            else cleaned_reports_df
        )

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
    _set_reports_df_with_row_ids(session, "reports_df", preview_df)
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
    reports_df, dashboard_filters = _resolve_reports_for_ai_action(request)
    if reports_df.empty:
        if any(dashboard_filters[field] for field in ("coroner", "area", "receiver")):
            messages.error(
                request,
                "Your manual dashboard filters returned no reports. Adjust or discard them, then try again.",
            )
        else:
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
    _set_reports_df_with_row_ids(session, "reports_df", result_df)
    session["reports_df_modified"] = True
    clear_preview_state(session)
    session["active_action"] = None
    messages.success(request, "Tagging complete. The working dataset has been updated.")


def _bundle_download_response(request: HttpRequest) -> HttpResponse:
    session = request.session
    reports_df = get_dataframe(session, "reports_df")
    excluded_reports_df = _get_excluded_reports_df(session)
    theme_summary_df = get_dataframe(session, "theme_summary_table")
    feature_grid_df = _get_feature_grid_df(request, use_default=False)

    include_dataset = _bool_from_post(request, "download_include_dataset", default=True)
    include_excluded = _bool_from_post(request, "download_include_excluded", default=False)
    include_theme = _bool_from_post(request, "download_include_theme", default=False)
    include_feature_grid = _bool_from_post(request, "download_include_feature_grid", default=False)
    include_script = _bool_from_post(request, "download_include_script", default=False)

    files: list[tuple[str, bytes]] = []
    if include_dataset and not reports_df.empty:
        files.append(
            (
                "pfd_reports.csv",
                reports_df.drop(columns=[WORKBENCH_ROW_ID_COL], errors="ignore").to_csv(index=False).encode("utf-8"),
            )
        )
    excluded_export_df = _excluded_reports_export_df(excluded_reports_df)
    if include_excluded and not excluded_export_df.empty:
        files.append(("pfd_excluded_reports.csv", excluded_export_df.to_csv(index=False).encode("utf-8")))
    if include_theme and not theme_summary_df.empty:
        files.append(("theme_summary.txt", theme_summary_df.to_csv(index=False, sep="\t").encode("utf-8")))
    if include_feature_grid and not feature_grid_df.empty:
        files.append(("custom_feature_grid.txt", feature_grid_df.to_csv(index=False, sep="\t").encode("utf-8")))
    if include_script:
        files.append(("reproducible_workspace.py", get_repro_script_text(session).encode("utf-8")))

    if not files:
        raise ValueError("Select at least one resource above to enable the download.")

    # If only the dataset is selected, return CSV directly instead of a ZIP archive.
    if len(files) == 1 and files[0][0] in {"pfd_reports.csv", "pfd_excluded_reports.csv"}:
        response = HttpResponse(files[0][1], content_type="text/csv; charset=utf-8")
        response["Content-Disposition"] = f'attachment; filename="{files[0][0]}"'
        return response

    zip_buffer = BytesIO()
    with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
        for filename, payload in files:
            zip_file.writestr(filename, payload)
    zip_buffer.seek(0)

    response = HttpResponse(zip_buffer.getvalue(), content_type="application/zip")
    response["Content-Disposition"] = 'attachment; filename="pfd_research_bundle.zip"'
    return response


def _build_shared_dataset_context(
    request: HttpRequest,
    reports_df: pd.DataFrame,
    *,
    panel_base: str,
    browser_base: str,
    prefix: str = "",
) -> dict[str, Any]:
    reports_df, _ = _ensure_workbench_row_ids(reports_df)
    dashboard_filters = _get_explore_dashboard_filters(request)
    filtered_df = _apply_explore_dashboard_filters(reports_df, dashboard_filters)

    reports_count = len(filtered_df)
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
    reports_page_df = filtered_df.iloc[page_start:page_end].copy()
    reports_columns, reports_rows = _dataset_table_rows(reports_page_df, include_reason=False)

    earliest_display = "-"
    latest_display = "-"
    if not filtered_df.empty and "date" in filtered_df.columns:
        date_series = pd.to_datetime(filtered_df["date"], errors="coerce")
        if not date_series.empty:
            earliest = date_series.min()
            latest = date_series.max()
            if pd.notna(earliest):
                earliest_display = earliest.strftime("%d %b %Y")
            if pd.notna(latest):
                latest_display = latest.strftime("%d %b %Y")

    dashboard_filter_query = urlencode(
        [
            *[("coroner", value) for value in dashboard_filters["coroner"]],
            *[("area", value) for value in dashboard_filters["area"]],
            *[("receiver", value) for value in dashboard_filters["receiver"]],
        ],
        doseq=True,
    )

    return {
        f"{prefix}reports_df": filtered_df,
        f"{prefix}reports_count": reports_count,
        f"{prefix}reports_columns": reports_columns,
        f"{prefix}reports_rows": reports_rows,
        f"{prefix}dataset_editable": False,
        f"{prefix}reports_page": page,
        f"{prefix}reports_total_pages": total_pages,
        f"{prefix}reports_page_from": (page_start + 1) if reports_count else 0,
        f"{prefix}reports_page_to": min(page_end, reports_count),
        f"{prefix}reports_prev_page": page - 1 if page > 1 else None,
        f"{prefix}reports_next_page": page + 1 if page < total_pages else None,
        f"{prefix}earliest_display": earliest_display,
        f"{prefix}latest_display": latest_display,
        f"{prefix}dashboard_selected_coroners": dashboard_filters["coroner"],
        f"{prefix}dashboard_selected_areas": dashboard_filters["area"],
        f"{prefix}dashboard_selected_receivers": dashboard_filters["receiver"],
        f"{prefix}dashboard_filter_query": dashboard_filter_query,
        f"{prefix}dataset_available": not filtered_df.empty,
        f"{prefix}dataset_panel_base": panel_base,
        f"{prefix}dataset_browser_base": browser_base,
    }


def _bundle_download_from_workbook(workbook: Workbook, request: HttpRequest) -> HttpResponse:
    snapshot = workbook.snapshot if isinstance(workbook.snapshot, dict) else {}
    reports_df = _snapshot_dataframe_from_payload(snapshot.get("reports_df"))
    excluded_reports_df, _ = _normalise_excluded_reports_df(
        _snapshot_dataframe_from_payload(snapshot.get("excluded_reports_df"))
    )
    theme_summary_df = _snapshot_dataframe_from_payload(snapshot.get("theme_summary_df"))

    include_dataset = _bool_from_post(request, "download_include_dataset", default=True)
    include_excluded = _bool_from_post(request, "download_include_excluded", default=False)
    include_theme = _bool_from_post(request, "download_include_theme", default=False)

    files: list[tuple[str, bytes]] = []
    if include_dataset and not reports_df.empty:
        files.append(
            (
                "pfd_reports.csv",
                reports_df.drop(columns=[WORKBENCH_ROW_ID_COL], errors="ignore").to_csv(index=False).encode("utf-8"),
            )
        )
    excluded_export_df = _excluded_reports_export_df(excluded_reports_df)
    if include_excluded and not excluded_export_df.empty:
        files.append(("pfd_excluded_reports.csv", excluded_export_df.to_csv(index=False).encode("utf-8")))
    if include_theme and not theme_summary_df.empty:
        files.append(("theme_summary.txt", theme_summary_df.to_csv(index=False, sep="\t").encode("utf-8")))
    if not files:
        raise ValueError("Select at least one resource above to enable the download.")

    if len(files) == 1 and files[0][0] in {"pfd_reports.csv", "pfd_excluded_reports.csv"}:
        response = HttpResponse(files[0][1], content_type="text/csv; charset=utf-8")
        response["Content-Disposition"] = f'attachment; filename="{files[0][0]}"'
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
    _set_reports_df_with_row_ids(session, "reports_df", initial_df)
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

    _set_reports_df_with_row_ids(session, "reports_df", initial_df)
    clear_outputs_for_new_dataset(session)
    messages.success(request, "Reverted to the initially loaded reports.")


def _handle_exclude_report(request: HttpRequest) -> None:
    session = request.session
    row_id = _normalise_row_id(request.POST.get("report_row_id"))
    if not row_id:
        messages.error(request, "Select a report to exclude.")
        return

    reports_df = _get_reports_df_with_row_ids(session)
    if reports_df.empty or WORKBENCH_ROW_ID_COL not in reports_df.columns:
        messages.info(request, "No reports are available to exclude.")
        return

    row_mask = reports_df[WORKBENCH_ROW_ID_COL].astype(str) == row_id
    if not bool(row_mask.any()):
        messages.info(request, "That report is no longer available in the active dataset.")
        return

    exclusion_reason = (request.POST.get("exclusion_reason") or "").strip()

    push_history_snapshot(session)
    removed_rows = reports_df.loc[row_mask].copy()
    removed_rows[EXCLUSION_REASON_COL] = exclusion_reason
    remaining_reports = reports_df.loc[~row_mask].reset_index(drop=True)

    excluded_df = _get_excluded_reports_df(session)
    if not excluded_df.empty and WORKBENCH_ROW_ID_COL in excluded_df.columns:
        excluded_df = excluded_df.loc[
            excluded_df[WORKBENCH_ROW_ID_COL].astype(str) != row_id
        ].copy()
    updated_excluded = pd.concat([removed_rows, excluded_df], ignore_index=True, sort=False)

    _set_reports_df_with_row_ids(session, "reports_df", remaining_reports)
    _set_excluded_reports_df(session, updated_excluded)
    clear_outputs_for_modified_dataset(session)
    session["active_action"] = None
    messages.success(request, "Report excluded from the working dataset.")


def _handle_restore_excluded_report(request: HttpRequest) -> None:
    session = request.session
    row_id = _normalise_row_id(request.POST.get("report_row_id"))
    if not row_id:
        messages.error(request, "Select a report to restore.")
        return

    excluded_df = _get_excluded_reports_df(session)
    if excluded_df.empty or WORKBENCH_ROW_ID_COL not in excluded_df.columns:
        messages.info(request, "There are no excluded reports to restore.")
        return

    restore_mask = excluded_df[WORKBENCH_ROW_ID_COL].astype(str) == row_id
    if not bool(restore_mask.any()):
        messages.info(request, "That excluded report is no longer available.")
        return

    push_history_snapshot(session)

    restored_rows = excluded_df.loc[restore_mask].drop(columns=[EXCLUSION_REASON_COL], errors="ignore").copy()
    remaining_excluded = excluded_df.loc[~restore_mask].reset_index(drop=True)

    reports_df = _get_reports_df_with_row_ids(session)
    if WORKBENCH_ROW_ID_COL in reports_df.columns:
        existing_ids = set(reports_df[WORKBENCH_ROW_ID_COL].astype(str))
        restored_rows = restored_rows.loc[
            ~restored_rows[WORKBENCH_ROW_ID_COL].astype(str).isin(existing_ids)
        ].copy()

    updated_reports = pd.concat([reports_df, restored_rows], ignore_index=True, sort=False)
    _set_reports_df_with_row_ids(session, "reports_df", updated_reports)
    _set_excluded_reports_df(session, remaining_excluded)
    clear_outputs_for_modified_dataset(session)
    session["active_action"] = None
    messages.success(request, "Excluded report restored to the working dataset.")


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
    elif action == "exclude_report":
        _handle_exclude_report(request)
    elif action == "restore_excluded_report":
        _handle_restore_excluded_report(request)
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


def _json_body(request: HttpRequest) -> dict[str, Any]:
    try:
        payload = json.loads(request.body.decode("utf-8"))
    except (json.JSONDecodeError, UnicodeDecodeError):
        return {}
    if not isinstance(payload, dict):
        return {}
    return payload


@require_http_methods(["POST"])
def workbook_create(request: HttpRequest) -> HttpResponse:
    init_state(request.session)
    _ensure_workspace_reports_loaded(request)

    payload = _json_body(request)
    title = _normalise_workbook_title(payload.get("title"))
    title_error = _workbook_title_error(title)
    if title_error:
        return JsonResponse({"ok": False, "error": title_error}, status=400)

    selected_filters = _normalise_workbook_filters(payload.get("filters"))
    snapshot = _build_workbook_snapshot(request, selected_filters)
    if not _workbook_payload_size_ok(snapshot):
        return JsonResponse({"ok": False, "error": "Workbook snapshot is too large to save."}, status=413)

    workbook = Workbook.objects.create(
        share_number=_allocate_workbook_share_number(),
        title=title,
        snapshot=snapshot,
    )

    return JsonResponse(
        {
            "ok": True,
            "workbook_id": str(workbook.public_id),
            "edit_token": str(workbook.edit_token),
            "share_url": _build_workbook_public_url(request, workbook),
            "edit_url": _build_workbook_edit_url(request, workbook),
            "title": workbook.title,
        }
    )


@require_http_methods(["POST"])
def workbook_save(request: HttpRequest, public_id: UUID) -> HttpResponse:
    init_state(request.session)
    _ensure_workspace_reports_loaded(request)

    workbook = get_object_or_404(Workbook, public_id=public_id)
    payload = _json_body(request)

    provided_edit_token = _normalise_workbook_token(payload.get("edit_token"))
    if not provided_edit_token or provided_edit_token != str(workbook.edit_token):
        return JsonResponse({"ok": False, "error": "Edit token is invalid."}, status=403)

    title = _normalise_workbook_title(payload.get("title"))
    title_error = _workbook_title_error(title)
    if title_error:
        return JsonResponse({"ok": False, "error": title_error}, status=400)

    selected_filters = _normalise_workbook_filters(payload.get("filters"))
    snapshot = _build_workbook_snapshot(request, selected_filters)
    if not _workbook_payload_size_ok(snapshot):
        return JsonResponse({"ok": False, "error": "Workbook snapshot is too large to save."}, status=413)

    workbook.title = title
    workbook.snapshot = snapshot
    workbook.save(update_fields=["title", "snapshot", "updated_at"])

    return JsonResponse(
        {
            "ok": True,
            "workbook_id": str(workbook.public_id),
            "share_url": _build_workbook_public_url(request, workbook),
            "edit_url": _build_workbook_edit_url(request, workbook),
            "title": workbook.title,
        }
    )


@require_http_methods(["GET"])
def workbook_public(request: HttpRequest, share_number: int, title_slug: str) -> HttpResponse:
    workbook = get_object_or_404(Workbook, share_number=share_number)
    canonical_slug = _workbook_title_slug(workbook.title)
    if title_slug != canonical_slug:
        return redirect("workbench:workbook_public", share_number=share_number, title_slug=canonical_slug)
    payload = {}
    if isinstance(workbook.snapshot, dict):
        payload = workbook.snapshot.get("dashboard_payload") or {}
    if not isinstance(payload, dict):
        payload = {}
    reports_df = pd.DataFrame()
    excluded_reports_df = pd.DataFrame()
    theme_summary_df = pd.DataFrame()
    if isinstance(workbook.snapshot, dict):
        reports_df = _snapshot_dataframe_from_payload(workbook.snapshot.get("reports_df"))
        excluded_reports_df = _snapshot_dataframe_from_payload(workbook.snapshot.get("excluded_reports_df"))
        theme_summary_df = _snapshot_dataframe_from_payload(workbook.snapshot.get("theme_summary_df"))
    panel_base = f"/workbooks/{workbook.share_number}-{canonical_slug}/dataset-panel/"
    browser_base = f"/workbooks/{workbook.share_number}-{canonical_slug}/?page="
    shared_context = _build_shared_dataset_context(
        request,
        reports_df,
        panel_base=panel_base,
        browser_base=browser_base,
    )
    theme_summary_sorted = theme_summary_df
    if not theme_summary_sorted.empty and {"Count", "Theme"}.issubset(theme_summary_sorted.columns):
        theme_summary_sorted = theme_summary_sorted.sort_values(by=["Count", "Theme"], ascending=[False, True])

    top_theme = None
    if not theme_summary_sorted.empty:
        row = theme_summary_sorted.iloc[0]
        top_theme = {
            "name": str(row.get("Theme", "-")).strip() or "-",
            "count": int(row.get("Count", 0)),
            "pct": float(row.get("%", 0.0)),
            "emoji": "*",
        }

    excluded_context = _build_excluded_reports_context(
        _normalise_excluded_reports_df(excluded_reports_df)[0],
        allow_restore=False,
    )

    context = {
        "current_page": "explore",
        **shared_context,
        **excluded_context,
        "explore_dashboard_payload": payload,
        "workbook": workbook,
        "has_theme_summary": not theme_summary_sorted.empty,
        "theme_summary_html": _df_to_html(theme_summary_sorted),
        "top_theme": top_theme,
        "workspace_label": "Shared worksheet dataset",
    }
    context.update(
        _build_seo_context(
            request,
            "workbook_public",
            canonical_path=reverse(
                "workbench:workbook_public",
                kwargs={"share_number": workbook.share_number, "title_slug": canonical_slug},
            ),
            title_override=f'{workbook.title} | Shared PFD Workbook',
            description_override=(
                f'Explore the shared workbook "{workbook.title}" with filtered Prevention of Future Deaths reports.'
            ),
        )
    )
    return render(request, "workbench/workbook_public.html", context)


@require_http_methods(["POST"])
def workbook_download(request: HttpRequest, share_number: int, title_slug: str) -> HttpResponse:
    workbook = get_object_or_404(Workbook, share_number=share_number)
    canonical_slug = _workbook_title_slug(workbook.title)
    if title_slug != canonical_slug:
        return redirect("workbench:workbook_public", share_number=share_number, title_slug=canonical_slug)
    try:
        return _bundle_download_from_workbook(workbook, request)
    except ValueError as exc:
        messages.info(request, str(exc))
        return redirect("workbench:workbook_public", share_number=share_number, title_slug=canonical_slug)


@require_http_methods(["GET"])
def workbook_dataset_panel(request: HttpRequest, share_number: int, title_slug: str) -> HttpResponse:
    workbook = get_object_or_404(Workbook, share_number=share_number)
    canonical_slug = _workbook_title_slug(workbook.title)
    if title_slug != canonical_slug:
        return redirect("workbench:workbook_public", share_number=share_number, title_slug=canonical_slug)
    reports_df = pd.DataFrame()
    if isinstance(workbook.snapshot, dict):
        reports_df = _snapshot_dataframe_from_payload(workbook.snapshot.get("reports_df"))
    panel_base = f"/workbooks/{workbook.share_number}-{canonical_slug}/dataset-panel/"
    browser_base = f"/workbooks/{workbook.share_number}-{canonical_slug}/?page="
    context = _build_shared_dataset_context(
        request,
        reports_df,
        panel_base=panel_base,
        browser_base=browser_base,
    )
    context["workspace_label"] = "Shared worksheet dataset"
    response = render(request, "workbench/_dataset_table_section.html", context)
    response["X-Robots-Tag"] = "noindex, nofollow"
    return response


@require_http_methods(["POST"])
def workbook_clone(request: HttpRequest, share_number: int, title_slug: str) -> HttpResponse:
    init_state(request.session)

    source_workbook = get_object_or_404(Workbook, share_number=share_number)
    canonical_slug = _workbook_title_slug(source_workbook.title)
    if title_slug != canonical_slug:
        return redirect("workbench:workbook_public", share_number=share_number, title_slug=canonical_slug)

    source_snapshot = source_workbook.snapshot if isinstance(source_workbook.snapshot, dict) else {}
    cloned_snapshot = copy.deepcopy(source_snapshot)
    if not _workbook_payload_size_ok(cloned_snapshot):
        messages.error(request, "This shared worksheet is too large to clone.")
        return redirect("workbench:workbook_public", share_number=share_number, title_slug=canonical_slug)

    cloned_workbook = Workbook.objects.create(
        share_number=_allocate_workbook_share_number(),
        title="",
        snapshot=cloned_snapshot,
    )

    _restore_workspace_from_snapshot(request, cloned_snapshot)
    selected_filters = _selected_filters_from_snapshot(cloned_snapshot)
    query_pairs = [
        ("workbook", str(cloned_workbook.public_id)),
        ("edit", str(cloned_workbook.edit_token)),
    ]
    query_pairs.extend(("coroner", value) for value in selected_filters["coroner"])
    query_pairs.extend(("area", value) for value in selected_filters["area"])
    query_pairs.extend(("receiver", value) for value in selected_filters["receiver"])
    query_string = urlencode(query_pairs, doseq=True)
    messages.success(
        request,
        f'Workbook "{source_workbook.title}" is now editable. Changes you make won\'t affect the shared URL.',
    )
    return redirect(f"{reverse('workbench:explore')}?{query_string}")


@require_http_methods(["GET"])
def index(request: HttpRequest) -> HttpResponse:
    init_state(request.session)
    _ensure_workspace_reports_loaded(request)
    context = _build_context(request)
    context["current_page"] = "home"
    context.update(
        _build_seo_context(
            request,
            "home",
            canonical_path=reverse("workbench:index"),
        )
    )
    return render(request, "workbench/home.html", context)


@require_http_methods(["GET"])
def home(request: HttpRequest) -> HttpResponse:
    return redirect("workbench:index", permanent=True)


@require_http_methods(["GET", "POST"])
def explore(request: HttpRequest) -> HttpResponse:
    init_state(request.session)
    _ensure_workspace_reports_loaded(request)

    if request.method == "POST":
        response = _handle_post_action(request)
        if response is not None:
            return response
        query = request.GET.urlencode()
        target = reverse("workbench:explore")
        if query:
            target = f"{target}?{query}"
        return redirect(target)

    context = _build_context(request)
    context["explore_dashboard_payload"] = _build_explore_dashboard_payload(
        _get_reports_df_with_row_ids(request.session)
    )
    context["explore_dashboard_payload"]["selected"] = {
        "coroner": context.get("dashboard_selected_coroners", []),
        "area": context.get("dashboard_selected_areas", []),
        "receiver": context.get("dashboard_selected_receivers", []),
    }
    active_workbook, workbook_editable = _resolve_active_workbook(request)
    context["active_workbook"] = active_workbook
    context["active_workbook_editable"] = workbook_editable
    context["active_workbook_id"] = str(active_workbook.public_id) if active_workbook else ""
    context["active_workbook_edit_token"] = (
        str(active_workbook.edit_token) if active_workbook and workbook_editable else ""
    )
    _set_explore_modal_flag(request, context, "explore")
    context["current_page"] = "explore"
    context.update(
        _build_seo_context(
            request,
            "explore",
            canonical_path=reverse("workbench:explore"),
        )
    )
    return render(request, "workbench/explore.html", context)


@require_http_methods(["GET", "POST"])
def filter_page(request: HttpRequest) -> HttpResponse:
    if request.method == "POST":
        init_state(request.session)
        _ensure_workspace_reports_loaded(request)
        response = _handle_post_action(request)
        if response is not None:
            return response
    return redirect("workbench:explore")


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
    context.update(
        _build_seo_context(
            request,
            "themes",
            canonical_path=reverse("workbench:themes"),
        )
    )
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
    context.update(
        _build_seo_context(
            request,
            "extract",
            canonical_path=reverse("workbench:extract"),
        )
    )
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
    context.update(
        _build_seo_context(
            request,
            "settings",
            canonical_path=reverse("workbench:settings"),
        )
    )
    return render(request, "workbench/settings.html", context)


def _render_for_coders_page(request: HttpRequest, doc_path: str) -> HttpResponse:
    normalised_doc_path = _normalise_docs_path(doc_path)
    if normalised_doc_path in DOCS_REMOVED_PATHS:
        return redirect(_docs_url_for_path(""))
    docs_error = ""
    try:
        payload = _load_docs_page_payload(normalised_doc_path)
    except (FileNotFoundError, ValueError) as exc:
        if normalised_doc_path:
            try:
                payload = _load_docs_page_payload("")
                docs_error = (
                    f'The page "{normalised_doc_path}" could not be loaded, so the documentation home page is shown.'
                )
            except (FileNotFoundError, ValueError):
                raise Http404(str(exc)) from exc
        else:
            raise Http404(str(exc)) from exc

    context = {
        "current_page": "for_coders",
        "docs_page_title": payload["page_title"],
        "docs_page_path": payload["doc_path"],
        "docs_article_html": payload["article_html"],
        "docs_nav_items": payload["nav_items"],
        "docs_toc_items": payload["toc_items"],
        "docs_error": docs_error,
    }
    context.update(
        _build_seo_context(
            request,
            "for_coders",
            canonical_path=_docs_url_for_path(payload["doc_path"]),
            title_override=f'{payload["page_title"]} | For Coders | PFD Toolkit',
        )
    )
    return render(request, "workbench/for_coders.html", context)


@require_http_methods(["GET"])
def for_coders(request: HttpRequest) -> HttpResponse:
    requested_doc_path = _normalise_docs_path(request.GET.get("doc", ""))
    return _render_for_coders_page(request, requested_doc_path)


@require_http_methods(["GET"])
def for_coders_page(request: HttpRequest, doc_path: str) -> HttpResponse:
    return redirect(_docs_url_for_path(_normalise_docs_path(doc_path)))


@require_http_methods(["GET"])
def for_coders_site_file(request: HttpRequest, file_path: str) -> HttpResponse:
    cleaned = str(file_path or "").lstrip("/")
    if not cleaned:
        raise Http404("Missing docs file path.")

    docs_root = DOCS_SOURCE_ROOT.resolve()
    candidate = (DOCS_SOURCE_ROOT / cleaned).resolve()
    if not _path_within_root(candidate, docs_root) or not candidate.is_file():
        raise Http404("Docs file was not found.")

    content_type, _ = mimetypes.guess_type(candidate.name)
    return FileResponse(candidate.open("rb"), content_type=content_type or "application/octet-stream")


@require_http_methods(["GET"])
def dataset_panel(request: HttpRequest) -> HttpResponse:
    init_state(request.session)
    _ensure_workspace_reports_loaded(request)
    context = _build_context(request)
    response = render(request, "workbench/_dataset_table_section.html", context)
    response["X-Robots-Tag"] = "noindex, nofollow"
    return response
