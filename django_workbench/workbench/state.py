"""Session-backed workspace state helpers for the Django workbench."""
from __future__ import annotations

import copy
import json
import tempfile
from collections import OrderedDict
from datetime import date
from io import StringIO
from pathlib import Path
from typing import Any, Dict, Iterable, Optional
from uuid import uuid4

import pandas as pd

REPRO_SCRIPT_KEY = "repro_script_lines"
REPRO_ACTION_COUNTS_KEY = "repro_action_counts"
LLM_SIGNATURE_KEY = "llm_config_signature"
FRAME_PAYLOAD_PREFIX = "framefile:"
FRAME_CACHE_DIR = Path(tempfile.gettempdir()) / "pfd_workbench_frames"
FRAME_MEMORY_CACHE: "OrderedDict[str, pd.DataFrame]" = OrderedDict()
FRAME_MEMORY_CACHE_MAX = 6

SNAPSHOT_KEYS = [
    "reports_df",
    "reports_df_initial",
    "excluded_reports_df",
    "reports_df_modified",
    "screener_result",
    "extractor_result",
    "summary_result",
    "theme_model_schema",
    "theme_summary_table",
    "seed_topics_last",
    "feature_grid",
    "active_action",
    "preview_state",
    REPRO_SCRIPT_KEY,
    REPRO_ACTION_COUNTS_KEY,
    LLM_SIGNATURE_KEY,
]


def _initial_repro_script_lines() -> list[str]:
    return [
        "# -----------------------------------------------------------------------------",
        "# Reproducible workspace script",
        "# This script contains the Python code for the actions you performed in",
        "# PFD Toolkit Workbench.",
        "# To replay locally, first install the Toolkit:",
        "#     pip install pfd_toolkit",
        "# Workbench does not save your API key. Add it manually before running.",
        "",
        "from pfd_toolkit import load_reports",
        "from pfd_toolkit import LLM",
        "from pfd_toolkit import Screener",
        "from pfd_toolkit import Extractor",
        "",
    ]


def init_state(session: dict[str, Any]) -> None:
    """Ensure the workspace keys exist in session."""

    defaults: Dict[str, Any] = {
        "reports_df": None,
        "reports_df_initial": None,
        "excluded_reports_df": None,
        "history": [],
        "redo_history": [],
        "active_action": None,
        "preview_state": None,
        "reports_df_modified": False,
        "screener_result": None,
        "extractor_result": None,
        "summary_result": None,
        "theme_model_schema": None,
        "theme_summary_table": None,
        "seed_topics_last": None,
        "feature_grid": None,
        "openrouter_api_key": "",
        "openrouter_base_url": "https://openrouter.ai/api/v1",
        "openai_api_key": "",
        "openai_base_url": "",
        "provider_override": "OpenAI",
        "model_name": "gpt-4.1-mini",
        "max_parallel_workers": 8,
        "report_start_date": date(2013, 1, 1).isoformat(),
        "report_end_date": date.today().isoformat(),
        "report_limit": None,
        REPRO_SCRIPT_KEY: _initial_repro_script_lines(),
        REPRO_ACTION_COUNTS_KEY: {},
        LLM_SIGNATURE_KEY: None,
        "theme_emoji_cache": {},
    }
    for key, value in defaults.items():
        session.setdefault(key, value)


def dataframe_to_payload(df: Optional[pd.DataFrame]) -> Optional[str]:
    if df is None:
        return None
    if not isinstance(df, pd.DataFrame):
        try:
            df = pd.DataFrame(df)
        except ValueError:
            return None
    FRAME_CACHE_DIR.mkdir(parents=True, exist_ok=True)
    frame_path = FRAME_CACHE_DIR / f"{uuid4().hex}.pkl"
    df.to_pickle(frame_path)
    # Keep the cache bounded to avoid unbounded growth in /tmp.
    cache_files = sorted(FRAME_CACHE_DIR.glob("*.pkl"), key=lambda item: item.stat().st_mtime, reverse=True)
    for stale_file in cache_files[800:]:
        try:
            stale_file.unlink()
        except OSError:
            continue
    return f"{FRAME_PAYLOAD_PREFIX}{frame_path}"


def dataframe_from_payload(payload: Any) -> pd.DataFrame:
    if not payload or not isinstance(payload, str):
        return pd.DataFrame()
    if payload.startswith(FRAME_PAYLOAD_PREFIX):
        frame_key = payload
        cached = FRAME_MEMORY_CACHE.get(frame_key)
        if isinstance(cached, pd.DataFrame):
            FRAME_MEMORY_CACHE.move_to_end(frame_key)
            return cached

        frame_path = Path(payload[len(FRAME_PAYLOAD_PREFIX):])
        if not frame_path.exists():
            return pd.DataFrame()
        try:
            df = pd.read_pickle(frame_path)
            FRAME_MEMORY_CACHE[frame_key] = df
            FRAME_MEMORY_CACHE.move_to_end(frame_key)
            while len(FRAME_MEMORY_CACHE) > FRAME_MEMORY_CACHE_MAX:
                FRAME_MEMORY_CACHE.popitem(last=False)
            return df
        except Exception:
            return pd.DataFrame()
    try:
        return pd.read_json(StringIO(payload), orient="split")
    except ValueError:
        return pd.DataFrame()


def set_dataframe(session: dict[str, Any], key: str, df: Optional[pd.DataFrame]) -> None:
    session[key] = dataframe_to_payload(df)


def get_dataframe(session: dict[str, Any], key: str) -> pd.DataFrame:
    return dataframe_from_payload(session.get(key))


def reset_repro_tracking(session: dict[str, Any]) -> None:
    session[REPRO_SCRIPT_KEY] = _initial_repro_script_lines()
    session[REPRO_ACTION_COUNTS_KEY] = {}
    session[LLM_SIGNATURE_KEY] = None


def ensure_repro_script(session: dict[str, Any], *, reset: bool = False) -> list[str]:
    lines = session.get(REPRO_SCRIPT_KEY)
    if reset or not isinstance(lines, list):
        lines = _initial_repro_script_lines()
        session[REPRO_SCRIPT_KEY] = lines
    return lines


def get_repro_script_text(session: dict[str, Any]) -> str:
    lines = ensure_repro_script(session)
    return "\n".join(lines) + "\n"


def format_call(prefix: str, kwargs: Dict[str, Any], raw_parameters: Iterable[str] = ()) -> str:
    lines = [f"{prefix}("]
    raw = set(raw_parameters)
    for key, value in kwargs.items():
        rendered = value if key in raw else repr(value)
        lines.append(f"    {key}={rendered},")
    lines.append(")")
    return "\n".join(lines)


def record_repro_action(
    session: dict[str, Any], action_key: str, base_comment: str, code_block: str
) -> None:
    lines = ensure_repro_script(session)
    counters = session.setdefault(REPRO_ACTION_COUNTS_KEY, {})
    count = int(counters.get(action_key, 0)) + 1
    counters[action_key] = count
    comment = base_comment if count == 1 else f"{base_comment} (run {count})"

    if lines and lines[-1] != "":
        lines.append("")
    lines.append(f"# {comment}")
    lines.extend(code_block.splitlines())


def snapshot_state(session: dict[str, Any]) -> Dict[str, Any]:
    snapshot: Dict[str, Any] = {}
    for key in SNAPSHOT_KEYS:
        snapshot[key] = copy.deepcopy(session.get(key))
    return snapshot


def restore_state(session: dict[str, Any], snapshot: Dict[str, Any]) -> None:
    for key, value in snapshot.items():
        session[key] = copy.deepcopy(value)


def push_history_snapshot(session: dict[str, Any]) -> None:
    history = list(session.get("history", []))
    history.append(snapshot_state(session))
    if len(history) > 10:
        history = history[-10:]
    session["history"] = history
    session["redo_history"] = []


def undo_last_change(session: dict[str, Any]) -> bool:
    history = list(session.get("history", []))
    if not history:
        return False

    redo_history = list(session.get("redo_history", []))
    redo_history.append(snapshot_state(session))
    if len(redo_history) > 10:
        redo_history = redo_history[-10:]

    snapshot = history.pop()
    session["history"] = history
    session["redo_history"] = redo_history
    restore_state(session, snapshot)
    session["active_action"] = None
    return True


def redo_last_change(session: dict[str, Any]) -> bool:
    redo_history = list(session.get("redo_history", []))
    if not redo_history:
        return False

    history = list(session.get("history", []))
    history.append(snapshot_state(session))
    if len(history) > 10:
        history = history[-10:]

    snapshot = redo_history.pop()
    session["history"] = history
    session["redo_history"] = redo_history
    restore_state(session, snapshot)
    session["active_action"] = None
    return True


def clear_preview_state(session: dict[str, Any]) -> None:
    session["preview_state"] = None


def clear_outputs_for_new_dataset(session: dict[str, Any]) -> None:
    session["reports_df_modified"] = False
    session["excluded_reports_df"] = None
    session["screener_result"] = None
    session["extractor_result"] = None
    session["summary_result"] = None
    session["theme_model_schema"] = None
    session["theme_summary_table"] = None
    session["seed_topics_last"] = None
    session["active_action"] = None
    clear_preview_state(session)


def clear_outputs_for_modified_dataset(session: dict[str, Any]) -> None:
    session["reports_df_modified"] = True
    session["extractor_result"] = None
    session["summary_result"] = None
    session["theme_model_schema"] = None
    session["theme_summary_table"] = None
    session["seed_topics_last"] = None
    clear_preview_state(session)


def parse_optional_non_negative_int(value: str, field_name: str) -> Optional[int]:
    value = (value or "").strip()
    if not value:
        return None
    try:
        parsed = int(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an integer if provided.") from exc
    if parsed < 0:
        raise ValueError(f"{field_name} cannot be negative.")
    return parsed


def format_theme_description(raw: Any) -> str:
    if isinstance(raw, dict):
        for key in ("description", "details", "detail", "text"):
            if key in raw:
                return format_theme_description(raw[key])
        return json.dumps(raw, ensure_ascii=False)

    if isinstance(raw, list):
        return ", ".join(filter(None, (format_theme_description(item) for item in raw)))

    if isinstance(raw, str):
        stripped = raw.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                parsed = json.loads(stripped)
            except json.JSONDecodeError:
                return raw
            if isinstance(parsed, (dict, list)):
                return format_theme_description(parsed)
        return raw

    if raw is None:
        return ""

    return str(raw)


def build_theme_summary_table(
    extracted_reports: pd.DataFrame, theme_schema: Optional[Dict[str, Any]]
) -> pd.DataFrame:
    if extracted_reports.empty or not isinstance(theme_schema, dict):
        return pd.DataFrame(columns=["Theme", "Description", "Count", "%"])

    properties = theme_schema.get("properties")
    if not isinstance(properties, dict):
        return pd.DataFrame(columns=["Theme", "Description", "Count", "%"])

    rows = []
    total_reports = len(extracted_reports)

    def _is_true(value: Any) -> bool:
        if pd.isna(value):
            return False
        if isinstance(value, bool):
            return value
        if isinstance(value, (int, float)):
            return value == 1
        return str(value).strip().lower() in {"true", "1", "yes"}

    for theme_key, config in properties.items():
        if theme_key not in extracted_reports.columns:
            continue
        count = int(extracted_reports[theme_key].map(_is_true).sum())
        percentage = round((count / total_reports * 100.0) if total_reports else 0.0, 1)
        display_name = (
            config.get("title") if isinstance(config, dict) else None
        ) or theme_key.replace("_", " ").title()
        description = (
            format_theme_description(config.get("description"))
            if isinstance(config, dict)
            else ""
        )
        rows.append(
            {
                "Theme": display_name,
                "Description": description,
                "Count": count,
                "%": percentage,
            }
        )

    if not rows:
        return pd.DataFrame(columns=["Theme", "Description", "Count", "%"])

    return pd.DataFrame(rows).sort_values(by=["Count", "Theme"], ascending=[False, True])


def has_api_key(session: dict[str, Any]) -> bool:
    provider = session.get("provider_override", "OpenAI")
    if provider == "OpenRouter":
        return bool((session.get("openrouter_api_key") or "").strip())
    return bool((session.get("openai_api_key") or "").strip())


def workspace_has_activity(session: dict[str, Any]) -> bool:
    if session.get("reports_df_modified", False):
        return True
    if len(session.get("history", [])):
        return True

    for key in (
        "excluded_reports_df",
        "screener_result",
        "extractor_result",
        "summary_result",
        "theme_model_schema",
        "theme_summary_table",
        "seed_topics_last",
        "feature_grid",
        "preview_state",
    ):
        value = session.get(key)
        if value is None:
            continue
        if isinstance(value, (list, tuple, set, dict)) and not value:
            continue
        if isinstance(value, str) and not value.strip():
            continue
        return True

    return False
