"""Interactive Streamlit dashboard for the PFD Toolkit API."""
from __future__ import annotations

import ast
import copy
import json
import sys
import zipfile
from html import escape
from datetime import date
from io import BytesIO
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

import pandas as pd
import streamlit as st
from streamlit.delta_generator import DeltaGenerator
from pydantic import Field, create_model

ROOT_DIR = Path(__file__).resolve().parent
SRC_DIR = ROOT_DIR / "src"
if SRC_DIR.exists():
    src_str = str(SRC_DIR)
    if src_str not in sys.path:
        sys.path.insert(0, src_str)

from pfd_toolkit.extractor import Extractor
from pfd_toolkit.loader import load_reports
from pfd_toolkit.llm import LLM
from pfd_toolkit.screener import Screener

LOGO_PATH = Path("docs/assets/badge-circle.png")
OPENROUTER_API_BASE = "https://openrouter.ai/api/v1"

DEFAULT_THEME_EMOJI = "💡"
THEME_EMOJI_BANNED_TOKENS = {
    "🩸",
    "🩹",
    "🔪",
    "🗡",
    "🗡️",
    "🪓",
    "⚔",
    "⚔️",
    "💣",
    "🧨",
    "🔫",
    "☠",
    "☠️",
    "💀",
    "🧟",
    "⚰️",
    "🪦",
    "🧛", "🧛‍♂️", "🧛‍♀️",
    "🥀",
    "⚱️",
    "🕷️", "🕸️",
    "🕊️",
    "🕯️",
}

ThemeEmojiModel = create_model(
    "ThemeEmojiModel",
    emoji=(
        str,
        Field(
            min_length=1,
            max_length=8,
            pattern=r"^\S+$",
            description="A single emoji with no surrounding text or whitespace.",
            json_schema_extra={"examples": ["🗣️"]},
        ),
    ),
)


# Keys for the reproducible script caching feature
REPRO_SCRIPT_KEY = "repro_script_lines"
REPRO_ACTION_COUNTS_KEY = "repro_action_counts"
LLM_SIGNATURE_KEY = "llm_config_signature"


# ---------------------------------------------------------------------------
# Reproducible script helpers
# ---------------------------------------------------------------------------

def _initial_repro_script_lines() -> List[str]:
    """Return the initial lines for the reproducible workspace script."""

    return [
        "# -----------------------------------------------------------------------------",
        "# Reproducible workspace script",
        "# This script contains the Python code for the various actions you performed in",
        "# PFD Toolkit Workbench.",
        "# To replay them locally, first install the Toolkit before running the script:",
        "#     pip install pfd_toolkit",
        "# Workbench does not save your API key. Make sure to add this to the script."
        ""
        "from pfd_toolkit import load_reports",
        "from pfd_toolkit import LLM",
        "from pfd_toolkit import Screener",
        "from pfd_toolkit import Extractor",
        "",
    ]


def _ensure_repro_script(reset: bool = False) -> List[str]:
    """Ensure the reproducible script list exists and optionally reset it."""

    lines = st.session_state.get(REPRO_SCRIPT_KEY)
    if reset or not isinstance(lines, list):
        lines = _initial_repro_script_lines()
        st.session_state[REPRO_SCRIPT_KEY] = lines
    return lines


def _reset_repro_tracking() -> None:
    """Reset the cached script and action counters."""

    _ensure_repro_script(reset=True)
    st.session_state[REPRO_ACTION_COUNTS_KEY] = {}
    st.session_state[LLM_SIGNATURE_KEY] = None


def _format_call(prefix: str, kwargs: Dict[str, Any], raw_parameters: Iterable[str] = ()) -> str:
    """Return a formatted multi-line call string for the reproducible script."""

    lines = [f"{prefix}("]
    for key, value in kwargs.items():
        if key in raw_parameters:
            rendered = value
        else:
            rendered = repr(value)
        lines.append(f"    {key}={rendered},")
    lines.append(")")
    return "\n".join(lines)


def _record_repro_action(action_key: str, base_comment: str, code_block: str) -> None:
    """Append an action to the reproducible script with run counters."""

    lines = _ensure_repro_script()
    counters: Dict[str, int] = st.session_state.setdefault(
        REPRO_ACTION_COUNTS_KEY, {}
    )
    count = counters.get(action_key, 0) + 1
    counters[action_key] = count
    comment = base_comment if count == 1 else f"{base_comment} (run {count})"

    if lines and lines[-1] != "":
        lines.append("")
    lines.append(f"# {comment}")
    lines.extend(code_block.splitlines())


def _get_repro_script_text() -> str:
    """Return the reproducible script as a single text blob."""

    lines = _ensure_repro_script()
    return "\n".join(lines) + "\n"


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _init_session_state() -> None:
    """Ensure default keys exist in ``st.session_state``."""
    defaults = {
        "reports_df": pd.DataFrame(),
        "reports_df_initial": None,
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
        "llm_client": None,
        "extractor": None,
        "extractor_source_signature": None,
        "seed_topics_last": None,
        "feature_grid": None,
        FLASH_MESSAGE_KEY: None,
        "openrouter_api_key": "",
        "openrouter_base_url": OPENROUTER_API_BASE,
        "openai_api_key": "",
        "openai_base_url": "",
        "provider_override": "OpenAI",
        "provider_override_select": "OpenAI",
        REPORTS_LOADING_FLAG_KEY: False,
        REPRO_SCRIPT_KEY: _initial_repro_script_lines(),
        REPRO_ACTION_COUNTS_KEY: {},
        LLM_SIGNATURE_KEY: None,
        "theme_emoji_cache": {},
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _styled_metric(label: str, value: Any) -> None:
    """Render a metric with consistent styling."""

    st.markdown(
        f"""
        <div class="metric-card">
            <span class="metric-label">{label}</span>
            <span class="metric-value">{value}</span>
        </div>
        """,
        unsafe_allow_html=True,
    )


class LoadingIndicator:
    """Utility to display the animated loading indicator."""

    def __init__(self, placeholder: DeltaGenerator, message: str = "Loading…") -> None:
        self._placeholder = placeholder
        self.update(message)

    def update(self, message: str) -> None:
        """Update the indicator with ``message``."""

        safe_message = escape(message)
        self._placeholder.markdown(
            f"""
            <div class="loading-indicator">
                <div class="loading-indicator__dots">
                    <span class="loading-indicator__dot"></span>
                    <span class="loading-indicator__dot"></span>
                    <span class="loading-indicator__dot"></span>
                </div>
                <span class="loading-indicator__label">{safe_message}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    def close(self) -> None:
        """Remove the indicator from the UI."""

        self._placeholder.empty()


FLASH_MESSAGE_KEY = "flash_message"

REPORTS_LOADING_OVERLAY_KEY = "_reports_loading_overlay_placeholder"
REPORTS_LOADING_FLAG_KEY = "reports_loading_active"


def _get_loading_overlay_placeholder() -> DeltaGenerator:
    """Return the placeholder used for the reports loading scrim."""

    placeholder: Optional[DeltaGenerator] = st.session_state.get(REPORTS_LOADING_OVERLAY_KEY)
    if placeholder is None:
        placeholder = st.empty()
        st.session_state[REPORTS_LOADING_OVERLAY_KEY] = placeholder
    return placeholder


def _ensure_loading_overlay_placeholder() -> DeltaGenerator:
    """Ensure the loading overlay placeholder exists and is reset when idle."""

    placeholder = _get_loading_overlay_placeholder()
    if not st.session_state.get(REPORTS_LOADING_FLAG_KEY, False):
        placeholder.empty()
    return placeholder


def _show_reports_loading_overlay(message: str = "Loading reports…") -> None:
    """Display the blur scrim while reports are loading."""

    placeholder = _get_loading_overlay_placeholder()
    st.session_state[REPORTS_LOADING_FLAG_KEY] = True
    safe_message = escape(message)
    placeholder.markdown(
        f"""
        <div class="reports-loading-scrim reports-loading-scrim--active" role="status" aria-live="assertive">
            <div class="reports-loading-scrim__content">
                <span class="reports-loading-scrim__halo"></span>
                <span class="reports-loading-scrim__ring"></span>
                <span class="reports-loading-scrim__label">{safe_message}</span>
            </div>
        </div>
        """,
        unsafe_allow_html=True,
    )


def _hide_reports_loading_overlay() -> None:
    """Remove the reports loading scrim from view."""

    placeholder = st.session_state.get(REPORTS_LOADING_OVERLAY_KEY)
    if placeholder is not None:
        placeholder.empty()
    st.session_state[REPORTS_LOADING_FLAG_KEY] = False


SNAPSHOT_KEYS = [
    "reports_df",
    "screener_result",
    "extractor_result",
    "summary_result",
    "theme_model_schema",
    "theme_summary_table",
    "seed_topics_last",
    "extractor",
    "extractor_source_signature",
    "feature_grid",
    "active_action",
    "preview_state",
]


def _deepcopy_value(value: Any) -> Any:
    """Return a deep copy of ``value`` preserving DataFrame contents."""

    if isinstance(value, pd.DataFrame):
        return value.copy(deep=True)
    try:
        return copy.deepcopy(value)
    except Exception:
        return value


def snapshot_state() -> Dict[str, Any]:
    """Capture a snapshot of the interactive state for undo support."""

    snapshot: Dict[str, Any] = {}
    for key in SNAPSHOT_KEYS:
        snapshot[key] = _deepcopy_value(st.session_state.get(key))
    return snapshot


def restore_state(snapshot: Dict[str, Any]) -> None:
    """Restore Streamlit session state from ``snapshot``."""

    for key, value in snapshot.items():
        st.session_state[key] = _deepcopy_value(value)


def _coerce_dataframe(value: Any) -> Optional[pd.DataFrame]:
    """Return ``value`` as a DataFrame when possible."""

    if isinstance(value, pd.DataFrame):
        return value.copy(deep=True)
    if isinstance(value, (list, tuple)):
        try:
            return pd.DataFrame(value)
        except ValueError:
            return None
    if isinstance(value, dict):
        try:
            return pd.DataFrame(value)
        except ValueError:
            return None
    return None


def push_history_snapshot() -> None:
    """Push the current state onto the undo history (max depth 10)."""

    history = list(st.session_state.get("history", []))
    history.append(snapshot_state())
    if len(history) > 10:
        history = history[-10:]
    st.session_state["history"] = history
    st.session_state["redo_history"] = []


def clear_preview_state() -> None:
    """Remove any pending preview artefacts."""

    st.session_state["preview_state"] = None


def _parse_optional_non_negative_int(value: str, field_name: str) -> Optional[int]:
    """Convert ``value`` to ``int`` ensuring it is not negative."""

    value = value.strip()
    if not value:
        return None
    try:
        parsed_value = int(value)
    except ValueError as exc:
        raise ValueError(f"{field_name} must be an integer if provided.") from exc

    if parsed_value < 0:
        raise ValueError(f"{field_name} cannot be negative.")

    return parsed_value


def _format_theme_description(raw: Any) -> str:
    """Return a user-friendly description extracted from ``raw``."""

    if isinstance(raw, dict):
        for key in ("description", "details", "detail", "text"):
            if key in raw:
                return _format_theme_description(raw[key])
        return json.dumps(raw, ensure_ascii=False)

    if isinstance(raw, list):
        return ", ".join(filter(None, (_format_theme_description(item) for item in raw)))

    if isinstance(raw, str):
        stripped = raw.strip()
        if stripped.startswith("{") and stripped.endswith("}"):
            try:
                parsed = ast.literal_eval(stripped)
            except (SyntaxError, ValueError):
                return raw
            else:
                if isinstance(parsed, (dict, list)):
                    return _format_theme_description(parsed)
        return raw

    if raw is None:
        return ""

    return str(raw)


def _build_feature_model(schema_text: str):
    """Return a ``pydantic`` model built from the user supplied JSON."""
    schema_text = schema_text.strip()
    if not schema_text:
        raise ValueError("Please provide a JSON schema describing your features.")

    try:
        schema_dict = json.loads(schema_text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Schema must be valid JSON: {exc}") from exc

    if not isinstance(schema_dict, dict):
        raise ValueError("Schema must be a JSON object mapping feature names to metadata.")

    type_mapping = {
        "string": str,
        "str": str,
        "text": str,
        "bool": bool,
        "boolean": bool,
        "int": int,
        "integer": int,
        "float": float,
        "number": float,
    }

    fields: Dict[str, Tuple[type, Field]] = {}
    for feature, config in schema_dict.items():
        if not isinstance(config, dict):
            raise ValueError("Each feature must map to an object containing at least a 'type'.")

        feature_type = str(config.get("type", "string")).lower()
        python_type = type_mapping.get(feature_type)
        if python_type is None:
            raise ValueError(
                f"Unsupported type '{feature_type}' for feature '{feature}'.\n"
                "Use one of: string, bool, int, float."
            )
        description = str(config.get("description", "")).strip() or ""
        required = bool(config.get("required", True))
        default_value = config.get("default") if not required else ...
        fields[feature] = (
            python_type,
            Field(
                default_value,
                description=description,
                title=config.get("title", feature.replace("_", " ").title()),
            ),
        )

    model_name = schema_dict.get("__model_name__", "UserFeatures")
    return create_model(model_name, **fields)


def _build_feature_model_from_grid(feature_rows: pd.DataFrame):
    """Create a ``pydantic`` model from the feature editor grid."""

    if feature_rows.empty:
        raise ValueError("Please add at least one feature to extract.")

    type_mapping: Dict[str, type] = {
        "Free text": str,
        "Conditional (True/False)": bool,
        "Whole number": int,
        "Decimal number": float,
    }

    fields: Dict[str, Tuple[type, Field]] = {}
    for row in feature_rows.to_dict(orient="records"):
        raw_name = str(row.get("Field name", "")).strip()
        description = str(row.get("Description", "")).strip()
        type_label = str(row.get("Type", "")).strip()

        if not raw_name:
            continue

        if type_label not in type_mapping:
            raise ValueError(f"Select a valid type for '{raw_name}'.")

        if raw_name in fields:
            raise ValueError(f"Duplicate feature name '{raw_name}'. Each feature must be unique.")

        python_type = type_mapping[type_label]
        fields[raw_name] = (
            python_type,
            Field(
                ...,
                description=description,
                title=raw_name.replace("_", " ").title(),
            ),
        )

    if not fields:
        raise ValueError("Please provide a name and type for each feature you want to extract.")

    return create_model("SelectedFeatures", **fields)


def _display_dataframe(df: pd.DataFrame, caption: str) -> None:
    """Render a ``DataFrame`` with a consistent caption."""
    if df is None or df.empty:
        st.info("No rows to display yet. Load or generate data to see results here.")
        return
    st.markdown(f"<div class='section-caption'>{caption}</div>", unsafe_allow_html=True)
    st.dataframe(df, width="stretch", hide_index=True)


def _render_reports_overview(
    caption: str = "Loaded Prevention of Future Death reports", *, key_suffix: str = ""
) -> None:
    """Display the active reports with an optional revert button."""

    reports_df: pd.DataFrame = st.session_state.get("reports_df", pd.DataFrame())
    initial_df: Optional[pd.DataFrame] = st.session_state.get("reports_df_initial")

    if initial_df is None and not reports_df.empty:
        initial_df = reports_df.copy(deep=True)
        st.session_state["reports_df_initial"] = initial_df

    modified = False
    if initial_df is not None:
        if reports_df.empty and not initial_df.empty:
            modified = True
        elif not reports_df.empty:
            try:
                modified = not reports_df.equals(initial_df)
            except Exception:
                modified = True

    st.session_state["reports_df_modified"] = modified

    if modified:
        if st.button(
            "Revert changes",
            key=f"revert_reports_{key_suffix}" if key_suffix else "revert_reports",
            width="stretch",
        ):
            st.session_state["reports_df"] = initial_df.copy()
            st.session_state["reports_df_modified"] = False
            st.session_state["screener_result"] = None
            st.session_state["summary_result"] = None
            st.session_state["extractor_result"] = None
            st.session_state["theme_model_schema"] = None
            st.session_state["theme_summary_table"] = None
            st.session_state["seed_topics_last"] = None
            st.session_state["extractor"] = None
            st.session_state["extractor_source_signature"] = None
            reports_df = st.session_state["reports_df"]

    _display_dataframe(reports_df, caption)


# ---------------------------------------------------------------------------
# Sidebar controls
# ---------------------------------------------------------------------------

def _build_sidebar() -> None:
    st.sidebar.image(str(LOGO_PATH), width=120)
    st.sidebar.markdown(
        "<h2 class='sidebar-title'>PFD Toolkit Workbench <span class='sidebar-beta'>Beta</span></h2>",
        unsafe_allow_html=True,
    )
    st.sidebar.caption(
        "Configure your language model and reporting window, then load the source data before running screeners or extractors."
    )

    st.sidebar.markdown("### API credentials")

    provider_options = ["OpenAI", "OpenRouter"]
    current_provider = st.session_state.get("provider_override_select", st.session_state.get("provider_override", "OpenAI"))
    st.session_state["provider_override"] = current_provider

    if current_provider == "OpenAI":
        openai_api_key = st.sidebar.text_input(
            "OpenAI API key",
            value=st.session_state.get("openai_api_key", ""),
            type="password",
            help="Required to access OpenAI models.",
        )
        st.session_state["openai_api_key"] = openai_api_key
    else:
        openrouter_api_key = st.sidebar.text_input(
            "OpenRouter API key",
            value=st.session_state.get("openrouter_api_key", ""),
            type="password",
            help="Paste your OpenRouter key to use their API.",
        )
        st.session_state["openrouter_api_key"] = openrouter_api_key

    with st.sidebar.expander("Advanced model provider options", expanded=False):
        provider_index = (
            provider_options.index(current_provider)
            if current_provider in provider_options
            else 0
        )
        provider_choice = st.selectbox(
            "Model provider",
            provider_options,
            index=provider_index,
            key="provider_override_select",
        )
        st.session_state["provider_override"] = provider_choice

        if provider_choice == "OpenAI":
            openai_base_url = st.text_input(
                "Custom OpenAI base URL (optional)",
                value=st.session_state.get("openai_base_url", ""),
                help="Leave blank to use the official OpenAI endpoint.",
            )
            st.session_state["openai_base_url"] = openai_base_url
        else:
            openrouter_base_url = st.text_input(
                "OpenRouter API base",
                value=st.session_state.get("openrouter_base_url", OPENROUTER_API_BASE),
                help="Override when using an OpenRouter-compatible proxy.",
            )
            st.session_state["openrouter_base_url"] = openrouter_base_url

    provider = st.session_state.get("provider_override", "OpenAI")

    if provider == "OpenRouter":
        base_url = (st.session_state.get("openrouter_base_url") or OPENROUTER_API_BASE).strip()
        api_key = (st.session_state.get("openrouter_api_key") or "").strip()
    else:
        base_url = (st.session_state.get("openai_base_url") or "").strip() or None
        api_key = (st.session_state.get("openai_api_key") or "").strip()

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model configuration")
    model_display_options = [
        "GPT 4.1 Mini (+speed, -accuracy)",
        "GPT 4.1 (+accuracy, -speed)",
    ]
    model_value_map = {
        "GPT 4.1 Mini (+speed, -accuracy)": "gpt-4.1-mini",
        "GPT 4.1 (+accuracy, -speed)": "gpt-4.1",
    }
    current_model_value = st.session_state.get("model_name", "gpt-4.1-mini")
    current_display = next(
        (label for label, value in model_value_map.items() if value == current_model_value),
        model_display_options[0],
    )
    model_display = st.sidebar.selectbox(
        "Choose your AI model",
        model_display_options,
        index=model_display_options.index(current_display)
        if current_display in model_display_options
        else 0,
    )
    model_name = model_value_map[model_display]
    st.session_state["model_name"] = model_name
    with st.sidebar.expander("Advanced options"):
        max_workers = st.number_input(
            "Max parallel workers",
            min_value=1,
            max_value=32,
            value=int(st.session_state.get("max_parallel_workers", 8)),
            key="max_parallel_workers_input",
        )
    st.session_state["max_parallel_workers"] = int(max_workers)
    validation_attempts = 2

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Report window")
    default_start = date(2013, 1, 1)
    start_date = st.sidebar.date_input(
        "Report start date",
        value=st.session_state.get("report_start_date", default_start),
        min_value=date(2000, 1, 1),
        max_value=date.today(),
    )
    st.session_state["report_start_date"] = start_date
    end_date = st.sidebar.date_input(
        "Report end date",
        value=st.session_state.get("report_end_date", date.today()),
        min_value=start_date,
        max_value=date.today(),
    )
    st.session_state["report_end_date"] = end_date
    if end_date < start_date:
        st.sidebar.error("End date must be on or after the start date.")

    n_reports_raw = st.sidebar.text_input(
        "Limit number of recent reports (optional)",
        value="",
        placeholder="e.g. 1000",
        help="Leave blank to load all matching reports.",
    )
    n_reports: Optional[int] = None
    if n_reports_raw.strip():
        try:
            n_reports = int(n_reports_raw.strip())
            if n_reports < 0:
                raise ValueError
        except ValueError:
            st.sidebar.error("Please enter a whole number for the report limit.")
            n_reports = None
    refresh = st.sidebar.checkbox(
        "Force refresh from remote dataset", value=True, help="Disable to reuse the cached CSV if available."
    )

    load_button = st.sidebar.button("Load in reports", width="stretch")

    if load_button:
        if n_reports_raw.strip() and n_reports is None:
            st.sidebar.error("Enter a valid integer before loading reports.")
            return
        if end_date < start_date:
            st.sidebar.error("Fix the report date range before loading reports.")
            return
        try:
            _show_reports_loading_overlay()
            with st.spinner("Fetching reports from server..."):
                df = load_reports(
                    start_date=start_date.isoformat(),
                    end_date=end_date.isoformat(),
                    n_reports=n_reports,
                    refresh=refresh,
                )
            st.session_state["reports_df"] = df.copy(deep=True)
            st.session_state["reports_df_initial"] = df.copy(deep=True)
            st.session_state["reports_df_modified"] = False
            st.session_state["screener_result"] = None
            st.session_state["extractor_result"] = None
            st.session_state["summary_result"] = None
            st.session_state["theme_model_schema"] = None
            st.session_state["theme_summary_table"] = None
            st.session_state["extractor"] = None
            st.session_state["extractor_source_signature"] = None
            st.session_state["seed_topics_last"] = None
            st.session_state["history"] = []
            st.session_state["redo_history"] = []
            st.session_state["active_action"] = None
            clear_preview_state()
            _reset_repro_tracking()
            load_kwargs = {
                "start_date": start_date.isoformat(),
                "end_date": end_date.isoformat(),
                "n_reports": n_reports,
                "refresh": refresh,
            }
            _record_repro_action(
                "load_reports",
                "Load in reports",
                _format_call("reports_df = load_reports", load_kwargs),
            )
            st.success(f"Loaded {len(df)} reports into the workspace.")
        except Exception as exc:  # pragma: no cover - UI feedback
            st.error(f"Could not load reports: {exc}")
        finally:
            _hide_reports_loading_overlay()

    if api_key:
        llm_kwargs: Dict[str, Any] = {
            "api_key": api_key.strip(),
            "model": model_name.strip() or "gpt-4.1",
            "max_workers": int(max_workers),
            "temperature": 0.0,
            "validation_attempts": int(validation_attempts),
            "seed": 123,
            "timeout": 30,
        }
        if provider == "OpenRouter":
            llm_kwargs["base_url"] = base_url or OPENROUTER_API_BASE
        elif base_url:
            llm_kwargs["base_url"] = base_url
        try:
            st.session_state["llm_client"] = LLM(**llm_kwargs)
            st.sidebar.success("LLM client initialised.")
            signature_payload = llm_kwargs.copy()
            signature_payload["api_key"] = bool(signature_payload.get("api_key"))
            signature = tuple(signature_payload.items())
            if st.session_state.get(LLM_SIGNATURE_KEY) != signature:
                st.session_state[LLM_SIGNATURE_KEY] = signature
                st.session_state["theme_emoji_cache"] = {}
                llm_script_kwargs = llm_kwargs.copy()
                if "api_key" in llm_script_kwargs:
                    llm_script_kwargs["api_key"] = "<redacted>"
                _record_repro_action(
                    "init_llm",
                    "Set up the language model client",
                    _format_call("llm_client = LLM", llm_script_kwargs),
                )
        except Exception as exc:  # pragma: no cover - depends on credentials
            st.sidebar.error(f"Failed to create LLM client: {exc}")
    else:
        st.sidebar.info("Provide an API key to unlock the Screener and Extractor tools.")
        st.session_state["llm_client"] = None


# ---------------------------------------------------------------------------
# Main layout helpers
# ---------------------------------------------------------------------------

def _get_reports_df() -> pd.DataFrame:
    """Return the current working dataset."""

    reports_df: pd.DataFrame = st.session_state.get("reports_df", pd.DataFrame())
    return reports_df


def _render_header(container: Optional[DeltaGenerator] = None) -> None:
    """Render the hero header and dataset snapshot."""

    ctx = container or st

    reports_df = _get_reports_df()
    reports_count = len(reports_df)
    date_series = (
        pd.to_datetime(
            reports_df.get("date", pd.Series(dtype="datetime64[ns]")),
            errors="coerce",
        )
        if not reports_df.empty
        else pd.Series(dtype="datetime64[ns]")
    )

    earliest_display = "—"
    latest_display = "—"
    if not reports_df.empty and not date_series.empty:
        earliest = date_series.min()
        latest = date_series.max()
        if pd.notna(earliest):
            earliest_display = earliest.strftime("%d %b %Y")
        if pd.notna(latest):
            latest_display = latest.strftime("%d %b %Y")

    hero_html = f"""
    <div class="beta-banner">
        <div class="beta-banner__glow"></div>
        <div class="beta-banner__content">
            <span class="beta-banner__pill">Beta</span>
            <div class="beta-banner__text">
                <span class="beta-banner__headline">Welcome to the beta version of PFD Toolkit Workbench.</span>
                <span class="beta-banner__subtext">
                Because this app is still in development, there might be some bugs we haven't squashed yet. Please bare with us while we iron out the kinks! </span>
            </div>
            <div class="beta-banner__sparks">
                <span class="beta-banner__spark"></span>
                <span class="beta-banner__spark"></span>
                <span class="beta-banner__spark"></span>
            </div>
        </div>
    </div>
    <section class="hero-section">
        <div class="hero-copy">
            <span class="hero-badge">PFD Toolkit · AI Workbench <span class="hero-badge-beta">Beta</span></span>
            <h1>Command the narrative of Prevention of Future Death reports.</h1>
            <p>
                Load official reports, triage them against strategic focus areas, and extract
                structured evidence in minutes. The streamlined workspace keeps your data,
                models, and insights aligned from discovery through delivery.
            </p>
            <div class="hero-checklist">
                <span class="hero-pill">① Load and curate source data</span>
                <span class="hero-pill">② Screen topics and themes with LLMs</span>
                <span class="hero-pill">③ Capture structured insights for reporting</span>
            </div>
        </div>
        <div class="hero-visual">
            <div class="hero-orb orb-primary"></div>
            <div class="hero-orb orb-secondary"></div>
            <div class="hero-kpi">
                <span class="hero-kpi-value">{reports_count:,}</span>
                <span class="hero-kpi-label">Reports currently in view</span>
                <span class="hero-kpi-range">{earliest_display} – {latest_display}</span>
            </div>
        </div>
    </section>
    """

    ctx.markdown(hero_html, unsafe_allow_html=True)

    if reports_df.empty:
        onboarding_html = """
        <section class="empty-onboarding">
            <div class="empty-onboarding__scene" aria-hidden="true">
                <div class="empty-onboarding__backdrop"></div>
                <div class="empty-onboarding__halo empty-onboarding__halo--outer"></div>
                <div class="empty-onboarding__halo empty-onboarding__halo--inner"></div>
                <div class="empty-onboarding__cta-ring">
                    <span class="empty-onboarding__cta">Use the sidebar</span>
                </div>
            </div>
            <div class="empty-onboarding__content">
                <span class="empty-onboarding__eyebrow">Workspace locked</span>
                <h2>Load reports to activate your workspace.</h2>
                <p>Use the sidebar to import Prevention of Future Death reports and unlock metrics, datasets, and guided actions.</p>
                <div class="empty-onboarding__steps" role="list">
                    <span class="empty-onboarding__step" role="listitem">Set your API key</span>
                    <span class="empty-onboarding__step" role="listitem">Choose your reporting window</span>
                    <span class="empty-onboarding__step" role="listitem">Load reports</span>
                </div>
            </div>
        </section>
        """

        ctx.markdown(onboarding_html, unsafe_allow_html=True)
        return

    metric_row = ctx.container()
    metric_row.markdown("<div class='metric-row'>", unsafe_allow_html=True)
    col1, col2, col3 = metric_row.columns(3)
    with col1:
        _styled_metric("Reports in view", f"{reports_count:,}")
    with col2:
        _styled_metric("Earliest report", earliest_display)
    with col3:
        _styled_metric("Latest report", latest_display)
    metric_row.markdown("</div>", unsafe_allow_html=True)

    dataset_card = ctx.container()
    dataset_card.markdown(
        """
        <div class="section-card data-card">
            <div class="section-card-header">
                <span class="section-kicker">Workspace</span>
                <h3>Current working dataset</h3>
            </div>
        """,
        unsafe_allow_html=True,
    )

    dataset_card.dataframe(reports_df, width="stretch", hide_index=True)

    dataset_card.markdown("</div>", unsafe_allow_html=True)

    _render_theme_summary_panel(ctx)


def _resolve_theme_emoji(theme_name: str, llm_client: Optional[LLM]) -> str:
    """Return a single emoji representing ``theme_name`` using the configured LLM."""

    theme_clean = (theme_name or "").strip()
    if not theme_clean:
        return DEFAULT_THEME_EMOJI

    if llm_client is None:
        return DEFAULT_THEME_EMOJI

    cache: Dict[Any, str] = st.session_state.setdefault("theme_emoji_cache", {})
    signature = st.session_state.get(LLM_SIGNATURE_KEY)
    cache_key = (signature, theme_clean.lower())
    cached = cache.get(cache_key)
    if isinstance(cached, str) and cached:
        return cached

    prompt = (
        "You choose exactly one emoji to represent a theme extracted from Prevention "
        "of Future Death reports.\n"
        f"Theme: \"{theme_clean}\"\n\n"
        "Rules:\n"
        "- Select precisely one emoji that captures the theme's core idea.\n"
        "- Avoid violent, graphic, or harmful imagery such as knives, guns, bombs, "
        "blood, skulls, zombies, or anything implying injury.\n"
        f"- If nothing is suitable, respond with the light bulb emoji {DEFAULT_THEME_EMOJI}.\n"
        "- Do not engage in any death-related euphemism whatsoever. Use the light "
        "bulb emoji for any theme specific to death. "
        "- Do not include text, spaces, or additional punctuation.\n"
        'Respond only with JSON that matches the schema {"emoji": "🙂"}.\n'
    )

    candidate = DEFAULT_THEME_EMOJI
    try:
        responses = llm_client.generate([prompt], response_format=ThemeEmojiModel)
        result = responses[0] if responses else None
        if isinstance(result, ThemeEmojiModel):
            candidate = result.emoji.strip() or DEFAULT_THEME_EMOJI
    except Exception:
        candidate = DEFAULT_THEME_EMOJI

    if any(token in candidate for token in THEME_EMOJI_BANNED_TOKENS):
        candidate = DEFAULT_THEME_EMOJI

    if any(char.isalnum() for char in candidate):
        candidate = DEFAULT_THEME_EMOJI

    cache[cache_key] = candidate or DEFAULT_THEME_EMOJI
    return cache[cache_key]


def _render_theme_summary_panel(container: Optional[DeltaGenerator] = None) -> None:
    """Render a collapsible card summarising accepted themes."""

    llm_client: Optional[LLM] = st.session_state.get("llm_client")

    theme_df = _coerce_dataframe(st.session_state.get("theme_summary_table"))
    if theme_df is None or theme_df.empty:
        return

    ctx = container or st

    display_df = theme_df.copy(deep=True)
    if {"Count", "Theme"}.issubset(display_df.columns):
        display_df = display_df.sort_values(
            by=["Count", "Theme"], ascending=[False, True]
        ).reset_index(drop=True)

    reports_df = _get_reports_df()
    total_reports = len(reports_df)
    theme_count = len(display_df)

    top_theme = display_df.iloc[0] if not display_df.empty else {}
    top_theme_name_raw = (
        str(top_theme.get("Theme", "")).strip() if isinstance(top_theme, pd.Series) else ""
    )
    top_theme_name = escape(top_theme_name_raw or "—")
    top_theme_count = (
        int(top_theme.get("Count", 0)) if isinstance(top_theme, pd.Series) else 0
    )
    top_theme_percentage_raw = (
        float(top_theme.get("%", 0.0)) if isinstance(top_theme, pd.Series) else 0.0
    )
    top_theme_percentage = f"{top_theme_percentage_raw:.1f}%"

    theme_emoji = _resolve_theme_emoji(top_theme_name_raw, llm_client)
    theme_emoji_display = escape(theme_emoji or DEFAULT_THEME_EMOJI)

    theme_label = "theme" if theme_count == 1 else "themes"
    reports_label = "report" if total_reports == 1 else "reports"
    top_theme_reports_label = "report" if top_theme_count == 1 else "reports"

    panel = ctx.container()
    panel.markdown(
        f"""
        <div class="section-card theme-summary-card">
            <div class="section-card-header theme-summary-card__header">
                <span class="section-kicker">Theme discovery</span>
                <div class="theme-summary-card__title">
                    <h3>Thematic snapshot</h3>
                    <p>Revisit the thematic patterns surfaced by PFD Toolkit.</p>
                </div>
                <div class="theme-summary-card__meta">
                    <span class="theme-summary-chip">{theme_count} {theme_label}</span>
                    <span class="theme-summary-chip theme-summary-chip--accent">{total_reports:,} {reports_label} analysed</span>
                </div>
            </div>
            <div class="theme-summary-highlight">
                <div class="theme-summary-highlight__icon">{theme_emoji_display}</div>
                <div class="theme-summary-highlight__content">
                    <span class="theme-summary-highlight__label">Most prominent theme</span>
                    <span class="theme-summary-highlight__value">{top_theme_name}</span>
                    <span class="theme-summary-highlight__meta">{top_theme_count} {top_theme_reports_label} · {top_theme_percentage}</span>
                </div>
            </div>
            <div class="theme-summary-card__expander">
        """,
        unsafe_allow_html=True,
    )

    expander = panel.expander("See all identified themes", expanded=False)
    expander.markdown(
        "<div class='theme-summary-table-caption'>Theme assignments across the working dataset</div>",
        unsafe_allow_html=True,
    )
    expander.dataframe(display_df, width="stretch", hide_index=True)

    panel.markdown("""</div></div>""", unsafe_allow_html=True)


def _value_has_content(value: Any) -> bool:
    """Return ``True`` when ``value`` contains meaningful state."""

    if value is None:
        return False
    if isinstance(value, (pd.DataFrame, pd.Series)):
        return not value.empty
    if isinstance(value, (list, tuple, set, dict)):
        return len(value) > 0
    try:
        return bool(value)
    except ValueError:
        return True
    except TypeError:
        return True


def _workspace_has_activity() -> bool:
    """Return ``True`` if the workspace has actions beyond the initial load."""

    if st.session_state.get("reports_df_modified", False):
        return True
    if len(st.session_state.get("history", [])):
        return True

    activity_keys = [
        "screener_result",
        "extractor_result",
        "summary_result",
        "theme_model_schema",
        "theme_summary_table",
        "seed_topics_last",
        "extractor",
        "feature_grid",
        "preview_state",
    ]

    for key in activity_keys:
        if _value_has_content(st.session_state.get(key)):
            return True

    return False


def _render_workspace_footer(
    flash_payload: Optional[Tuple[str, str]] = None,
) -> None:
    """Render the bottom workspace footer with flash messaging and utilities."""

    reports_df = _get_reports_df()
    dataset_available = not reports_df.empty

    if not dataset_available and not flash_payload:
        return

    history_depth = len(st.session_state.get("history", []))
    redo_depth = len(st.session_state.get("redo_history", []))
    initial_df = st.session_state.get("reports_df_initial")

    undo_disabled = history_depth == 0
    if history_depth == 1:
        undo_hint = "(1 step available)"
    elif history_depth > 1:
        undo_hint = f"({history_depth} steps available)"
    else:
        undo_hint = ""

    redo_disabled = redo_depth == 0
    if redo_depth == 1:
        redo_hint = "(1 step available)"
    elif redo_depth > 1:
        redo_hint = f"({redo_depth} steps available)"
    else:
        redo_hint = ""

    has_initial_dataset = isinstance(initial_df, pd.DataFrame) and not initial_df.empty
    workspace_active = _workspace_has_activity()
    start_over_disabled = not (has_initial_dataset and workspace_active)

    footer = st.container()
    footer.markdown("<div class='workspace-footer'>", unsafe_allow_html=True)

    if flash_payload:
        message, level = flash_payload
        icon_map = {"success": "✅", "info": "ℹ️", "warning": "⚠️", "error": "❌"}
        level_key = level if level in icon_map else "info"
        icon = icon_map[level_key]
        safe_message = escape(message)
        footer.markdown(
            f"""
            <div class="workspace-message workspace-message--{level_key}">
                <span class="workspace-message-icon">{icon}</span>
                <span class="workspace-message-text">{safe_message}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

    if not dataset_available:
        footer.markdown("</div>", unsafe_allow_html=True)
        return

    undo_label = "↶ Undo" if not undo_hint else f"↶ Undo\n{undo_hint}"
    redo_label = "↷ Redo" if not redo_hint else f"↷ Redo\n{redo_hint}"
    undo_col, redo_col, start_over_col = footer.columns(3, gap="large")

    with undo_col:
        if st.button(
            undo_label,
            key="footer_undo",
            width="stretch",
            disabled=undo_disabled,
        ):
            _undo_last_change()

    with redo_col:
        if st.button(
            redo_label,
            key="footer_redo",
            width="stretch",
            disabled=redo_disabled,
        ):
            _redo_last_change()

    with start_over_col:
        if st.button(
            "↻ Start over",
            key="footer_reset",
            width="stretch",
            disabled=start_over_disabled,
        ):
            _start_again()

    footer.markdown("</div>", unsafe_allow_html=True)


def _build_theme_summary_table(
    extracted_reports: pd.DataFrame, theme_schema: Optional[Dict[str, Any]]
) -> pd.DataFrame:
    """Return a summary table of theme assignments for preview/apply flows."""

    if extracted_reports.empty or not isinstance(theme_schema, dict):
        return pd.DataFrame(columns=["Theme", "Description", "Count", "%"])

    properties = theme_schema.get("properties") if isinstance(theme_schema, dict) else None
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
        column_name = theme_key
        if column_name not in extracted_reports.columns:
            continue
        column = extracted_reports[column_name]
        count = int(column.map(_is_true).sum())
        percentage = round((count / total_reports * 100.0) if total_reports else 0.0, 1)
        display_name = (
            config.get("title") if isinstance(config, dict) else None
        ) or column_name.replace("_", " ").title()
        description = (
            _format_theme_description(config.get("description"))
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


def _queue_status_message(message: str, level: str = "success") -> None:
    """Store a transient status message to display on the next rerun."""

    st.session_state[FLASH_MESSAGE_KEY] = {"message": message, "level": level}


def _trigger_rerun() -> None:
    """Trigger a Streamlit rerun across Streamlit versions."""

    rerun_fn = getattr(st, "rerun", None)
    if callable(rerun_fn):
        rerun_fn()
        return

    rerun_fn = getattr(st, "experimental_rerun", None)
    if callable(rerun_fn):
        rerun_fn()
        return

    raise RuntimeError("Streamlit rerun is not available in this environment.")


def _consume_flash_message() -> Optional[Tuple[str, str]]:
    """Return and clear any queued status message for later rendering."""

    flash_payload = st.session_state.pop(FLASH_MESSAGE_KEY, None)
    if not flash_payload:
        return None

    message = flash_payload.get("message")
    if not message:
        return None

    level = str(flash_payload.get("level", "info"))
    return message, level


def _render_action_tiles() -> None:
    """Render the next-step action grid with gradient tiles."""

    reports_df = _get_reports_df()
    dataset_available = not reports_df.empty
    llm_ready = st.session_state.get("llm_client") is not None
    if not dataset_available:
        if st.session_state.get("active_action") is not None:
            st.session_state["active_action"] = None
        return

    actions = [
        {
            "label": "Download reports",
            "description": "Export the curated reports to your local device.",
            "key": "tile_save",
            "disabled": not dataset_available,
            "icon": "💾",
            "target": "save",
        },
        {
            "label": "Filter reports",
            "description": "Screen reports against your custom search query. "
            "The AI bot will crawl through each report, showing matches.",
            "key": "tile_filter",
            "disabled": not (dataset_available and llm_ready),
            "icon": "🧭",
            "target": "filter",
        },
        {
            "label": "Discover recurring themes",
            "description": "Surface thematic clusters to see how systemic issues recur across cases.",
            "key": "tile_discover",
            "disabled": not (dataset_available and llm_ready),
            "icon": "🌐",
            "target": "discover",
        },
        {
            "label": "Pull out structured information",
            "description": "Extract fields, tags, and custom attributes ready for dashboards and briefs.",
            "key": "tile_extract",
            "disabled": not (dataset_available and llm_ready),
            "icon": "🧾",
            "target": "extract",
        },
    ]

    section = st.container()
    section.markdown(
        """
        <div class="section-card action-section">
            <div class="section-card-header">
                <span class="section-kicker">Next steps</span>
                <h3>Where would you like to go next?</h3>
                <p>Choose a workflow to continue refining, tagging, or exporting your Prevention of Future Death reports.</p>
            </div>
        """,
        unsafe_allow_html=True,
    )

    grid_container = section.container()
    for i in range(0, len(actions), 2):
        cols = grid_container.columns(2, gap="large")
        for col, action in zip(cols, actions[i : i + 2]):
            with col:
                card = col.container()
                card.markdown(
                    f"""
                    <div class="action-card {'is-disabled' if action['disabled'] else ''}">
                        <div class="action-card-content">
                            <div class="action-card-heading">
                                <span class="action-card-icon">{action['icon']}</span>
                                <span class="action-card-title">{action['label']}</span>
                            </div>
                            <p class="action-card-copy">{action['description']}</p>
                        </div>
                    """,
                    unsafe_allow_html=True,
                )
                if card.button(
                    f"{action['icon']} {action['label']}",
                    key=action["key"],
                    width="stretch",
                    disabled=action["disabled"],
                ):
                    st.session_state["active_action"] = action["target"]
                card.markdown(
                    """
                    </div>
                    """,
                    unsafe_allow_html=True,
                )

    section.markdown("</div>", unsafe_allow_html=True)

def _undo_last_change() -> None:
    """Pop the latest snapshot from history and restore it."""

    history = list(st.session_state.get("history", []))
    if not history:
        return

    redo_history = list(st.session_state.get("redo_history", []))
    redo_history.append(snapshot_state())
    if len(redo_history) > 10:
        redo_history = redo_history[-10:]

    snapshot = history.pop()
    st.session_state["history"] = history
    st.session_state["redo_history"] = redo_history
    restore_state(snapshot)
    st.session_state["active_action"] = None
    _queue_status_message("Reverted to the previous state.")
    _trigger_rerun()


def _redo_last_change() -> None:
    """Restore the most recently undone snapshot."""

    redo_history = list(st.session_state.get("redo_history", []))
    if not redo_history:
        return

    history = list(st.session_state.get("history", []))
    history.append(snapshot_state())
    if len(history) > 10:
        history = history[-10:]

    snapshot = redo_history.pop()
    st.session_state["history"] = history
    st.session_state["redo_history"] = redo_history
    restore_state(snapshot)
    st.session_state["active_action"] = None
    _queue_status_message("Reapplied the next state.")
    _trigger_rerun()


def _start_again() -> None:
    """Reset the workspace to the post-load dataset."""

    initial_df = st.session_state.get("reports_df_initial")
    if not isinstance(initial_df, pd.DataFrame):
        return

    push_history_snapshot()
    st.session_state["reports_df"] = initial_df.copy(deep=True)
    st.session_state["reports_df_modified"] = False
    st.session_state["screener_result"] = None
    st.session_state["extractor_result"] = None
    st.session_state["summary_result"] = None
    st.session_state["theme_model_schema"] = None
    st.session_state["theme_summary_table"] = None
    st.session_state["seed_topics_last"] = None
    st.session_state["extractor"] = None
    st.session_state["extractor_source_signature"] = None
    st.session_state["feature_grid"] = None
    clear_preview_state()
    st.session_state["history"] = []
    st.session_state["redo_history"] = []
    st.session_state["active_action"] = None
    _queue_status_message("Workspace restored to the post-load dataset.")
    _trigger_rerun()


def _render_active_action() -> None:
    """Render the inline form for the selected action."""

    action = st.session_state.get("active_action")
    if action is None:
        return

    if action == "save":
        _render_save_action()
    elif action == "filter":
        _render_filter_action()
    elif action == "discover":
        _render_discover_action()
    elif action == "extract":
        _render_extract_action()


def _render_save_action() -> None:
    """Render the save action allowing users to download the dataset."""

    reports_df = _get_reports_df()
    st.markdown("#### Download research bundle")
    if reports_df.empty:
        st.info("No reports available to download yet.")
        return

    csv_bytes = reports_df.to_csv(index=False).encode("utf-8")

    theme_summary_df = _coerce_dataframe(st.session_state.get("theme_summary_table"))
    feature_grid_df = _coerce_dataframe(st.session_state.get("feature_grid"))

    include_theme_summary = theme_summary_df is not None and not theme_summary_df.empty
    include_feature_grid = feature_grid_df is not None and not feature_grid_df.empty

    dataset_selected = st.checkbox(
        "Working dataset",
        value=True,
        key="download_include_dataset",
    )

    theme_help = None
    if not include_theme_summary:
        theme_help = "Run Discover themes to generate the theme table before downloading it."
    else:
        theme_help = "Downloads your list of themes, definitions and counts."
    theme_selected = st.checkbox(
        "Theme table",
        value=include_theme_summary,
        disabled=not include_theme_summary,
        help=theme_help,
        key="download_include_theme",
    )

    feature_help = None
    if not include_feature_grid:
        feature_help = "Use Pull out structured info to create a custom feature grid."
    else:
        feature_help = "Downloads the custom attributes you defined in 'Pull out structured info'"
    feature_selected = st.checkbox(
        "Custom feature grid",
        value=include_feature_grid,
        disabled=not include_feature_grid,
        help=feature_help,
        key="download_include_feature_grid",
    )

    script_selected = st.checkbox(
        "Reproducible Python script",
        value=True,
        key="download_include_script",
        help="Saves a reproducible script of every action you've taken on this Workbench session.",
    )

    bundle_files: List[Tuple[str, bytes]] = []
    if dataset_selected:
        bundle_files.append(("pfd_reports.csv", csv_bytes))
    if theme_selected and include_theme_summary and isinstance(theme_summary_df, pd.DataFrame):
        theme_text = theme_summary_df.to_csv(index=False, sep="\t").encode("utf-8")
        bundle_files.append(("theme_summary.txt", theme_text))
    if feature_selected and include_feature_grid and isinstance(feature_grid_df, pd.DataFrame):
        feature_text = feature_grid_df.to_csv(index=False, sep="\t").encode("utf-8")
        bundle_files.append(("custom_feature_grid.txt", feature_text))
    if script_selected:
        script_bytes = _get_repro_script_text().encode("utf-8")
        bundle_files.append(("reproducible_workspace.py", script_bytes))

    has_selection = bool(bundle_files)

    zip_bytes = b""
    if has_selection:
        zip_buffer = BytesIO()
        with zipfile.ZipFile(zip_buffer, "w", compression=zipfile.ZIP_DEFLATED) as zip_file:
            for filename, payload in bundle_files:
                zip_file.writestr(filename, payload)
        zip_buffer.seek(0)
        zip_bytes = zip_buffer.getvalue()

    if not has_selection:
        st.info("Select at least one resource above to enable the download.")

    st.download_button(
        "Download research bundle",
        data=zip_bytes,
        file_name="pfd_research_bundle.zip",
        mime="application/zip",
        width="stretch",
        disabled=not has_selection,
    )


def _get_extractor(reports_df: pd.DataFrame) -> Optional[Extractor]:
    """Return a cached extractor for the current dataset, creating it if needed."""

    llm_client: Optional[LLM] = st.session_state.get("llm_client")
    if llm_client is None or reports_df.empty:
        return None

    signature = (len(reports_df), tuple(reports_df.columns))
    extractor: Optional[Extractor] = st.session_state.get("extractor")

    if (
        extractor is None
        or st.session_state.get("extractor_source_signature") != signature
    ):
        try:
            extractor = Extractor(
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
            extractor_kwargs = {
                "llm": "llm_client",
                "reports": "reports_df",
                "include_date": True,
                "include_coroner": True,
                "include_area": True,
                "include_receiver": True,
                "include_investigation": True,
                "include_circumstances": True,
                "include_concerns": True,
                "verbose": False,
            }
            _record_repro_action(
                "init_extractor",
                "Initialise the extractor",
                _format_call(
                    "extractor = Extractor",
                    extractor_kwargs,
                    raw_parameters={"llm", "reports"},
                ),
            )
        except Exception as exc:  # pragma: no cover - depends on live API
            st.error(f"Could not initialise extractor: {exc}")
            st.session_state["extractor"] = None
            st.session_state["extractor_source_signature"] = None
            return None

        st.session_state["extractor"] = extractor
        st.session_state["extractor_source_signature"] = signature

    return extractor


def _render_filter_action() -> None:
    """Render the inline Screener form."""

    reports_df = _get_reports_df()
    llm_client: Optional[LLM] = st.session_state.get("llm_client")

    st.markdown("#### Filter reports")
    if llm_client is None:
        st.warning("Add a valid API key in the sidebar to enable the Screener.")
        return
    if reports_df.empty:
        st.info("Load reports from the sidebar before screening.")
        return

    with st.form("filter_reports_form", enter_to_submit=False):
        search_query = st.text_area(
            "Describe what you want to find",
            placeholder="e.g. Reports mentioning delays in ambulance response times",
        )
        cols = st.columns(2)
        filter_df = cols[0].checkbox(
            "Only keep matching reports",
            value=True,
        )

        with st.expander("Advanced options"):
            produce_spans = st.checkbox(
                "Return supporting quotes from the reports",
                value=False,
                key="screener_produce_spans",
            )
            drop_spans = st.checkbox(
                "Remove the quotes column from the results",
                value=False,
                key="screener_drop_spans",
            )

        with st.container():
            submitted = st.form_submit_button(
                "Run Screener",
                width="stretch",
                type="primary",
            )

    if not submitted:
        return

    if not search_query.strip():
        st.error("Describe what the Screener should look for.")
        return

    push_history_snapshot()
    initial_report_count = len(reports_df)
    match_column_name = "matches_query"

    loading_placeholder = st.empty()
    loading_indicator = LoadingIndicator(loading_placeholder, "Preparing screener…")
    try:
        loading_indicator.update("Configuring screener…")
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
        screener_kwargs = {
            "llm": "llm_client",
            "reports": "reports_df",
            "verbose": False,
            "include_date": True,
            "include_coroner": True,
            "include_area": True,
            "include_receiver": True,
            "include_investigation": True,
            "include_circumstances": True,
            "include_concerns": True,
        }
        _record_repro_action(
            "init_screener",
            "Initialise the screener",
            _format_call(
                "screener = Screener",
                screener_kwargs,
                raw_parameters={"llm", "reports"},
            ),
        )
        loading_indicator.update("Running the screener…")
        result_df = screener.screen_reports(
            search_query=search_query or None,
            filter_df=filter_df,
            result_col_name=match_column_name,
            produce_spans=produce_spans,
            drop_spans=drop_spans,
        )
        screen_kwargs = {
            "search_query": search_query or None,
            "filter_df": filter_df,
            "result_col_name": match_column_name,
            "produce_spans": produce_spans,
            "drop_spans": drop_spans,
        }
        _record_repro_action(
            "run_screener",
            "Screen the reports",
            _format_call(
                "result_df = screener.screen_reports",
                screen_kwargs,
            ),
        )
        loading_indicator.update("Finalising results…")
    except Exception as exc:  # pragma: no cover - relies on live API
        st.error(f"Screening failed: {exc}")
        return
    finally:
        loading_indicator.close()

    st.session_state["screener_result"] = result_df
    st.session_state["reports_df"] = result_df.copy(deep=True)
    st.session_state["reports_df_modified"] = True
    st.session_state["extractor"] = None
    st.session_state["extractor_source_signature"] = None
    st.session_state["extractor_result"] = None
    st.session_state["summary_result"] = None
    st.session_state["theme_model_schema"] = None
    st.session_state["theme_summary_table"] = None
    st.session_state["seed_topics_last"] = None
    clear_preview_state()
    st.session_state["active_action"] = None

    if filter_df:
        matched_report_count = len(result_df)
    elif isinstance(result_df, pd.DataFrame) and match_column_name in result_df.columns:
        matched_series = result_df[match_column_name]
        matched_report_count = int(matched_series.fillna(False).astype(bool).sum())
    else:
        matched_report_count = len(result_df)

    _queue_status_message(
        f"Screening successful! From the initial {initial_report_count:,} reports, "
        f"{matched_report_count:,} matched your search query."
    )
    _trigger_rerun()


def _render_discover_action() -> None:
    """Render the discovery preview/apply workflow."""

    reports_df = _get_reports_df()
    extractor = _get_extractor(reports_df)
    if extractor is None:
        return

    st.markdown("#### Discover recurring themes")
    st.write(
        "We will summarise each report and then surface recurring themes before applying them to the dataset."
    )

    with st.form("discover_themes_form", enter_to_submit=False):
        extra_theme_instructions = st.text_area(
            "Add any extra guidance for the themes (optional)",
            placeholder="e.g. Focus on system-level safety issues.",
            key="discover_extra_instructions",
        )
        with st.expander("Advanced options"):
            trim_labels = {
                "Detailed paragraph": "low",
                "Concise summary": "medium",
                "Short summary": "high",
                "One or two sentences": "very high",
            }
            trim_choice = st.selectbox(
                "How concise should the summaries be?",
                list(trim_labels.keys()),
                index=1,
                key="discover_trim_choice",
            )
            warning_threshold = st.number_input(
                "Warn if the token estimate exceeds",
                min_value=1000,
                value=100000,
                step=1000,
                key="discover_warning_threshold",
            )
            error_threshold = st.number_input(
                "Stop if the token estimate exceeds",
                min_value=1000,
                value=500000,
                step=1000,
                key="discover_error_threshold",
            )
            min_col, max_col = st.columns(2)
            min_themes_raw = min_col.text_input(
                "Minimum number of themes (optional)",
                value="",
                placeholder="e.g. 5",
                key="discover_min_themes",
            )
            max_themes_raw = max_col.text_input(
                "Maximum number of themes (optional)",
                value="",
                placeholder="e.g. 10",
                key="discover_max_themes",
            )
            seed_topics_text = st.text_area(
                "Seed topics (optional)",
                value=st.session_state.get("seed_topics_last") or "",
                placeholder='["Communication", "Medication safety", "Staff training"]',
                key="discover_seed_topics",
            )

        preview_requested = st.form_submit_button(
            "Discover recurring themes", width="stretch"
        )

    if preview_requested:
        try:
            min_themes_value = _parse_optional_non_negative_int(
                min_themes_raw, "Minimum number of themes"
            )
        except ValueError as exc:
            st.error(str(exc))
            min_themes_value = None
        try:
            max_themes_value = _parse_optional_non_negative_int(
                max_themes_raw, "Maximum number of themes"
            )
        except ValueError as exc:
            st.error(str(exc))
            max_themes_value = None

        if min_themes_value is not None or max_themes_value is not None or (
            not min_themes_raw.strip() and not max_themes_raw.strip()
        ):
            loading_placeholder = st.empty()
            loading_indicator = LoadingIndicator(
                loading_placeholder, "Summarising the reports…"
            )
            try:
                summary_col_name = extractor.summary_col or "summary"
                summary_df = extractor.summarise(
                    result_col_name=summary_col_name,
                    trim_intensity=trim_labels[trim_choice],
                )
                summarise_kwargs = {
                    "result_col_name": summary_col_name,
                    "trim_intensity": trim_labels[trim_choice],
                }
                _record_repro_action(
                    "summarise_reports",
                    "Summarise the reports",
                    _format_call(
                        "summary_df = extractor.summarise",
                        summarise_kwargs,
                    ),
                )

                seed_topics: Optional[Any] = None
                if seed_topics_text.strip():
                    try:
                        seed_topics = json.loads(seed_topics_text)
                    except json.JSONDecodeError:
                        seed_topics = [
                            line.strip()
                            for line in seed_topics_text.splitlines()
                            if line.strip()
                        ]

                loading_indicator.update("Identifying themes…")
                ThemeModel = extractor.discover_themes(
                    warn_exceed=int(warning_threshold),
                    error_exceed=int(error_threshold),
                    max_themes=max_themes_value,
                    min_themes=min_themes_value,
                    extra_instructions=extra_theme_instructions or None,
                    seed_topics=seed_topics,
                )
                discover_kwargs = {
                    "warn_exceed": int(warning_threshold),
                    "error_exceed": int(error_threshold),
                    "max_themes": max_themes_value,
                    "min_themes": min_themes_value,
                    "extra_instructions": extra_theme_instructions or None,
                    "seed_topics": seed_topics,
                }
                _record_repro_action(
                    "discover_themes",
                    "Discover recurring themes",
                    _format_call(
                        "ThemeModel = extractor.discover_themes",
                        discover_kwargs,
                    ),
                )

                if ThemeModel is None or not hasattr(ThemeModel, "model_json_schema"):
                    loading_indicator.update("Theme discovery finished.")
                    st.warning(
                        "Theme discovery completed but did not return a schema."
                    )
                    clear_preview_state()
                else:
                    loading_indicator.update("Assigning themes to reports…")
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
                    _record_repro_action(
                        "assign_themes",
                        "Assign discovered themes to the reports",
                        _format_call(
                            "preview_df = extractor.extract_features",
                            theme_extract_kwargs,
                            raw_parameters={"feature_model"},
                        ),
                    )
                    theme_summary_df = _build_theme_summary_table(
                        preview_df, theme_schema
                    )
                    st.session_state["preview_state"] = {
                        "type": "discover",
                        "summary_df": summary_df,
                        "preview_df": preview_df,
                        "theme_schema": theme_schema,
                        "theme_summary": theme_summary_df,
                        "seed_topics": seed_topics or None,
                    }
                    loading_indicator.update("Preview ready.")
                    st.success(
                        "Preview ready. Review the results below and apply them when happy."
                    )
            except Exception as exc:  # pragma: no cover - depends on live API
                st.error(f"Theme discovery failed: {exc}")
                clear_preview_state()
            finally:
                loading_indicator.close()

    preview_state = st.session_state.get("preview_state")
    if not isinstance(preview_state, dict) or preview_state.get("type") != "discover":
        return

    theme_summary_df = preview_state.get("theme_summary")
    if isinstance(theme_summary_df, pd.DataFrame) and not theme_summary_df.empty:
        _display_dataframe(theme_summary_df, "Theme assignments preview")
    else:
        st.info("No theme assignments were returned in the preview.")

    preview_df = preview_state.get("preview_df")
    theme_schema = preview_state.get("theme_schema")

    actions_col1, actions_col2 = st.columns(2)
    if actions_col1.button("Accept themes", width="stretch"):
        if not isinstance(preview_df, pd.DataFrame):
            st.error("No preview data available to apply.")
            return
        push_history_snapshot()
        st.session_state["reports_df"] = preview_df.copy(deep=True)
        st.session_state["reports_df_modified"] = True
        st.session_state["summary_result"] = _deepcopy_value(
            preview_state.get("summary_df")
        )
        st.session_state["extractor_result"] = preview_df.copy(deep=True)
        st.session_state["theme_model_schema"] = _deepcopy_value(theme_schema)
        st.session_state["theme_summary_table"] = (
            theme_summary_df.copy(deep=True)
            if isinstance(theme_summary_df, pd.DataFrame)
            else None
        )
        st.session_state["seed_topics_last"] = preview_state.get("seed_topics")
        st.session_state["extractor"] = None
        st.session_state["extractor_source_signature"] = None
        clear_preview_state()
        st.session_state["active_action"] = None
        _queue_status_message("Themes applied to the working dataset.")
        _trigger_rerun()

    if actions_col2.button("Discard themes", width="stretch"):
        clear_preview_state()
        st.session_state["active_action"] = None
        _queue_status_message("Theme preview discarded.", level="info")
        _trigger_rerun()


def _render_extract_action() -> None:
    """Render the feature tagging workflow."""

    reports_df = _get_reports_df()
    extractor = _get_extractor(reports_df)
    if extractor is None:
        return

    st.markdown("#### Pull out structured information")
    st.write(
        "Describe the fields you want to capture and the Extractor will populate them for each report."
    )

    default_grid = pd.DataFrame(
        [
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
    )

    stored_grid = st.session_state.get("feature_editor")
    if stored_grid is None:
        stored_grid = st.session_state.get("feature_grid")

    if isinstance(stored_grid, pd.DataFrame):
        feature_grid_df = stored_grid.copy(deep=True)
    else:
        try:
            feature_grid_df = pd.DataFrame(stored_grid)
        except ValueError:
            feature_grid_df = pd.DataFrame()

    if feature_grid_df.empty:
        feature_grid_df = default_grid.copy(deep=True)

    missing_columns = [
        column
        for column in ("Field name", "Description", "Type")
        if column not in feature_grid_df.columns
    ]
    if missing_columns:
        feature_grid_df = feature_grid_df.reindex(columns=default_grid.columns)

    feature_grid = st.data_editor(
        feature_grid_df,
        num_rows="dynamic",
        width="stretch",
        column_config={
            "Field name": st.column_config.TextColumn(
                "Field name",
                help="This becomes the column name in the output.",
            ),
            "Description": st.column_config.TextColumn(
                "Description",
                help="Optional guidance for the language model.",
            ),
            "Type": st.column_config.SelectboxColumn(
                "Type",
                options=[
                    "Free text",
                    "Conditional (True/False)",
                    "Whole number",
                    "Decimal number",
                ],
                help="Choose the data type the extractor should return.",
            ),
        },
        key="feature_editor",
    )

    st.session_state["feature_grid"] = feature_grid.copy(deep=True)

    with st.form("extract_features_form", enter_to_submit=False):
        with st.expander("Advanced options"):
            produce_spans = st.checkbox(
                "Return supporting quotes from the reports",
                value=False,
                key="extract_produce_spans",
            )
            drop_spans = st.checkbox(
                "Remove the quote columns from the results",
                value=False,
                key="extract_drop_spans",
            )
            force_assign = st.checkbox(
                "Always assign a value even when unsure",
                value=False,
                key="extract_force_assign",
            )
            allow_multiple = st.checkbox(
                "Allow multiple categories for a single field",
                value=False,
                key="extract_allow_multiple",
            )
            extra_instructions = st.text_area(
                "Anything else the model should know? (optional)",
                key="extract_extra_instructions",
            )
            skip_if_present = st.checkbox(
                "Skip rows that already contain extracted values",
                value=True,
                key="extract_skip_if_present",
            )

        extract_submitted = st.form_submit_button(
            "Tag the reports", width="stretch"
        )

    if not extract_submitted:
        return

    push_history_snapshot()
    loading_placeholder = st.empty()
    loading_indicator = LoadingIndicator(
        loading_placeholder, "Configuring extraction…"
    )
    try:
        feature_model = _build_feature_model_from_grid(feature_grid)
        loading_indicator.update("Extracting structured data…")
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
        _record_repro_action(
            "extract_features",
            "Pull structured information",
            _format_call(
                "result_df = extractor.extract_features",
                extract_kwargs,
                raw_parameters={"reports", "feature_model"},
            ),
        )
        loading_indicator.update("Finalising dataset…")
    except Exception as exc:  # pragma: no cover - depends on live API
        st.error(f"Extraction failed: {exc}")
        return
    finally:
        loading_indicator.close()

    st.session_state["extractor_result"] = result_df
    st.session_state["reports_df"] = result_df.copy(deep=True)
    st.session_state["reports_df_modified"] = True
    clear_preview_state()
    st.session_state["active_action"] = None
    _queue_status_message("Tagging complete. The working dataset has been updated.")
    _trigger_rerun()
# ---------------------------------------------------------------------------
# Entrypoint
# ---------------------------------------------------------------------------

def main() -> None:
    st.set_page_config(
        page_title="PFD Toolkit AI Workbench · Beta",
        page_icon=":material/build:",
        layout="wide",
    )
    _init_session_state()

    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@400;500;600;700;800&family=Space+Grotesk:wght@500;600&display=swap');

            :root {
                color-scheme: dark;
            }

            html, body, .stApp {
                font-family: 'Plus Jakarta Sans', sans-serif;
            }

            .stApp {
                background: linear-gradient(130deg, #040716 0%, #131b46 35%, #2b1564 70%, #461b8b 100%);
                min-height: 100vh;
                color: #e7ecff;
                position: relative;
            }

            .stApp::before {
                content: "";
                position: fixed;
                inset: 0;
                background:
                    radial-gradient(circle at 20% 20%, rgba(96, 165, 250, 0.28), transparent 55%),
                    radial-gradient(circle at 80% 0%, rgba(192, 132, 252, 0.38), transparent 50%),
                    radial-gradient(circle at 50% 95%, rgba(45, 212, 191, 0.18), transparent 55%);
                pointer-events: none;
                z-index: -2;
            }

            .stApp::after {
                content: "";
                position: fixed;
                inset: 0;
                background-image: radial-gradient(rgba(255, 255, 255, 0.06) 1px, transparent 0);
                background-size: 40px 40px;
                opacity: 0.35;
                pointer-events: none;
                z-index: -1;
            }

            .stApp a {
                color: #8fdcff;
            }

            .stApp p, .stApp label, .stApp li, .stApp span, .stApp .stMarkdown {
                color: rgba(230, 234, 255, 0.88);
            }

            div[data-testid="stHeader"] {
                background: linear-gradient(135deg, rgba(8, 12, 36, 0.9), rgba(24, 18, 54, 0.85));
                border-bottom: 1px solid rgba(99, 102, 241, 0.28);
                box-shadow: none;
                backdrop-filter: blur(18px);
            }

            div[data-testid="stHeader"] * {
                color: #dfe5ff !important;
            }

            div[data-testid="stToolbar"] button[title*="theme"],
            div[data-testid="stToolbar"] button[aria-label*="theme"] {
                display: none;
            }

            .stApp h1, .stApp h2, .stApp h3, .stApp h4, .stApp h5 {
                color: #f8f9ff;
                font-family: 'Space Grotesk', sans-serif;
            }

            section[data-testid="stSidebar"] {
                background: rgba(8, 12, 36, 0.85);
                backdrop-filter: blur(20px);
                border-right: 1px solid rgba(148, 163, 255, 0.2);
            }

            section[data-testid="stSidebar"] h2.sidebar-title {
                margin-bottom: 0.2rem;
                display: inline-flex;
                align-items: center;
                gap: 0.5rem;
            }

            .sidebar-beta {
                display: inline-flex;
                align-items: center;
                padding: 0.2rem 0.55rem;
                border-radius: 999px;
                background: linear-gradient(135deg, rgba(56, 189, 248, 0.65), rgba(192, 132, 252, 0.7));
                font-size: 0.6em;
                letter-spacing: 0.12em;
                text-transform: uppercase;
                color: #050b24;
                font-weight: 700;
                box-shadow: 0 8px 18px rgba(7, 10, 34, 0.35);
            }

            section[data-testid="stSidebar"] .stMarkdown, section[data-testid="stSidebar"] label, section[data-testid="stSidebar"] span {
                color: rgba(226, 232, 255, 0.88);
            }

            section[data-testid="stSidebar"] input, section[data-testid="stSidebar"] textarea, section[data-testid="stSidebar"] select {
                background: rgba(15, 23, 42, 0.65);
                border: 1px solid rgba(148, 163, 255, 0.45);
                color: #f8f9ff;
                border-radius: 12px;
            }

            section[data-testid="stSidebar"] .stButton button {
                background: linear-gradient(135deg, rgba(99, 102, 241, 0.6), rgba(56, 189, 248, 0.45));
                border-radius: 999px;
                border: 1px solid rgba(148, 163, 255, 0.4);
                color: #0f172a;
                font-weight: 700;
            }

            .workspace-footer {
                margin-top: 3rem;
                padding: 0;
                border-radius: 0;
                background: transparent;
                border: none;
                box-shadow: none;
                display: flex;
                flex-direction: column;
                gap: 1.2rem;
            }

            .workspace-message {
                display: flex;
                align-items: center;
                gap: 1rem;
                padding: 1rem 1.3rem;
                border-radius: 18px;
                background: rgba(30, 41, 82, 0.72);
                border: 1px solid rgba(99, 102, 241, 0.45);
                font-size: 0.98rem;
                line-height: 1.5;
                color: rgba(233, 238, 255, 0.92);
            }

            .workspace-message--success {
                background: rgba(16, 185, 129, 0.22);
                border-color: rgba(45, 212, 191, 0.5);
            }

            .workspace-message--info {
                background: rgba(56, 189, 248, 0.2);
                border-color: rgba(59, 130, 246, 0.5);
            }

            .workspace-message--warning {
                background: rgba(251, 191, 36, 0.2);
                border-color: rgba(245, 158, 11, 0.5);
            }

            .workspace-message--error {
                background: rgba(248, 113, 113, 0.22);
                border-color: rgba(248, 113, 113, 0.5);
            }

            .workspace-message-icon {
                font-size: 1.5rem;
            }

            .workspace-message-text {
                flex: 1 1 auto;
            }

            .workspace-footer div[data-testid="column"] {
                display: flex;
                flex-direction: column;
            }

            .workspace-footer div[data-testid="column"] div[data-testid="stButton"] {
                margin: 0;
                flex: 1 1 auto;
            }

            .workspace-footer div[data-testid="column"] div[data-testid="stButton"] > button {
                display: flex;
                flex-direction: column;
                align-items: flex-start;
                justify-content: center;
                gap: 0.35rem;
                padding: 1.05rem 1.35rem;
                border-radius: 20px;
                border: none;
                font-weight: 700;
                font-size: 1.02rem;
                letter-spacing: 0.01em;
                white-space: pre-line;
                text-align: left;
                width: 100%;
                color: #041026;
                background: linear-gradient(135deg, rgba(56, 189, 248, 0.85), rgba(96, 165, 250, 0.95));
                box-shadow: 0 26px 52px rgba(12, 18, 54, 0.5);
                transition: transform 0.2s ease, box-shadow 0.2s ease;
            }

            .workspace-footer div[data-testid="column"]:last-of-type div[data-testid="stButton"] > button {
                background: linear-gradient(135deg, rgba(192, 132, 252, 0.9), rgba(129, 140, 248, 0.95));
            }

            .workspace-footer div[data-testid="column"] div[data-testid="stButton"] > button:hover:not(:disabled) {
                transform: translateY(-2px);
                box-shadow: 0 32px 60px rgba(12, 18, 54, 0.55);
            }

            .workspace-footer div[data-testid="column"] div[data-testid="stButton"] > button:disabled {
                opacity: 0.48;
                cursor: not-allowed;
                background: rgba(30, 41, 82, 0.65);
                color: rgba(226, 232, 255, 0.6);
                box-shadow: none;
            }

            .stApp div[data-baseweb="input"] > div,
            .stApp div[data-baseweb="textarea"] > div,
            .stApp div[data-baseweb="select"] > div {
                background: rgba(13, 21, 52, 0.75);
                border: 1px solid rgba(148, 163, 255, 0.45);
                border-radius: 14px;
                box-shadow: inset 0 0 0 1px rgba(5, 10, 28, 0.55);
            }

            .stApp div[data-baseweb="input"] input,
            .stApp div[data-baseweb="textarea"] textarea,
            .stApp div[data-baseweb="select"] div[role="combobox"],
            .stApp .stTextInput input,
            .stApp .stTextArea textarea {
                color: #f8f9ff !important;
            }

            .stApp div[data-baseweb="input"] input::placeholder,
            .stApp div[data-baseweb="textarea"] textarea::placeholder {
                color: rgba(226, 232, 255, 0.55);
            }

            .stApp div[data-baseweb="select"] svg {
                color: rgba(226, 232, 255, 0.72);
            }

            .status-label {
                text-transform: uppercase;
                letter-spacing: 0.12em;
                font-size: 0.7rem;
                color: rgba(226, 232, 255, 0.72);
            }

            .status-value {
                font-weight: 600;
                color: #f8faff;
            }

            .beta-banner {
                margin-top: 1.4rem;
                position: relative;
                overflow: hidden;
                padding: 1.4rem 1.8rem;
                border-radius: 28px;
                background: linear-gradient(135deg, rgba(14, 18, 54, 0.9), rgba(56, 189, 248, 0.18));
                border: 1px solid rgba(148, 163, 255, 0.42);
                box-shadow: 0 32px 60px rgba(5, 10, 30, 0.55);
                isolation: isolate;
            }

            .beta-banner__glow {
                position: absolute;
                inset: -35% -20% -20% -25%;
                background:
                    radial-gradient(circle at 15% 25%, rgba(96, 165, 250, 0.45), transparent 55%),
                    radial-gradient(circle at 85% 20%, rgba(192, 132, 252, 0.42), transparent 60%),
                    radial-gradient(circle at 50% 85%, rgba(45, 212, 191, 0.35), transparent 65%);
                filter: blur(12px);
                opacity: 0.9;
                transform: scale(1.02);
                animation: betaGlow 12s ease-in-out infinite;
                z-index: 0;
            }

            @keyframes betaGlow {
                0%, 100% {
                    transform: scale(1.02) translate3d(0, 0, 0);
                    opacity: 0.82;
                }
                50% {
                    transform: scale(1.08) translate3d(1%, -2%, 0);
                    opacity: 1;
                }
            }

            .beta-banner__content {
                position: relative;
                z-index: 1;
                display: flex;
                align-items: center;
                gap: 1.4rem;
            }

            .beta-banner__pill {
                display: inline-flex;
                align-items: center;
                gap: 0.4rem;
                padding: 0.45rem 1rem;
                border-radius: 999px;
                background: linear-gradient(135deg, rgba(99, 102, 241, 0.9), rgba(56, 189, 248, 0.85));
                color: #050b24;
                font-size: 0.78rem;
                letter-spacing: 0.16em;
                text-transform: uppercase;
                font-weight: 700;
                box-shadow: 0 18px 35px rgba(5, 10, 30, 0.45);
            }

            .beta-banner__text {
                display: flex;
                flex-direction: column;
                gap: 0.25rem;
                max-width: 38rem;
            }

            .beta-banner__headline {
                font-size: 1.08rem;
                font-weight: 600;
                color: #f8faff;
            }

            .beta-banner__subtext {
                font-size: 0.94rem;
                color: rgba(224, 231, 255, 0.8);
            }

            .beta-banner__sparks {
                display: flex;
                align-items: center;
                gap: 0.45rem;
                margin-left: auto;
            }

            .beta-banner__spark {
                width: 16px;
                height: 16px;
                border-radius: 50%;
                background: linear-gradient(135deg, rgba(56, 189, 248, 0.92), rgba(192, 132, 252, 0.92));
                box-shadow: 0 0 18px rgba(148, 163, 255, 0.85);
                opacity: 0.8;
                animation: betaSpark 2.8s ease-in-out infinite;
            }

            .beta-banner__spark:nth-child(2) {
                animation-delay: 0.6s;
            }

            .beta-banner__spark:nth-child(3) {
                animation-delay: 1.2s;
            }

            @keyframes betaSpark {
                0%, 100% {
                    transform: scale(0.75) translateY(0);
                    opacity: 0.55;
                }
                50% {
                    transform: scale(1.25) translateY(-6px);
                    opacity: 1;
                }
            }

            .hero-section {
                margin-top: 2.2rem;
                position: relative;
                display: grid;
                grid-template-columns: minmax(0, 3fr) minmax(0, 2fr);
                gap: 3rem;
                padding: 3rem 3.5rem;
                border-radius: 32px;
                background: rgba(12, 17, 46, 0.78);
                border: 1px solid rgba(148, 163, 255, 0.25);
                box-shadow: 0 35px 70px rgba(6, 10, 32, 0.55);
                overflow: hidden;
            }

            .hero-section::after {
                content: "";
                position: absolute;
                inset: 0;
                background: linear-gradient(135deg, rgba(79, 70, 229, 0.25), transparent 65%);
                pointer-events: none;
            }

            .hero-copy {
                position: relative;
                z-index: 1;
                display: flex;
                flex-direction: column;
                gap: 1rem;
            }

            .hero-badge {
                display: inline-flex;
                align-items: center;
                gap: 0.4rem;
                padding: 0.4rem 0.9rem;
                border-radius: 999px;
                background: rgba(56, 189, 248, 0.16);
                border: 1px solid rgba(148, 163, 255, 0.35);
                font-size: 0.8rem;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: rgba(224, 231, 255, 0.9);
            }

            .hero-badge-beta {
                display: inline-flex;
                align-items: center;
                padding: 0.2rem 0.6rem;
                border-radius: 999px;
                background: linear-gradient(135deg, rgba(148, 163, 255, 0.92), rgba(56, 189, 248, 0.95));
                color: #050b24;
                font-size: 0.7rem;
                letter-spacing: 0.14em;
                text-transform: uppercase;
                font-weight: 700;
                box-shadow: 0 12px 24px rgba(6, 10, 32, 0.45);
            }

            .hero-section h1 {
                font-size: clamp(2.6rem, 4vw, 3.6rem);
                line-height: 1.1;
                margin: 0.5rem 0 0.75rem;
            }

            .hero-section p {
                max-width: 34rem;
                font-size: 1.05rem;
                color: rgba(226, 232, 255, 0.78);
            }

            .hero-checklist {
                display: flex;
                flex-wrap: wrap;
                gap: 0.6rem;
            }

            .hero-pill {
                padding: 0.55rem 1rem;
                border-radius: 999px;
                background: rgba(148, 163, 255, 0.22);
                border: 1px solid rgba(180, 198, 255, 0.35);
                font-size: 0.85rem;
                color: rgba(230, 234, 255, 0.9);
            }

            .hero-visual {
                position: relative;
                display: flex;
                align-items: center;
                justify-content: center;
            }

            .hero-orb {
                position: absolute;
                border-radius: 50%;
                filter: blur(0px);
                opacity: 0.55;
                animation: float 12s ease-in-out infinite;
            }

            .hero-orb.orb-primary {
                width: 260px;
                height: 260px;
                background: radial-gradient(circle at 30% 30%, rgba(96, 165, 250, 0.6), transparent 65%);
                animation-delay: 0s;
            }

            .hero-orb.orb-secondary {
                width: 360px;
                height: 360px;
                background: radial-gradient(circle at 70% 40%, rgba(192, 132, 252, 0.45), transparent 70%);
                animation-delay: 4s;
            }

            @keyframes float {
                0%, 100% { transform: translate3d(0, -8px, 0); }
                50% { transform: translate3d(0, 12px, 0); }
            }

            .hero-kpi {
                position: relative;
                z-index: 1;
                padding: 2.2rem 2rem;
                border-radius: 24px;
                background: rgba(12, 15, 45, 0.78);
                border: 1px solid rgba(129, 140, 248, 0.4);
                box-shadow: 0 22px 45px rgba(9, 10, 34, 0.55);
                text-align: center;
                display: flex;
                flex-direction: column;
                gap: 0.6rem;
            }

            .hero-kpi-value {
                font-size: 2.8rem;
                font-weight: 600;
            }

            .hero-kpi-label {
                font-size: 0.95rem;
                letter-spacing: 0.06em;
                text-transform: uppercase;
                color: rgba(224, 231, 255, 0.72);
            }

            .hero-kpi-range {
                font-size: 0.95rem;
                color: rgba(226, 232, 255, 0.68);
            }

            .empty-onboarding {
                margin-top: 3rem;
                display: grid;
                grid-template-columns: minmax(240px, 1fr) minmax(320px, 1.05fr);
                gap: clamp(1.5rem, 5vw, 3.5rem);
                padding: clamp(2.4rem, 5vw, 3.8rem);
                border-radius: 40px;
                background: radial-gradient(140% 150% at 18% 20%, rgba(99, 102, 241, 0.22), transparent 60%),
                    radial-gradient(120% 140% at 78% 18%, rgba(244, 114, 182, 0.18), transparent 65%),
                    linear-gradient(135deg, rgba(15, 23, 42, 0.96), rgba(17, 24, 39, 0.78));
                border: 1px solid rgba(148, 163, 255, 0.42);
                box-shadow: 0 48px 95px rgba(8, 12, 44, 0.6);
                position: relative;
                overflow: hidden;
                isolation: isolate;
            }

            .empty-onboarding__scene {
                position: relative;
                min-height: clamp(240px, 32vw, 340px);
                display: flex;
                flex-direction: column;
                align-items: center;
                justify-content: center;
                gap: clamp(1.6rem, 3vw, 2.6rem);
                color: rgba(224, 231, 255, 0.88);
            }

            .empty-onboarding__backdrop {
                position: absolute;
                inset: -60% -50% -55% -60%;
                background: radial-gradient(circle at 35% 55%, rgba(56, 189, 248, 0.55), transparent 70%);
                filter: blur(80px);
                opacity: 0.9;
                z-index: -3;
            }

            .empty-onboarding__halo {
                position: absolute;
                border-radius: 50%;
                border: 1px solid rgba(165, 180, 252, 0.35);
                mix-blend-mode: screen;
            }

            .empty-onboarding__halo--outer {
                width: clamp(260px, 34vw, 360px);
                height: clamp(260px, 34vw, 360px);
                background: radial-gradient(circle, rgba(59, 130, 246, 0.25), transparent 68%);
                z-index: -2;
                animation: haloPulse 8s ease-in-out infinite;
            }

            .empty-onboarding__halo--inner {
                width: clamp(180px, 24vw, 260px);
                height: clamp(180px, 24vw, 260px);
                background: radial-gradient(circle, rgba(244, 114, 182, 0.25), transparent 70%);
                z-index: -1;
                animation: haloPulse 6s ease-in-out infinite reverse;
            }

            @keyframes haloPulse {
                0%,
                100% {
                    transform: scale(0.92);
                    opacity: 0.85;
                }

                50% {
                    transform: scale(1.05);
                    opacity: 1;
                }
            }

            .empty-onboarding__cta-ring {
                position: relative;
                display: flex;
                align-items: center;
                justify-content: center;
                width: clamp(190px, 26vw, 260px);
                height: clamp(190px, 26vw, 260px);
                border-radius: 50%;
                background: radial-gradient(circle, rgba(30, 64, 175, 0.55), rgba(15, 23, 42, 0.4));
                border: 1px solid rgba(148, 163, 255, 0.4);
                box-shadow: inset 0 0 60px rgba(96, 165, 250, 0.35), 0 30px 70px rgba(8, 12, 44, 0.65);
                backdrop-filter: blur(24px);
                overflow: hidden;
            }

            .empty-onboarding__cta-ring::before {
                content: "";
                position: absolute;
                inset: 14%;
                border-radius: 50%;
                background: radial-gradient(circle, rgba(96, 165, 250, 0.65), transparent 70%);
                filter: blur(6px);
                opacity: 0.65;
                animation: ctaGlow 5.5s ease-in-out infinite;
            }

            .empty-onboarding__cta-ring::after {
                content: "";
                position: absolute;
                width: 120%;
                height: 120%;
                background: conic-gradient(from 120deg, rgba(59, 130, 246, 0.55), rgba(147, 197, 253, 0.2), rgba(236, 72, 153, 0.4), rgba(59, 130, 246, 0.55));
                opacity: 0.18;
                animation: orbitSweep 18s linear infinite;
            }

            @keyframes ctaGlow {
                0%,
                100% {
                    transform: scale(0.94);
                    opacity: 0.5;
                }

                50% {
                    transform: scale(1.05);
                    opacity: 0.85;
                }
            }

            @keyframes orbitSweep {
                0% {
                    transform: rotate(0deg);
                }

                100% {
                    transform: rotate(360deg);
                }
            }

            .empty-onboarding__cta {
                position: relative;
                display: inline-flex;
                align-items: center;
                gap: 0.65rem;
                padding: 0.75rem 1.6rem;
                border-radius: 999px;
                text-transform: uppercase;
                letter-spacing: 0.32em;
                font-size: 0.82rem;
                font-weight: 600;
                background: rgba(15, 23, 42, 0.78);
                border: 1px solid rgba(191, 219, 254, 0.28);
                box-shadow: 0 20px 46px rgba(2, 6, 23, 0.55);
                z-index: 1;
            }

            .empty-onboarding__cta::before {
                content: "↖";
                font-size: 1.4rem;
                line-height: 1;
            }

            .empty-onboarding__content {
                position: relative;
                display: flex;
                flex-direction: column;
                gap: clamp(1.2rem, 2.4vw, 2.2rem);
                justify-content: center;
            }

            .empty-onboarding__content::after {
                content: "";
                position: absolute;
                inset: 12% -12% auto auto;
                width: clamp(180px, 32vw, 260px);
                height: clamp(180px, 32vw, 260px);
                background: radial-gradient(circle, rgba(244, 114, 182, 0.28), transparent 70%);
                opacity: 0.8;
                pointer-events: none;
                z-index: -1;
            }

            .empty-onboarding__eyebrow {
                align-self: flex-start;
                padding: 0.55rem 1.4rem;
                border-radius: 999px;
                background: rgba(30, 64, 175, 0.38);
                border: 1px solid rgba(129, 140, 248, 0.55);
                text-transform: uppercase;
                letter-spacing: 0.2em;
                font-size: 0.78rem;
                font-weight: 600;
                color: rgba(239, 246, 255, 0.92);
                box-shadow: 0 16px 36px rgba(9, 12, 34, 0.55);
            }

            .empty-onboarding__content h2 {
                margin: 0;
                font-size: clamp(2.2rem, 4.6vw, 3rem);
                line-height: 1.05;
                max-width: 32rem;
            }

            .empty-onboarding__content p {
                margin: 0;
                font-size: 1.04rem;
                color: rgba(226, 232, 255, 0.85);
                max-width: 33rem;
            }

            .empty-onboarding__steps {
                display: flex;
                flex-wrap: wrap;
                gap: 0.85rem;
            }

            .empty-onboarding__step {
                display: inline-flex;
                align-items: center;
                gap: 0.5rem;
                padding: 0.65rem 1.05rem;
                border-radius: 16px;
                background: rgba(37, 99, 235, 0.18);
                border: 1px solid rgba(129, 140, 248, 0.4);
                font-size: 0.96rem;
                letter-spacing: 0.01em;
                color: rgba(224, 231, 255, 0.9);
                box-shadow: 0 20px 40px rgba(9, 12, 34, 0.5);
            }

            .empty-onboarding__step::before {
                content: "";
                width: 8px;
                height: 8px;
                border-radius: 50%;
                background: linear-gradient(135deg, rgba(56, 189, 248, 0.95), rgba(59, 130, 246, 0.95));
                box-shadow: 0 0 14px rgba(56, 189, 248, 0.7);
            }

            .reports-loading-scrim {
                position: fixed;
                inset: 0;
                display: flex;
                align-items: center;
                justify-content: center;
                background: rgba(6, 10, 28, 0.6);
                backdrop-filter: blur(18px) saturate(120%);
                z-index: 1000;
                pointer-events: none;
                opacity: 0;
                transition: opacity 0.35s ease;
            }

            .reports-loading-scrim--active {
                opacity: 1;
                pointer-events: auto;
            }

            .reports-loading-scrim__content {
                position: relative;
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 1.2rem;
                padding: 2.4rem 3rem;
                border-radius: 30px;
                background: rgba(15, 23, 42, 0.8);
                border: 1px solid rgba(148, 163, 255, 0.32);
                box-shadow: 0 32px 90px rgba(2, 6, 23, 0.6);
                overflow: hidden;
            }

            .reports-loading-scrim__content::before {
                content: "";
                position: absolute;
                inset: -40% -50% auto -50%;
                height: 140%;
                background: radial-gradient(circle at top, rgba(96, 165, 250, 0.35), transparent 60%);
                opacity: 0.75;
                pointer-events: none;
            }

            .reports-loading-scrim__halo {
                position: absolute;
                width: 220px;
                height: 220px;
                border-radius: 50%;
                background: radial-gradient(circle, rgba(59, 130, 246, 0.35), rgba(14, 165, 233, 0.12), transparent 72%);
                filter: blur(2px);
                animation: scrimPulse 7s ease-in-out infinite;
            }

            .reports-loading-scrim__ring {
                position: relative;
                width: 82px;
                height: 82px;
                border-radius: 50%;
                border: 4px solid rgba(148, 163, 255, 0.25);
                border-top-color: rgba(96, 165, 250, 0.95);
                border-right-color: rgba(59, 130, 246, 0.65);
                animation: scrimSpin 1.2s linear infinite;
                box-shadow: 0 0 30px rgba(59, 130, 246, 0.45);
            }

            .reports-loading-scrim__label {
                font-size: 1rem;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: rgba(226, 232, 255, 0.85);
            }

            @keyframes scrimSpin {
                0% {
                    transform: rotate(0deg);
                }

                100% {
                    transform: rotate(360deg);
                }
            }

            @keyframes scrimPulse {
                0%,
                100% {
                    transform: scale(0.92);
                    opacity: 0.7;
                }

                50% {
                    transform: scale(1);
                    opacity: 1;
                }
            }

            .loading-indicator {
                display: flex;
                flex-direction: column;
                align-items: center;
                gap: 0.85rem;
                padding: 1.6rem 0;
                margin: 0 auto;
                text-align: center;
                width: fit-content;
            }

            .loading-indicator__dots {
                display: flex;
                gap: 0.55rem;
            }

            .loading-indicator__dot {
                width: 18px;
                height: 18px;
                border-radius: 50%;
                background: linear-gradient(135deg, rgba(99, 102, 241, 0.9), rgba(14, 165, 233, 0.9));
                box-shadow: 0 0 18px rgba(56, 189, 248, 0.65);
                opacity: 0.75;
                animation: loadingPulse 1.9s ease-in-out infinite;
            }

            .loading-indicator__dot:nth-child(2) {
                animation-delay: 0.25s;
            }

            .loading-indicator__dot:nth-child(3) {
                animation-delay: 0.5s;
            }

            @keyframes loadingPulse {
                0%, 100% {
                    transform: scale(0.72) translateY(0);
                    opacity: 0.45;
                }
                50% {
                    transform: scale(1.18) translateY(-6px);
                    opacity: 1;
                }
            }

            .loading-indicator__label {
                font-size: 0.95rem;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                color: rgba(226, 232, 255, 0.82);
            }

            .metric-row {
                margin-top: 2.4rem;
            }

            .metric-row [data-testid="column"] {
                padding: 0.4rem;
            }

            .metric-card {
                width: 100%;
                background: linear-gradient(135deg, rgba(148, 163, 255, 0.22), rgba(56, 189, 248, 0.18));
                border-radius: 20px;
                padding: 1.5rem 1.6rem;
                border: 1px solid rgba(180, 198, 255, 0.35);
                box-shadow: 0 24px 40px rgba(5, 10, 30, 0.45);
                display: flex;
                flex-direction: column;
                gap: 0.3rem;
            }

            .metric-label {
                font-size: 0.75rem;
                text-transform: uppercase;
                letter-spacing: 0.14em;
                color: rgba(229, 234, 255, 0.72);
            }

            .metric-value {
                font-family: 'Space Grotesk', sans-serif;
                font-size: 1.9rem;
                font-weight: 600;
                color: #f9fbff;
            }

            .section-card {
                margin-top: 2.8rem;
                padding: 2.4rem 2.6rem;
                border-radius: 28px;
                background: rgba(12, 17, 44, 0.78);
                border: 1px solid rgba(148, 163, 255, 0.25);
                box-shadow: 0 32px 60px rgba(6, 10, 32, 0.5);
                backdrop-filter: blur(18px);
            }

            .section-card-header h3 {
                margin: 0.4rem 0 0.7rem;
                font-size: 1.6rem;
            }

            .section-card-header p {
                margin: 0;
                color: rgba(224, 231, 255, 0.72);
                font-size: 0.98rem;
            }

            .section-kicker {
                display: inline-flex;
                align-items: center;
                padding: 0.35rem 0.9rem;
                border-radius: 999px;
                text-transform: uppercase;
                letter-spacing: 0.12em;
                font-size: 0.75rem;
                color: rgba(228, 233, 255, 0.85);
                background: rgba(99, 102, 241, 0.25);
                border: 1px solid rgba(148, 163, 255, 0.35);
            }

            .data-card div[data-testid="stDataFrame"] {
                margin-top: 1.8rem;
                border-radius: 22px;
                overflow: hidden;
                box-shadow: 0 18px 40px rgba(6, 10, 32, 0.55);
                background: rgba(9, 14, 36, 0.92);
                border: 1px solid rgba(129, 140, 248, 0.32);
            }

            .data-card div[data-testid="stDataFrame"] table {
                color: rgba(227, 233, 255, 0.92);
            }

            .data-card div[data-testid="stDataFrame"] thead tr th {
                background: rgba(15, 23, 42, 0.95);
                color: rgba(239, 246, 255, 0.82);
                border-bottom: 1px solid rgba(129, 140, 248, 0.35);
            }

            .data-card div[data-testid="stDataFrame"] tbody tr td {
                background: rgba(11, 18, 44, 0.76);
                border-bottom: 1px solid rgba(71, 85, 139, 0.35);
            }

            .data-card div[data-testid="stDataFrame"] tbody tr:hover td {
                background: rgba(32, 41, 86, 0.85);
            }

            .theme-summary-card {
                position: relative;
                overflow: hidden;
            }

            .theme-summary-card::before {
                content: "";
                position: absolute;
                inset: 0;
                background: radial-gradient(circle at 0% 0%, rgba(192, 132, 252, 0.22), transparent 55%),
                    radial-gradient(circle at 85% 15%, rgba(56, 189, 248, 0.18), transparent 55%);
                opacity: 0.9;
                pointer-events: none;
            }

            .theme-summary-card__header {
                position: relative;
                display: grid;
                gap: 1.4rem;
                grid-template-columns: minmax(0, 1fr);
            }

            .theme-summary-card__title h3 {
                margin: 0.2rem 0 0.4rem;
                font-size: 1.5rem;
            }

            .theme-summary-card__meta {
                display: inline-flex;
                flex-wrap: wrap;
                gap: 0.6rem;
            }

            .theme-summary-chip {
                display: inline-flex;
                align-items: center;
                gap: 0.35rem;
                padding: 0.45rem 0.85rem;
                border-radius: 999px;
                font-size: 0.78rem;
                letter-spacing: 0.08em;
                text-transform: uppercase;
                background: rgba(148, 163, 255, 0.18);
                border: 1px solid rgba(148, 163, 255, 0.38);
                color: rgba(226, 232, 255, 0.82);
            }

            .theme-summary-chip--accent {
                background: linear-gradient(135deg, rgba(59, 130, 246, 0.28), rgba(192, 132, 252, 0.32));
                border-color: rgba(96, 165, 250, 0.5);
            }

            .theme-summary-highlight {
                position: relative;
                margin-top: 1.8rem;
                padding: 1.2rem 1.4rem;
                border-radius: 20px;
                background: rgba(15, 23, 42, 0.78);
                border: 1px solid rgba(148, 163, 255, 0.24);
                display: flex;
                gap: 1rem;
                align-items: center;
            }

            .theme-summary-highlight__icon {
                font-size: 1.5rem;
            }

            .theme-summary-highlight__content {
                display: flex;
                flex-direction: column;
                gap: 0.25rem;
            }

            .theme-summary-highlight__label {
                font-size: 0.75rem;
                text-transform: uppercase;
                letter-spacing: 0.14em;
                color: rgba(226, 232, 255, 0.72);
            }

            .theme-summary-highlight__value {
                font-family: 'Space Grotesk', sans-serif;
                font-size: 1.3rem;
                color: #f9fbff;
            }

            .theme-summary-highlight__meta {
                font-size: 0.9rem;
                color: rgba(226, 232, 255, 0.7);
            }

            .theme-summary-card__expander {
                position: relative;
                margin-top: 1.6rem;
            }

            .theme-summary-card__expander div[data-testid="stExpander"] {
                border-radius: 18px;
                border: 1px solid rgba(148, 163, 255, 0.3);
                background: rgba(15, 23, 42, 0.74);
                box-shadow: 0 18px 40px rgba(6, 10, 32, 0.4);
            }

            .theme-summary-card__expander button[aria-expanded="true"],
            .theme-summary-card__expander button[aria-expanded="false"] {
                color: rgba(224, 231, 255, 0.85);
                font-weight: 600;
                font-size: 0.95rem;
            }

            .theme-summary-table-caption {
                font-size: 0.82rem;
                text-transform: uppercase;
                letter-spacing: 0.14em;
                color: rgba(199, 210, 254, 0.7);
                margin-bottom: 0.6rem;
            }

            .theme-summary-card__expander div[data-testid="stDataFrame"] {
                margin-top: 0.4rem;
                border-radius: 16px;
                overflow: hidden;
                border: 1px solid rgba(129, 140, 248, 0.28);
                background: rgba(9, 14, 36, 0.9);
            }

            .theme-summary-card__expander div[data-testid="stDataFrame"] thead tr th {
                background: rgba(15, 23, 42, 0.94);
            }

            .theme-summary-card__expander div[data-testid="stDataFrame"] tbody tr td {
                background: rgba(17, 24, 54, 0.82);
            }

            .action-section {
                margin-top: 3rem;
            }

            .action-section [data-testid="column"] {
                padding: 0.5rem;
            }

            .action-card {
                position: relative;
                padding: 1.7rem 1.8rem 1.6rem;
                border-radius: 24px;
                background: linear-gradient(135deg, rgba(15, 23, 42, 0.88), rgba(37, 69, 180, 0.6));
                border: 1px solid rgba(99, 102, 241, 0.42);
                box-shadow: 0 26px 52px rgba(4, 8, 28, 0.65);
                backdrop-filter: blur(18px);
                display: flex;
                flex-direction: column;
                justify-content: space-between;
                min-height: 10rem;
                transition: transform 0.28s ease, box-shadow 0.28s ease;
                isolation: isolate;
            }

            .action-card::after {
                content: "";
                position: absolute;
                inset: 0;
                border-radius: inherit;
                background: linear-gradient(135deg, rgba(79, 70, 229, 0.08), rgba(59, 130, 246, 0.14));
                opacity: 0;
                transition: opacity 0.28s ease;
                z-index: 0;
            }

            .action-card:not(.is-disabled):hover,
            .action-card:not(.is-disabled):has(+ div[data-testid="stElementContainer"] button:focus-visible) {
                transform: translateY(-3px);
                box-shadow: 0 32px 60px rgba(16, 20, 58, 0.65);
            }

            .action-card:not(.is-disabled):hover::after,
            .action-card:not(.is-disabled):has(+ div[data-testid="stElementContainer"] button:focus-visible)::after {
                opacity: 1;
            }

            .action-card.is-disabled::after {
                opacity: 0.4;
                background: rgba(12, 18, 43, 0.72);
            }

            .action-card-content {
                position: relative;
                z-index: 1;
                display: flex;
                flex-direction: column;
                gap: 1.05rem;
                flex: 1 1 auto;
            }

            .action-card-heading {
                display: flex;
                align-items: center;
                gap: 0.7rem;
                font-weight: 700;
                font-size: 1.08rem;
                letter-spacing: 0.01em;
                color: #f8fafc;
            }

            .action-card-icon {
                font-size: 1.35rem;
                line-height: 1;
            }

            .action-card-title {
                display: inline-flex;
                align-items: center;
            }

            div[data-testid="stVerticalBlock"]:has(.action-card) {
                position: relative;
                width: 100%;
                overflow: visible;
                height: 100%;
            }

            div[data-testid="stElementContainer"]:has(.action-card) {
                position: relative;
                z-index: 1;
                height: 100%;
            }

            div[data-testid="stVerticalBlock"]:has(.action-card) > div[data-testid="stElementContainer"]:has(div[data-testid="stButton"]) {
                position: absolute;
                inset: 0;
                margin: 0;
                z-index: 2;
            }

            div[data-testid="stVerticalBlock"]:has(.action-card) > div[data-testid="stElementContainer"]:has(div[data-testid="stButton"]) > div[data-testid="stButton"] {
                width: 100%;
                height: 100%;
            }

            div[data-testid="stVerticalBlock"]:has(.action-card) > div[data-testid="stElementContainer"]:has(div[data-testid="stButton"]) button {
                width: 100%;
                height: 100%;
                border: none;
                background: transparent;
                color: transparent;
                cursor: pointer;
                opacity: 0;
            }

            div[data-testid="stVerticalBlock"]:has(.action-card) > div[data-testid="stElementContainer"]:has(div[data-testid="stButton"]) button:focus-visible {
                outline: none;
            }

            div[data-testid="stVerticalBlock"]:has(.action-card.is-disabled) > div[data-testid="stElementContainer"]:has(div[data-testid="stButton"]) button {
                pointer-events: none;
                cursor: not-allowed;
            }

            .action-section div[data-testid="column"]:has(.action-card) {
                display: flex;
                flex-direction: column;
            }

            .action-section div[data-testid="column"]:has(.action-card) > div[data-testid="stVerticalBlock"] {
                display: flex;
                flex-direction: column;
                flex: 1 1 auto;
            }

            .action-section div[data-testid="column"]:has(.action-card) > div[data-testid="stVerticalBlock"] > div[data-testid="stVerticalBlock"] {
                flex: 1 1 auto;
            }

            .action-card-copy {
                color: rgba(224, 231, 255, 0.76);
                font-size: 0.94rem;
                margin: 0;
            }

            div[data-testid="stAlert"] {
                border-radius: 18px;
                border: 1px solid rgba(148, 163, 255, 0.35);
                background: rgba(12, 22, 54, 0.85);
            }

            div[data-testid="stAlert"] p {
                color: rgba(226, 232, 255, 0.88);
            }

            @media (max-width: 1200px) {
                .beta-banner__content {
                    flex-direction: column;
                    align-items: flex-start;
                    gap: 1rem;
                }

                .beta-banner__sparks {
                    margin-left: 0;
                }

                .hero-section {
                    grid-template-columns: 1fr;
                    padding: 2.6rem;
                }

                .hero-visual {
                    margin-top: 2rem;
                }

                .empty-onboarding {
                    grid-template-columns: 1fr;
                    justify-items: center;
                    text-align: center;
                    padding: 2.8rem 2.4rem;
                }

                .empty-onboarding__scene {
                    min-height: clamp(220px, 40vw, 320px);
                }

                .empty-onboarding__content {
                    align-items: center;
                }

                .empty-onboarding__content::after {
                    inset: auto auto -10% 50%;
                    transform: translateX(-50%);
                }

                .empty-onboarding__eyebrow {
                    align-self: center;
                }

                .empty-onboarding__content h2,
                .empty-onboarding__content p {
                    max-width: 38rem;
                }

                .workspace-footer div[data-testid="column"] {
                    flex: 1 1 100% !important;
                }
            }

            @media (max-width: 900px) {
                .beta-banner {
                    padding: 1.3rem 1.5rem;
                }

                .beta-banner__text {
                    max-width: 100%;
                }

                .empty-onboarding {
                    padding: 2.4rem;
                }

                .empty-onboarding__cta {
                    letter-spacing: 0.28em;
                }

                .empty-onboarding__cta-ring {
                    width: clamp(180px, 44vw, 240px);
                    height: clamp(180px, 44vw, 240px);
                }

                .metric-row [data-testid="column"],
                .action-section [data-testid="column"] {
                    flex: 1 1 100% !important;
                    width: 100% !important;
                }
            }

            @media (max-width: 600px) {
                .beta-banner {
                    padding: 1.1rem 1.25rem;
                    border-radius: 22px;
                }

                .beta-banner__headline {
                    font-size: 1rem;
                }

                .beta-banner__subtext {
                    font-size: 0.88rem;
                }

                .hero-section {
                    padding: 2.2rem;
                }

                .hero-section h1 {
                    font-size: 2.2rem;
                }

                .empty-onboarding {
                    padding: 1.85rem;
                    border-radius: 28px;
                }

                .empty-onboarding__cta {
                    font-size: 0.76rem;
                    letter-spacing: 0.26em;
                }

                .empty-onboarding__cta-ring {
                    width: clamp(170px, 58vw, 220px);
                    height: clamp(170px, 58vw, 220px);
                }

                .empty-onboarding__content h2 {
                    font-size: 2.05rem;
                }

                .empty-onboarding__steps {
                    justify-content: center;
                }

                .metric-row {
                    margin-top: 2rem;
                }
            }

        </style>
        """,
        unsafe_allow_html=True,
    )

    _ensure_loading_overlay_placeholder()

    _build_sidebar()
    header_container = st.container()
    _render_header(header_container)
    flash_payload = _consume_flash_message()
    _render_action_tiles()
    _render_active_action()
    _render_workspace_footer(flash_payload)


if __name__ == "__main__":
    main()
