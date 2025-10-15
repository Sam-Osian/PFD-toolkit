"""Interactive Streamlit dashboard for the PFD Toolkit API."""
from __future__ import annotations

import ast
import copy
import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

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


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _init_session_state() -> None:
    """Ensure default keys exist in ``st.session_state``."""
    defaults = {
        "reports_df": pd.DataFrame(),
        "reports_df_initial": None,
        "history": [],
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
    }
    for key, value in defaults.items():
        st.session_state.setdefault(key, value)


def _styled_metric(label: str, value: Any) -> None:
    """Render a metric with consistent styling."""
    st.markdown(
        f"""
        <div class="metric-card">
            <div class="metric-label">{label}</div>
            <div class="metric-value">{value}</div>
        </div>
        """,
        unsafe_allow_html=True,
    )


FLASH_MESSAGE_KEY = "flash_message"


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


def push_history_snapshot() -> None:
    """Push the current state onto the undo history (max depth 10)."""

    history = list(st.session_state.get("history", []))
    history.append(snapshot_state())
    if len(history) > 10:
        history = history[-10:]
    st.session_state["history"] = history


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
        "Text": str,
        "Boolean": bool,
        "Integer": int,
        "Decimal": float,
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
    st.dataframe(df, use_container_width=True, hide_index=True)


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
            use_container_width=True,
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
        "<h2 style='margin-bottom:0.2rem;'>PFD Toolkit Workbench</h2>",
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
        "Chat model",
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
        placeholder="e.g. 50",
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

    load_button = st.sidebar.button("Load in reports", use_container_width=True)

    if load_button:
        if n_reports_raw.strip() and n_reports is None:
            st.sidebar.error("Enter a valid integer before loading reports.")
            return
        if end_date < start_date:
            st.sidebar.error("Fix the report date range before loading reports.")
            return
        try:
            with st.spinner("Downloading and filtering reports..."):
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
            st.session_state["active_action"] = None
            clear_preview_state()
            st.success(f"Loaded {len(df)} reports into the workspace.")
        except Exception as exc:  # pragma: no cover - UI feedback
            st.error(f"Could not load reports: {exc}")

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
    """Render the pinned header containing KPIs and the active dataset."""

    ctx = container or st

    ctx.title("PFD Toolkit AI Workbench")
    ctx.markdown(
        """
        <p class="lead">
            Explore Prevention of Future Death (PFD) reports, screen them against custom topics,
            and extract structured insights using the Screener and Extractor APIs.
            Configure your data and model in the sidebar, then work through the guided flows below.
        </p>
        """,
        unsafe_allow_html=True,
    )

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

    col1, col2, col3 = ctx.columns(3)
    with col1:
        _styled_metric("Reports in view", f"{reports_count:,}")
    with col2:
        _styled_metric("Earliest date", earliest_display)
    with col3:
        _styled_metric("Latest date", latest_display)

    ctx.markdown("#### Current working dataset")
    if reports_df.empty:
        ctx.info(
            "No reports loaded yet. Use the **sidebar** to configure and load reports."
        )
    else:
        ctx.dataframe(reports_df, use_container_width=True, hide_index=True)

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


def _render_flash_message() -> None:
    """Display and clear any queued status message."""

    flash_payload = st.session_state.pop(FLASH_MESSAGE_KEY, None)
    if not flash_payload:
        return

    message = flash_payload.get("message")
    if not message:
        return

    level = flash_payload.get("level", "info")
    display_fn = getattr(st, level, st.info)
    display_fn(message)


def _render_action_tiles() -> None:
    """Render the single-column action tiles."""

    reports_df = _get_reports_df()
    dataset_available = not reports_df.empty
    llm_ready = st.session_state.get("llm_client") is not None
    history = st.session_state.get("history", [])
    initial_df = st.session_state.get("reports_df_initial")

    if not dataset_available and st.session_state.get("active_action") is not None:
        st.session_state["active_action"] = None

    st.markdown("### What would you like to do next?")

    if st.button(
        "Save working dataset to file",
        key="tile_save",
        use_container_width=True,
        disabled=not dataset_available,
    ):
        st.session_state["active_action"] = "save"

    if st.button(
        "Filter reports (LLM Screener)",
        key="tile_filter",
        use_container_width=True,
        disabled=not (dataset_available and llm_ready),
    ):
        st.session_state["active_action"] = "filter"

    if st.button(
        "Discover recurring themes (Extractor)",
        key="tile_discover",
        use_container_width=True,
        disabled=not (dataset_available and llm_ready),
    ):
        st.session_state["active_action"] = "discover"

    if st.button(
        "Pull out structured information (Extractor)",
        key="tile_extract",
        use_container_width=True,
        disabled=not (dataset_available and llm_ready),
    ):
        st.session_state["active_action"] = "extract"

    if history:
        undo_container = st.container()
        undo_container.markdown(
            "<div class='pfd-undo-button'>",
            unsafe_allow_html=True,
        )
        if undo_container.button(
            "↶ Undo most recent change",
            key="tile_undo",
            use_container_width=True,
        ):
            _undo_last_change()
        undo_container.markdown("</div>", unsafe_allow_html=True)

    start_again_disabled = not (
        isinstance(initial_df, pd.DataFrame) and not initial_df.empty
    )
    restart_container = st.container()
    restart_container.markdown(
        "<div class='pfd-start-button'>",
        unsafe_allow_html=True,
    )
    if restart_container.button(
        "↻ Start over",
        key="tile_reset",
        use_container_width=True,
        disabled=start_again_disabled,
    ):
        _start_again()
    restart_container.markdown("</div>", unsafe_allow_html=True)


def _undo_last_change() -> None:
    """Pop the latest snapshot from history and restore it."""

    history = list(st.session_state.get("history", []))
    if not history:
        return

    snapshot = history.pop()
    st.session_state["history"] = history
    restore_state(snapshot)
    st.session_state["active_action"] = None
    _queue_status_message("Reverted to the previous state.")
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
    st.markdown("#### Save working dataset to file")
    if reports_df.empty:
        st.info("No reports available to download yet.")
        return

    st.download_button(
        "Download working dataset as CSV",
        data=reports_df.to_csv(index=False).encode("utf-8"),
        file_name="pfd_reports.csv",
        mime="text/csv",
        use_container_width=True,
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

    st.markdown("#### Filter reports (LLM Screener)")
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
                use_container_width=True,
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

    progress_placeholder = st.empty()
    progress_bar = progress_placeholder.progress(0)
    try:
        progress_bar.progress(10)
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
        progress_bar.progress(25)
        with st.spinner("Running the screener..."):
            result_df = screener.screen_reports(
                search_query=search_query or None,
                filter_df=filter_df,
                result_col_name=match_column_name,
                produce_spans=produce_spans,
                drop_spans=drop_spans,
            )
        progress_bar.progress(100)
    except Exception as exc:  # pragma: no cover - relies on live API
        st.error(f"Screening failed: {exc}")
        return
    finally:
        progress_placeholder.empty()

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
            "Preview recurring themes", use_container_width=True
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
            progress_placeholder = st.empty()
            progress_bar = progress_placeholder.progress(0)
            try:
                progress_bar.progress(10)
                summary_col_name = extractor.summary_col or "summary"
                with st.spinner("Summarising the reports..."):
                    summary_df = extractor.summarise(
                        result_col_name=summary_col_name,
                        trim_intensity=trim_labels[trim_choice],
                    )
                progress_bar.progress(55)

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

                progress_bar.progress(70)
                with st.spinner("Identifying themes..."):
                    ThemeModel = extractor.discover_themes(
                        warn_exceed=int(warning_threshold),
                        error_exceed=int(error_threshold),
                        max_themes=max_themes_value,
                        min_themes=min_themes_value,
                        extra_instructions=extra_theme_instructions or None,
                        seed_topics=seed_topics,
                    )

                if ThemeModel is None or not hasattr(ThemeModel, "model_json_schema"):
                    st.warning(
                        "Theme discovery completed but did not return a schema."
                    )
                    clear_preview_state()
                else:
                    theme_schema = ThemeModel.model_json_schema()
                    with st.spinner("Assigning themes to reports..."):
                        preview_df = extractor.extract_features(
                            feature_model=ThemeModel,
                            force_assign=True,
                            allow_multiple=True,
                            skip_if_present=False,
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
                    st.success(
                        "Preview ready. Review the results below and apply them when happy."
                    )
            except Exception as exc:  # pragma: no cover - depends on live API
                st.error(f"Theme discovery failed: {exc}")
                clear_preview_state()
            finally:
                progress_placeholder.empty()

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
    if actions_col1.button("Apply themes", use_container_width=True):
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

    if actions_col2.button("Cancel preview", use_container_width=True):
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

    default_grid = st.session_state.get("feature_grid")
    if default_grid is None:
        default_grid = pd.DataFrame(
            [
                {
                    "Field name": "risk_factor",
                    "Description": "Primary risk factor contributing to the death.",
                    "Type": "Text",
                },
                {
                    "Field name": "is_healthcare",
                    "Description": "True if the report involves a healthcare setting.",
                    "Type": "Boolean",
                },
            ]
        )

    feature_grid = st.data_editor(
        default_grid,
        num_rows="dynamic",
        use_container_width=True,
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
                options=["Text", "Boolean", "Integer", "Decimal"],
                help="Choose the data type the extractor should return.",
            ),
        },
        key="feature_editor",
    )
    st.session_state["feature_grid"] = feature_grid

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
            "Tag the reports", use_container_width=True
        )

    if not extract_submitted:
        return

    push_history_snapshot()
    progress_placeholder = st.empty()
    progress_bar = progress_placeholder.progress(0)
    try:
        progress_bar.progress(10)
        feature_model = _build_feature_model_from_grid(feature_grid)
        target_df = reports_df
        with st.spinner("Extracting structured data..."):
            result_df = extractor.extract_features(
                reports=target_df,
                feature_model=feature_model,
                produce_spans=produce_spans,
                drop_spans=drop_spans,
                force_assign=force_assign,
                allow_multiple=allow_multiple,
                schema_detail="minimal",
                extra_instructions=extra_instructions or None,
                skip_if_present=skip_if_present,
            )
        progress_bar.progress(100)
    except Exception as exc:  # pragma: no cover - depends on live API
        st.error(f"Extraction failed: {exc}")
        return
    finally:
        progress_placeholder.empty()

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
        page_title="PFD Toolkit AI Workbench",
        page_icon="📊",
        layout="wide",
    )
    _init_session_state()

    st.markdown(
        """
        <style>
            .stApp {background: radial-gradient(circle at top, #f0f4ff 0%, #ffffff 60%);}
            .metric-card {
                background: linear-gradient(135deg, #1b64f2 0%, #4f46e5 100%);
                padding: 1.2rem;
                border-radius: 1rem;
                color: white;
                text-align: center;
                box-shadow: 0 8px 20px rgba(79, 70, 229, 0.25);
            }
            .metric-label {font-size: 0.9rem; opacity: 0.85;}
            .metric-value {font-size: 1.6rem; font-weight: 600; margin-top: 0.2rem;}
            .lead {font-size: 1.1rem; color: #1f2937;}
            div[data-testid="stButton"] {margin-bottom: 0.75rem;}
            div[data-testid="stButton"] > button {
                border-radius: 0.9rem;
                padding: 1rem 1.2rem;
                font-size: 1rem;
                font-weight: 600;
                border: 1px solid rgba(79, 70, 229, 0.35);
                background: linear-gradient(90deg, #eef2ff 0%, #e0e7ff 100%);
                color: #1f2937;
            }
            div[data-testid="stButton"] > button:disabled {
                cursor: not-allowed;
                opacity: 0.55;
            }
            .pfd-undo-button button {
                border: 1px solid rgba(148, 163, 184, 0.6);
                background: linear-gradient(90deg, #f9fafb 0%, #e5e7eb 100%);
                color: #1f2937;
            }
            .pfd-undo-button button:hover {
                background: linear-gradient(90deg, #e5e7eb 0%, #d1d5db 100%);
            }
            .pfd-start-button button {
                border: 1px solid rgba(148, 163, 184, 0.6);
                background: linear-gradient(90deg, #f9fafb 0%, #e5e7eb 100%);
                color: #1f2937;
            }
            .pfd-start-button button:hover {
                background: linear-gradient(90deg, #e5e7eb 0%, #d1d5db 100%);
            }
        </style>
        """,
        unsafe_allow_html=True,
    )

    _build_sidebar()
    header_container = st.container()
    _render_header(header_container)
    st.markdown("---")
    _render_flash_message()
    _render_action_tiles()
    _render_active_action()


if __name__ == "__main__":
    main()
