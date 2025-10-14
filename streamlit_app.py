"""Interactive Streamlit dashboard for the PFD Toolkit API."""
from __future__ import annotations

import json
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import pandas as pd
import streamlit as st
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
        "reports_df_modified": False,
        "screener_result": None,
        "extractor_result": None,
        "summary_result": None,
        "theme_model_schema": None,
        "llm_client": None,
        "extractor": None,
        "extractor_source_signature": None,
        "extractor_verbose": False,
        "seed_topics_last": None,
        "feature_grid": None,
        "extractor_mode": "Discover themes in the reports",
        "active_tab": "Load in reports",
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
    model_display_options = ["GPT 4.1", "GPT 4.1 Mini"]
    model_value_map = {"GPT 4.1": "gpt-4.1", "GPT 4.1 Mini": "gpt-4.1-mini"}
    current_model_value = st.session_state.get("model_name", "gpt-4.1")
    current_display = next(
        (label for label, value in model_value_map.items() if value == current_model_value),
        "GPT 4.1",
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
    max_workers = st.sidebar.number_input("Max parallel workers", min_value=1, max_value=32, value=8)
    validation_attempts = 2

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Report window")
    default_start = date(2013, 1, 1)
    date_range = st.sidebar.date_input(
        "Report date range",
        value=(default_start, date.today()),
        min_value=date(2000, 1, 1),
        max_value=date.today(),
    )
    if isinstance(date_range, tuple):
        start_date, end_date = date_range
    else:
        start_date = date_range
        end_date = date.today()

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
            st.session_state["extractor"] = None
            st.session_state["extractor_source_signature"] = None
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

def _render_load_reports_tab() -> None:
    st.title("PFD Toolkit AI Workbench")
    st.markdown(
        """
        <p class="lead">
            Explore Prevention of Future Death (PFD) reports, screen them against custom topics,
            and extract structured insights using the Screener and Extractor APIs.
            Configure your data and model in the sidebar, then walk through the guided flows below.
        </p>
        """,
        unsafe_allow_html=True,
    )

    reports_df: pd.DataFrame = st.session_state.get("reports_df", pd.DataFrame())
    if not reports_df.empty:
        col1, col2, col3 = st.columns(3)
        with col1:
            _styled_metric("Reports in view", f"{len(reports_df):,}")
        with col2:
            date_min = reports_df.get("date", pd.Series(dtype="datetime64[ns]")).min()
            if pd.notna(date_min):
                _styled_metric("Earliest date", date_min.strftime("%d %b %Y"))
        with col3:
            date_max = reports_df.get("date", pd.Series(dtype="datetime64[ns]")).max()
            if pd.notna(date_max):
                _styled_metric("Latest date", date_max.strftime("%d %b %Y"))

    _render_reports_overview(
        "Loaded Prevention of Future Death reports", key_suffix="load_tab"
    )


def _render_screener_tab() -> None:
    st.subheader("Screen reports")
    st.markdown(
        """
        Ask the Screener to search every report for a topic or scenario. All report fields are sent to the
        language model so you only need to describe what you are looking for.
        """
    )
    st.caption(
        "Example: *Find reports describing medication errors in care homes.*"
    )

    llm_client: Optional[LLM] = st.session_state.get("llm_client")
    reports_df: pd.DataFrame = st.session_state.get("reports_df", pd.DataFrame())

    if llm_client is None:
        st.warning("Add a valid API key in the sidebar to enable the Screener.")
        return
    if reports_df.empty:
        st.info("Load reports from the sidebar before screening.")
        return

    reports_container = st.container()
    with reports_container:
        _render_reports_overview("Loaded reports in workspace", key_suffix="screener")

    with st.form("screener_form", enter_to_submit=False):
        st.markdown("#### Screening request")
        search_query = st.text_area(
            "Describe what you want to find",
            placeholder="e.g. Reports mentioning delays in ambulance response times",
        )
        cols3 = st.columns(2)
        filter_df = cols3[0].checkbox("Only keep matching reports", value=True)
        result_col_name = cols3[1].text_input("Name for the match column", value="matches_query")

        with st.expander("Advanced options"):
            verbose = st.checkbox("Show detailed logs in the terminal", value=False)
            produce_spans = st.checkbox("Return supporting quotes from the reports", value=False)
            drop_spans = st.checkbox(
                "Remove the quotes column from the results", value=False
            )

        submitted = st.form_submit_button("Run Screener", use_container_width=True)

    if submitted:
        if not search_query.strip():
            st.error("Describe what the Screener should look for.")
            with reports_container:
                _render_reports_overview(
                    "Loaded reports in workspace", key_suffix="screener"
                )
            return
        progress_placeholder = st.empty()
        progress_bar = progress_placeholder.progress(0)
        try:
            progress_bar.progress(10)
            screener = Screener(
                llm=llm_client,
                reports=reports_df,
                verbose=verbose,
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
                    result_col_name=result_col_name or "matches_query",
                    produce_spans=produce_spans,
                    drop_spans=drop_spans,
                )
            progress_bar.progress(100)
            st.session_state["screener_result"] = result_df
            st.session_state["reports_df"] = result_df.copy(deep=True)
            st.session_state["extractor"] = None
            st.session_state["extractor_source_signature"] = None
            st.session_state["extractor_result"] = None
            st.session_state["summary_result"] = None
            st.session_state["theme_model_schema"] = None
            st.success("Screening complete. Review the results below.")
        except Exception as exc:  # pragma: no cover - relies on live API
            st.error(f"Screening failed: {exc}")
        finally:
            progress_placeholder.empty()

        with reports_container:
            _render_reports_overview("Loaded reports in workspace", key_suffix="screener")

    result_df = st.session_state.get("screener_result")
    if isinstance(result_df, pd.DataFrame):
        _display_dataframe(result_df, "Screener output")
        st.download_button(
            "Download screener results as CSV",
            data=result_df.to_csv(index=False).encode("utf-8"),
            file_name="pfd_screener_results.csv",
            mime="text/csv",
            use_container_width=True,
        )



def _render_extractor_tab() -> None:
    st.subheader("Extract insights")
    st.markdown(
        """
        Use the Extractor to surface new themes automatically or to tag reports with an existing classification scheme.
        All report fields are sent to the language model for richer context.
        """
    )

    llm_client: Optional[LLM] = st.session_state.get("llm_client")
    reports_df: pd.DataFrame = st.session_state.get("reports_df", pd.DataFrame())

    if llm_client is None:
        st.warning("Add a valid API key in the sidebar to enable the Extractor.")
        return
    if reports_df.empty:
        st.info("Load reports from the sidebar before extracting insights.")
        return

    screened_df = st.session_state.get("screener_result")
    has_screened_df = isinstance(screened_df, pd.DataFrame) and not screened_df.empty

    if has_screened_df:
        dataset_label_map = {
            "Use screened reports": "screened",
            "Use all loaded reports": "loaded",
        }
        dataset_choice = st.radio(
            "Select the reports to analyse",
            list(dataset_label_map.keys()),
            index=0,
            horizontal=True,
            key="extractor_dataset_choice",
        )
        active_reports_df = screened_df if dataset_label_map[dataset_choice] == "screened" else reports_df
    else:
        if isinstance(screened_df, pd.DataFrame) and screened_df.empty:
            st.warning(
                "The screener did not retain any reports. Falling back to the loaded dataset."
            )
        active_reports_df = reports_df

    st.caption(
        f"Working with {len(active_reports_df)} report(s) based on the current selection."
    )

    reports_container = st.container()
    with reports_container:
        _render_reports_overview("Loaded reports in workspace", key_suffix="extractor")

    previous_verbose = bool(st.session_state.get("extractor_verbose", False))
    verbose = st.checkbox(
        "Show detailed logs in the terminal",
        value=previous_verbose,
    )
    st.session_state["extractor_verbose"] = verbose

    extractor_signature = (len(active_reports_df), tuple(active_reports_df.columns))
    extractor: Optional[Extractor] = st.session_state.get("extractor")

    if (
        extractor is None
        or st.session_state.get("extractor_source_signature") != extractor_signature
        or previous_verbose != verbose
    ):
        try:
            extractor = Extractor(
                llm=llm_client,
                reports=active_reports_df,
                include_date=True,
                include_coroner=True,
                include_area=True,
                include_receiver=True,
                include_investigation=True,
                include_circumstances=True,
                include_concerns=True,
                verbose=verbose,
            )
            st.session_state["extractor"] = extractor
            st.session_state["extractor_source_signature"] = extractor_signature
        except Exception as exc:  # pragma: no cover - depends on live API
            st.error(f"Could not initialise extractor: {exc}")
            return

    extractor = st.session_state.get("extractor")
    if extractor is None:
        return

    st.markdown("---")

    analysis_mode = st.radio(
        "How would you like to work with the reports?",
        ("Discover themes in the reports", "Tag reports with existing themes"),
        horizontal=True,
        key="extractor_mode",
    )

    if analysis_mode == "Discover themes in the reports":
        st.markdown("#### Discover themes in the reports")
        st.write(
            "We will summarise each report and then ask the model to surface recurring themes."
        )

        with st.form("discover_themes_form", enter_to_submit=False):
            extra_theme_instructions = st.text_area(
                "Add any extra guidance for the themes (optional)",
                placeholder="e.g. Focus on system-level safety issues.",
            )
            seed_topics_text = ""

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
                )
                warning_threshold = st.number_input(
                    "Warn if the token estimate exceeds",
                    min_value=1000,
                    value=100000,
                    step=1000,
                )
                error_threshold = st.number_input(
                    "Stop if the token estimate exceeds",
                    min_value=1000,
                    value=500000,
                    step=1000,
                )
                max_col, min_col = st.columns(2)
                max_themes_raw = max_col.text_input(
                    "Maximum number of themes (optional)",
                    value="",
                    placeholder="Leave blank to let the model decide.",
                )
                min_themes_raw = min_col.text_input(
                    "Minimum number of themes (optional)",
                    value="",
                    placeholder="Leave blank to let the model decide.",
                )
                seed_topics_text = st.text_area(
                    "Seed topics (optional)",
                    value=seed_topics_text,
                    placeholder='["Communication", "Medication safety", "Staff training"]',
                )

            themes_submitted = st.form_submit_button(
                "Discover themes", use_container_width=True
            )

        if themes_submitted:
            st.session_state["seed_topics_last"] = None
            input_error = False
            try:
                max_themes_value = _parse_optional_non_negative_int(
                    max_themes_raw,
                    "Maximum number of themes",
                )
            except ValueError as exc:
                st.error(str(exc))
                input_error = True
                max_themes_value = None

            try:
                min_themes_value = _parse_optional_non_negative_int(
                    min_themes_raw,
                    "Minimum number of themes",
                )
            except ValueError as exc:
                st.error(str(exc))
                input_error = True
                min_themes_value = None

            if not input_error:
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
                        st.session_state["summary_result"] = summary_df
                        st.session_state["reports_df"] = summary_df.copy(deep=True)
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
                    st.session_state["seed_topics_last"] = seed_topics or None
                    if ThemeModel is None or not hasattr(ThemeModel, "model_json_schema"):
                        st.warning(
                            "Theme discovery completed but did not return a schema. "
                            "Check the application logs for more details."
                        )
                    else:
                        st.session_state["theme_model_schema"] = ThemeModel.model_json_schema()
                    progress_bar.progress(100)
                    st.success(
                        "Theme discovery complete. The schema below can be reused when tagging reports."
                    )
                except Exception as exc:  # pragma: no cover - depends on live API
                    st.error(f"Theme discovery failed: {exc}")
                finally:
                    progress_placeholder.empty()

                with reports_container:
                    _render_reports_overview(
                        "Loaded reports in workspace", key_suffix="extractor"
                    )

        seed_topics_last = st.session_state.get("seed_topics_last")
        if seed_topics_last:
            st.markdown("##### Seed topics in use")
            if isinstance(seed_topics_last, (list, tuple, set)):
                for topic in seed_topics_last:
                    st.markdown(f"- {topic}")
            else:
                st.json(seed_topics_last)

        summary_df = st.session_state.get("summary_result")
        if isinstance(summary_df, pd.DataFrame) and not summary_df.empty:
            _display_dataframe(summary_df, "Report summaries")
            st.download_button(
                "Download summaries as CSV",
                data=summary_df.to_csv(index=False).encode("utf-8"),
                file_name="pfd_summaries.csv",
                mime="text/csv",
                use_container_width=True,
            )

        theme_schema = st.session_state.get("theme_model_schema")
        if theme_schema:
            st.markdown("##### Theme schema")
            st.json(theme_schema)

    else:
        st.markdown("#### Tag reports with existing themes")
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

        uploaded_df: Optional[pd.DataFrame] = None
        with st.form("extract_features_form", enter_to_submit=False):
            dataset_choice = st.radio(
                "Which reports should be processed?",
                ("Use the loaded reports", "Upload a CSV"),
                index=0,
            )
            if dataset_choice == "Upload a CSV":
                uploaded_file = st.file_uploader(
                    "Upload a CSV file", type="csv", accept_multiple_files=False
                )
                if uploaded_file is not None:
                    uploaded_df = pd.read_csv(uploaded_file)

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
                schema_detail_choice = st.selectbox(
                    "Level of schema detail shared with the model",
                    options=["Minimal", "Full"],
                    index=0,
                    key="extract_schema_detail",
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

        if extract_submitted:
            progress_placeholder = st.empty()
            progress_bar = progress_placeholder.progress(0)
            try:
                progress_bar.progress(10)
                feature_model = _build_feature_model_from_grid(feature_grid)
                target_df = uploaded_df if uploaded_df is not None else None
                with st.spinner("Extracting structured data..."):
                    result_df = extractor.extract_features(
                        reports=target_df,
                        feature_model=feature_model,
                        produce_spans=produce_spans,
                        drop_spans=drop_spans,
                        force_assign=force_assign,
                        allow_multiple=allow_multiple,
                        schema_detail=schema_detail_choice.lower(),  # type: ignore[arg-type]
                        extra_instructions=extra_instructions or None,
                        skip_if_present=skip_if_present,
                    )
                progress_bar.progress(100)
                st.session_state["extractor_result"] = result_df
                st.session_state["reports_df"] = result_df.copy(deep=True)
                st.success("Tagging complete. Review the extracted fields below.")
            except Exception as exc:  # pragma: no cover - depends on live API
                st.error(f"Extraction failed: {exc}")
            finally:
                progress_placeholder.empty()

            with reports_container:
                _render_reports_overview("Loaded reports in workspace", key_suffix="extractor")

        result_df = st.session_state.get("extractor_result")
        if isinstance(result_df, pd.DataFrame):
            _display_dataframe(result_df, "Tagged reports")
            st.download_button(
                "Download extracted fields as CSV",
                data=result_df.to_csv(index=False).encode("utf-8"),
                file_name="pfd_extracted_features.csv",
                mime="text/csv",
                use_container_width=True,
            )


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
            .section-caption {font-weight: 600; font-size: 1.05rem; margin: 1.2rem 0 0.5rem; color: #111827;}
        </style>
        """,
        unsafe_allow_html=True,
    )

    _build_sidebar()

    tab_labels = ["Load in reports", "Screen reports", "Extract insights"]
    st.session_state.setdefault("active_tab", tab_labels[0])

    if "main_navigation" not in st.session_state:
        st.session_state["main_navigation"] = st.session_state["active_tab"]

    selected_tab = st.radio(
        "Choose a workflow",
        tab_labels,
        key="main_navigation",
        horizontal=True,
        label_visibility="collapsed",
    )
    st.session_state["active_tab"] = selected_tab

    if selected_tab == tab_labels[0]:
        _render_load_reports_tab()
    elif selected_tab == tab_labels[1]:
        _render_screener_tab()
    else:
        _render_extractor_tab()


if __name__ == "__main__":
    main()
