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
DEFAULT_OPENROUTER_URL = "https://openrouter.ai/api/v1"


# ---------------------------------------------------------------------------
# Helper utilities
# ---------------------------------------------------------------------------

def _init_session_state() -> None:
    """Ensure default keys exist in ``st.session_state``."""
    defaults = {
        "reports_df": pd.DataFrame(),
        "screener_result": None,
        "extractor_result": None,
        "summary_result": None,
        "token_estimate": None,
        "theme_model_schema": None,
        "llm_client": None,
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


def _parse_optional_int(value: str) -> Optional[int]:
    """Convert ``value`` to ``int`` when possible."""
    value = value.strip()
    if not value:
        return None
    try:
        return int(value)
    except ValueError:
        st.error("Seed must be an integer if provided.")
        return None


def _parse_optional_float(value: str) -> Optional[float]:
    """Convert ``value`` to ``float`` when possible."""
    value = value.strip()
    if not value:
        return None
    try:
        return float(value)
    except ValueError:
        st.error("Timeout must be numeric (seconds).")
        return None


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


def _display_dataframe(df: pd.DataFrame, caption: str) -> None:
    """Render a ``DataFrame`` with a consistent caption."""
    if df is None or df.empty:
        st.info("No rows to display yet. Load or generate data to see results here.")
        return
    st.markdown(f"<div class='section-caption'>{caption}</div>", unsafe_allow_html=True)
    st.dataframe(df, use_container_width=True, hide_index=True)


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

    provider = st.sidebar.selectbox("Model provider", ["OpenAI", "OpenRouter"], index=0)
    api_key = st.sidebar.text_input("API key", type="password", help="Required for LLM-powered features.")

    if provider == "OpenRouter":
        base_url = st.sidebar.text_input(
            "API base URL",
            value=DEFAULT_OPENROUTER_URL,
            help="Override the API endpoint if you are using a compatible OpenRouter proxy.",
        )
    else:
        base_url = st.sidebar.text_input(
            "Custom base URL (optional)",
            value="",
            help="Leave blank to use the official OpenAI endpoint.",
        ).strip() or None

    st.sidebar.markdown("---")
    st.sidebar.markdown("### Model configuration")
    model_name = st.sidebar.text_input("Chat model", value="gpt-4.1")
    temperature = st.sidebar.slider("Temperature", min_value=0.0, max_value=2.0, value=0.0, step=0.1)
    max_workers = st.sidebar.number_input("Max parallel workers", min_value=1, max_value=32, value=8)
    seed_raw = st.sidebar.text_input("Deterministic seed (optional)")
    timeout_raw = st.sidebar.text_input("Request timeout seconds (optional)", value="120")
    validation_attempts = st.sidebar.number_input("Validation attempts", min_value=1, max_value=5, value=2)

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

    n_reports = st.sidebar.number_input(
        "Limit number of recent reports (optional)",
        min_value=0,
        max_value=5000,
        value=0,
        help="Keep 0 to load all matching reports.",
    )
    refresh = st.sidebar.checkbox(
        "Force refresh from remote dataset", value=True, help="Disable to reuse the cached CSV if available."
    )

    load_button = st.sidebar.button("Load in reports", use_container_width=True)

    seed = _parse_optional_int(seed_raw) if seed_raw else None
    timeout = _parse_optional_float(timeout_raw) if timeout_raw else None

    if load_button:
        try:
            with st.spinner("Downloading and filtering reports..."):
                df = load_reports(
                    start_date=start_date.isoformat(),
                    end_date=end_date.isoformat(),
                    n_reports=int(n_reports) if n_reports else None,
                    refresh=refresh,
                )
            st.session_state["reports_df"] = df
            st.session_state["screener_result"] = None
            st.session_state["extractor_result"] = None
            st.session_state["summary_result"] = None
            st.session_state["token_estimate"] = None
            st.session_state["theme_model_schema"] = None
            st.success(f"Loaded {len(df)} reports into the workspace.")
        except Exception as exc:  # pragma: no cover - UI feedback
            st.error(f"Could not load reports: {exc}")

    if api_key:
        llm_kwargs: Dict[str, Any] = {
            "api_key": api_key.strip(),
            "model": model_name.strip() or "gpt-4.1",
            "max_workers": int(max_workers),
            "temperature": float(temperature),
            "validation_attempts": int(validation_attempts),
        }
        if seed is not None:
            llm_kwargs["seed"] = seed
        if timeout is not None:
            llm_kwargs["timeout"] = timeout
        if provider == "OpenRouter":
            llm_kwargs["base_url"] = base_url or DEFAULT_OPENROUTER_URL
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

def _render_overview() -> None:
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

    _display_dataframe(reports_df, "Loaded Prevention of Future Death reports")


def _render_screener_tab() -> None:
    st.subheader("Screener")
    st.markdown(
        """
        Use the Screener to classify reports against a topic. Provide a clear search query, choose which
        report fields feed into the prompt, and decide whether to filter the results or append a classification column.
        """
    )
    st.caption(
        "Example query: *Identify reports relating to medication errors involving care homes.*"
    )

    llm_client: Optional[LLM] = st.session_state.get("llm_client")
    reports_df: pd.DataFrame = st.session_state.get("reports_df", pd.DataFrame())

    if llm_client is None:
        st.warning("Add a valid API key in the sidebar to enable the Screener.")
        return
    if reports_df.empty:
        st.info("Load reports from the sidebar before screening.")
        return

    with st.form("screener_form", enter_to_submit=False):
        st.markdown("#### Configuration")
        cols = st.columns(4)
        verbose = cols[0].checkbox("Verbose logging", value=False)
        include_date = cols[1].checkbox("Include date", value=False)
        include_coroner = cols[2].checkbox("Include coroner", value=False)
        include_area = cols[3].checkbox("Include area", value=False)

        cols2 = st.columns(4)
        include_receiver = cols2[0].checkbox("Include receiver", value=False)
        include_investigation = cols2[1].checkbox("Include investigation", value=True)
        include_circumstances = cols2[2].checkbox("Include circumstances", value=True)
        include_concerns = cols2[3].checkbox("Include concerns", value=True)

        st.markdown("#### Screening request")
        search_query = st.text_input(
            "Primary search query", placeholder="e.g. Reports mentioning delays in ambulance response times"
        )
        user_query = st.text_input(
            "Deprecated user_query parameter (optional)",
            help="Only use if you rely on legacy code that still passes 'user_query'.",
        )
        cols3 = st.columns(3)
        filter_df = cols3[0].checkbox("Filter to matches", value=True)
        result_col_name = cols3[1].text_input("Result column", value="matches_query")
        produce_spans = cols3[2].checkbox("Return supporting spans", value=False)
        drop_spans = st.checkbox("Drop spans column from the result", value=False)

        submitted = st.form_submit_button("Run Screener", use_container_width=True)

    if submitted:
        if not search_query and not user_query:
            st.error("Provide at least one of 'search_query' or 'user_query'.")
            return
        try:
            screener = Screener(
                llm=llm_client,
                reports=reports_df,
                verbose=verbose,
                include_date=include_date,
                include_coroner=include_coroner,
                include_area=include_area,
                include_receiver=include_receiver,
                include_investigation=include_investigation,
                include_circumstances=include_circumstances,
                include_concerns=include_concerns,
            )
            result_df = screener.screen_reports(
                search_query=search_query or None,
                user_query=user_query or None,
                filter_df=filter_df,
                result_col_name=result_col_name or "matches_query",
                produce_spans=produce_spans,
                drop_spans=drop_spans,
            )
            st.session_state["screener_result"] = result_df
            st.success("Screening complete. Review the results below.")
        except Exception as exc:  # pragma: no cover - relies on live API
            st.error(f"Screening failed: {exc}")

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
    st.subheader("Extractor")
    st.markdown(
        """
        The Extractor turns unstructured reports into tailored datasets. Begin by instantiating an Extractor with the
        report fields you want to provide, then choose from the available public methods to extract features, summarise,
        estimate tokens, or discover themes.
        """
    )
    st.caption(
        "Tip: start with `summarise()` to generate concise summaries, then run `discover_themes()` to build a theme model that feeds into `extract_features()`."
    )

    llm_client: Optional[LLM] = st.session_state.get("llm_client")
    reports_df: pd.DataFrame = st.session_state.get("reports_df", pd.DataFrame())

    if llm_client is None:
        st.warning("Add a valid API key in the sidebar to enable the Extractor.")
        return
    if reports_df.empty:
        st.info("Load reports from the sidebar before extracting features.")
        return

    with st.form("extractor_setup_form", enter_to_submit=False):
        st.markdown("#### Initial configuration")
        cols = st.columns(4)
        include_date = cols[0].checkbox("Include date", value=False)
        include_coroner = cols[1].checkbox("Include coroner", value=False)
        include_area = cols[2].checkbox("Include area", value=False)
        include_receiver = cols[3].checkbox("Include receiver", value=False)

        cols2 = st.columns(4)
        include_investigation = cols2[0].checkbox("Include investigation", value=True)
        include_circumstances = cols2[1].checkbox("Include circumstances", value=True)
        include_concerns = cols2[2].checkbox("Include concerns", value=True)
        verbose = cols2[3].checkbox("Verbose logging", value=False)

        initialise = st.form_submit_button("Initialise extractor", use_container_width=True)

    if initialise or "extractor" not in st.session_state:
        try:
            st.session_state["extractor"] = Extractor(
                llm=llm_client,
                reports=reports_df,
                include_date=include_date,
                include_coroner=include_coroner,
                include_area=include_area,
                include_receiver=include_receiver,
                include_investigation=include_investigation,
                include_circumstances=include_circumstances,
                include_concerns=include_concerns,
                verbose=verbose,
            )
            if initialise:
                st.success("Extractor initialised with the chosen settings.")
        except Exception as exc:  # pragma: no cover - depends on live API
            st.error(f"Could not initialise extractor: {exc}")
            return

    extractor: Optional[Extractor] = st.session_state.get("extractor")
    if extractor is None:
        return

    st.markdown("---")
    st.markdown("#### extract_features")
    st.write(
        "Define the features you want to capture using JSON (field name ➜ type, description). "
        "Supported types: string, bool, int, float. Set `required` to false to keep optional fields."
    )
    default_schema = json.dumps(
        {
            "risk_factor": {
                "type": "string",
                "description": "Primary risk factor contributing to the death.",
            },
            "is_healthcare": {
                "type": "bool",
                "description": "True if the report involves a healthcare setting.",
            },
        },
        indent=2,
    )

    with st.form("extract_features_form", enter_to_submit=False):
        schema_text = st.text_area("Feature schema (JSON)", value=default_schema, height=220)
        use_loaded = st.radio(
            "Reports to process",
            options=("Use loaded reports", "Upload CSV"),
            index=0,
            help="`extract_features` also accepts a custom DataFrame per call.",
        )
        uploaded_df: Optional[pd.DataFrame] = None
        if use_loaded == "Upload CSV":
            uploaded_file = st.file_uploader("Upload CSV file", type="csv")
            if uploaded_file is not None:
                uploaded_df = pd.read_csv(uploaded_file)
        produce_spans = st.checkbox("Return supporting spans", value=False)
        drop_spans = st.checkbox("Drop spans columns on return", value=False)
        force_assign = st.checkbox("Force assign values", value=False)
        allow_multiple = st.checkbox("Allow multiple categories per feature", value=False)
        schema_detail = st.selectbox("Schema detail", options=["minimal", "full"], index=0)
        extra_instructions = st.text_area("Extra prompt instructions", placeholder="Add any additional guidance for the LLM.")
        skip_if_present = st.checkbox(
            "Skip rows that already contain feature values", value=True
        )
        extract_submitted = st.form_submit_button("Run extract_features", use_container_width=True)

    if extract_submitted:
        try:
            feature_model = _build_feature_model(schema_text)
            target_df = uploaded_df if uploaded_df is not None else None
            result_df = extractor.extract_features(
                reports=target_df,
                feature_model=feature_model,
                produce_spans=produce_spans,
                drop_spans=drop_spans,
                force_assign=force_assign,
                allow_multiple=allow_multiple,
                schema_detail=schema_detail,  # type: ignore[arg-type]
                extra_instructions=extra_instructions or None,
                skip_if_present=skip_if_present,
            )
            st.session_state["extractor_result"] = result_df
            st.success("Feature extraction complete.")
        except Exception as exc:  # pragma: no cover - depends on live API
            st.error(f"Extraction failed: {exc}")

    result_df = st.session_state.get("extractor_result")
    if isinstance(result_df, pd.DataFrame):
        _display_dataframe(result_df, "Extractor output")
        st.download_button(
            "Download extracted features as CSV",
            data=result_df.to_csv(index=False).encode("utf-8"),
            file_name="pfd_extracted_features.csv",
            mime="text/csv",
            use_container_width=True,
        )

    st.markdown("---")
    st.markdown("#### summarise")
    st.write(
        "Generate concise summaries of each report. Choose how aggressively to trim details and optionally append your own instructions."
    )

    with st.form("summarise_form", enter_to_submit=False):
        result_col_name = st.text_input("Summary column name", value="summary")
        trim_intensity = st.selectbox(
            "Trim intensity",
            options=["low", "medium", "high", "very high"],
            index=1,
        )
        extra_instructions = st.text_area(
            "Extra summary instructions",
            placeholder="e.g. Highlight safeguarding lessons learned.",
        )
        summarise_submitted = st.form_submit_button("Run summarise", use_container_width=True)

    if summarise_submitted:
        try:
            summary_df = extractor.summarise(
                result_col_name=result_col_name or "summary",
                trim_intensity=trim_intensity,  # type: ignore[arg-type]
                extra_instructions=extra_instructions or None,
            )
            st.session_state["summary_result"] = summary_df
            st.success("Summaries generated.")
        except Exception as exc:  # pragma: no cover - depends on live API
            st.error(f"Summarisation failed: {exc}")

    summary_df = st.session_state.get("summary_result")
    if isinstance(summary_df, pd.DataFrame):
        _display_dataframe(summary_df, "Summarised reports")

    st.markdown("---")
    st.markdown("#### estimate_tokens")
    st.write(
        "Estimate token usage for a summary column before prompting large batches."
    )

    with st.form("estimate_tokens_form", enter_to_submit=False):
        col_name = st.text_input(
            "Column to measure",
            value="",
            help="Leave blank to reuse the latest summary column name.",
        )
        return_series = st.checkbox("Return per-row series instead of a total", value=False)
        tokens_submitted = st.form_submit_button("Estimate tokens", use_container_width=True)

    if tokens_submitted:
        try:
            token_result = extractor.estimate_tokens(
                col_name=col_name or None,
                return_series=return_series,
            )
            st.session_state["token_estimate"] = token_result
            if isinstance(token_result, pd.Series):
                st.success("Token counts calculated for each row.")
            else:
                st.success(f"Estimated total tokens: {token_result:,}")
        except Exception as exc:  # pragma: no cover - depends on live API
            st.error(f"Token estimation failed: {exc}")

    token_result = st.session_state.get("token_estimate")
    if isinstance(token_result, pd.Series):
        st.dataframe(token_result.to_frame(), use_container_width=True)

    st.markdown("---")
    st.markdown("#### discover_themes")
    st.write(
        "Automatically identify recurring themes from summaries. The resulting pydantic model can be fed back into `extract_features()`."
    )

    with st.form("discover_themes_form", enter_to_submit=False):
        warn_exceed = st.number_input("Warning threshold", min_value=1000, value=100000, step=1000)
        error_exceed = st.number_input("Error threshold", min_value=1000, value=500000, step=1000)
        col1, col2 = st.columns(2)
        max_themes = col1.number_input("Maximum themes", min_value=0, value=0, help="Set 0 to leave unset.")
        min_themes = col2.number_input("Minimum themes", min_value=0, value=0, help="Set 0 to leave unset.")
        extra_instructions = st.text_area(
            "Extra theme instructions",
            placeholder="e.g. Prioritise system-level safety issues.",
        )
        seed_topics_text = st.text_area(
            "Seed topics (JSON list or newline separated)",
            placeholder="[\n  \"care_home\",\n  \"medication_errors\"\n]",
        )
        themes_submitted = st.form_submit_button("Run discover_themes", use_container_width=True)

    if themes_submitted:
        try:
            seed_topics: Optional[Any]
            seed_topics = None
            if seed_topics_text.strip():
                try:
                    seed_topics = json.loads(seed_topics_text)
                except json.JSONDecodeError:
                    seed_topics = [line.strip() for line in seed_topics_text.splitlines() if line.strip()]
            ThemeModel = extractor.discover_themes(
                warn_exceed=int(warn_exceed),
                error_exceed=int(error_exceed),
                max_themes=int(max_themes) or None,
                min_themes=int(min_themes) or None,
                extra_instructions=extra_instructions or None,
                seed_topics=seed_topics,
            )
            st.session_state["theme_model_schema"] = ThemeModel.model_json_schema()
            st.success("Theme discovery complete. Use the schema below when defining features.")
        except Exception as exc:  # pragma: no cover - depends on live API
            st.error(f"Theme discovery failed: {exc}")

    theme_schema = st.session_state.get("theme_model_schema")
    if theme_schema:
        st.json(theme_schema)


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

    overview_tab, screener_tab, extractor_tab = st.tabs([
        "Overview",
        "Screener",
        "Extractor",
    ])

    with overview_tab:
        _render_overview()
    with screener_tab:
        _render_screener_tab()
    with extractor_tab:
        _render_extractor_tab()


if __name__ == "__main__":
    main()
