"""Domain services for the Django workbench."""
from __future__ import annotations

import json
from datetime import date
from typing import Any, Optional

import pandas as pd
from pydantic import Field, create_model

from pfd_toolkit.extractor import Extractor
from pfd_toolkit.llm import LLM
from pfd_toolkit.loader import load_reports
from pfd_toolkit.screener import Screener

from .state import LLM_SIGNATURE_KEY, format_call, record_repro_action

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
    "🧛",
    "🧛‍♂️",
    "🧛‍♀️",
    "🥀",
    "⚱️",
    "🕷️",
    "🕸️",
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


def get_llm_config(session: dict[str, Any]) -> tuple[Optional[dict[str, Any]], Optional[str]]:
    provider = session.get("provider_override", "OpenAI")
    model_name = (session.get("model_name") or "gpt-4.1-mini").strip() or "gpt-4.1-mini"

    if provider == "OpenRouter":
        api_key = (session.get("openrouter_api_key") or "").strip()
        base_url = (session.get("openrouter_base_url") or OPENROUTER_API_BASE).strip()
        if not api_key:
            return None, "Provide an OpenRouter API key to continue."
    else:
        api_key = (session.get("openai_api_key") or "").strip()
        base_url = (session.get("openai_base_url") or "").strip() or None
        if not api_key:
            return None, "Provide an OpenAI API key to continue."

    llm_kwargs: dict[str, Any] = {
        "api_key": api_key,
        "model": model_name,
        "max_workers": int(session.get("max_parallel_workers", 8) or 8),
        "temperature": 0.0,
        "validation_attempts": 2,
        "seed": 123,
        "timeout": 30,
    }
    if provider == "OpenRouter":
        llm_kwargs["base_url"] = base_url or OPENROUTER_API_BASE
    elif base_url:
        llm_kwargs["base_url"] = base_url

    return llm_kwargs, None


def build_llm(session: dict[str, Any]) -> LLM:
    llm_kwargs, error = get_llm_config(session)
    if error or not llm_kwargs:
        raise ValueError(error or "Missing LLM configuration.")

    signature_payload = llm_kwargs.copy()
    signature_payload["api_key"] = bool(signature_payload.get("api_key"))
    signature = tuple(signature_payload.items())

    if session.get(LLM_SIGNATURE_KEY) != signature:
        session[LLM_SIGNATURE_KEY] = signature
        session["theme_emoji_cache"] = {}

        llm_script_kwargs = llm_kwargs.copy()
        llm_script_kwargs["api_key"] = "<redacted>"
        record_repro_action(
            session,
            "init_llm",
            "Set up the language model client",
            format_call("llm_client = LLM", llm_script_kwargs),
        )

    return LLM(**llm_kwargs)


def load_reports_dataframe(
    *,
    start_date: date,
    end_date: date,
    n_reports: Optional[int],
    refresh: bool,
) -> pd.DataFrame:
    return load_reports(
        start_date=start_date.isoformat(),
        end_date=end_date.isoformat(),
        n_reports=n_reports,
        refresh=refresh,
    )


def build_screener(llm_client: LLM, reports_df: pd.DataFrame, session: dict[str, Any]) -> Screener:
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
    record_repro_action(
        session,
        "init_screener",
        "Initialise the screener",
        format_call("screener = Screener", screener_kwargs, raw_parameters={"llm", "reports"}),
    )
    return screener


def build_extractor(llm_client: LLM, reports_df: pd.DataFrame, session: dict[str, Any]) -> Extractor:
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
    record_repro_action(
        session,
        "init_extractor",
        "Initialise the extractor",
        format_call("extractor = Extractor", extractor_kwargs, raw_parameters={"llm", "reports"}),
    )
    return extractor


def build_feature_model_from_rows(feature_rows: pd.DataFrame):
    if feature_rows.empty:
        raise ValueError("Please add at least one feature to extract.")

    type_mapping = {
        "Free text": str,
        "Conditional (True/False)": bool,
        "Whole number": int,
        "Decimal number": float,
    }

    fields: dict[str, tuple[type, Field]] = {}
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


def parse_seed_topics(seed_topics_text: str) -> Optional[Any]:
    text = (seed_topics_text or "").strip()
    if not text:
        return None
    try:
        return json.loads(text)
    except json.JSONDecodeError:
        lines = [line.strip() for line in text.splitlines() if line.strip()]
        return lines or None


def resolve_theme_emoji(theme_name: str, session: dict[str, Any]) -> str:
    """Return a single emoji for a theme, using cached LLM output when available."""

    theme_clean = (theme_name or "").strip()
    if not theme_clean:
        return DEFAULT_THEME_EMOJI

    cache = session.setdefault("theme_emoji_cache", {})
    signature = session.get(LLM_SIGNATURE_KEY)
    cache_key = (str(signature), theme_clean.lower())
    cached = cache.get(str(cache_key))
    if isinstance(cached, str) and cached:
        return cached

    try:
        llm_client = build_llm(session)
    except Exception:
        return DEFAULT_THEME_EMOJI

    prompt = (
        "You choose exactly one emoji to represent a theme extracted from Prevention "
        "of Future Death reports.\n"
        f'Theme: "{theme_clean}"\n\n'
        "Rules:\n"
        "- Select precisely one emoji that captures the theme's core idea.\n"
        "- Avoid violent, graphic, or harmful imagery.\n"
        f"- If nothing is suitable, respond with {DEFAULT_THEME_EMOJI}.\n"
        "- Do not include text, spaces, or punctuation.\n"
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

    cache[str(cache_key)] = candidate
    return candidate
