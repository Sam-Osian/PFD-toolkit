import logging
from dataclasses import dataclass
from difflib import SequenceMatcher
from functools import lru_cache
import re
import warnings

import pandas as pd
from pydantic import BaseModel, Field, ConfigDict, field_validator
from tqdm import tqdm
from tqdm import TqdmWarning

from pfd_toolkit.llm import LLM
from pfd_toolkit.config import GeneralConfig

# -----------------------------------------------------------------------------
# Logging Configuration:
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, force=True)

# Set the log level for the 'httpx' library to WARNING to reduce verbosity
logging.getLogger("httpx").setLevel(logging.WARNING)
# Silence the OpenAI client’s info-level logs (as in llm.py)
logging.getLogger("openai").setLevel(logging.WARNING)

warnings.filterwarnings("ignore", category=TqdmWarning)


def _normalise_area_for_matching(area: str) -> str:
    """Return a simplified representation used only for area matching."""
    if not isinstance(area, str):
        return ""

    normalised = area.strip().casefold()
    normalised = normalised.replace("&", " and ")
    normalised = re.sub(r"[^a-z0-9]+", " ", normalised)
    return " ".join(normalised.split())


RECEIVER_DROP_VALUES = {
    "chief coroner",
    "the chief coroner",
    "his majesty s chief coroner",
    "her majesty s chief coroner",
    "hm chief coroner",
    "chief coroner of england and wales",
}

RECEIVER_ROLE_PREFIXES = {
    "ceo",
    "chief executive",
    "chief executive officer",
    "medical director",
    "clinical director",
    "managing director",
    "director",
    "minister",
    "secretary of state",
    "trust chair",
    "chair",
    "coroner",
    "assistant coroner",
    "senior coroner",
}

RECEIVER_ORG_CUES = (
    "nhs",
    "trust",
    "foundation trust",
    "hospital",
    "council",
    "department",
    "ministry",
    "agency",
    "care",
    "health",
    "board",
    "services",
    "service",
    "authority",
    "office",
    "group",
    "limited",
    "ltd",
    "plc",
    "university",
    "college",
    "cardinal healthcare",
)

RECEIVER_CANONICAL_MAP = {
    "department of health": "Department of Health and Social Care",
    "department of health and social care": "Department of Health and Social Care",
    "nhs england and nhs improvement": "NHS England",
    "highways agency": "National Highways",
    "highways england": "National Highways",
}

RECEIVER_ROLE_PREFIX_TEXTS = (
    "the chief executive officer of ",
    "chief executive officer of ",
    "the chief executive of ",
    "chief executive of ",
    "the chief executive officer ",
    "chief executive officer ",
    "the chief executive ",
    "chief executive ",
)


def _normalise_receiver_match_key(text: str) -> str:
    """Return a permissive key used only for receiver cleanup rules."""
    normalised = text.strip().casefold()
    normalised = normalised.replace("&", " and ")
    normalised = re.sub(r"[^a-z0-9]+", " ", normalised)
    return " ".join(normalised.split())


def _looks_like_receiver_organisation(text: str) -> bool:
    """Heuristic for detecting organisation-like recipient segments."""
    key = _normalise_receiver_match_key(text)
    if not key:
        return False
    return any(cue in key for cue in RECEIVER_ORG_CUES) or text.isupper()


def _normalise_receiver_formatting(text: str) -> str:
    """Apply low-risk receiver formatting cleanup while preserving casing."""
    cleaned = " ".join((text or "").split()).strip(" ,;:.")
    if not cleaned:
        return ""

    cleaned = cleaned.replace("’", "'").replace("‘", "'")
    cleaned = re.sub(r"^[\-\–\)\(:\s]+", "", cleaned)
    cleaned = re.sub(r"^\s*The\s+", "", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\bDept\.?\b", "Department", cleaned, flags=re.IGNORECASE)
    cleaned = re.sub(r"\s*&\s*", " and ", cleaned)
    cleaned = re.sub(
        r"^Secretary of State for (?:the )?Department(?: of)? ",
        "Secretary of State for ",
        cleaned,
        flags=re.IGNORECASE,
    )
    cleaned = re.sub(r"\s{2,}", " ", cleaned)
    cleaned = re.sub(r"\s*\(([A-Za-z0-9&./' -]{2,15})\)\s*$", "", cleaned)
    cleaned = re.sub(r'\s*\((?:"?the Trust"?|Highways Department|Local Government)\)\s*$', "", cleaned, flags=re.IGNORECASE)
    return cleaned.strip(" ,;:.")


def _canonicalise_receiver_segment(text: str) -> str:
    """Apply explicit receiver canonical mappings while preserving formatting."""
    key = _normalise_receiver_match_key(text)
    return RECEIVER_CANONICAL_MAP.get(key, text)


def _clean_receiver_segment(segment: str) -> str:
    """Normalise a single receiver segment while preserving useful casing."""
    text = " ".join((segment or "").replace("\n", " ").split()).strip(" ,;:.")
    if not text:
        return ""

    text = re.sub(r"^\[REDACTED\][,:\- ]*", "", text, flags=re.IGNORECASE)
    text = re.sub(r'^[A-Z][^,;]{0,80}\bMP\b[,:\- ]*', "", text)
    text = _normalise_receiver_formatting(text)

    if _normalise_receiver_match_key(text) in RECEIVER_DROP_VALUES:
        return ""

    parts = [part.strip(" ,;:.") for part in text.split(",") if part.strip(" ,;:.")]
    if parts:
        if _normalise_receiver_match_key(parts[0]) in RECEIVER_DROP_VALUES:
            return ""
        if len(parts) >= 2 and _normalise_receiver_match_key(parts[0]) in RECEIVER_ROLE_PREFIXES:
            text = ", ".join(parts[1:])
            parts = [part.strip(" ,;:.") for part in text.split(",") if part.strip(" ,;:.")]
        elif len(parts) >= 3 and _looks_like_receiver_organisation(parts[-1]):
            text = parts[-1]
            parts = [text]
        elif len(parts) >= 2 and _looks_like_receiver_organisation(parts[-1]) and not _looks_like_receiver_organisation(parts[0]):
            text = parts[-1]
            parts = [text]

    normalised_role_form = _normalise_receiver_formatting(text)
    role_form_key = _normalise_receiver_match_key(normalised_role_form)
    for prefix in RECEIVER_ROLE_PREFIX_TEXTS:
        prefix_key = _normalise_receiver_match_key(prefix)
        if role_form_key.startswith(prefix_key):
            candidate = normalised_role_form[len(prefix):]
            candidate = _normalise_receiver_formatting(candidate)
            if candidate and _looks_like_receiver_organisation(candidate):
                text = candidate
                break

    if ";" not in text and not _looks_like_receiver_organisation(text):
        semicolon_parts = [part.strip(" ,;:.") for part in text.split(";") if part.strip(" ,;:.")]
        if len(semicolon_parts) == 2 and _normalise_receiver_match_key(semicolon_parts[0]) in RECEIVER_ROLE_PREFIXES:
            candidate = _normalise_receiver_formatting(semicolon_parts[1])
            if _looks_like_receiver_organisation(candidate):
                text = candidate

    text = _normalise_receiver_formatting(text)
    if _normalise_receiver_match_key(text) in RECEIVER_DROP_VALUES:
        return ""
    return _canonicalise_receiver_segment(text.strip(" ,;:."))


def _normalise_receiver_output(receiver_text: str) -> str:
    """Clean receiver output while preserving semicolon-separated formatting."""
    if not isinstance(receiver_text, str):
        return receiver_text

    trust_variants: dict[str, str] = {}
    segments = []
    seen = set()
    for segment in receiver_text.split(";"):
        cleaned = _clean_receiver_segment(segment)
        if not cleaned:
            continue

        trust_match = re.fullmatch(
            r"(.+?)\s+NHS(?:\s+Foundation)?\s+Trust",
            cleaned,
            flags=re.IGNORECASE,
        )
        if trust_match:
            trust_stem = _normalise_receiver_match_key(trust_match.group(1))
            cleaned = trust_variants.setdefault(
                trust_stem,
                f"{trust_match.group(1).strip()} NHS Foundation Trust",
            )

        key = _normalise_receiver_match_key(cleaned)
        if not key or key in seen:
            continue
        seen.add(key)
        segments.append(cleaned)

    return "; ".join(segments)


DEFAULT_AREA_MATCH_STRATEGY = "exact_then_fuzzy"
AREA_MATCH_STRATEGIES = ("exact_only", "exact_then_fuzzy", "fuzzy_only")
AREA_FUZZY_MIN_SCORE = 0.88
AREA_FUZZY_MIN_MARGIN = 0.03


@dataclass(frozen=True)
class AreaMatchResult:
    canonical_area: str | None
    match_method: str
    score: float | None = None
    matched_value: str | None = None


@lru_cache(maxsize=1)
def _build_area_match_lookups() -> tuple[dict[str, str], dict[str, str], dict[str, str]]:
    """Build cached normalized lookups for canonical areas and synonyms."""
    allowed_lookup = {
        _normalise_area_for_matching(candidate): candidate
        for candidate in GeneralConfig.ALLOWED_AREAS
    }
    synonym_lookup = {
        _normalise_area_for_matching(alias): canonical
        for alias, canonical in GeneralConfig.AREA_SYNONYMS.items()
        if canonical in GeneralConfig.ALLOWED_AREAS
    }
    legacy_lookup = {
        _normalise_area_for_matching(alias): canonical
        for alias, canonical in GeneralConfig.LEGACY_AREA_SYNONYMS.items()
        if canonical in GeneralConfig.ALLOWED_AREAS
    }
    return allowed_lookup, synonym_lookup, legacy_lookup


def _exact_area_match(area: str) -> AreaMatchResult:
    """Resolve exact and normalized matches against canonical areas."""
    if not isinstance(area, str):
        return AreaMatchResult(None, "invalid_input")

    stripped = area.strip()
    if not stripped:
        return AreaMatchResult(None, "empty")

    direct_synonym = GeneralConfig.AREA_SYNONYMS.get(stripped)
    if direct_synonym in GeneralConfig.ALLOWED_AREAS:
        return AreaMatchResult(direct_synonym, "exact_synonym", 1.0, stripped)

    direct_legacy = GeneralConfig.LEGACY_AREA_SYNONYMS.get(stripped)
    if direct_legacy in GeneralConfig.ALLOWED_AREAS:
        return AreaMatchResult(direct_legacy, "legacy_synonym", 1.0, stripped)

    if stripped in GeneralConfig.ALLOWED_AREAS:
        return AreaMatchResult(stripped, "exact_allowed", 1.0, stripped)

    allowed_lookup, synonym_lookup, legacy_lookup = _build_area_match_lookups()
    match_key = _normalise_area_for_matching(stripped)
    if not match_key:
        return AreaMatchResult(None, "empty")

    if match_key in synonym_lookup:
        return AreaMatchResult(
            synonym_lookup[match_key],
            "normalised_synonym",
            1.0,
            match_key,
        )
    if match_key in legacy_lookup:
        return AreaMatchResult(
            legacy_lookup[match_key],
            "legacy_synonym",
            1.0,
            match_key,
        )
    if match_key in allowed_lookup:
        return AreaMatchResult(
            allowed_lookup[match_key],
            "normalised_allowed",
            1.0,
            match_key,
        )
    return AreaMatchResult(None, "unmatched")


def _fuzzy_area_match(area: str) -> AreaMatchResult:
    """Resolve approximate matches against canonical areas and synonyms."""
    if not isinstance(area, str):
        return AreaMatchResult(None, "invalid_input")

    stripped = area.strip()
    match_key = _normalise_area_for_matching(stripped)
    if not match_key:
        return AreaMatchResult(None, "empty")

    allowed_lookup, synonym_lookup, legacy_lookup = _build_area_match_lookups()
    candidates = []
    for candidate_key, canonical in allowed_lookup.items():
        candidates.append((candidate_key, canonical, "fuzzy_allowed"))
    for candidate_key, canonical in synonym_lookup.items():
        candidates.append((candidate_key, canonical, "fuzzy_synonym"))
    for candidate_key, canonical in legacy_lookup.items():
        candidates.append((candidate_key, canonical, "fuzzy_legacy_synonym"))

    if not candidates:
        return AreaMatchResult(None, "unmatched")

    scored = []
    for candidate_key, canonical, method in candidates:
        score = SequenceMatcher(None, match_key, candidate_key).ratio()
        scored.append((score, canonical, method, candidate_key))

    scored.sort(key=lambda item: item[0], reverse=True)
    best_score, best_canonical, best_method, best_value = scored[0]
    next_score = scored[1][0] if len(scored) > 1 else 0.0
    if (
        best_score >= AREA_FUZZY_MIN_SCORE
        and (best_score - next_score) >= AREA_FUZZY_MIN_MARGIN
    ):
        return AreaMatchResult(best_canonical, best_method, best_score, best_value)
    return AreaMatchResult(None, "unmatched")


def match_area(area: str, strategy: str = "exact_only") -> AreaMatchResult:
    """Resolve a raw area string into a canonical area using the given strategy."""
    if strategy not in AREA_MATCH_STRATEGIES:
        raise ValueError(
            f"Unknown area_match_strategy '{strategy}'. Valid options are: {', '.join(AREA_MATCH_STRATEGIES)}."
        )

    if strategy == "fuzzy_only":
        return _fuzzy_area_match(area)

    exact_result = _exact_area_match(area)
    if exact_result.canonical_area is not None:
        return exact_result

    if strategy == "exact_then_fuzzy":
        return _fuzzy_area_match(area)

    return exact_result


def _canonicalise_area(
    area: str, strategy: str = DEFAULT_AREA_MATCH_STRATEGY
) -> str:
    """Map a raw area string to a canonical allowed area where possible."""
    match = match_area(area, strategy=strategy)
    return match.canonical_area or "Other"


# ---------------------------------------------------------------------------
# Area validation model
# ---------------------------------------------------------------------------


def create_area_model(
    area_match_strategy: str = DEFAULT_AREA_MATCH_STRATEGY,
) -> type[BaseModel]:
    """Return a Pydantic model that canonicalises areas with the given strategy."""
    if area_match_strategy not in AREA_MATCH_STRATEGIES:
        raise ValueError(
            f"Unknown area_match_strategy '{area_match_strategy}'. Valid options are: {', '.join(AREA_MATCH_STRATEGIES)}."
        )

    class AreaModel(BaseModel):
        """Pydantic model restricting the area field."""

        area: str = Field(..., description="Name of the coroner area")

        model_config = ConfigDict(extra="forbid")

        @field_validator("area", mode="before")
        @classmethod
        def apply_synonyms(cls, v: str) -> str:
            """Map raw area text to a canonical allowed area where possible."""
            return _canonicalise_area(v, strategy=area_match_strategy)

        @field_validator("area")
        @classmethod
        def validate_area(cls, v: str) -> str:
            """Ensure the area is one of the allowed values."""
            if v not in GeneralConfig.ALLOWED_AREAS:
                return "Other"
            return v

    AreaModel.__name__ = f"AreaModel_{area_match_strategy}"
    return AreaModel


AreaModel = create_area_model()


class Cleaner:
    """Batch-clean PFD report fields with an LLM.

    The cleaner loops over selected columns, builds field-specific prompts and
    writes the returned text back into a copy of the DataFrame.

    Parameters
    ----------
    reports : pandas.DataFrame
        Input DataFrame to clean.
    llm : LLM
        Instance of the ``LLM`` helper used for prompting.
    include_coroner : bool, optional
        Clean the ``coroner`` column. Defaults to ``True``.
    include_receiver : bool, optional
        Clean the ``receiver`` column. Defaults to ``True``.
    include_area : bool, optional
        Clean the ``area`` column. Defaults to ``True``.
    include_investigation : bool, optional
        Clean the ``investigation`` column. Defaults to ``True``.
    include_circumstances : bool, optional
        Clean the ``circumstances`` column. Defaults to ``True``.
    include_concerns : bool, optional
        Clean the ``concerns`` column. Defaults to ``True``.
    coroner_prompt : str or None, optional
        Custom prompt for the coroner field. Defaults to ``None``.
    area_prompt : str or None, optional
        Custom prompt for the area field. Defaults to ``None``.
    receiver_prompt : str or None, optional
        Custom prompt for the receiver field. Defaults to ``None``.
    investigation_prompt : str or None, optional
        Custom prompt for the investigation field. Defaults to ``None``.
    circumstances_prompt : str or None, optional
        Custom prompt for the circumstances field. Defaults to ``None``.
    concerns_prompt : str or None, optional
        Custom prompt for the concerns field. Defaults to ``None``.
    verbose : bool, optional
        Emit info-level logs for each batch when ``True``. Defaults to ``False``.

    Attributes
    ----------
    cleaned_reports : pandas.DataFrame
        Result of the last call to ``clean_reports``.
    coroner_prompt_template, area_prompt_template, ... : str
        Finalised prompt strings actually sent to the model.

    Examples
    --------

        cleaner = Cleaner(df, llm, include_coroner=False, verbose=True)
        cleaned_df = cleaner.clean_reports()
        cleaned_df.head()
    """

    # DataFrame column names
    COL_URL = GeneralConfig.COL_URL
    COL_ID = GeneralConfig.COL_ID
    COL_DATE = GeneralConfig.COL_DATE
    COL_CORONER_NAME = GeneralConfig.COL_CORONER_NAME
    COL_AREA = GeneralConfig.COL_AREA
    COL_RECEIVER = GeneralConfig.COL_RECEIVER
    COL_INVESTIGATION = GeneralConfig.COL_INVESTIGATION
    COL_CIRCUMSTANCES = GeneralConfig.COL_CIRCUMSTANCES
    COL_CONCERNS = GeneralConfig.COL_CONCERNS

    @classmethod
    def map_area_synonym(cls, area: str) -> str:
        """Return canonical name for an area synonym."""
        return _canonicalise_area(area)

    @classmethod
    def match_area(cls, area: str, strategy: str = "exact_only") -> AreaMatchResult:
        """Return detailed area matching output for debugging and comparison."""
        return match_area(area, strategy=strategy)

    # Base prompt template used for all cleaning operations
    CLEANER_BASE_PROMPT = (
        "You are an expert in extracting and cleaning specific information from UK Coronal Prevention of Future Deaths (PFD) reports.\n\n"
        "Task:\n"
        "1. **Extract** only the information related to {field_description}.\n"
        "2. **Clean** the input text by fixing typos and removing clearly spurious characters (e.g. rogue numbers, stray punctuation, HTML tags). Do not delete any valid sentences or shorten the text.\n"
        "3. **Correct** any misspellings, ensuring the text is in sentence-case **British English**. Keep any existing acronyms if used; do not expand them.\n"
        "4. **Return** exactly and only the cleaned data for {field_contents_and_rules}. You must only return the cleaned string, without adding additional commentary, summarisation, or headings.\n"
        f"5. **If extraction fails**, return only and exactly: {GeneralConfig.NOT_FOUND_TEXT}\n"
        "6. **Do not** remove or summarise any of the original content other than the minimal fixes described above.\n\n"
        "Extra instructions:\n"
        "{extra_instructions}\n\n"
        "Input Text:\n"
        '"""'
    )

    # Field-specific configuration for prompt substitution
    CLEANER_PROMPT_CONFIG = {
        "Coroner": {
            "field_description": "the name of the Coroner who presided over the inquest",
            "field_contents_and_rules": "this name of the Coroner - nothing else",
            "extra_instructions": (
                "Remove all reference to titles & middle name(s), if present, and replace the first name with an initial. "
                'For example, if the string is "Mr. Joe E Bloggs", return "J. Bloggs". '
                'If the string is "Joe Bloggs Senior Coroner for West London", return "J. Bloggs". '
                'If the string is "J. Bloggs", just return "J. Bloggs" (no modification). '
            ),
        },
        "Area": {
            "field_description": "the area where the coroner's inquest took place",
            "field_contents_and_rules": "exactly one coroner area chosen from the allowed list below",
            "extra_instructions": (
                "Choose the best logical match from the allowed area list below. "
                "The input text does not need to exactly match an allowed label. "
                f"If no reasonable coroner area can be identified, return exactly: {GeneralConfig.NOT_FOUND_TEXT}. "
                'For example, if the string is "Area: West London", return "London West". '
            ),
        },
        "Receiver": {
            "field_description": "the name(s)/organisation(s) of the receiver(s) of the report",
            "field_contents_and_rules": "only the recipient organisation name(s) -- nothing else",
            "extra_instructions": (
                "Separate multiple recipients with semicolons (;). "
                "Do not use a numbered list. "
                "Remove reference to family altogether. "
                "Remove address(es) if given (i.e. just include the recipient organisation). "
                "If a personal name or job title is given alongside an organisation, return the organisation only. "
                "If the recipient is a named government office-holder or minister, return the relevant department rather than the person or office title. "
                "Do not return acronyms as the main recipient label when the full organisation name is available; return the full expanded name instead. "
                'For example, if the string is "CEO, Cardinal Healthcare", return "Cardinal Healthcare". '
                'If the string is "Jane Smith, Chief Executive, NHS England", return "NHS England". '
                'If the string is "John Smith MP, Secretary of State for Justice", return "Ministry of Justice". '
                'If the string is "Secretary of State for Health and Social Care", return "Department of Health and Social Care". '
                'If the string is "Secretary of State for Transport", return "Department for Transport". '
                'If the string is "Chief Executive of NHS England", return "NHS England". '
                'If the string is "The Chief Executive of Aneurin Bevan University Health Board", return "Aneurin Bevan University Health Board". '
                'If the string is "NICE (National Institute for Health and Care Excellence)", return "National Institute for Health and Care Excellence". '
                'If a redacted personal name appears before an office or organisation, remove the person and keep the office or organisation only. '
                'For example, if the string is "[REDACTED], Secretary of State for Health and Social Care", return "Department of Health and Social Care". '
                'If the string is "[REDACTED], Chief Executive NHS England", return "NHS England". '
                'If "Chief Coroner" appears as a recipient, remove it altogether rather than returning it. '
            ),
        },
        "InvestigationAndInquest": {
            "field_description": "the details of the investigation and inquest",
            "field_contents_and_rules": "the entire text",
            "extra_instructions": (
                "If the string appears to need no cleaning, return it as is. "
                "If a date is used, put it in numerical form (e.g. '1 January 2024'). "
                "Keep any existing paragraph formatting (e.g. spacing). "
                "Do not summarise or shorten the text."
            ),
        },
        "CircumstancesOfDeath": {
            "field_description": "the circumstances of death",
            "field_contents_and_rules": "the entire text",
            "extra_instructions": (
                "If the string appears to need no cleaning, return it as is. "
                "If a date is used, put it in numerical form (e.g. '1 January 2024'). "
                "Keep any existing paragraph formatting (e.g. spacing). "
                "Do not summarise or shorten the text."
            ),
        },
        "MattersOfConcern": {
            "field_description": "the matters of concern",
            "field_contents_and_rules": "the entire text",
            "extra_instructions": (
                'Remove reference to boilerplate text, if any occurs. This is usually 1 or 2 non-specific sentences at the start of the string often ending with "...The Matters of Concern are as follows:" (which should also be removed). '
                "If the string appears to need no cleaning, return it as is. "
                "If a date is used, put it in numerical form (e.g. '1 January 2024'). "
                "Keep any existing paragraph formatting (e.g. spacing). "
                "Do not summarise or shorten the text."
            ),
        },
    }

    def __init__(
        self,
        # Input DataFrame containing PFD reports
        reports: pd.DataFrame,
        llm: LLM,
        # Fields to clean
        include_coroner: bool = True,
        include_receiver: bool = True,
        include_area: bool = True,
        include_investigation: bool = True,
        include_circumstances: bool = True,
        include_concerns: bool = True,
        # Custom prompts for each field; defaults to None
        coroner_prompt: str = None,
        area_prompt: str = None,
        receiver_prompt: str = None,
        investigation_prompt: str = None,
        circumstances_prompt: str = None,
        concerns_prompt: str = None,
        verbose: bool = False,
    ) -> None:
        self.reports = reports
        self.llm = llm

        # Flags for which fields to clean
        self.include_coroner = include_coroner
        self.include_receiver = include_receiver
        self.include_area = include_area
        self.include_investigation = include_investigation
        self.include_circumstances = include_circumstances
        self.include_concerns = include_concerns

        # Prompt templates
        self.coroner_prompt_template = coroner_prompt or self._get_prompt_for_field(
            "Coroner"
        )
        self.area_prompt_template = area_prompt or self._build_area_prompt_template()
        self.receiver_prompt_template = receiver_prompt or self._get_prompt_for_field(
            "Receiver"
        )
        self.investigation_prompt_template = (
            investigation_prompt
            or self._get_prompt_for_field("InvestigationAndInquest")
        )
        self.circumstances_prompt_template = (
            circumstances_prompt or self._get_prompt_for_field("CircumstancesOfDeath")
        )
        self.concerns_prompt_template = concerns_prompt or self._get_prompt_for_field(
            "MattersOfConcern"
        )

        self.verbose = verbose

        # -----------------------------------------------------------------------------
        # Error and Warning Handling for Initialisation Parameters
        # -----------------------------------------------------------------------------

        ### Errors
        # If the reports parameter is not a DataFrame
        if not isinstance(reports, pd.DataFrame):
            raise TypeError("The 'reports' parameter must be a pandas DataFrame.")

        # If the input DataFrame does not contain the necessary columns
        required_df_columns = []
        if self.include_coroner:
            required_df_columns.append(self.COL_CORONER_NAME)
        if self.include_area:
            required_df_columns.append(self.COL_AREA)
        if self.include_receiver:
            required_df_columns.append(self.COL_RECEIVER)
        if self.include_investigation:
            required_df_columns.append(self.COL_INVESTIGATION)
        if self.include_circumstances:
            required_df_columns.append(self.COL_CIRCUMSTANCES)
        if self.include_concerns:
            required_df_columns.append(self.COL_CONCERNS)

        # Get unique column names in case user mapped multiple flags to the same df column
        required_df_columns = list(set(required_df_columns))

        missing_columns = [
            col for col in required_df_columns if col not in self.reports.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Cleaner could not find the following DataFrame columns: {missing_columns}."
            )

    def _get_prompt_for_field(self, field_name: str) -> str:
        """Generates a complete prompt template for a given PFD report field."""
        # Access prompt configuration stored on this class
        config = self.CLEANER_PROMPT_CONFIG[field_name]
        return self.CLEANER_BASE_PROMPT.format(
            field_description=config["field_description"],
            field_contents_and_rules=config["field_contents_and_rules"],
            extra_instructions=config["extra_instructions"],
        )

    def _build_area_prompt_template(self) -> str:
        """Return the area prompt with the current allowed list embedded."""
        base_prompt = self._get_prompt_for_field("Area")
        allowed_areas = ", ".join(
            area for area in GeneralConfig.ALLOWED_AREAS if area != "Other"
        )
        insertion = (
            f"Allowed area labels:\n{allowed_areas}\n\n"
            f"Return exactly one label from this list. If no logical match exists, return exactly: {GeneralConfig.NOT_FOUND_TEXT}."
        )
        return base_prompt.replace("\n\nInput Text:\n", f"\n\n{insertion}\n\nInput Text:\n")

    def generate_prompt_template(self) -> dict[str, str]:
        """Return the prompt templates used for each field.

        The returned dictionary maps DataFrame column names to the full prompt
        text with a ``[TEXT]`` placeholder appended to illustrate how the
        prompt will look during ``clean_reports``.
        """

        return {
            self.COL_CORONER_NAME: f"{self.coroner_prompt_template}\n[TEXT]",
            self.COL_AREA: f"{self.area_prompt_template}\n[TEXT]",
            self.COL_RECEIVER: f"{self.receiver_prompt_template}\n[TEXT]",
            self.COL_INVESTIGATION: f"{self.investigation_prompt_template}\n[TEXT]",
            self.COL_CIRCUMSTANCES: f"{self.circumstances_prompt_template}\n[TEXT]",
            self.COL_CONCERNS: f"{self.concerns_prompt_template}\n[TEXT]",
        }

    def clean_reports(self, anonymise: bool = False) -> pd.DataFrame:
        """Run LLM-based cleaning for the configured columns.

        The method operates **in place on a copy** of ``self.reports`` so the
        original DataFrame is never mutated.

        Returns
        -------
        pandas.DataFrame
            A new DataFrame in which the selected columns have been
            replaced by the LLM output (or left unchanged when the model
            returns an error marker).

        Parameters
        ----------
        anonymise : bool, optional
            When ``True`` append an instruction to anonymise names and pronouns
            in the investigation, circumstances and concerns fields. Defaults to
            ``False``.

        Examples
        --------
            cleaner = Cleaner(llm=llm_client, reports=reports)
            cleaned = cleaner.clean_reports()

        """
        cleaned_df = self.reports.copy()  # Work on a copy

        # Optional anonymisation instruction
        anonymise_instruction = (
            "Replace all personal names and pronouns with they/them/their."
        )

        investigation_prompt = self.investigation_prompt_template
        circumstances_prompt = self.circumstances_prompt_template
        concerns_prompt = self.concerns_prompt_template

        if anonymise:
            # Insert the instruction just before the input text portion so the
            # LLM treats it as guidance rather than part of the text to clean
            insertion_point = "\n\nInput Text:"
            investigation_prompt = investigation_prompt.replace(
                insertion_point,
                f"\n{anonymise_instruction}{insertion_point}",
            )
            circumstances_prompt = circumstances_prompt.replace(
                insertion_point,
                f"\n{anonymise_instruction}{insertion_point}",
            )
            concerns_prompt = concerns_prompt.replace(
                insertion_point,
                f"\n{anonymise_instruction}{insertion_point}",
            )

        # Define fields to process: (Config Key, Process Flag, DF Column Name, Prompt Template)
        field_processing_config = [
            (
                "Coroner",
                self.include_coroner,
                self.COL_CORONER_NAME,
                self.coroner_prompt_template,
            ),
            ("Area", self.include_area, self.COL_AREA, self.area_prompt_template),
            (
                "Receiver",
                self.include_receiver,
                self.COL_RECEIVER,
                self.receiver_prompt_template,
            ),
            (
                "InvestigationAndInquest",
                self.include_investigation,
                self.COL_INVESTIGATION,
                investigation_prompt,
            ),
            (
                "CircumstancesOfDeath",
                self.include_circumstances,
                self.COL_CIRCUMSTANCES,
                circumstances_prompt,
            ),
            (
                "MattersOfConcern",
                self.include_concerns,
                self.COL_CONCERNS,
                concerns_prompt,
            ),
        ]

        # Use tqdm for the outer loop over fields
        for config_key, process_flag, column_name, prompt_template in tqdm(
            field_processing_config, desc="Processing Fields", position=0, leave=True
        ):
            if not process_flag:
                continue

            if column_name not in cleaned_df.columns:
                # This case should ideally be caught by __init__ checks, but good to have defence here
                logger.warning(
                    f"Column '{column_name}' for field '{config_key}' not found at cleaning time. Skipping."
                )
                continue
            if self.verbose:
                logger.info(
                    f"Preparing batch for column: '{column_name}' (Field: {config_key})"
                )

            # Ensure column is treated as string for processing
            # Handle cases where column might be all NaNs or mixed type before attempting string operations
            if cleaned_df[column_name].notna().any():
                if not pd.api.types.is_string_dtype(cleaned_df[column_name]):
                    cleaned_df[column_name] = cleaned_df[column_name].astype(str)
            else:
                logger.info(
                    f"Column '{column_name}' contains all NaN values. No text to clean."
                )
                continue  # Skip to next field if column is all NaN

            # Select non-null texts to clean and their original indices
            # Ensure we are working with string representations for LLM processing
            texts_to_clean_series = cleaned_df[column_name][
                cleaned_df[column_name].notna()
            ].astype(str)

            original_indices = texts_to_clean_series.index
            original_texts_list = texts_to_clean_series.tolist()

            if texts_to_clean_series.empty:
                logger.info(
                    f"No actual text data to clean in column '{column_name}' after filtering NaNs. Skipping."
                )
                continue

            # Construct prompts for the batch
            # Each prompt consists of the field-specific template followed by the actual text
            prompts_for_batch = [
                f"{prompt_template}\n{text}" for text in original_texts_list
            ]

            if (
                not prompts_for_batch
            ):  # Should not happen if texts_to_clean_series was not empty
                logger.info(
                    f"No prompts generated for column '{column_name}'. Skipping LLM call."
                )
                continue

            if self.verbose:
                logger.info(
                    f"First prompt for '{column_name}' batch: {prompts_for_batch[0][:250]}..."
                )  # Log snippet of first prompt

            # Call LLM in batch
            if self.verbose:
                logger.info(
                    f"Sending {len(prompts_for_batch)} text items to LLM for column '{column_name}'."
                )

            inner_tqdm_config = {
                "desc": f"LLM: Cleaning {config_key}",
                "position": 1,
                "leave": False,
            }

            response_model = AreaModel if config_key == "Area" else None
            cleaned_results_batch = self.llm.generate(
                prompts=prompts_for_batch,
                tqdm_extra_kwargs=inner_tqdm_config,
                response_format=response_model,
            )

            if len(cleaned_results_batch) != len(prompts_for_batch):
                logger.error(
                    f"Mismatch in results count for '{column_name}'. "
                    f"Expected {len(prompts_for_batch)}, got {len(cleaned_results_batch)}. "
                    "Skipping update for this column to prevent data corruption."
                )
                continue  # Skip if counts don't match

            # Process results and update DataFrame
            modifications_count = 0
            for i, cleaned_text_output in enumerate(cleaned_results_batch):
                original_text = original_texts_list[i]
                df_index = original_indices[i]

                if isinstance(cleaned_text_output, BaseModel):
                    cleaned_text_output = getattr(cleaned_text_output, "area", "")

                final_text_to_write = cleaned_text_output  # Assume success initially

                # Logic to revert to original if cleaning "failed" or LLM indicated "N/A"
                if isinstance(cleaned_text_output, str) or pd.isna(cleaned_text_output):
                    if (
                        pd.isna(cleaned_text_output)
                        or cleaned_text_output.startswith("Error:")
                        or cleaned_text_output.startswith("N/A: LLM Error")
                        or cleaned_text_output.startswith("N/A: Unexpected LLM output")
                    ):
                        if self.verbose:
                            logger.info(
                                f"Reverting to original for column '{column_name}', index {df_index}. LLM output: '{cleaned_text_output}'"
                            )
                        final_text_to_write = original_text
                    elif config_key == "Receiver":
                        final_text_to_write = _normalise_receiver_output(cleaned_text_output)
                    elif cleaned_text_output != original_text:
                        modifications_count += 1
                elif cleaned_text_output is None and original_text is not None:
                    logger.warning(
                        f"LLM returned None for non-null original text (index {df_index}, col '{column_name}'). Reverting to original."
                    )
                    final_text_to_write = original_text

                cleaned_df.loc[df_index, column_name] = final_text_to_write

            if self.verbose:
                logger.info(
                    f"Finished batch cleaning for '{column_name}'. {modifications_count} entries were actively modified by the LLM."
                )

        self.cleaned_reports = cleaned_df
        return cleaned_df
