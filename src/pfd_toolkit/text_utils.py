import logging
import re
from dateutil import parser as date_parser
import pandas as pd

logger = logging.getLogger(__name__)


def normalise_apostrophes(text: str) -> str:
    """Replace curly apostrophes with straight ones."""
    return text.replace("’", "'").replace("‘", "'")


def clean_text(text: str) -> str:
    """Collapse whitespace and normalise apostrophes."""
    normalised = normalise_apostrophes(text)
    return " ".join(normalised.split())


def normalise_date(raw_date_str: str, verbose: bool = False) -> str:
    """Convert a fuzzy date string into ``YYYY-MM-DD`` format if possible."""
    text_being_processed = clean_text(raw_date_str).strip()
    final_text_to_parse = text_being_processed
    try:
        match = re.match(r"(.+?)(Ref[:\s]|$)", text_being_processed, re.IGNORECASE)
        if match:
            potential = match.group(1).strip()
            if potential:
                final_text_to_parse = potential
        if not final_text_to_parse:
            if verbose:
                logger.warning(
                    "Date string empty after trying to remove 'Ref...' from '%s'. Keeping raw.",
                    text_being_processed,
                )
            return text_being_processed
        dt = date_parser.parse(final_text_to_parse, fuzzy=True, dayfirst=True)
        return dt.strftime("%Y-%m-%d")
    except Exception as e:
        if verbose:
            logger.warning(
                "Date parse failed for raw '%s' (processed to '%s', attempted '%s') – keeping raw. Error: %s",
                raw_date_str,
                text_being_processed,
                final_text_to_parse,
                e,
            )
        return text_being_processed


def process_extracted_field(
    text: str,
    strings_to_remove: list[str],
    not_found_text: object,
    *,
    min_len: int | None = None,
    max_len: int | None = None,
    is_date: bool = False,
    verbose: bool = False,
) -> str:
    """Apply cleaning and validation to an extracted text field."""
    if pd.isna(not_found_text):
        if pd.isna(text) or text is None or text == "":
            return not_found_text
    else:
        if text == not_found_text:
            return not_found_text
    processed_text = text
    for s_to_remove in strings_to_remove:
        processed_text = processed_text.replace(s_to_remove, "")
    processed_text = processed_text.strip()
    processed_text = clean_text(processed_text)
    if is_date:
        return normalise_date(processed_text, verbose=verbose)
    if not processed_text and min_len is not None and min_len > 0:
        return not_found_text
    if min_len is not None and len(processed_text) < min_len:
        return not_found_text
    if max_len is not None and len(processed_text) > max_len:
        return not_found_text
    return processed_text
