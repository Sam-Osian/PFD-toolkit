from __future__ import annotations

import re

import pandas as pd


COLLECTION_COLUMNS: dict[str, str] = {
    "nhs": "theme_sent_to_nhs_bodies",
    "gov_department": "theme_sent_to_government_departments",
    "prisons": "theme_sent_to_prisons",
    "health_regulators": "theme_sent_to_health_regulators",
    "local_gov": "theme_sent_to_local_government",
}

_GOVERNMENT_DEPARTMENTS = (
    "cabinet office",
    "home office",
    "ministry of justice",
    "attorney general's office",
    "welsh government",
)

_PRISON_PATTERNS = (
    " hmp ",
    "hmp ",
    " hm prison",
    " prison",
    " young offender institution",
    " yoi ",
    "yoi ",
    " secure training centre",
    " hm prison and probation service",
)

_HEALTH_REGULATORS = (
    "care quality commission",
    "national institute for health and care excellence",
    "medicines and healthcare products regulatory agency",
    "general medical council",
    "nursing and midwifery council",
    "health and care professions council",
    "general pharmaceutical council",
)

_LOCAL_GOVERNMENT_PATTERNS = (
    " county council",
    " city council",
    " borough council",
    " district council",
    " county borough council",
    " metropolitan borough council",
    " london borough of ",
    " unitary authority",
    " local authority",
)

_LOCAL_GOVERNMENT_COUNCIL_SUFFIXES = (" council",)

_LOCAL_GOVERNMENT_COUNCIL_EXCLUSIONS = (
    "general medical council",
    "nursing and midwifery council",
    "general pharmaceutical council",
    "health and care professions council",
    "national police chiefs council",
    "royal college",
)


def _split_receiver_segments(receiver: str) -> list[str]:
    if not isinstance(receiver, str):
        return []
    return [segment.strip() for segment in receiver.split(";") if segment.strip()]


def _normalise_segment(segment: str) -> str:
    cleaned = segment.casefold()
    cleaned = cleaned.replace("&", " and ")
    cleaned = cleaned.replace("’", "'")
    cleaned = re.sub(r"[^a-z0-9']+", " ", cleaned)
    return f" {' '.join(cleaned.split())} "


def _segment_matches_any(segment: str, patterns: tuple[str, ...]) -> bool:
    normalised = _normalise_segment(segment)
    return any(pattern in normalised for pattern in patterns)


def _segment_startswith_any(segment: str, prefixes: tuple[str, ...]) -> bool:
    normalised = _normalise_segment(segment).strip()
    return any(normalised.startswith(prefix) for prefix in prefixes)


def _segment_endswith_any(segment: str, suffixes: tuple[str, ...]) -> bool:
    normalised = _normalise_segment(segment).strip()
    return any(normalised.endswith(suffix) for suffix in suffixes)


def _match_nhs_bodies(segment: str) -> bool:
    return _segment_matches_any(
        segment,
        (
            " nhs ",
            " integrated care board",
            " health board",
        ),
    )


def _match_government_departments(segment: str) -> bool:
    return _segment_startswith_any(
        segment,
        (
            "department of ",
            "department for ",
        ),
    ) or _segment_matches_any(segment, _GOVERNMENT_DEPARTMENTS)


def _match_prisons(segment: str) -> bool:
    return _segment_matches_any(segment, _PRISON_PATTERNS)


def _match_health_regulators(segment: str) -> bool:
    return _segment_matches_any(segment, _HEALTH_REGULATORS)


def _match_local_government(segment: str) -> bool:
    normalised = _normalise_segment(segment).strip()
    if _segment_matches_any(segment, _LOCAL_GOVERNMENT_PATTERNS):
        return True
    if _segment_endswith_any(segment, _LOCAL_GOVERNMENT_COUNCIL_SUFFIXES):
        return not any(
            exclusion in normalised for exclusion in _LOCAL_GOVERNMENT_COUNCIL_EXCLUSIONS
        )
    return False


def _apply_collection_rule(
    reports: pd.DataFrame,
    *,
    collection_column: str,
    matcher,
    receiver_column: str,
    row_mask: pd.Series | None,
) -> None:
    target_mask = (
        row_mask.reindex(reports.index, fill_value=False)
        if row_mask is not None
        else pd.Series(True, index=reports.index)
    )
    if collection_column not in reports.columns:
        reports[collection_column] = False
    reports.loc[target_mask, collection_column] = reports.loc[target_mask, receiver_column].apply(
        lambda value: any(matcher(segment) for segment in _split_receiver_segments(value))
    )


def apply_collection_columns(
    reports: pd.DataFrame,
    *,
    receiver_column: str = "receiver",
    row_mask: pd.Series | None = None,
) -> pd.DataFrame:
    """Populate rule-based collection columns from a receiver column."""
    if receiver_column not in reports.columns:
        raise ValueError(f"Receiver column '{receiver_column}' is not present in the dataset.")

    _apply_collection_rule(
        reports,
        collection_column=COLLECTION_COLUMNS["nhs"],
        matcher=_match_nhs_bodies,
        receiver_column=receiver_column,
        row_mask=row_mask,
    )
    _apply_collection_rule(
        reports,
        collection_column=COLLECTION_COLUMNS["gov_department"],
        matcher=_match_government_departments,
        receiver_column=receiver_column,
        row_mask=row_mask,
    )
    _apply_collection_rule(
        reports,
        collection_column=COLLECTION_COLUMNS["prisons"],
        matcher=_match_prisons,
        receiver_column=receiver_column,
        row_mask=row_mask,
    )
    _apply_collection_rule(
        reports,
        collection_column=COLLECTION_COLUMNS["health_regulators"],
        matcher=_match_health_regulators,
        receiver_column=receiver_column,
        row_mask=row_mask,
    )
    _apply_collection_rule(
        reports,
        collection_column=COLLECTION_COLUMNS["local_gov"],
        matcher=_match_local_government,
        receiver_column=receiver_column,
        row_mask=row_mask,
    )
    return reports
