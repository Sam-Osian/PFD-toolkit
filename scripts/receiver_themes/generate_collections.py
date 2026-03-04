#!/usr/bin/env python3
"""Generate rule-based receiver theme columns for a PFD dataset."""

from __future__ import annotations

import re
from pathlib import Path

import pandas as pd


INPUT_CSV = Path("scripts/data/all_reports.csv")
OUTPUT_CSV = Path("scripts/data/all_reports_with_collections.csv")
OVERWRITE_INPUT = False


def _split_receiver_segments(receiver: str) -> list[str]:
    if not isinstance(receiver, str):
        return []
    return [segment.strip() for segment in receiver.split(";") if segment.strip()]


def _normalise_segment(segment: str) -> str:
    cleaned = segment.casefold()
    cleaned = cleaned.replace("&", " and ")
    cleaned = cleaned.replace("’", "'")
    cleaned = re.sub(r"[^a-z0-9']+", " ", cleaned)
    return " ".join(cleaned.split())


def _segment_matches_any(segment: str, patterns: tuple[str, ...]) -> bool:
    normalised = _normalise_segment(segment)
    return any(pattern in normalised for pattern in patterns)


def _segment_startswith_any(segment: str, prefixes: tuple[str, ...]) -> bool:
    normalised = _normalise_segment(segment)
    return any(normalised.startswith(prefix) for prefix in prefixes)


def _segment_endswith_any(segment: str, suffixes: tuple[str, ...]) -> bool:
    normalised = _normalise_segment(segment)
    return any(normalised.endswith(suffix) for suffix in suffixes)


def _apply_theme_rule(
    reports: pd.DataFrame,
    *,
    theme_name: str,
    matcher,
) -> None:
    reports[theme_name] = reports["receiver"].apply(
        lambda value: any(matcher(segment) for segment in _split_receiver_segments(value))
    )


def main() -> None:
    input_path = INPUT_CSV
    output_path = input_path if OVERWRITE_INPUT else OUTPUT_CSV

    reports = pd.read_csv(input_path)

    # --------------------
    # NHS bodies
    # --------------------
    def match_nhs_bodies(segment: str) -> bool:
        return _segment_matches_any(
            segment,
            (
                " nhs ",
                "nhs ",
                " integrated care board",
                " health board",
            ),
        )

    _apply_theme_rule(
        reports,
        theme_name="theme_sent_to_nhs_bodies",
        matcher=match_nhs_bodies,
    )

    # --------------------
    # Governmental departments
    # --------------------
    GOVERNMENT_DEPARTMENTS = (
        "cabinet office",
        "home office",
        "ministry of justice",
        "attorney general's office",
        "welsh government",
    )

    def match_government_departments(segment: str) -> bool:
        return _segment_startswith_any(
            segment,
            (
                "department of ",
                "department for ",
            ),
        ) or _segment_matches_any(segment, GOVERNMENT_DEPARTMENTS)

    _apply_theme_rule(
        reports,
        theme_name="theme_sent_to_government_departments",
        matcher=match_government_departments,
    )

    # --------------------
    # Prisons
    # --------------------
    PRISON_PATTERNS = (
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

    def match_prisons(segment: str) -> bool:
        return _segment_matches_any(segment, PRISON_PATTERNS)

    _apply_theme_rule(
        reports,
        theme_name="theme_sent_to_prisons",
        matcher=match_prisons,
    )

    # --------------------
    # Health regulators
    # --------------------
    HEALTH_REGULATORS = (
        "care quality commission",
        "national institute for health and care excellence",
        "medicines and healthcare products regulatory agency",
        "general medical council",
        "nursing and midwifery council",
        "health and care professions council",
        "general pharmaceutical council",
    )

    def match_health_regulators(segment: str) -> bool:
        return _segment_matches_any(segment, HEALTH_REGULATORS)

    _apply_theme_rule(
        reports,
        theme_name="theme_sent_to_health_regulators",
        matcher=match_health_regulators,
    )

    # --------------------
    # Local government
    # --------------------
    LOCAL_GOVERNMENT_PATTERNS = (
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

    LOCAL_GOVERNMENT_COUNCIL_SUFFIXES = (
        " council",
    )

    LOCAL_GOVERNMENT_COUNCIL_EXCLUSIONS = (
        "general medical council",
        "nursing and midwifery council",
        "general pharmaceutical council",
        "health and care professions council",
        "national police chiefs council",
        "royal college",
    )

    def match_local_government(segment: str) -> bool:
        normalised = _normalise_segment(segment)
        if _segment_matches_any(segment, LOCAL_GOVERNMENT_PATTERNS):
            return True
        if _segment_endswith_any(segment, LOCAL_GOVERNMENT_COUNCIL_SUFFIXES):
            return not any(
                exclusion in normalised for exclusion in LOCAL_GOVERNMENT_COUNCIL_EXCLUSIONS
            )
        return False

    _apply_theme_rule(
        reports,
        theme_name="theme_sent_to_local_government",
        matcher=match_local_government,
    )

    reports.to_csv(output_path, index=False)

    theme_columns = [column for column in reports.columns if column.startswith("theme_")]
    print(f"Rows processed: {len(reports)}")
    print(f"Wrote themed dataset to: {output_path}")
    for column in theme_columns:
        print(f"{column}: {int(reports[column].fillna(False).sum())}")


if __name__ == "__main__":
    main()
