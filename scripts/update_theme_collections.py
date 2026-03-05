#!/usr/bin/env python3
"""Apply approved theme collections to the report dataset."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import Field, create_model

from pfd_toolkit import Extractor, LLM


def _load_openai_key() -> str:
    env_value = (os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_API_KEY") or "").strip()
    if env_value:
        return env_value

    project_root = Path(__file__).resolve().parents[1]
    candidate_paths = [
        project_root / "api.env",
        project_root / "src" / "pfd_toolkit" / "api.env",
    ]
    for path in candidate_paths:
        if not path.exists():
            continue
        raw = path.read_text(encoding="utf-8")
        for line in raw.splitlines():
            token = line.strip()
            if not token or token.startswith("#"):
                continue
            if "=" not in token:
                continue
            left, right = token.split("=", 1)
            key_name = left.strip()
            if key_name not in {"OPENAI_API_KEY", "OPEN_API_KEY"}:
                continue
            value = right.strip().strip('"').strip("'")
            if value:
                return value

    raise ValueError("OPENAI_API_KEY is required.")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Apply curated themes to all reports using the LLM extractor."
    )
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("all_reports.csv"),
    )
    parser.add_argument("--schema", type=Path, default=Path("scripts/theme_collections/approved_themes.json"))
    parser.add_argument("--model", default="gpt-4.1")
    parser.add_argument("--max-workers", type=int, default=18)
    parser.add_argument(
        "--recompute-all",
        action="store_true",
        help="Recompute theme columns for every row, not only rows with missing values.",
    )
    parser.add_argument(
        "--summary-file",
        type=Path,
        default=Path(".github/workflows/update_summary.txt"),
    )
    return parser.parse_args()


def _load_theme_schema(path: Path) -> dict[str, str]:
    if not path.exists():
        raise FileNotFoundError(f"Theme schema file not found: {path}")
    payload = json.loads(path.read_text(encoding="utf-8"))
    themes = payload.get("themes")
    if not isinstance(themes, dict) or not themes:
        raise ValueError("Theme schema JSON must include a non-empty 'themes' object.")
    cleaned: dict[str, str] = {}
    for key, value in themes.items():
        name = str(key or "").strip()
        if not name:
            continue
        cleaned[name] = str(value or "").strip()
    if not cleaned:
        raise ValueError("No valid themes found in schema file.")
    return cleaned


def _build_theme_model(themes: dict[str, str]):
    fields = {
        name: (bool, Field(description=description))
        for name, description in themes.items()
    }
    return create_model("ApprovedThemeCollections", **fields)


def _coerce_boolean_series(series: pd.Series) -> pd.Series:
    def _coerce(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        if value is None:
            return False
        if isinstance(value, (int, float)):
            return bool(value)
        token = str(value).strip().lower()
        if token in {"true", "1", "yes", "y", "t"}:
            return True
        if token in {"false", "0", "no", "n", "f", "", "none", "nan", "not_found"}:
            return False
        return False

    return series.map(_coerce)


def _row_mask_for_update(reports: pd.DataFrame, theme_columns: list[str], recompute_all: bool) -> pd.Series:
    if recompute_all:
        return pd.Series(True, index=reports.index)

    missing_columns = [column for column in theme_columns if column not in reports.columns]
    if missing_columns:
        return pd.Series(True, index=reports.index)

    missing_value_mask = reports[theme_columns].isna().any(axis=1)
    return missing_value_mask


def _append_summary(summary_file: Path, message: str) -> None:
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with summary_file.open("a", encoding="utf-8") as handle:
        handle.write("\n")
        handle.write(message)
        if not message.endswith("\n"):
            handle.write("\n")


def main() -> None:
    args = _parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"{args.input} not found.")

    api_key = _load_openai_key()

    themes = _load_theme_schema(args.schema)
    theme_columns = list(themes.keys())
    reports = pd.read_csv(args.input)
    row_mask = _row_mask_for_update(reports, theme_columns, recompute_all=args.recompute_all)
    rows_to_update = int(row_mask.sum())

    if rows_to_update == 0:
        print("No theme collection updates required.")
        _append_summary(args.summary_file, "Theme collections already up to date (no rows updated).")
        return

    subset = reports.loc[row_mask].copy().reset_index(drop=True)
    llm_client = LLM(
        api_key=api_key,
        model=args.model,
        max_workers=max(1, args.max_workers),
        temperature=0.0,
        timeout=30,
        seed=123,
        validation_attempts=2,
    )
    extractor = Extractor(
        llm=llm_client,
        reports=subset,
        include_date=True,
        include_coroner=True,
        include_area=True,
        include_receiver=True,
        include_investigation=True,
        include_circumstances=True,
        include_concerns=True,
        verbose=False,
    )
    theme_model = _build_theme_model(themes)
    assigned_subset = extractor.extract_features(
        feature_model=theme_model,
        force_assign=True,
        allow_multiple=True,
        skip_if_present=False,
    )

    for column in theme_columns:
        assigned_subset[column] = _coerce_boolean_series(assigned_subset[column])
        reports.loc[row_mask, column] = assigned_subset[column].values
        reports[column] = _coerce_boolean_series(reports[column])

    reports.to_csv(args.input, index=False)

    print(f"Theme collections updated for {rows_to_update} row(s).")
    for column in theme_columns:
        print(f"{column}: {int(reports[column].sum())}")

    _append_summary(
        args.summary_file,
        f"Theme collections updated for {rows_to_update} row(s) using {len(theme_columns)} approved themes.",
    )


if __name__ == "__main__":
    main()
