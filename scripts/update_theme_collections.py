#!/usr/bin/env python3
"""Apply approved theme collections to the report dataset."""

from __future__ import annotations

import argparse
import json
import os
import hashlib
from pathlib import Path
from typing import Any
import math

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
        "--theme-batches",
        type=int,
        default=3,
        help="Number of theme batches to process sequentially. Defaults to 3.",
    )
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
    parser.add_argument(
        "--checkpoint-file",
        type=Path,
        default=Path("scripts/theme_collections/update_theme_collections_checkpoint.csv"),
        help="CSV file used for incremental checkpointing and resume.",
    )
    parser.add_argument(
        "--checkpoint-metadata-file",
        type=Path,
        default=Path("scripts/theme_collections/update_theme_collections_checkpoint.meta.json"),
        help="Metadata file used to validate checkpoint compatibility on resume.",
    )
    parser.add_argument(
        "--no-resume",
        action="store_true",
        help="Ignore any existing checkpoint and recompute from scratch.",
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


def _chunk_theme_map(themes: dict[str, str], batches: int) -> list[dict[str, str]]:
    if batches <= 0:
        raise ValueError("--theme-batches must be at least 1.")
    items = list(themes.items())
    if not items:
        return []
    batch_size = max(1, math.ceil(len(items) / batches))
    chunks: list[dict[str, str]] = []
    for start in range(0, len(items), batch_size):
        chunk_items = items[start : start + batch_size]
        chunks.append(dict(chunk_items))
    return chunks


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


def _metadata_signature(
    input_path: Path,
    schema_path: Path,
    theme_columns: list[str],
    source_indices: pd.Index,
) -> dict[str, Any]:
    digest = hashlib.sha256()
    digest.update(str(input_path.resolve()).encode("utf-8"))
    digest.update(str(schema_path.resolve()).encode("utf-8"))
    digest.update("|".join(theme_columns).encode("utf-8"))
    digest.update(",".join(str(int(idx)) for idx in source_indices.tolist()).encode("utf-8"))
    return {
        "input_path": str(input_path.resolve()),
        "schema_path": str(schema_path.resolve()),
        "theme_columns": theme_columns,
        "source_index_hash": digest.hexdigest(),
        "row_count": int(len(source_indices)),
    }


def _save_checkpoint(
    checkpoint_file: Path,
    metadata_file: Path,
    source_indices: pd.Index,
    assigned_subset: pd.DataFrame,
    theme_columns: list[str],
    metadata: dict[str, Any],
) -> None:
    checkpoint_file.parent.mkdir(parents=True, exist_ok=True)
    metadata_file.parent.mkdir(parents=True, exist_ok=True)

    checkpoint_payload = pd.DataFrame({"__source_index": source_indices.to_list()})
    for column in theme_columns:
        if column in assigned_subset.columns:
            checkpoint_payload[column] = assigned_subset[column]

    tmp_checkpoint = checkpoint_file.with_suffix(f"{checkpoint_file.suffix}.tmp")
    tmp_metadata = metadata_file.with_suffix(f"{metadata_file.suffix}.tmp")
    checkpoint_payload.to_csv(tmp_checkpoint, index=False)
    tmp_metadata.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    tmp_checkpoint.replace(checkpoint_file)
    tmp_metadata.replace(metadata_file)


def _load_checkpoint_if_compatible(
    checkpoint_file: Path,
    metadata_file: Path,
    metadata: dict[str, Any],
) -> pd.DataFrame | None:
    if not checkpoint_file.exists() or not metadata_file.exists():
        return None

    try:
        checkpoint_metadata = json.loads(metadata_file.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return None

    if checkpoint_metadata != metadata:
        return None

    checkpoint = pd.read_csv(checkpoint_file)
    if "__source_index" not in checkpoint.columns:
        return None
    if len(checkpoint) != metadata["row_count"]:
        return None
    return checkpoint


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

    subset_source_indices = reports.index[row_mask]
    subset = reports.loc[row_mask].copy().reset_index(drop=True)
    checkpoint_metadata = _metadata_signature(
        input_path=args.input,
        schema_path=args.schema,
        theme_columns=theme_columns,
        source_indices=subset_source_indices,
    )
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
    theme_batches = _chunk_theme_map(themes, args.theme_batches)
    assigned_subset = subset.copy()

    if args.no_resume:
        if args.checkpoint_file.exists():
            print(f"Ignoring checkpoint due to --no-resume: {args.checkpoint_file}")
    else:
        checkpoint = _load_checkpoint_if_compatible(
            checkpoint_file=args.checkpoint_file,
            metadata_file=args.checkpoint_metadata_file,
            metadata=checkpoint_metadata,
        )
        if checkpoint is None and args.checkpoint_file.exists():
            print("Existing checkpoint is incompatible with current run; recomputing from scratch.")
        if checkpoint is not None:
            print(f"Resuming from checkpoint: {args.checkpoint_file}")
            for column in theme_columns:
                if column in checkpoint.columns:
                    assigned_subset[column] = _coerce_boolean_series(checkpoint[column])

    total_batches = len(theme_batches)
    try:
        for batch_idx, batch_themes in enumerate(theme_batches, start=1):
            batch_columns = list(batch_themes.keys())
            already_complete = all(
                column in assigned_subset.columns and assigned_subset[column].notna().all()
                for column in batch_columns
            )
            if already_complete:
                print(
                    f"Skipping batch {batch_idx}/{total_batches} "
                    f"({len(batch_columns)} themes) from checkpoint."
                )
                continue

            print(
                f"Running theme assignment batch {batch_idx}/{total_batches} "
                f"({len(batch_columns)} themes)."
            )
            batch_model = _build_theme_model(batch_themes)
            batch_assigned = extractor.extract_features(
                feature_model=batch_model,
                force_assign=True,
                allow_multiple=True,
                skip_if_present=False,
            )
            for column in batch_columns:
                assigned_subset[column] = _coerce_boolean_series(batch_assigned[column])
            _save_checkpoint(
                checkpoint_file=args.checkpoint_file,
                metadata_file=args.checkpoint_metadata_file,
                source_indices=subset_source_indices,
                assigned_subset=assigned_subset,
                theme_columns=theme_columns,
                metadata=checkpoint_metadata,
            )
            print(f"Checkpoint saved: {args.checkpoint_file}")
    except (KeyboardInterrupt, Exception):
        _save_checkpoint(
            checkpoint_file=args.checkpoint_file,
            metadata_file=args.checkpoint_metadata_file,
            source_indices=subset_source_indices,
            assigned_subset=assigned_subset,
            theme_columns=theme_columns,
            metadata=checkpoint_metadata,
        )
        print(f"Interrupted; partial checkpoint saved to {args.checkpoint_file}")
        raise

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
        (
            f"Theme collections updated for {rows_to_update} row(s) using "
            f"{len(theme_columns)} approved themes across {len(theme_batches)} batch(es)."
        ),
    )


if __name__ == "__main__":
    main()
