#!/usr/bin/env python3
"""Update rule-based and LLM-based collection columns in the report dataset."""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

import pandas as pd
from pydantic import Field, create_model

from pfd_toolkit import Extractor, LLM
from pfd_toolkit.collections import COLLECTION_COLUMNS, apply_collection_columns


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
        description="Update regex/rule collections and approved LLM theme collections.",
    )
    parser.add_argument("--input", type=Path, default=Path("all_reports.csv"))
    parser.add_argument(
        "--schema",
        type=Path,
        default=Path("scripts/theme_collections/approved_themes.json"),
    )
    parser.add_argument("--model", default="gpt-4.1")
    parser.add_argument("--max-workers", type=int, default=18)
    parser.add_argument(
        "--theme-batches",
        type=int,
        default=3,
        help="Number of LLM theme batches to process sequentially.",
    )
    parser.add_argument(
        "--recompute-all",
        action="store_true",
        help="Recompute both regex and LLM collection columns for all rows.",
    )
    parser.add_argument(
        "--recompute-all-rules",
        action="store_true",
        help="Recompute regex/rule-based collection columns for all rows.",
    )
    parser.add_argument(
        "--recompute-all-llm",
        action="store_true",
        help="Recompute approved LLM theme columns for all rows.",
    )
    parser.add_argument(
        "--skip-rule-based",
        action="store_true",
        help="Skip regex/rule-based collection updates.",
    )
    parser.add_argument(
        "--skip-llm",
        action="store_true",
        help="Skip LLM-based collection updates.",
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


def _append_summary(summary_file: Path, message: str) -> None:
    summary_file.parent.mkdir(parents=True, exist_ok=True)
    with summary_file.open("a", encoding="utf-8") as handle:
        handle.write("\n")
        handle.write(message)
        if not message.endswith("\n"):
            handle.write("\n")


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
        if token in {"false", "0", "no", "n", "f", "", "none", "nan", "not_found", "<na>"}:
            return False
        return False

    return series.map(_coerce)


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
    fields = {name: (bool, Field(description=description)) for name, description in themes.items()}
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


def _row_mask_for_update(reports: pd.DataFrame, columns: list[str], recompute_all: bool) -> pd.Series:
    if recompute_all:
        return pd.Series(True, index=reports.index)

    missing_columns = [column for column in columns if column not in reports.columns]
    if missing_columns:
        return pd.Series(True, index=reports.index)

    return reports[columns].isna().any(axis=1)


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


def _run_rule_updates(reports: pd.DataFrame, *, recompute_all_rules: bool) -> tuple[int, dict[str, int]]:
    rule_columns = list(COLLECTION_COLUMNS.values())
    row_mask = _row_mask_for_update(reports, rule_columns, recompute_all=recompute_all_rules)
    rows_to_update = int(row_mask.sum())
    if rows_to_update == 0:
        return 0, {column: int(_coerce_boolean_series(reports[column]).sum()) for column in rule_columns if column in reports.columns}

    apply_collection_columns(reports, recompute_all=recompute_all_rules)

    counts: dict[str, int] = {}
    for column in rule_columns:
        reports[column] = _coerce_boolean_series(reports[column])
        counts[column] = int(reports[column].sum())
    return rows_to_update, counts


def _run_llm_updates(
    reports: pd.DataFrame,
    *,
    input_path: Path,
    schema_path: Path,
    model: str,
    max_workers: int,
    theme_batches_count: int,
    recompute_all_llm: bool,
    checkpoint_file: Path,
    checkpoint_metadata_file: Path,
    no_resume: bool,
) -> tuple[int, dict[str, int]]:
    themes = _load_theme_schema(schema_path)
    theme_columns = list(themes.keys())

    row_mask = _row_mask_for_update(reports, theme_columns, recompute_all=recompute_all_llm)
    rows_to_update = int(row_mask.sum())

    if rows_to_update == 0:
        return 0, {column: int(_coerce_boolean_series(reports[column]).sum()) for column in theme_columns if column in reports.columns}

    subset_source_indices = reports.index[row_mask]
    subset = reports.loc[row_mask].copy().reset_index(drop=True)
    checkpoint_metadata = _metadata_signature(
        input_path=input_path,
        schema_path=schema_path,
        theme_columns=theme_columns,
        source_indices=subset_source_indices,
    )

    api_key = _load_openai_key()
    llm_client = LLM(
        api_key=api_key,
        model=model,
        max_workers=max(1, max_workers),
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
    theme_batches = _chunk_theme_map(themes, theme_batches_count)
    assigned_subset = subset.copy()

    if no_resume:
        if checkpoint_file.exists():
            print(f"Ignoring checkpoint due to --no-resume: {checkpoint_file}")
    else:
        checkpoint = _load_checkpoint_if_compatible(
            checkpoint_file=checkpoint_file,
            metadata_file=checkpoint_metadata_file,
            metadata=checkpoint_metadata,
        )
        if checkpoint is None and checkpoint_file.exists():
            print("Existing checkpoint is incompatible with current run; recomputing from scratch.")
        if checkpoint is not None:
            print(f"Resuming from checkpoint: {checkpoint_file}")
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
                checkpoint_file=checkpoint_file,
                metadata_file=checkpoint_metadata_file,
                source_indices=subset_source_indices,
                assigned_subset=assigned_subset,
                theme_columns=theme_columns,
                metadata=checkpoint_metadata,
            )
            print(f"Checkpoint saved: {checkpoint_file}")
    except (KeyboardInterrupt, Exception):
        _save_checkpoint(
            checkpoint_file=checkpoint_file,
            metadata_file=checkpoint_metadata_file,
            source_indices=subset_source_indices,
            assigned_subset=assigned_subset,
            theme_columns=theme_columns,
            metadata=checkpoint_metadata,
        )
        print(f"Interrupted; partial checkpoint saved to {checkpoint_file}")
        raise

    counts: dict[str, int] = {}
    for column in theme_columns:
        assigned_subset[column] = _coerce_boolean_series(assigned_subset[column])
        reports.loc[row_mask, column] = assigned_subset[column].values
        reports[column] = _coerce_boolean_series(reports[column])
        counts[column] = int(reports[column].sum())

    return rows_to_update, counts


def main() -> None:
    args = _parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"{args.input} not found.")

    recompute_all_rules = bool(args.recompute_all or args.recompute_all_rules)
    recompute_all_llm = bool(args.recompute_all or args.recompute_all_llm)

    reports = pd.read_csv(args.input)

    any_updates = False

    if args.skip_rule_based:
        rule_rows_updated = 0
        rule_counts = {column: int(_coerce_boolean_series(reports[column]).sum()) for column in COLLECTION_COLUMNS.values() if column in reports.columns}
    else:
        rule_rows_updated, rule_counts = _run_rule_updates(
            reports,
            recompute_all_rules=recompute_all_rules,
        )
        any_updates = any_updates or rule_rows_updated > 0

    if args.skip_llm:
        llm_rows_updated = 0
        llm_counts: dict[str, int] = {}
    else:
        llm_rows_updated, llm_counts = _run_llm_updates(
            reports,
            input_path=args.input,
            schema_path=args.schema,
            model=args.model,
            max_workers=args.max_workers,
            theme_batches_count=args.theme_batches,
            recompute_all_llm=recompute_all_llm,
            checkpoint_file=args.checkpoint_file,
            checkpoint_metadata_file=args.checkpoint_metadata_file,
            no_resume=args.no_resume,
        )
        any_updates = any_updates or llm_rows_updated > 0

    if any_updates:
        reports.to_csv(args.input, index=False)

    if args.skip_rule_based:
        print("Rule-based collection updates skipped.")
    elif rule_rows_updated == 0:
        print("No rule-based collection updates required.")
    else:
        print(f"Rule-based collection columns updated for {rule_rows_updated} row(s).")
    for column in COLLECTION_COLUMNS.values():
        if column in rule_counts:
            print(f"{column}: {rule_counts[column]}")

    if args.skip_llm:
        print("LLM theme collection updates skipped.")
    elif llm_rows_updated == 0:
        print("No LLM theme collection updates required.")
    else:
        print(f"LLM theme collections updated for {llm_rows_updated} row(s).")
    for column, count in llm_counts.items():
        print(f"{column}: {count}")

    if not any_updates:
        _append_summary(
            args.summary_file,
            "Collection update already up to date (no rows updated).",
        )
        return

    _append_summary(
        args.summary_file,
        (
            f"Rule-based collection columns updated for {rule_rows_updated} row(s); "
            f"LLM theme collections updated for {llm_rows_updated} row(s)."
        ),
    )


if __name__ == "__main__":
    main()
