#!/usr/bin/env python3
"""Interactively discover and curate theme collections for PFD reports."""

from __future__ import annotations

import argparse
import ast
import json
import os
import re
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

from pfd_toolkit import Extractor, LLM


DEFAULT_SEED_TOPICS = [
    "Suicide-related deaths",
    "Deaths in hospital settings",
    "Medication purchased online",
    "Medication-related deaths",
    "Deaths in care home settings",
]


def _load_local_openai_key() -> str:
    project_root = Path(__file__).resolve().parents[1]
    candidate_paths = [
        project_root / "api.env",
        project_root / "src" / "pfd_toolkit" / "api.env",
        project_root / "OPEN_API_KEY",
    ]

    for key_path in candidate_paths:
        if not key_path.exists():
            continue
        raw = key_path.read_text(encoding="utf-8").strip()
        if not raw:
            continue

        for line in raw.splitlines():
            token = line.strip()
            if not token or token.startswith("#"):
                continue
            if "=" in token:
                left, right = token.split("=", 1)
                if left.strip() in {"OPEN_API_KEY", "OPENAI_API_KEY"}:
                    value = right.strip().strip('"').strip("'")
                    if value:
                        return value
            else:
                return token

    env_value = (os.getenv("OPENAI_API_KEY") or os.getenv("OPEN_API_KEY") or "").strip()
    if env_value:
        return env_value

    raise FileNotFoundError(
        "OpenAI key not found. Expected one of: "
        "api.env, src/pfd_toolkit/api.env, OPEN_API_KEY, or OPENAI_API_KEY env var."
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Discover candidate themes from a sample, then curate them interactively."
    )
    parser.add_argument("--input", type=Path, default=Path("all_reports.csv"))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("scripts/theme_collections/approved_themes.json"),
        help="Where to save the approved theme schema JSON.",
    )
    parser.add_argument("--sample-size", type=int, default=3000)
    parser.add_argument("--sample-seed", type=int, default=67)
    parser.add_argument("--min-themes", type=int, default=20)
    parser.add_argument("--max-themes", type=int, default=30)
    parser.add_argument("--model", default="gpt-4.1")
    parser.add_argument("--max-workers", type=int, default=18)
    parser.add_argument("--summarise-intensity", default="high")
    parser.add_argument(
        "--extra-instructions",
        default=(
            "Prioritise recurring, actionable prevention topics in coroners' concerns. "
            "Avoid creating near-duplicate themes."
        ),
    )
    return parser.parse_args()


def _normalise_identifier(text: str) -> str:
    cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", (text or "").strip().lower()).strip("_")
    if not cleaned:
        cleaned = "theme"
    if cleaned[0].isdigit():
        cleaned = f"theme_{cleaned}"
    if not cleaned.startswith("theme_"):
        cleaned = f"theme_{cleaned}"
    return cleaned


def _ensure_unique_theme_name(name: str, existing: set[str]) -> str:
    candidate = _normalise_identifier(name)
    if candidate not in existing:
        return candidate
    suffix = 2
    while f"{candidate}_{suffix}" in existing:
        suffix += 1
    return f"{candidate}_{suffix}"


def _stratified_sample(df: pd.DataFrame, sample_size: int, seed: int) -> pd.DataFrame:
    if sample_size <= 0 or sample_size >= len(df):
        return df.copy().reset_index(drop=True)

    if "date" not in df.columns:
        return df.sample(n=sample_size, random_state=seed).reset_index(drop=True)

    dated = df.copy()
    dated["__sample_year"] = pd.to_datetime(dated["date"], errors="coerce").dt.year.fillna(-1).astype(int)
    groups = dated.groupby("__sample_year", dropna=False)

    total = len(dated)
    allocations: list[tuple[pd.DataFrame, int]] = []
    for _, group in groups:
        raw = sample_size * (len(group) / total)
        take = max(1, int(round(raw)))
        allocations.append((group, min(len(group), take)))

    assigned = sum(take for _, take in allocations)
    while assigned > sample_size:
        for idx, (group, take) in enumerate(allocations):
            if assigned <= sample_size:
                break
            if take > 1:
                allocations[idx] = (group, take - 1)
                assigned -= 1

    selected_indices: list[int] = []
    rng_offset = 0
    for group, take in allocations:
        if take <= 0:
            continue
        sampled = group.sample(n=take, random_state=seed + rng_offset)
        selected_indices.extend(sampled.index.tolist())
        rng_offset += 1

    if len(selected_indices) < sample_size:
        remaining = sample_size - len(selected_indices)
        remaining_pool = dated.loc[~dated.index.isin(selected_indices)]
        if not remaining_pool.empty:
            top_up = remaining_pool.sample(n=min(remaining, len(remaining_pool)), random_state=seed + 1000)
            selected_indices.extend(top_up.index.tolist())

    sample = dated.loc[selected_indices].drop(columns=["__sample_year"], errors="ignore")
    if len(sample) > sample_size:
        sample = sample.sample(n=sample_size, random_state=seed + 2000)
    return sample.reset_index(drop=True)


def _discover_themes(args: argparse.Namespace, reports_sample: pd.DataFrame) -> dict[str, str]:
    api_key = _load_local_openai_key()

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
        reports=reports_sample,
        include_date=True,
        include_coroner=True,
        include_area=True,
        include_receiver=True,
        include_investigation=True,
        include_circumstances=True,
        include_concerns=True,
        verbose=False,
    )
    theme_model = extractor.discover_themes(
        min_themes=args.min_themes,
        max_themes=args.max_themes,
        extra_instructions=args.extra_instructions,
        seed_topics=DEFAULT_SEED_TOPICS,
        trim_approach="summarise",
        summarise_intensity=args.summarise_intensity,
    )

    if theme_model is not None and hasattr(theme_model, "model_json_schema"):
        schema = theme_model.model_json_schema()
        properties = schema.get("properties", {})
        discovered: dict[str, str] = {}
        for name, meta in properties.items():
            if not str(name).strip():
                continue
            description = str((meta or {}).get("description") or "").strip()
            discovered[str(name).strip()] = description
        return discovered

    recovered = _recover_themes_from_raw_output(getattr(extractor, "identified_themes", None))
    if recovered:
        print("Recovered themes from non-strict JSON output.")
        return recovered
    raise ValueError("Theme discovery returned unparsable output.")


def _recover_themes_from_raw_output(raw_output: Any) -> dict[str, str]:
    if isinstance(raw_output, dict):
        return {str(k).strip(): str(v or "").strip() for k, v in raw_output.items() if str(k).strip()}
    if raw_output is None:
        return {}

    text = str(raw_output or "").strip()
    if not text:
        return {}

    fenced = re.match(r"^```(?:json)?\s*(?P<body>[\s\S]*?)\s*```$", text, flags=re.IGNORECASE)
    if fenced:
        text = fenced.group("body").strip()

    def _as_theme_map(value: Any) -> dict[str, str]:
        if not isinstance(value, dict):
            return {}
        return {
            str(key).strip(): str(desc or "").strip()
            for key, desc in value.items()
            if str(key).strip()
        }

    try:
        parsed = json.loads(text)
        mapped = _as_theme_map(parsed)
        if mapped:
            return mapped
    except Exception:
        pass

    repaired = re.sub(
        r'([{\s,])([A-Za-z_][A-Za-z0-9_]*)\s*:',
        r'\1"\2":',
        text,
    )
    repaired = re.sub(r",\s*}", "}", repaired)
    repaired = re.sub(r",\s*]", "]", repaired)

    try:
        parsed = json.loads(repaired)
        mapped = _as_theme_map(parsed)
        if mapped:
            return mapped
    except Exception:
        pass

    try:
        parsed = ast.literal_eval(repaired)
        mapped = _as_theme_map(parsed)
        if mapped:
            return mapped
    except Exception:
        pass

    return {}


def _print_themes(themes: list[dict[str, str]]) -> None:
    print("\nCandidate themes:")
    for idx, item in enumerate(themes, start=1):
        print(f"{idx:>2}. {item['name']} -> {item['description']}")


def _parse_drop_input(raw: str, max_index: int) -> list[int]:
    selected: set[int] = set()
    for chunk in (raw or "").split(","):
        token = chunk.strip()
        if not token:
            continue
        if "-" in token:
            left, right = token.split("-", 1)
            if not left.strip().isdigit() or not right.strip().isdigit():
                continue
            start = int(left.strip())
            end = int(right.strip())
            if start > end:
                start, end = end, start
            for value in range(start, end + 1):
                if 1 <= value <= max_index:
                    selected.add(value)
            continue
        if token.isdigit():
            value = int(token)
            if 1 <= value <= max_index:
                selected.add(value)
    return sorted(selected)


def _interactive_curation(discovered: dict[str, str]) -> list[dict[str, str]]:
    themes = [
        {"name": str(name).strip(), "description": str(description or "").strip()}
        for name, description in discovered.items()
        if str(name).strip()
    ]
    themes.sort(key=lambda item: item["name"])

    while True:
        _print_themes(themes)
        raw = input(
            "\nEnter theme numbers to remove (e.g. '2,5-7'), or press Enter to keep all: "
        ).strip()
        if not raw:
            break
        to_drop = set(_parse_drop_input(raw, len(themes)))
        if not to_drop:
            print("No valid indices found. Try again.")
            continue
        themes = [item for idx, item in enumerate(themes, start=1) if idx not in to_drop]
        print(f"Removed {len(to_drop)} theme(s).")

    while True:
        add_more = input("\nAdd a manual theme? [y/N]: ").strip().lower()
        if add_more not in {"y", "yes"}:
            break
        name = input("Theme name: ").strip()
        if not name:
            print("Skipped: empty theme name.")
            continue
        description = input("Theme description: ").strip()
        themes.append({"name": name, "description": description})

    _print_themes(themes)
    confirm = input("\nSave this curated list? [Y/n]: ").strip().lower()
    if confirm in {"n", "no"}:
        raise SystemExit("Aborted; no file written.")
    return themes


def _save_approved_themes(
    output_path: Path,
    themes: list[dict[str, str]],
    *,
    args: argparse.Namespace,
    input_rows: int,
    sample_rows: int,
) -> None:
    approved: dict[str, str] = {}
    used_names: set[str] = set()
    for item in themes:
        raw_name = item.get("name", "")
        description = item.get("description", "")
        unique_name = _ensure_unique_theme_name(raw_name, used_names)
        used_names.add(unique_name)
        approved[unique_name] = str(description or "").strip()

    payload: dict[str, Any] = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "input_path": str(args.input),
        "input_rows": int(input_rows),
        "sample_rows": int(sample_rows),
        "sample_seed": int(args.sample_seed),
        "model": args.model,
        "trim_approach": "summarise",
        "summarise_intensity": args.summarise_intensity,
        "min_themes": int(args.min_themes),
        "max_themes": int(args.max_themes),
        "seed_topics": DEFAULT_SEED_TOPICS,
        "extra_instructions": args.extra_instructions,
        "themes": approved,
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True) + "\n", encoding="utf-8")
    print(f"\nSaved {len(approved)} approved themes to: {output_path}")


def main() -> None:
    args = _parse_args()
    if not args.input.exists():
        raise FileNotFoundError(f"{args.input} not found.")

    reports = pd.read_csv(args.input)
    if reports.empty:
        raise ValueError("Input dataset is empty.")

    sample = _stratified_sample(reports, sample_size=args.sample_size, seed=args.sample_seed)
    print(f"Loaded {len(reports)} reports. Running discovery on sample of {len(sample)}.")
    print("Seed topics:")
    for topic in DEFAULT_SEED_TOPICS:
        print(f" - {topic}")

    discovered = _discover_themes(args, sample)
    curated = _interactive_curation(discovered)
    _save_approved_themes(
        args.output,
        curated,
        args=args,
        input_rows=len(reports),
        sample_rows=len(sample),
    )


if __name__ == "__main__":
    main()
