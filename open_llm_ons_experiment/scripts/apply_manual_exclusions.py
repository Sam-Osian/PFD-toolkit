#!/usr/bin/env python3
"""
Apply manual exclusions to discovery outputs.

Reads:
- eligible_tags_latest.csv
- excluded_tags_latest.csv
- manual_exclusions.csv

Writes:
- eligible_tags_latest.csv (updated)
- excluded_tags_latest.csv (updated)
- exclusion_reason_counts_latest.csv (recomputed)
"""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


@dataclass
class ManualExclusion:
    family: str
    tag: str
    stage: str
    reason_code: str
    rationale: str

    def matches(self, row: pd.Series) -> bool:
        row_family = str(row.get("family", ""))
        row_tag = str(row.get("tag", ""))
        family = self.family.strip()
        tag = self.tag.strip()

        if family and tag:
            return row_family == family and row_tag == tag
        if family:
            return row_family == family
        if tag:
            return row_tag == tag
        return False


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--eligible",
        default="open_llm_ons_experiment/artifacts/discovery/eligible_tags_latest.csv",
    )
    p.add_argument(
        "--excluded",
        default="open_llm_ons_experiment/artifacts/discovery/excluded_tags_latest.csv",
    )
    p.add_argument(
        "--manual-exclusions",
        default="open_llm_ons_experiment/config/manual_exclusions.csv",
    )
    p.add_argument(
        "--counts-out",
        default="open_llm_ons_experiment/artifacts/discovery/exclusion_reason_counts_latest.csv",
    )
    return p.parse_args()


def load_manual_exclusions(path: Path) -> list[ManualExclusion]:
    rows: list[ManualExclusion] = []
    if not path.exists():
        return rows
    with path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for r in reader:
            family = (r.get("family") or "").strip()
            tag = (r.get("tag") or "").strip()
            stage = (r.get("stage") or "manual_exclusion").strip()
            reason_code = (r.get("reason_code") or "").strip()
            rationale = (r.get("rationale") or "").strip()
            if (not family and not tag) or family.startswith("#"):
                continue
            if stage not in {"preprocessing", "manual_exclusion"}:
                raise ValueError(
                    f"Invalid stage '{stage}' for manual exclusion row family='{family}' tag='{tag}'. "
                    "Stage must be 'preprocessing' or 'manual_exclusion'."
                )
            rows.append(
                ManualExclusion(
                    family=family,
                    tag=tag,
                    stage=stage,
                    reason_code=reason_code,
                    rationale=rationale,
                )
            )
    return rows


def append_reason(existing: str, new_reason: str) -> str:
    if existing is None or (isinstance(existing, float) and pd.isna(existing)) or str(existing).strip().lower() == "nan":
        existing_text = ""
    else:
        existing_text = str(existing)
    parts = [p.strip() for p in existing_text.split(";") if p.strip() and p.strip().lower() != "nan"]
    if new_reason not in parts:
        parts.append(new_reason)
    return ";".join(parts)


def _reason_stage(reason: str) -> str:
    preprocessing = {"latest_alias_tag", "unknown_update_date"}
    if reason.startswith("manual_exclusion:"):
        return "manual_exclusion"
    if reason in preprocessing:
        return "preprocessing"
    return "eligibility"


def build_exclusion_reason_counts(excluded_df: pd.DataFrame) -> pd.DataFrame:
    counts: dict[tuple[str, str], int] = {}
    for reasons in excluded_df.get("exclusion_reasons", pd.Series(dtype=str)).fillna(""):
        for reason in [p.strip() for p in str(reasons).split(";") if p.strip()]:
            if reason.lower() == "nan":
                continue
            stage = _reason_stage(reason)
            code = reason
            if reason.startswith("manual_exclusion:"):
                code = reason.split(":", 1)[1].strip() or "unspecified_manual_reason"
            key = (code, stage)
            counts[key] = counts.get(key, 0) + 1
    if not counts:
        return pd.DataFrame(columns=["reason_code", "stage", "excluded_tag_count"])
    out = pd.DataFrame(
        {
            "reason_code": [k[0] for k in counts.keys()],
            "stage": [k[1] for k in counts.keys()],
            "excluded_tag_count": list(counts.values()),
        }
    )
    return out.sort_values(
        by=["stage", "excluded_tag_count", "reason_code"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    eligible_path = Path(args.eligible)
    excluded_path = Path(args.excluded)
    manual_path = Path(args.manual_exclusions)
    counts_out_path = Path(args.counts_out)

    eligible = pd.read_csv(eligible_path)
    excluded = pd.read_csv(excluded_path)
    manual = load_manual_exclusions(manual_path)

    moved_rows: list[pd.Series] = []
    keep_mask: list[bool] = []

    for _, row in eligible.iterrows():
        matched = None
        for ex in manual:
            if ex.matches(row):
                matched = ex
                break

        if matched is None:
            keep_mask.append(True)
            continue

        row = row.copy()
        row["eligible"] = False
        manual_code = matched.reason_code or "unspecified_manual_reason"
        if matched.stage == "manual_exclusion":
            reason_token = f"manual_exclusion:{manual_code}"
        else:
            reason_token = manual_code
        row["exclusion_reasons"] = append_reason(row.get("exclusion_reasons", ""), reason_token)
        if "manual_rationale" in row.index:
            row["manual_rationale"] = matched.rationale
        moved_rows.append(row)
        keep_mask.append(False)

    eligible_updated = eligible[pd.Series(keep_mask, index=eligible.index)].copy()

    if moved_rows:
        moved_df = pd.DataFrame(moved_rows)
        excluded_updated = pd.concat([excluded, moved_df], ignore_index=True)
        excluded_updated = excluded_updated.drop_duplicates(subset=["tag"], keep="last").reset_index(drop=True)
    else:
        excluded_updated = excluded.copy()

    counts = build_exclusion_reason_counts(excluded_updated)

    eligible_updated.to_csv(eligible_path, index=False)
    excluded_updated.to_csv(excluded_path, index=False)
    counts.to_csv(counts_out_path, index=False)

    print(f"Manual exclusions loaded: {len(manual)}")
    print(f"Moved from eligible to excluded: {len(moved_rows)}")
    print(f"Wrote: {eligible_path}")
    print(f"Wrote: {excluded_path}")
    print(f"Wrote: {counts_out_path}")


if __name__ == "__main__":
    main()
