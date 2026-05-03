#!/usr/bin/env python3
"""
Discover and filter eligible Ollama library models for the open LLM experiment.

This script:
1) Crawls family pages from https://ollama.com/library
2) Expands to tag-level variants
3) Applies protocol filters
4) Applies manual exclusions from a CSV template
5) Writes discovered / eligible / excluded CSV outputs
"""

from __future__ import annotations

import argparse
import csv
import re
from dataclasses import dataclass
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Iterable
from urllib.parse import quote, unquote

import pandas as pd
import requests
from bs4 import BeautifulSoup


LIBRARY_URL = "https://ollama.com/library"

# Protocol defaults
DEFAULT_START_DATE = "2024-05-01"
DEFAULT_END_DATE = "2026-05-01"
DEFAULT_MIN_PARAMS_B = 5.0
DEFAULT_DENSE_MAX_PARAMS_B = 120.0
DEFAULT_MOE_MAX_ACTIVE_PARAMS_B = 40.0
DEFAULT_MOE_MAX_TOTAL_PARAMS_B = 400.0
DEFAULT_MAX_MODEL_SIZE_GB = 100.0

# Family/tag keyword heuristics for specialised model exclusions.
# Keep this conservative to reduce false positives.
SPECIALISED_PATTERNS = [
    re.compile(p, flags=re.IGNORECASE)
    for p in [
        r"\bcoder\b",
        r"\bcodeqwen\b",
        r"\bcodellama\b",
        r"\bcodestral\b",
        r"\bcodegemma\b",
        r"\bdeepseek-coder\b",
        r"\bgranite-code\b",
        r"\bstarcoder\b",
        r"\bwizardcoder\b",
        r"\byi-coder\b",
        r"\bdolphincoder\b",
        r"\bsqlcoder\b",
        r"\bqwen2-math\b",
        r"\bwizard-math\b",
        r"\bllama-guard\b",
        r"\bembedding\b",
        r"\bembed\b",
        r"\btranslategemma\b",
    ]
]

# VLM-specialised families commonly marketed as vision-first endpoints.
VISION_SPECIALISED_FAMILIES = {
    "llava",
    "llava-llama3",
    "minicpm-v",
    "moondream",
    "qwen2.5vl",
    "qwen3-vl",
    "llama3.2-vision",
    "granite3.2-vision",
}


@dataclass
class ManualExclusion:
    family: str
    tag: str
    stage: str
    reason_code: str
    rationale: str

    def matches(self, row: pd.Series) -> bool:
        row_tag = str(row.get("tag", ""))
        row_family = str(row.get("family", ""))
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
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--start-date", default=DEFAULT_START_DATE, help="YYYY-MM-DD")
    parser.add_argument("--end-date", default=DEFAULT_END_DATE, help="YYYY-MM-DD")
    parser.add_argument("--min-params-b", type=float, default=DEFAULT_MIN_PARAMS_B)
    parser.add_argument("--dense-max-params-b", type=float, default=DEFAULT_DENSE_MAX_PARAMS_B)
    parser.add_argument("--moe-max-active-params-b", type=float, default=DEFAULT_MOE_MAX_ACTIVE_PARAMS_B)
    parser.add_argument("--moe-max-total-params-b", type=float, default=DEFAULT_MOE_MAX_TOTAL_PARAMS_B)
    parser.add_argument("--max-model-size-gb", type=float, default=DEFAULT_MAX_MODEL_SIZE_GB)
    parser.add_argument(
        "--manual-exclusions",
        default="open_llm_ons_experiment/config/manual_exclusions.csv",
        help="CSV file of manual exclusions.",
    )
    parser.add_argument(
        "--out-dir",
        default="open_llm_ons_experiment/artifacts/discovery",
        help="Directory for CSV outputs.",
    )
    parser.add_argument("--timeout", type=int, default=30)
    return parser.parse_args()


def _http_get_text(url: str, timeout: int) -> str:
    r = requests.get(url, timeout=timeout)
    r.raise_for_status()
    return r.text


def _clean_text(text: str) -> str:
    return " ".join((text or "").split())


def _parse_relative_age_to_date(relative: str, reference_date: date) -> date | None:
    if not relative:
        return None
    m = re.search(r"(\d+)\s+(day|days|week|weeks|month|months|year|years)\s+ago", relative)
    if not m:
        return None
    value = int(m.group(1))
    unit = m.group(2)
    if "day" in unit:
        delta = timedelta(days=value)
    elif "week" in unit:
        delta = timedelta(weeks=value)
    elif "month" in unit:
        delta = timedelta(days=value * 30)
    else:
        delta = timedelta(days=value * 365)
    return reference_date - delta


def _parse_size_gb(text: str) -> float | None:
    m = re.search(r"(\d+(?:\.\d+)?)\s*(KB|MB|GB|TB)\b", text, flags=re.IGNORECASE)
    if not m:
        return None
    value = float(m.group(1))
    unit = m.group(2).upper()
    if unit == "KB":
        return value / (1024 * 1024)
    if unit == "MB":
        return value / 1024
    if unit == "GB":
        return value
    if unit == "TB":
        return value * 1024
    return None


def _parse_params_b_from_tag(tag: str) -> float | None:
    t = (tag or "").lower()
    tag_part = t.split(":", 1)[1] if ":" in t else t

    # 235b-a22b
    m = re.search(r"(\d+(?:\.\d+)?)b[-_]?a(\d+(?:\.\d+)?)b", tag_part)
    if m:
        return float(m.group(1))

    # 8x22b
    m = re.search(r"(\d+)x(\d+(?:\.\d+)?)b", tag_part)
    if m:
        return float(m.group(1)) * float(m.group(2))

    # 72b / 270m / 1.7b (normalise to billions)
    m = re.search(r"(\d+(?:\.\d+)?)([mb])(?:[^a-z0-9]|$)", tag_part)
    if m:
        value = float(m.group(1))
        unit = m.group(2)
        return value / 1000.0 if unit == "m" else value

    return None


def _infer_moe_fields(tag: str, detail_blob: str, total_params_b: float | None) -> tuple[bool, float | None, float | None]:
    # Keep parsing tag-scoped to prevent family-level prose from contaminating
    # per-tag parameter assignments.
    tag_l = (tag or "").lower()
    text = tag_l

    is_moe = False
    active_params_b = None
    total_params_b_out = total_params_b

    if any(x in text for x in ["moe", "mixtral", "a22b", "a3b", "a16b"]):
        is_moe = True

    m = re.search(r"(\d+(?:\.\d+)?)b[-_]?a(\d+(?:\.\d+)?)b", text)
    if m:
        total_params_b_out = float(m.group(1))
        active_params_b = float(m.group(2))
        is_moe = True

    m = re.search(r"(\d+)x(\d+(?:\.\d+)?)b", text)
    if m:
        experts = float(m.group(1))
        expert_size = float(m.group(2))
        if total_params_b_out is None:
            total_params_b_out = experts * expert_size
        is_moe = True

    return is_moe, active_params_b, total_params_b_out


def _is_specialised(row: pd.Series) -> bool:
    # Restrict matching to stable identifiers only.
    corpus = " ".join([str(row.get("tag", "")), str(row.get("family", ""))]).lower()
    return any(p.search(corpus) for p in SPECIALISED_PATTERNS)


def _is_vision_specialised(row: pd.Series) -> bool:
    family = str(row.get("family", "")).lower()
    return family in VISION_SPECIALISED_FAMILIES


def _load_manual_exclusions(path: Path) -> list[ManualExclusion]:
    if not path.exists():
        return []
    rows: list[ManualExclusion] = []
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


def discover_family_pages(timeout: int) -> list[dict]:
    html = _http_get_text(LIBRARY_URL, timeout=timeout)
    soup = BeautifulSoup(html, "html.parser")

    seen: set[str] = set()
    families: list[dict] = []
    for a in soup.select("a[href]"):
        href = a.get("href", "")
        if not href.startswith("/library/"):
            continue
        slug = href.split("/library/", 1)[1]
        if ":" in slug or "/" in slug or not slug:
            continue
        if slug in seen:
            continue
        seen.add(slug)
        families.append(
            {
                "family": slug,
                "family_url": f"https://ollama.com{href}",
                "family_card_text": _clean_text(a.get_text(" ", strip=True)),
            }
        )
    return families


def discover_tags_for_family(family: str, timeout: int) -> list[dict]:
    family_url = f"https://ollama.com/library/{family}"
    html = _http_get_text(family_url, timeout=timeout)
    soup = BeautifulSoup(html, "html.parser")

    by_tag: dict[str, str] = {}
    for a in soup.select("a[href]"):
        href = a.get("href", "")
        prefix = f"/library/{family}:"
        if not href.startswith(prefix):
            continue
        tag = unquote(href.split("/library/", 1)[1])
        text = _clean_text(a.get_text(" ", strip=True))
        old = by_tag.get(tag, "")
        if text.count("·") > old.count("·"):
            by_tag[tag] = text
        elif tag not in by_tag:
            by_tag[tag] = text

    rows: list[dict] = []
    for tag, row_text in sorted(by_tag.items()):
        segments = [s.strip() for s in row_text.split("·")] if row_text else []
        modality = None
        updated_relative = None
        is_cloud_tag = False
        if segments:
            for seg in segments:
                sl = seg.lower()
                if sl in {"text", "vision", "embedding", "audio"}:
                    modality = seg
                if sl == "cloud":
                    is_cloud_tag = True
                if sl.endswith("ago"):
                    updated_relative = seg

        rows.append(
            {
                "family": family,
                "tag": tag,
                "tag_url": f"https://ollama.com/library/{quote(tag, safe='')}",
                "row_text": row_text,
                "size_gb": _parse_size_gb(row_text),
                "modality": modality,
                "is_cloud_tag": is_cloud_tag,
                "updated_relative": updated_relative,
            }
        )
    return rows


def enrich_tag_details(df: pd.DataFrame, timeout: int) -> pd.DataFrame:
    out = df.copy()
    params_from_tag: list[float | None] = []
    quants: list[str | None] = []
    detail_blobs: list[str] = []

    for _, row in out.iterrows():
        url = row["tag_url"]
        try:
            html = _http_get_text(url, timeout=timeout)
            text = _clean_text(BeautifulSoup(html, "html.parser").get_text(" ", strip=True))
        except Exception:
            text = ""
        detail_blobs.append(text)
        params_from_tag.append(_parse_params_b_from_tag(str(row.get("tag", ""))))
        m_quant = re.search(r"\bQ\d(?:_[A-Z0-9]+)?\b|\bFP8\b|\bFP16\b", text, flags=re.IGNORECASE)
        quants.append(m_quant.group(0).upper() if m_quant else None)

    out["detail_text"] = detail_blobs
    out["params_total_b_raw"] = params_from_tag
    out["quantisation"] = quants
    return out


def apply_filters(
    df: pd.DataFrame,
    start_date: date,
    end_date: date,
    min_params_b: float,
    dense_max_params_b: float,
    moe_max_active_params_b: float,
    moe_max_total_params_b: float,
    max_model_size_gb: float,
    manual_exclusions: Iterable[ManualExclusion],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    now = date.today()
    work = df.copy()

    work["updated_date_approx"] = work["updated_relative"].apply(
        lambda x: _parse_relative_age_to_date(str(x or ""), reference_date=now)
    )

    inferred = work.apply(
        lambda r: _infer_moe_fields(
            tag=str(r["tag"]),
            detail_blob=str(r.get("detail_text", "")),
            total_params_b=r.get("params_total_b_raw"),
        ),
        axis=1,
    )
    work["is_moe"] = inferred.apply(lambda t: t[0])
    work["params_active_b"] = inferred.apply(lambda t: t[1])
    work["params_total_b"] = inferred.apply(lambda t: t[2])

    # Defensive fallback from tag if needed.
    work["params_total_b"] = work.apply(
        lambda r: r["params_total_b"] if pd.notna(r["params_total_b"]) else _parse_params_b_from_tag(str(r["tag"])),
        axis=1,
    )

    reasons: list[list[str]] = []
    keep: list[bool] = []
    manual_reason: list[str] = []
    manual_rationale: list[str] = []

    for _, row in work.iterrows():
        row_reasons: list[str] = []

        tag = str(row["tag"])
        tag_l = tag.lower()
        family_text = str(row.get("family_card_text", "")).lower()

        # Pre-processing exclusions
        if tag_l.endswith(":latest") or tag_l == "latest":
            row_reasons.append("latest_alias_tag")

        if "cloud" in tag_l or bool(row.get("is_cloud_tag", False)):
            row_reasons.append("cloud_model_excluded")

        # Eligibility exclusions
        if str(row.get("modality", "")).lower() == "embedding" or "embedding" in family_text:
            row_reasons.append("embedding_model_excluded")

        if _is_specialised(row):
            row_reasons.append("specialised_model_excluded")

        if _is_vision_specialised(row):
            row_reasons.append("vision_specialised_model_excluded")

        updated_date = row.get("updated_date_approx")
        if isinstance(updated_date, date):
            if updated_date < start_date or updated_date > end_date:
                row_reasons.append("outside_date_window")
        else:
            row_reasons.append("unknown_update_date")

        size_gb = row.get("size_gb")
        if pd.notna(size_gb) and float(size_gb) > max_model_size_gb:
            row_reasons.append("exceeds_max_model_size_gb")

        params_total = row.get("params_total_b")
        params_active = row.get("params_active_b")
        is_moe = bool(row.get("is_moe"))

        if pd.notna(params_total):
            if is_moe:
                # Do not apply dense min floor to MoE total parameters.
                if float(params_total) > moe_max_total_params_b:
                    row_reasons.append("exceeds_moe_max_total_params_b")
            else:
                if float(params_total) < min_params_b:
                    row_reasons.append("below_min_params_b")
                if float(params_total) > dense_max_params_b:
                    row_reasons.append("exceeds_dense_max_params_b")

        # Fix 2: do NOT apply min floor to MoE active params. Keep max-active gate.
        if is_moe and pd.notna(params_active):
            if float(params_active) > moe_max_active_params_b:
                row_reasons.append("exceeds_moe_max_active_params_b")

        matched_manual_reason = ""
        matched_manual_rationale = ""
        for ex in manual_exclusions:
            if ex.matches(row):
                manual_code = ex.reason_code or "unspecified_manual_reason"
                if ex.stage == "manual_exclusion":
                    reason_token = f"manual_exclusion:{manual_code}"
                else:
                    reason_token = manual_code
                row_reasons.append(reason_token)
                matched_manual_reason = reason_token
                matched_manual_rationale = ex.rationale
                break

        # Unknown update-date metadata is informational only.
        blocking_reasons = [r for r in row_reasons if r != "unknown_update_date"]
        keep.append(len(blocking_reasons) == 0)
        reasons.append(row_reasons)
        manual_reason.append(matched_manual_reason)
        manual_rationale.append(matched_manual_rationale)

    work["exclusion_reasons"] = [";".join(r) for r in reasons]
    work["manual_rationale"] = manual_rationale
    work["eligible"] = keep

    excluded = work[~work["eligible"]].copy()
    eligible = work[work["eligible"]].copy()
    return eligible, excluded


def _classify_reason(reason_code: str) -> str:
    if reason_code in {"latest_alias_tag", "unknown_update_date"}:
        return "preprocessing"
    if reason_code.startswith("manual_exclusion:"):
        return "manual"
    return "eligibility"


def build_exclusion_reason_counts(excluded_df: pd.DataFrame) -> pd.DataFrame:
    reason_counts: dict[tuple[str, str], int] = {}
    for reasons in excluded_df.get("exclusion_reasons", pd.Series(dtype=str)).fillna(""):
        for part in [p.strip() for p in str(reasons).split(";") if p.strip()]:
            if part.lower() == "nan":
                continue
            reason_group = _classify_reason(part)
            reason_key = part
            if part.startswith("manual_exclusion:"):
                reason_key = part.split(":", 1)[1].strip() or "unspecified_manual_reason"
            key = (reason_key, reason_group)
            reason_counts[key] = reason_counts.get(key, 0) + 1

    if not reason_counts:
        return pd.DataFrame(columns=["reason_code", "stage", "excluded_tag_count"])

    out = pd.DataFrame(
        {
            "reason_code": [k[0] for k in reason_counts.keys()],
            "stage": [k[1] for k in reason_counts.keys()],
            "excluded_tag_count": list(reason_counts.values()),
        }
    )
    return out[["reason_code", "stage", "excluded_tag_count"]].sort_values(
        by=["stage", "excluded_tag_count", "reason_code"],
        ascending=[True, False, True],
    ).reset_index(drop=True)


def main() -> None:
    args = parse_args()
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    start_date = datetime.strptime(args.start_date, "%Y-%m-%d").date()
    end_date = datetime.strptime(args.end_date, "%Y-%m-%d").date()

    families = discover_family_pages(timeout=args.timeout)
    families_df = pd.DataFrame(families)

    tag_rows: list[dict] = []
    for _, fam in families_df.iterrows():
        family = fam["family"]
        rows = discover_tags_for_family(family=family, timeout=args.timeout)
        for r in rows:
            r["family_card_text"] = fam["family_card_text"]
            r["family_url"] = fam["family_url"]
        tag_rows.extend(rows)

    discovered = pd.DataFrame(tag_rows).drop_duplicates(subset=["tag"]).reset_index(drop=True)
    discovered = enrich_tag_details(discovered, timeout=args.timeout)

    manual_exclusions = _load_manual_exclusions(Path(args.manual_exclusions))

    eligible, excluded = apply_filters(
        discovered,
        start_date=start_date,
        end_date=end_date,
        min_params_b=args.min_params_b,
        dense_max_params_b=args.dense_max_params_b,
        moe_max_active_params_b=args.moe_max_active_params_b,
        moe_max_total_params_b=args.moe_max_total_params_b,
        max_model_size_gb=args.max_model_size_gb,
        manual_exclusions=manual_exclusions,
    )

    # Keep exported CSVs lean: omit heavy/debug columns and redundant manual code.
    for frame in (discovered, eligible, excluded):
        for col in ("detail_text", "manual_reason_code"):
            if col in frame.columns:
                frame.drop(columns=[col], inplace=True)

    discovered_path = out_dir / "discovered_tags_latest.csv"
    eligible_path = out_dir / "eligible_tags_latest.csv"
    excluded_path = out_dir / "excluded_tags_latest.csv"
    exclusion_counts_path = out_dir / "exclusion_reason_counts_latest.csv"

    discovered.to_csv(discovered_path, index=False)
    eligible.to_csv(eligible_path, index=False)
    excluded.to_csv(excluded_path, index=False)

    exclusion_counts = build_exclusion_reason_counts(excluded)
    exclusion_counts.to_csv(exclusion_counts_path, index=False)

    print(f"Discovered tags: {len(discovered)}")
    print(f"Eligible tags:   {len(eligible)}")
    print(f"Excluded tags:   {len(excluded)}")
    print(f"Wrote: {discovered_path}")
    print(f"Wrote: {eligible_path}")
    print(f"Wrote: {excluded_path}")
    print(f"Wrote: {exclusion_counts_path}")
    print(f"Summary: discovered={len(discovered)}, eligible={len(eligible)}, excluded={len(excluded)}")


if __name__ == "__main__":
    main()
