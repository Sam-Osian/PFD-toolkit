#!/usr/bin/env python3
"""Archived: build candidate samples for automatic issue-family auditing."""

from __future__ import annotations

import argparse
import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


PILOT_VERSION = "issue-family-recall-pilot-v1"


@dataclass(frozen=True)
class Family:
    family_id: str
    label: str
    definition: str
    inclusions: str
    exclusions: str
    strict_pattern: str
    stages: tuple[str, ...] = ()
    themes: tuple[str, ...] = ()


FAMILIES = (
    Family(
        "continuity_coordination",
        "Failures in continuity and coordination of care",
        (
            "Service-side failures that interrupt, fragment, or leave unclear "
            "responsibility for care across staff, teams, organisations, settings, "
            "transfers, or time."
        ),
        (
            "Discontinuity after transfer or discharge; poorly coordinated or "
            "fragmented care; unclear ownership; absent joined-up working."
        ),
        (
            "Staff-retention continuity alone; generic poor treatment with no "
            "continuity or coordination element; record defects unless they caused "
            "fragmented care."
        ),
        r"\bcontinuity\b|joined[- ]up care|fragmented care|"
        r"(?:fail\w*|poor\w*|inadequate\w*|lack\w*)\W+(?:to )?coordinat\w*|"
        r"unclear responsibility for (?:ongoing )?care",
        stages=("care_coordination",),
        themes=("communication_handover",),
    ),
    Family(
        "records_information",
        "Medical and care-record failures",
        (
            "Failures to create, complete, maintain, integrate, retain, access, "
            "review, or use medical, clinical, nursing, care, or custody health records."
        ),
        (
            "Incomplete, inaccurate, delayed, inaccessible, fragmented, unavailable, "
            "or unread records and record systems."
        ),
        (
            "General verbal communication with no record component; unrelated "
            "commercial or administrative records; a clinical action merely mentioned "
            "in a record when the record itself did not fail."
        ),
        r"\b(?:medical|clinical|nursing|patient|care|health|custody) records?\b|"
        r"\brecord[- ]keeping\b|\bclinical documentation\b|\bcare notes?\b",
        stages=("information_record_management",),
        themes=("records_information",),
    ),
    Family(
        "staffing_capacity",
        "Insufficient staffing capacity and workforce availability",
        (
            "Insufficient numbers, availability, deployment, skill mix, recruitment, "
            "or retention of staff to operate a service safely."
        ),
        (
            "Understaffing, vacancies, unsafe rotas, workload pressure, unavailable "
            "specialist staff, recruitment and retention failures."
        ),
        (
            "Training or competence alone; individual misconduct; equipment or bed "
            "capacity without a workforce shortage."
        ),
        r"\bunderstaff\w*|\bstaff(?:ing)? (?:levels?|shortage|capacity|numbers?|"
        r"availability|vacanc\w*|rota)|\bworkforce (?:shortage|capacity|vacanc\w*|"
        r"recruit\w*|retention)|\binsufficient (?:numbers? of )?staff\b",
        stages=("workforce_management",),
        themes=("staffing_capacity",),
    ),
    Family(
        "discharge_transitions",
        "Unsafe or incomplete discharge and care-transition processes",
        (
            "Service-side failures in planning, deciding, documenting, communicating, "
            "or supporting discharge or transition between care settings."
        ),
        (
            "Unsafe discharge, absent discharge planning or summaries, inadequate "
            "post-discharge arrangements, and failed transition between services."
        ),
        (
            "A patient independently leaving care; transfer transport failures with no "
            "care-transition issue; later follow-up unrelated to discharge."
        ),
        r"\bdischarg\w*\b|\btransition(?:s|al)? (?:between|from|to|of) "
        r"(?:care|services?|settings?)\b",
        stages=("discharge",),
        themes=("communication_handover",),
    ),
    Family(
        "referral_follow_up",
        "Failures in referral and required follow-up pathways",
        (
            "Service-side failures to make, transmit, receive, process, act on, chase, "
            "or complete a referral or required follow-up."
        ),
        (
            "Missing, delayed, rejected, lost, or unacted referrals and omitted or "
            "delayed required follow-up."
        ),
        (
            "Patient non-attendance without a service-side failure; generic treatment "
            "delay unrelated to referral or follow-up; communication with no pathway."
        ),
        r"\breferr(?:al|ed|ing)\b|\bfollow[- ]?up\b|\bchase(?:d|s|ing)?\b",
        stages=("referral", "follow_up"),
        themes=(),
    ),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--occurrences-csv", required=True, type=Path)
    parser.add_argument("--embeddings-npy", required=True, type=Path)
    parser.add_argument("--assignments-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--semantic-top-k", type=int, default=1000)
    parser.add_argument("--recurring-sample", type=int, default=30)
    parser.add_argument("--review-sample", type=int, default=30)
    parser.add_argument("--isolated-sample", type=int, default=50)
    parser.add_argument("--semantic-only-sample", type=int, default=30)
    parser.add_argument("--max-per-current-group", type=int, default=2)
    parser.add_argument("--seed", type=int, default=20260730)
    return parser.parse_args()


def contains_pipe_value(series: pd.Series, values: tuple[str, ...]) -> pd.Series:
    wanted = set(values)
    if not wanted:
        return pd.Series(False, index=series.index)
    return series.astype(str).str.split("|").map(
        lambda items: bool(wanted.intersection(item.strip() for item in items))
    )


def family_masks(frame: pd.DataFrame, family: Family) -> tuple[pd.Series, pd.Series]:
    text = frame[
        ["canonical_issue", "issue_object", "evidence_quote"]
    ].astype(str).agg(" ".join, axis=1)
    strict = text.str.contains(
        re.compile(family.strict_pattern, flags=re.IGNORECASE), na=False
    )
    stage = frame["process_stage"].astype(str).isin(family.stages)
    theme = contains_pipe_value(frame["issue_themes"], family.themes)
    if family.family_id == "continuity_coordination":
        # Communication is only a retrieval hint for this family; requiring a
        # coordination/transition stage prevents the entire communication theme
        # from becoming the rule-selected pool.
        stage = stage | frame["process_stage"].astype(str).isin(
            ["handover", "transfer_transport", "discharge", "follow_up"]
        )
        rule = strict | (stage & theme)
    elif family.family_id == "discharge_transitions":
        rule = strict | stage
    else:
        rule = strict | stage | theme
    return strict, rule


def semantic_scores(
    embeddings: np.ndarray,
    strict_masks: list[np.ndarray],
) -> np.ndarray:
    centroids: list[np.ndarray] = []
    for mask in strict_masks:
        if not mask.any():
            raise ValueError("Every family requires at least one strict lexical seed")
        centroid = np.asarray(embeddings[mask], dtype=np.float32).mean(axis=0)
        centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
        centroids.append(centroid)
    return np.asarray(embeddings @ np.stack(centroids, axis=1), dtype=np.float32)


def status_stratum(row: pd.Series) -> str:
    if bool(row["semantic_only"]):
        return "semantic_only"
    status = str(row.get("refinement_status", ""))
    if status == "refined_recurring":
        return "refined_recurring"
    if status == "review_required":
        return "review_required"
    return "isolated_or_unassigned"


def group_balanced_sample(
    frame: pd.DataFrame,
    size: int,
    *,
    seed: int,
    max_per_group: int,
) -> pd.DataFrame:
    if size <= 0 or frame.empty:
        return frame.iloc[0:0].copy()
    shuffled = frame.sample(frac=1.0, random_state=seed)
    selected: list[int] = []
    counts: dict[str, int] = {}
    for index, row in shuffled.iterrows():
        group_id = str(row.get("refined_group_id", "")) or f"ungrouped:{row['issue_id']}"
        if counts.get(group_id, 0) >= max_per_group:
            continue
        selected.append(index)
        counts[group_id] = counts.get(group_id, 0) + 1
        if len(selected) >= size:
            break
    return frame.loc[selected].copy()


def main() -> None:
    args = parse_args()
    occurrences = pd.read_csv(args.occurrences_csv).fillna("")
    assignments = pd.read_csv(args.assignments_csv).fillna("")
    embeddings = np.load(args.embeddings_npy, mmap_mode="r")
    if len(occurrences) != len(embeddings):
        raise ValueError("Occurrence and embedding row counts do not match")
    if occurrences["issue_id"].astype(str).duplicated().any():
        raise ValueError("Occurrence issue IDs must be unique")
    membership = assignments[
        ["issue_id", "refined_group_id", "refinement_status"]
    ].drop_duplicates("issue_id")
    frame = occurrences.merge(
        membership, on="issue_id", how="left", validate="one_to_one"
    ).fillna("")

    strict_masks: list[np.ndarray] = []
    rule_masks: list[pd.Series] = []
    for family in FAMILIES:
        strict, rule = family_masks(frame, family)
        strict_masks.append(strict.to_numpy())
        rule_masks.append(rule)
    scores = semantic_scores(embeddings, strict_masks)

    pool_frames: list[pd.DataFrame] = []
    sample_frames: list[pd.DataFrame] = []
    manifest_families: dict[str, Any] = {}
    sample_sizes = {
        "refined_recurring": args.recurring_sample,
        "review_required": args.review_sample,
        "isolated_or_unassigned": args.isolated_sample,
        "semantic_only": args.semantic_only_sample,
    }
    for family_index, family in enumerate(FAMILIES):
        rule = rule_masks[family_index]
        top_count = min(args.semantic_top_k, len(frame))
        top_indices = np.argpartition(scores[:, family_index], -top_count)[-top_count:]
        semantic = pd.Series(False, index=frame.index)
        semantic.iloc[top_indices] = True
        selected = rule | semantic
        pool = frame[selected].copy()
        pool["family_id"] = family.family_id
        pool["family_label"] = family.label
        pool["strict_selected"] = strict[selected].to_numpy()
        pool["rule_selected"] = rule[selected].to_numpy()
        pool["semantic_selected"] = semantic[selected].to_numpy()
        pool["semantic_only"] = (
            pool["semantic_selected"] & ~pool["rule_selected"]
        )
        pool["family_similarity"] = scores[selected.to_numpy(), family_index]
        pool["recall_stratum"] = pool.apply(status_stratum, axis=1)
        pool_frames.append(pool)

        local_samples: list[pd.DataFrame] = []
        for stratum, size in sample_sizes.items():
            local = pool[pool["recall_stratum"].eq(stratum)]
            sampled = group_balanced_sample(
                local,
                size,
                seed=args.seed + family_index * 100 + list(sample_sizes).index(stratum),
                max_per_group=args.max_per_current_group,
            )
            local_samples.append(sampled)
        sample = pd.concat(local_samples, ignore_index=True)
        sample["sample_order"] = np.arange(1, len(sample) + 1)
        sample_frames.append(sample)
        manifest_families[family.family_id] = {
            "label": family.label,
            "definition": family.definition,
            "inclusions": family.inclusions,
            "exclusions": family.exclusions,
            "strict_seed_count": int(strict_masks[family_index].sum()),
            "rule_candidate_count": int(rule.sum()),
            "candidate_pool_count": len(pool),
            "candidate_report_count": int(pool["report_key"].nunique()),
            "candidate_counts_by_stratum": {
                str(key): int(value)
                for key, value in pool["recall_stratum"].value_counts().items()
            },
            "sample_count": len(sample),
            "sample_counts_by_stratum": {
                str(key): int(value)
                for key, value in sample["recall_stratum"].value_counts().items()
            },
        }

    output_columns = [
        "family_id",
        "family_label",
        "issue_id",
        "report_key",
        "refined_group_id",
        "refinement_status",
        "recall_stratum",
        "strict_selected",
        "rule_selected",
        "semantic_selected",
        "semantic_only",
        "family_similarity",
        "canonical_issue",
        "evidence_quote",
        "responsible_actor_role",
        "failed_action",
        "issue_object",
        "counterparty_role",
        "failure_state",
        "process_stage",
        "communication_direction",
        "issue_themes",
    ]
    pool_output = pd.concat(pool_frames, ignore_index=True)
    sample_output = pd.concat(sample_frames, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    pool_output[output_columns].to_csv(
        args.output_dir / "01_family_candidate_pool.csv", index=False
    )
    sample_output[["sample_order", *output_columns]].to_csv(
        args.output_dir / "02_family_recall_sample.csv", index=False
    )
    manifest = {
        "pilot_version": PILOT_VERSION,
        "occurrences_csv": str(args.occurrences_csv),
        "embeddings_npy": str(args.embeddings_npy),
        "assignments_csv": str(args.assignments_csv),
        "semantic_top_k": args.semantic_top_k,
        "seed": args.seed,
        "families": manifest_families,
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
