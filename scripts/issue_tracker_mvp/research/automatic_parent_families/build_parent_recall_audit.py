#!/usr/bin/env python3
"""Archived: build a report-level recall audit for automatic issue parents."""

from __future__ import annotations

import argparse
import hashlib
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd


AUDIT_VERSION = "parent-recall-audit-v1"


@dataclass(frozen=True)
class ParentSpec:
    family_id: str
    review_name: str
    source_theme_columns: tuple[str, ...]


def parse_parent_spec(value: str) -> ParentSpec:
    parts = value.split("::")
    if len(parts) != 3:
        raise ValueError("Parent spec must be FAMILY_ID::REVIEW_NAME::THEME_COLUMN[,THEME_COLUMN]")
    family_id, review_name, themes = (part.strip() for part in parts)
    columns = tuple(item.strip() for item in themes.split(",") if item.strip())
    if not family_id or not review_name or not columns:
        raise ValueError("Parent spec fields cannot be empty")
    return ParentSpec(family_id, review_name, columns)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--occurrences-csv", required=True, type=Path)
    parser.add_argument("--embeddings-npy", required=True, type=Path)
    parser.add_argument("--assignments-csv", required=True, type=Path)
    parser.add_argument("--memberships-csv", required=True, type=Path)
    parser.add_argument("--family-summary-csv", required=True, type=Path)
    parser.add_argument("--reports-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--parent", action="append", required=True, type=parse_parent_spec)
    parser.add_argument("--high-semantic", type=int, default=25)
    parser.add_argument("--source-theme", type=int, default=25)
    parser.add_argument("--random-source-theme", type=int, default=0)
    parser.add_argument("--minimum-random-similarity", type=float, default=0.0)
    parser.add_argument("--semantic-boundary", type=int, default=25)
    parser.add_argument("--seed", type=int, default=20260731)
    return parser.parse_args()


def unit(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("Cannot normalize a zero vector")
    return np.asarray(vector / norm, dtype=np.float32)


def stable_audit_id(family_id: str, report_key: str) -> str:
    digest = hashlib.sha256(f"{family_id}:{report_key}".encode()).hexdigest()[:16]
    return f"recall_{digest}"


def family_centroid(
    occurrences: pd.DataFrame,
    embeddings: np.ndarray,
    assignments: pd.DataFrame,
    core_child_ids: set[str],
) -> np.ndarray:
    positions = pd.Series(np.arange(len(occurrences)), index=occurrences["issue_id"].astype(str))
    child_centroids: list[np.ndarray] = []
    core = assignments[assignments["relational_group_id"].astype(str).isin(core_child_ids)]
    for _, local in core.groupby("relational_group_id"):
        indices = positions.loc[local["issue_id"].astype(str)].astype(int).to_numpy()
        child_centroids.append(unit(np.asarray(embeddings[indices], dtype=np.float32).mean(axis=0)))
    if not child_centroids:
        raise ValueError("Selected family has no core child occurrences")
    return unit(np.stack(child_centroids).mean(axis=0))


def report_mapping(occurrences: pd.DataFrame) -> pd.DataFrame:
    required = ["report_key", "report_id", "report_url", "report_date"]
    mapping = occurrences[required].drop_duplicates()
    if mapping["report_key"].astype(str).duplicated().any():
        raise ValueError("A report key maps to multiple report metadata rows")
    return mapping


def report_issue_summaries(
    occurrences: pd.DataFrame, scores: np.ndarray, limit: int = 6
) -> dict[str, str]:
    local = occurrences[["report_key", "issue_id", "canonical_issue"]].copy()
    local["_score"] = scores
    output: dict[str, str] = {}
    for report_key, rows in local.groupby("report_key"):
        chosen = rows.sort_values("_score", ascending=False).head(limit)
        output[str(report_key)] = " | ".join(
            f"{row.issue_id}: {row.canonical_issue}" for row in chosen.itertuples()
        )
    return output


def choose_strata(
    candidates: pd.DataFrame,
    *,
    high_semantic: int,
    source_theme: int,
    random_source_theme: int,
    minimum_random_similarity: float,
    semantic_boundary: int,
    seed: int,
) -> pd.DataFrame:
    ordered = candidates.sort_values(["family_similarity", "report_key"], ascending=[False, True])
    high = ordered.head(high_semantic).assign(recall_stratum="high_semantic")
    used = set(high["report_key"].astype(str))
    themed_pool = ordered[ordered["source_theme_selected"] & ~ordered["report_key"].astype(str).isin(used)]
    themed = themed_pool.head(source_theme).assign(recall_stratum="source_theme")
    used.update(themed["report_key"].astype(str))

    random_pool = ordered[
        ordered["source_theme_selected"]
        & ordered["family_similarity"].ge(minimum_random_similarity)
        & ~ordered["report_key"].astype(str).isin(used)
    ]
    random_count = min(random_source_theme, len(random_pool))
    random_themed = random_pool.sample(n=random_count, random_state=seed).assign(
        recall_stratum="random_source_theme"
    )
    used.update(random_themed["report_key"].astype(str))

    remainder = ordered[~ordered["report_key"].astype(str).isin(used)].copy()
    # Sample across the next 500 semantic candidates instead of taking only the
    # very top, so the audit observes the uncertain retrieval boundary.
    boundary_pool = remainder.head(min(500, len(remainder)))
    if semantic_boundary > 0 and not boundary_pool.empty:
        indices = np.linspace(0, len(boundary_pool) - 1, min(semantic_boundary, len(boundary_pool))).round().astype(int)
        boundary = boundary_pool.iloc[indices].assign(recall_stratum="semantic_boundary")
    else:
        boundary = boundary_pool.iloc[0:0].assign(recall_stratum="semantic_boundary")
    return pd.concat([high, themed, random_themed, boundary], ignore_index=True)


def build_parent_queue(
    spec: ParentSpec,
    occurrences: pd.DataFrame,
    embeddings: np.ndarray,
    assignments: pd.DataFrame,
    memberships: pd.DataFrame,
    summaries: pd.DataFrame,
    reports: pd.DataFrame,
    *,
    high_semantic: int,
    source_theme: int,
    random_source_theme: int,
    minimum_random_similarity: float,
    semantic_boundary: int,
    seed: int,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    family_rows = summaries[summaries["family_id"].astype(str).eq(spec.family_id)]
    if len(family_rows) != 1:
        raise ValueError(f"Family summary does not uniquely contain {spec.family_id}")
    missing_themes = set(spec.source_theme_columns) - set(reports.columns)
    if missing_themes:
        raise ValueError(f"Report source lacks theme columns: {sorted(missing_themes)}")
    core_ids = set(
        memberships.loc[
            memberships["family_id"].astype(str).eq(spec.family_id)
            & memberships["membership_type"].eq("core"),
            "child_group_id",
        ].astype(str)
    )
    centroid = family_centroid(occurrences, embeddings, assignments, core_ids)
    scores = np.asarray(embeddings @ centroid, dtype=np.float32)
    scored = occurrences[["issue_id", "report_key", "canonical_issue"]].copy()
    scored["family_similarity"] = scores
    representatives = scored.sort_values("family_similarity", ascending=False).drop_duplicates("report_key")

    core_assignment = assignments[assignments["relational_group_id"].astype(str).isin(core_ids)]
    existing_reports = set(core_assignment["report_key"].astype(str))
    report_meta = report_mapping(occurrences)
    source = reports.rename(columns={"url": "report_url", "date": "source_report_date"}).copy()
    source["report_url"] = source["report_url"].astype(str)
    # Report numbers are neither complete nor globally unique in the legacy
    # source. Canonical URLs are the stable key shared with the occurrence file.
    source = source[source["report_url"].str.strip().ne("")].copy()
    source["source_theme_selected"] = source[list(spec.source_theme_columns)].astype(bool).any(axis=1)
    source_columns = ["report_url", "concerns", "source_theme_selected", *spec.source_theme_columns]
    source = source[source_columns].drop_duplicates("report_url")

    candidates = representatives.merge(report_meta, on="report_key", how="left", validate="one_to_one")
    candidates = candidates.merge(source, on="report_url", how="left", validate="many_to_one").fillna("")
    candidates = candidates[~candidates["report_key"].astype(str).isin(existing_reports)].copy()
    candidates["source_theme_selected"] = candidates["source_theme_selected"].astype(bool)
    issue_summaries = report_issue_summaries(occurrences, scores)
    candidates["report_extracted_issues"] = candidates["report_key"].map(issue_summaries)

    assignment_lookup = assignments.drop_duplicates("issue_id").set_index("issue_id").to_dict("index")
    candidates["representative_group_id"] = candidates["issue_id"].map(
        lambda issue_id: assignment_lookup.get(str(issue_id), {}).get("relational_group_id", "")
    )
    candidates["representative_recurrence_status"] = candidates["issue_id"].map(
        lambda issue_id: assignment_lookup.get(str(issue_id), {}).get("recurrence_status", "unassigned")
    )
    queue = choose_strata(
        candidates,
        high_semantic=high_semantic,
        source_theme=source_theme,
        random_source_theme=random_source_theme,
        minimum_random_similarity=minimum_random_similarity,
        semantic_boundary=semantic_boundary,
        seed=seed,
    )
    family = family_rows.iloc[0]
    queue.insert(0, "family_id", spec.family_id)
    queue.insert(1, "parent_review_name", spec.review_name)
    queue.insert(2, "family_prototype", family["family_prototype"])
    queue.insert(3, "audit_id", [stable_audit_id(spec.family_id, str(value)) for value in queue["report_key"]])
    queue["supports_parent"] = ""
    queue["matched_issue_id"] = ""
    queue["miss_stage"] = ""
    queue["review_notes"] = ""
    metrics = {
        "family_id": spec.family_id,
        "review_name": spec.review_name,
        "core_child_groups": len(core_ids),
        "current_core_reports": len(existing_reports),
        "outside_report_candidates": len(candidates),
        "outside_source_theme_reports": int(candidates["source_theme_selected"].sum()),
        "sample_rows": len(queue),
        "sample_by_stratum": queue["recall_stratum"].value_counts().to_dict(),
    }
    return queue, metrics


def review_markdown(queue: pd.DataFrame) -> str:
    lines = [
        "# Parent issue recall audit",
        "",
        "Review reports currently outside the parent's tight core. Decide yes, no, or uncertain.",
        "For yes decisions, record the matching issue ID when an extracted issue captures the concern; otherwise use miss_stage `not_extracted`.",
        "",
    ]
    for family_id, family_rows in queue.groupby("family_id", sort=False):
        lines.extend([f"# {family_rows.iloc[0]['parent_review_name']}", ""])
        for row in family_rows.to_dict("records"):
            concerns = str(row.get("concerns", "")).strip()
            lines.extend(
                [
                    f"## {row['audit_id']} — {row['recall_stratum']}",
                    "",
                    f"Report: {row.get('report_id', '')} — {row.get('report_url', '')}",
                    "",
                    f"Similarity: {float(row['family_similarity']):.3f}",
                    "",
                    f"Representative extracted issue: {row['issue_id']}: {row['canonical_issue']}",
                    "",
                    f"Extracted issues: {row['report_extracted_issues']}",
                    "",
                    f"Original concerns: {concerns}",
                    "",
                    "Decision / matched issue / stage / notes:",
                    "",
                ]
            )
    return "\n".join(lines)


def main() -> None:
    args = parse_args()
    occurrences = pd.read_csv(args.occurrences_csv).fillna("")
    embeddings = np.load(args.embeddings_npy, mmap_mode="r")
    assignments = pd.read_csv(args.assignments_csv).fillna("")
    memberships = pd.read_csv(args.memberships_csv).fillna("")
    summaries = pd.read_csv(args.family_summary_csv).fillna("")
    reports = pd.read_csv(args.reports_csv).fillna("")
    if len(occurrences) != len(embeddings):
        raise ValueError("Occurrences and embeddings are not aligned")
    queues: list[pd.DataFrame] = []
    parent_metrics: list[dict[str, Any]] = []
    for index, spec in enumerate(args.parent):
        queue, metrics = build_parent_queue(
            spec, occurrences, embeddings, assignments, memberships, summaries, reports,
            high_semantic=args.high_semantic, source_theme=args.source_theme,
            random_source_theme=args.random_source_theme,
            minimum_random_similarity=args.minimum_random_similarity,
            semantic_boundary=args.semantic_boundary, seed=args.seed + index,
        )
        queues.append(queue)
        parent_metrics.append(metrics)
    output = pd.concat(queues, ignore_index=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    output.to_csv(args.output_dir / "01_parent_recall_review_queue.csv", index=False)
    (args.output_dir / "review_packet.md").write_text(review_markdown(output), encoding="utf-8")
    manifest = {
        "audit_version": AUDIT_VERSION,
        "parents": parent_metrics,
        "seed": args.seed,
        "interpretation": "The queue is deliberately enriched for possible misses and is not prevalence-weighted.",
    }
    (args.output_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
