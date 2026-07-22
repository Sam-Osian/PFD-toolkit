#!/usr/bin/env python3
"""Conservatively assign ungrouped issue occurrences to reviewed prototypes."""

from __future__ import annotations

import argparse
import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import build_issue_index as pipeline


@dataclass
class Prototype:
    prototype_id: str
    source: str
    status: str
    subject_domain: str
    issue_ids: list[str]
    report_ids: set[str]
    centroid: np.ndarray


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Assign ungrouped occurrences to recurring and emerging prototypes."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--tuning-dir", type=Path, default=None)
    parser.add_argument("--repair-dir", type=Path, default=None)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--similarity",
        type=float,
        default=0.845,
        help="Minimum cosine similarity; the reviewed default achieved 87.5%% precision.",
    )
    parser.add_argument(
        "--margin",
        type=float,
        default=0.03,
        help="Minimum lead over the second-best eligible prototype.",
    )
    parser.add_argument(
        "--domain-match",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Require occurrence and prototype subject_domain to match.",
    )
    return parser.parse_args()


def _prototype(
    prototype_id: str,
    source: str,
    status: str,
    subject_domain: str,
    issue_ids: list[str],
    occurrences: pd.DataFrame,
    embeddings: np.ndarray,
    issue_index: dict[str, int],
) -> Prototype:
    indices = [issue_index[issue_id] for issue_id in issue_ids]
    centroid = embeddings[indices].mean(axis=0)
    centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
    return Prototype(
        prototype_id=prototype_id,
        source=source,
        status=status,
        subject_domain=subject_domain,
        issue_ids=issue_ids,
        report_ids=set(occurrences.iloc[indices]["report_key"]),
        centroid=centroid,
    )


def build_prototypes(
    occurrences: pd.DataFrame,
    embeddings: np.ndarray,
    candidate_groups: pd.DataFrame,
    candidate_assignments: pd.DataFrame,
    repaired_groups: pd.DataFrame,
    repaired_assignments: pd.DataFrame,
) -> list[Prototype]:
    issue_index = {
        issue_id: position for position, issue_id in enumerate(occurrences["issue_id"])
    }
    prototypes: list[Prototype] = []
    repaired = repaired_groups[
        repaired_groups["recurrence_status"].isin(["recurring", "emerging"])
    ]
    for row in repaired.itertuples(index=False):
        issue_ids = repaired_assignments.loc[
            repaired_assignments["repaired_group_id"] == row.repaired_group_id, "issue_id"
        ].tolist()
        prototypes.append(
            _prototype(
                row.repaired_group_id,
                "review_repair",
                row.recurrence_status,
                row.subject_domain,
                issue_ids,
                occurrences,
                embeddings,
                issue_index,
            )
        )
    emerging = candidate_groups[candidate_groups["recurrence_status"] == "emerging"]
    for row in emerging.itertuples(index=False):
        issue_ids = candidate_assignments.loc[
            candidate_assignments["candidate_group_id"] == row.candidate_group_id, "issue_id"
        ].tolist()
        prototypes.append(
            _prototype(
                row.candidate_group_id,
                "recommended_emerging",
                "emerging",
                row.subject_domain,
                issue_ids,
                occurrences,
                embeddings,
                issue_index,
            )
        )
    return prototypes


def assign_occurrences(
    occurrences: pd.DataFrame,
    embeddings: np.ndarray,
    ungrouped_indices: np.ndarray,
    prototypes: list[Prototype],
    *,
    similarity_threshold: float,
    margin_threshold: float,
    domain_match: bool,
) -> pd.DataFrame:
    if not len(ungrouped_indices) or not prototypes:
        return pd.DataFrame()
    centroids = np.stack([prototype.centroid for prototype in prototypes])
    similarities = embeddings[ungrouped_indices] @ centroids.T
    for local_index, occurrence_index in enumerate(ungrouped_indices):
        occurrence = occurrences.iloc[occurrence_index]
        for prototype_index, prototype in enumerate(prototypes):
            occurrence_themes = set(
                pipeline.clean_text(occurrence["issue_themes"]).split("|")
            )
            if occurrence["report_key"] in prototype.report_ids or (
                domain_match and prototype.subject_domain not in occurrence_themes
            ):
                similarities[local_index, prototype_index] = -2.0
    order = np.argsort(similarities, axis=1)
    best_indices = order[:, -1]
    second_indices = order[:, -2]
    best_scores = similarities[np.arange(len(ungrouped_indices)), best_indices]
    second_scores = similarities[np.arange(len(ungrouped_indices)), second_indices]
    accepted = (best_scores >= similarity_threshold) & (
        best_scores - second_scores >= margin_threshold
    )
    rows: list[dict[str, Any]] = []
    for local_index in np.where(accepted)[0]:
        occurrence_index = int(ungrouped_indices[local_index])
        prototype = prototypes[int(best_indices[local_index])]
        rows.append(
            {
                "issue_id": occurrences.iloc[occurrence_index]["issue_id"],
                "report_key": occurrences.iloc[occurrence_index]["report_key"],
                "canonical_issue": occurrences.iloc[occurrence_index]["canonical_issue"],
                "subject_domain": pipeline.clean_text(
                    occurrences.iloc[occurrence_index]["issue_themes"]
                ).split("|")[0],
                "prototype_id": prototype.prototype_id,
                "prototype_source": prototype.source,
                "prototype_original_status": prototype.status,
                "best_similarity": float(best_scores[local_index]),
                "second_best_similarity": float(second_scores[local_index]),
                "similarity_margin": float(
                    best_scores[local_index] - second_scores[local_index]
                ),
            }
        )
    return pd.DataFrame(rows)


def summarise_assignments(
    occurrences: pd.DataFrame,
    prototypes: list[Prototype],
    assignments: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    issue_lookup = occurrences.set_index("issue_id")
    rows: list[dict[str, Any]] = []
    member_rows: list[dict[str, Any]] = []
    for prototype in prototypes:
        added = assignments[assignments["prototype_id"] == prototype.prototype_id]
        combined_issue_ids = prototype.issue_ids + added.get(
            "issue_id", pd.Series(dtype=str)
        ).tolist()
        combined_reports = prototype.report_ids | set(
            added.get("report_key", pd.Series(dtype=str))
        )
        new_status = pipeline.recurrence_status(len(combined_reports), 3)
        if added.empty:
            continue
        rows.append(
            {
                "prototype_id": prototype.prototype_id,
                "prototype_source": prototype.source,
                "original_status": prototype.status,
                "new_status": new_status,
                "original_report_count": len(prototype.report_ids),
                "new_report_count": len(combined_reports),
                "original_issue_count": len(prototype.issue_ids),
                "assigned_issue_count": len(added),
                "subject_domain": prototype.subject_domain,
                "original_issues": " | ".join(
                    issue_lookup.loc[prototype.issue_ids, "canonical_issue"].drop_duplicates()
                ),
                "assigned_issues": " | ".join(added["canonical_issue"].drop_duplicates()),
                "minimum_assignment_similarity": float(added["best_similarity"].min()),
                "minimum_assignment_margin": float(added["similarity_margin"].min()),
            }
        )
        for issue_id in combined_issue_ids:
            member_rows.append(
                {
                    "prototype_id": prototype.prototype_id,
                    "membership": "assigned" if issue_id in set(added["issue_id"]) else "original",
                    "issue_id": issue_id,
                    "report_key": issue_lookup.loc[issue_id, "report_key"],
                    "canonical_issue": issue_lookup.loc[issue_id, "canonical_issue"],
                }
            )
    return pd.DataFrame(rows), pd.DataFrame(member_rows)


def run_assignment(
    run_dir: Path,
    tuning_dir: Path,
    repair_dir: Path,
    output_dir: Path,
    *,
    similarity_threshold: float,
    margin_threshold: float,
    domain_match: bool,
) -> dict[str, Any]:
    occurrence_path = run_dir / "02_embedding_occurrences.csv"
    if not occurrence_path.exists():
        occurrence_path = run_dir / "01_issue_occurrences.csv"
    occurrences = pd.read_csv(occurrence_path).fillna("")
    embeddings = np.load(run_dir / "02_issue_embeddings.npy")
    candidate_groups = pd.read_csv(tuning_dir / "recommended_groups.csv").fillna("")
    candidate_assignments = pd.read_csv(
        tuning_dir / "recommended_assignments.csv"
    ).fillna("")
    repaired_groups = pd.read_csv(repair_dir / "repaired_groups.csv").fillna("")
    repaired_assignments = pd.read_csv(repair_dir / "repaired_assignments.csv").fillna("")
    prototypes = build_prototypes(
        occurrences,
        embeddings,
        candidate_groups,
        candidate_assignments,
        repaired_groups,
        repaired_assignments,
    )
    already_grouped = set(candidate_assignments["issue_id"])
    ungrouped_indices = occurrences.index[
        ~occurrences["issue_id"].isin(already_grouped)
    ].to_numpy()
    assignments = assign_occurrences(
        occurrences,
        embeddings,
        ungrouped_indices,
        prototypes,
        similarity_threshold=similarity_threshold,
        margin_threshold=margin_threshold,
        domain_match=domain_match,
    )
    summaries, members = summarise_assignments(occurrences, prototypes, assignments)
    promoted = summaries[
        (summaries["original_status"] == "emerging")
        & (summaries["new_status"] == "recurring")
    ].copy()
    promoted["coherent_after_assignment"] = ""
    promoted["assignment_correct"] = ""
    promoted["duplicate_of"] = ""
    promoted["review_notes"] = ""
    promoted["strict_automatic_candidate"] = (
        (promoted["minimum_assignment_similarity"] >= 0.845)
        & (promoted["minimum_assignment_margin"] >= 0.03)
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(output_dir / "assignment_candidates.csv", index=False)
    summaries.to_csv(output_dir / "affected_prototypes.csv", index=False)
    members.to_csv(output_dir / "affected_prototype_members.csv", index=False)
    review_path = output_dir / "promoted_groups_review_queue.csv"
    if review_path.exists():
        previous = pd.read_csv(review_path).fillna("")
        previous = previous.set_index("prototype_id")
        for column in (
            "coherent_after_assignment",
            "assignment_correct",
            "duplicate_of",
            "review_notes",
        ):
            promoted[column] = promoted.apply(
                lambda row: previous.at[row["prototype_id"], column]
                if row["prototype_id"] in previous.index and column in previous
                else row[column],
                axis=1,
            )
    promoted.to_csv(review_path, index=False)
    metrics = {
        "prototype_count": len(prototypes),
        "ungrouped_occurrences_considered": len(ungrouped_indices),
        "accepted_assignments": len(assignments),
        "affected_prototypes": len(summaries),
        "promoted_emerging_groups": len(promoted),
        "recurring_groups_before": sum(p.status == "recurring" for p in prototypes),
        "recurring_groups_after": sum(p.status == "recurring" for p in prototypes)
        + len(promoted),
    }
    (output_dir / "assignment_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    (output_dir / "assignment_config.json").write_text(
        json.dumps(
            {
                "similarity_threshold": similarity_threshold,
                "margin_threshold": margin_threshold,
                "domain_match": domain_match,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return metrics


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    tuning_dir = (
        args.tuning_dir.expanduser().resolve()
        if args.tuning_dir
        else run_dir / "05_tuning"
    )
    repair_dir = (
        args.repair_dir.expanduser().resolve()
        if args.repair_dir
        else run_dir / "06_review_repair"
    )
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else run_dir / "07_prototype_assignment"
    )
    metrics = run_assignment(
        run_dir,
        tuning_dir,
        repair_dir,
        output_dir,
        similarity_threshold=args.similarity,
        margin_threshold=args.margin,
        domain_match=args.domain_match,
    )
    print(f"Assignment artefacts: {output_dir}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
