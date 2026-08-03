#!/usr/bin/env python3
"""Archived: expand tight automatic issue-family cores with auditable overlap.

Precise recurring groups remain unchanged. Families are seeded from a strict
average-linkage cut, while child groups may receive a limited number of
secondary family memberships when their centroid is close to the alternative
core and the attachment has independent multi-resolution or facet evidence.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

import discover_issue_families as discovery


EXPANSION_VERSION = "overlapping-issue-family-expansion-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--occurrences-csv", required=True, type=Path)
    parser.add_argument("--embeddings-npy", required=True, type=Path)
    parser.add_argument("--assignments-csv", required=True, type=Path)
    parser.add_argument("--groups-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--core-similarity", type=float, default=0.84)
    parser.add_argument("--broad-similarity", type=float, default=0.82)
    parser.add_argument("--attachment-similarity", type=float, default=0.90)
    parser.add_argument("--maximum-primary-gap", type=float, default=0.04)
    parser.add_argument("--maximum-secondary-families", type=int, default=2)
    parser.add_argument("--minimum-core-children", type=int, default=2)
    parser.add_argument("--review-family-limit", type=int, default=40)
    return parser.parse_args()


def values(value: Any) -> set[str]:
    return set(discovery.split_pipe_values([value]))


def build_cores(
    child_features: pd.DataFrame,
    child_centroids: np.ndarray,
    strict_labels: np.ndarray,
    broad_labels: np.ndarray,
    *,
    minimum_core_children: int,
) -> tuple[list[dict[str, Any]], dict[int, int]]:
    cores: list[dict[str, Any]] = []
    child_owner: dict[int, int] = {}
    for strict_label in sorted(set(strict_labels)):
        positions = np.where(strict_labels == strict_label)[0]
        if len(positions) < minimum_core_children:
            continue
        centroid = discovery.unit_normalize(child_centroids[positions].mean(axis=0))
        similarities = child_centroids[positions] @ centroid
        medoid_position = positions[int(np.argmax(similarities))]
        family_id = discovery.stable_family_id(
            child_features.iloc[positions]["child_group_id"].astype(str)
        )
        broad_counts = pd.Series(broad_labels[positions]).value_counts()
        broad_label = int(broad_counts.index[0])
        core_index = len(cores)
        for position in positions:
            child_owner[int(position)] = core_index
        cores.append(
            {
                "family_id": family_id,
                "strict_label": int(strict_label),
                "broad_label": broad_label,
                "positions": positions,
                "centroid": centroid,
                "medoid_position": int(medoid_position),
                "themes": set().union(
                    *(
                        values(value)
                        for value in child_features.iloc[positions]["dominant_themes"]
                    )
                ),
                "stages": set().union(
                    *(
                        values(value)
                        for value in child_features.iloc[positions][
                            "dominant_process_stages"
                        ]
                    )
                ),
            }
        )
    return cores, child_owner


def attachment_rows(
    child_features: pd.DataFrame,
    child_centroids: np.ndarray,
    strict_labels: np.ndarray,
    broad_labels: np.ndarray,
    cores: list[dict[str, Any]],
    child_owner: dict[int, int],
    *,
    minimum_similarity: float,
    maximum_primary_gap: float,
    maximum_secondary_families: int,
) -> pd.DataFrame:
    if not 0.0 < minimum_similarity < 1.0:
        raise ValueError("Attachment similarity must be between zero and one")
    if not 0.0 <= maximum_primary_gap < 1.0:
        raise ValueError("Maximum primary gap must be in [0, 1)")
    if maximum_secondary_families < 0:
        raise ValueError("Maximum secondary families cannot be negative")
    core_centroids = np.stack([core["centroid"] for core in cores])
    similarities = child_centroids @ core_centroids.T
    rows: list[dict[str, Any]] = []
    for child_position, child in child_features.iterrows():
        owner = child_owner.get(int(child_position))
        primary_similarity = (
            float(similarities[child_position, owner]) if owner is not None else None
        )
        if owner is not None:
            core = cores[owner]
            rows.append(
                {
                    "family_id": core["family_id"],
                    "child_group_id": child["child_group_id"],
                    "membership_type": "core",
                    "similarity_to_family_centroid": primary_similarity,
                    "primary_family_id": core["family_id"],
                    "primary_similarity": primary_similarity,
                    "similarity_gap_from_primary": 0.0,
                    "same_broad_cluster": True,
                    "shared_themes": "|".join(
                        sorted(values(child["dominant_themes"]) & core["themes"])
                    ),
                    "shared_process_stages": "|".join(
                        sorted(
                            values(child["dominant_process_stages"])
                            & core["stages"]
                        )
                    ),
                    "attachment_evidence": "strict_core_membership",
                }
            )

        candidates: list[dict[str, Any]] = []
        for core_index in np.argsort(-similarities[child_position]):
            core_index = int(core_index)
            if core_index == owner:
                continue
            similarity = float(similarities[child_position, core_index])
            if similarity < minimum_similarity:
                break
            gap = (
                primary_similarity - similarity
                if primary_similarity is not None
                else None
            )
            if gap is not None and gap > maximum_primary_gap:
                continue
            core = cores[core_index]
            shared_themes = values(child["dominant_themes"]) & core["themes"]
            shared_stages = (
                values(child["dominant_process_stages"]) & core["stages"]
            )
            same_broad = int(broad_labels[child_position]) == core["broad_label"]
            if not (same_broad or (shared_themes and shared_stages)):
                continue
            evidence = []
            if same_broad:
                evidence.append("same_broad_cluster")
            if shared_themes and shared_stages:
                evidence.append("shared_theme_and_stage")
            candidates.append(
                {
                    "family_id": core["family_id"],
                    "child_group_id": child["child_group_id"],
                    "membership_type": (
                        "secondary" if owner is not None else "attached_singleton"
                    ),
                    "similarity_to_family_centroid": similarity,
                    "primary_family_id": (
                        cores[owner]["family_id"] if owner is not None else ""
                    ),
                    "primary_similarity": primary_similarity,
                    "similarity_gap_from_primary": gap,
                    "same_broad_cluster": same_broad,
                    "shared_themes": "|".join(sorted(shared_themes)),
                    "shared_process_stages": "|".join(sorted(shared_stages)),
                    "attachment_evidence": "|".join(evidence),
                }
            )
        rows.extend(candidates[:maximum_secondary_families])
    return pd.DataFrame(rows)


def expansion_grid(
    child_features: pd.DataFrame,
    child_centroids: np.ndarray,
    strict_labels: np.ndarray,
    broad_labels: np.ndarray,
    cores: list[dict[str, Any]],
    child_owner: dict[int, int],
    *,
    similarities: tuple[float, ...] = (0.89, 0.90, 0.91, 0.92),
    gaps: tuple[float, ...] = (0.02, 0.04, 0.06),
    maximum_secondary_families: int = 2,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for similarity in similarities:
        for gap in gaps:
            assignments = attachment_rows(
                child_features,
                child_centroids,
                strict_labels,
                broad_labels,
                cores,
                child_owner,
                minimum_similarity=similarity,
                maximum_primary_gap=gap,
                maximum_secondary_families=maximum_secondary_families,
            )
            attached = assignments[assignments["membership_type"].ne("core")]
            counts = attached["child_group_id"].value_counts()
            rows.append(
                {
                    "attachment_similarity": similarity,
                    "maximum_primary_gap": gap,
                    "attachment_rows": len(attached),
                    "attached_child_groups": attached["child_group_id"].nunique(),
                    "children_with_two_secondary_families": int(counts.ge(2).sum()),
                    "attached_singletons": int(
                        attached["membership_type"].eq("attached_singleton").sum()
                    ),
                    "cross_broad_attachments": int(
                        (~attached["same_broad_cluster"].astype(bool)).sum()
                    ),
                }
            )
    return pd.DataFrame(rows)


def family_outputs(
    child_features: pd.DataFrame,
    members: pd.DataFrame,
    cores: list[dict[str, Any]],
    memberships: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    child_lookup = child_features.set_index("child_group_id").to_dict("index")
    summary_rows: list[dict[str, Any]] = []
    member_rows: list[dict[str, Any]] = []
    centroid_lookup: dict[str, np.ndarray] = {}
    for core in cores:
        family_id = core["family_id"]
        local = memberships[memberships["family_id"].eq(family_id)].copy()
        child_ids = local["child_group_id"].astype(str).tolist()
        family_members = members[members["child_group_id"].astype(str).isin(child_ids)]
        core_ids = set(
            local.loc[local["membership_type"].eq("core"), "child_group_id"].astype(str)
        )
        core_members = members[members["child_group_id"].astype(str).isin(core_ids)]
        medoid = child_features.iloc[core["medoid_position"]]
        # Secondary children expand recall but must not silently redefine the
        # family label. Descriptive facets therefore come from the tight core.
        family_themes = discovery.dominant_values(
            child_lookup[child_id]["dominant_themes"] for child_id in core_ids
        )
        family_stages = discovery.dominant_values(
            child_lookup[child_id]["dominant_process_stages"]
            for child_id in core_ids
        )
        label_parts = [
            value.replace("_", " ")
            for value in (
                family_themes.split("|")[0] if family_themes else "",
                family_stages.split("|")[0] if family_stages else "",
            )
            if value
        ]
        core_reports = int(core_members["report_key"].astype(str).nunique())
        expanded_reports = int(family_members["report_key"].astype(str).nunique())
        secondary = local[local["membership_type"].ne("core")]
        secondary_count = int(local["membership_type"].eq("secondary").sum())
        singleton_count = int(
            local["membership_type"].eq("attached_singleton").sum()
        )
        report_expansion = expanded_reports / core_reports if core_reports else 0.0
        risk_reasons: list[str] = []
        if secondary_count + singleton_count > len(core_ids):
            risk_reasons.append("attachments_exceed_core_size")
        if report_expansion > 2.0:
            risk_reasons.append("report_count_more_than_doubled")
        cross_broad_count = int(
            (~secondary["same_broad_cluster"].astype(bool)).sum()
        )
        if len(secondary) and cross_broad_count / len(secondary) > 0.75:
            risk_reasons.append("mostly_cross_broad_attachments")
        summary_rows.append(
            {
                "family_id": family_id,
                "family_label_hint": " — ".join(label_parts),
                "family_prototype": medoid["prototype_canonical_issue"],
                "core_child_groups": len(core_ids),
                "secondary_child_groups": secondary_count,
                "attached_singletons": singleton_count,
                "total_child_groups": len(local),
                "core_report_count": core_reports,
                "expanded_report_count": expanded_reports,
                "added_report_count": expanded_reports - core_reports,
                "report_expansion": report_expansion,
                "same_broad_attachments": int(
                    secondary["same_broad_cluster"].astype(bool).sum()
                ),
                "cross_broad_attachments": cross_broad_count,
                "expansion_review_required": bool(risk_reasons),
                "expansion_risk_reasons": "|".join(risk_reasons),
                "dominant_themes": family_themes,
                "dominant_process_stages": family_stages,
            }
        )
        for row in local.to_dict("records"):
            row.update(
                {
                    "child_report_count": child_lookup[row["child_group_id"]][
                        "report_count"
                    ],
                    "prototype_canonical_issue": child_lookup[row["child_group_id"]][
                        "prototype_canonical_issue"
                    ],
                }
            )
            member_rows.append(row)
        centroid_lookup[family_id] = core["centroid"]
    summaries = pd.DataFrame(summary_rows).sort_values(
        ["expanded_report_count", "family_id"], ascending=[False, True]
    ).reset_index(drop=True)
    summaries["centroid_row"] = np.arange(len(summaries))
    family_memberships = pd.DataFrame(member_rows)
    order = {family_id: rank for rank, family_id in enumerate(summaries["family_id"])}
    family_memberships["family_rank"] = family_memberships["family_id"].map(order)
    family_memberships = family_memberships.sort_values(
        ["family_rank", "membership_type", "similarity_to_family_centroid"],
        ascending=[True, True, False],
    ).drop(columns="family_rank")
    family_centroids = np.stack(
        [centroid_lookup[family_id] for family_id in summaries["family_id"]]
    )
    return summaries, family_memberships, family_centroids


def review_markdown(
    summaries: pd.DataFrame,
    memberships: pd.DataFrame,
    *,
    limit: int,
) -> str:
    lines = [
        "# Overlapping issue-family expansion review",
        "",
        "Core membership is automatic and exclusive; secondary membership is auditable and non-exclusive.",
        "",
    ]
    for family in summaries.head(max(limit, 0)).to_dict("records"):
        family_id = family["family_id"]
        lines.extend(
            [
                f"## {family_id}: {family['family_label_hint']}",
                "",
                f"Prototype: {family['family_prototype']}",
                "",
                (
                    f"Core reports: {family['core_report_count']}; expanded reports: "
                    f"{family['expanded_report_count']}; core children: "
                    f"{family['core_child_groups']}; secondary children: "
                    f"{family['secondary_child_groups']}; attached singletons: "
                    f"{family['attached_singletons']}."
                ),
                "",
            ]
        )
        local = memberships[memberships["family_id"].eq(family_id)]
        for membership_type in ("core", "secondary", "attached_singleton"):
            selected = local[local["membership_type"].eq(membership_type)]
            if selected.empty:
                continue
            lines.extend([f"### {membership_type.replace('_', ' ').title()}", ""])
            for child in selected.to_dict("records"):
                lines.append(
                    f"- `{child['child_group_id']}` ({child['child_report_count']} reports; "
                    f"similarity {child['similarity_to_family_centroid']:.3f}; "
                    f"evidence {child['attachment_evidence']}) — "
                    f"{child['prototype_canonical_issue']}"
                )
            lines.append("")
    return "\n".join(lines)


def run_expansion(
    occurrences: pd.DataFrame,
    embeddings: np.ndarray,
    assignments: pd.DataFrame,
    groups: pd.DataFrame,
    *,
    core_similarity: float,
    broad_similarity: float,
    attachment_similarity: float,
    maximum_primary_gap: float,
    maximum_secondary_families: int,
    minimum_core_children: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    child_features, child_centroids, members = discovery.build_child_features(
        occurrences, embeddings, assignments, groups
    )
    strict_labels = discovery.cluster_labels(child_centroids, core_similarity)
    broad_labels = discovery.cluster_labels(child_centroids, broad_similarity)
    cores, child_owner = build_cores(
        child_features,
        child_centroids,
        strict_labels,
        broad_labels,
        minimum_core_children=minimum_core_children,
    )
    memberships = attachment_rows(
        child_features,
        child_centroids,
        strict_labels,
        broad_labels,
        cores,
        child_owner,
        minimum_similarity=attachment_similarity,
        maximum_primary_gap=maximum_primary_gap,
        maximum_secondary_families=maximum_secondary_families,
    )
    grid = expansion_grid(
        child_features,
        child_centroids,
        strict_labels,
        broad_labels,
        cores,
        child_owner,
        maximum_secondary_families=maximum_secondary_families,
    )
    summaries, memberships, family_centroids = family_outputs(
        child_features, members, cores, memberships
    )
    return child_features, grid, summaries, memberships, family_centroids


def main() -> None:
    args = parse_args()
    if not args.broad_similarity < args.core_similarity:
        raise ValueError("Broad similarity must be below core similarity")
    if args.minimum_core_children < 2:
        raise ValueError("Minimum core children must be at least two")
    occurrences = pd.read_csv(args.occurrences_csv).fillna("")
    embeddings = np.load(args.embeddings_npy, mmap_mode="r")
    assignments = pd.read_csv(args.assignments_csv).fillna("")
    groups = pd.read_csv(args.groups_csv).fillna("")
    outputs = run_expansion(
        occurrences,
        embeddings,
        assignments,
        groups,
        core_similarity=args.core_similarity,
        broad_similarity=args.broad_similarity,
        attachment_similarity=args.attachment_similarity,
        maximum_primary_gap=args.maximum_primary_gap,
        maximum_secondary_families=args.maximum_secondary_families,
        minimum_core_children=args.minimum_core_children,
    )
    child_features, grid, summaries, memberships, centroids = outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)
    child_features.to_csv(args.output_dir / "01_subissue_features.csv", index=False)
    grid.to_csv(args.output_dir / "02_attachment_sweep.csv", index=False)
    summaries.to_csv(args.output_dir / "03_expanded_family_summary.csv", index=False)
    memberships.to_csv(args.output_dir / "04_overlapping_memberships.csv", index=False)
    np.save(args.output_dir / "05_family_core_centroids.npy", centroids)
    (args.output_dir / "review_packet.md").write_text(
        review_markdown(summaries, memberships, limit=args.review_family_limit),
        encoding="utf-8",
    )
    secondary = memberships[memberships["membership_type"].ne("core")]
    child_parent_counts = memberships.groupby("child_group_id")["family_id"].nunique()
    manifest = {
        "expansion_version": EXPANSION_VERSION,
        "core_similarity": args.core_similarity,
        "broad_similarity": args.broad_similarity,
        "attachment_similarity": args.attachment_similarity,
        "maximum_primary_gap": args.maximum_primary_gap,
        "maximum_secondary_families": args.maximum_secondary_families,
        "minimum_core_children": args.minimum_core_children,
        "input_recurring_child_groups": len(child_features),
        "family_cores": len(summaries),
        "core_memberships": int(memberships["membership_type"].eq("core").sum()),
        "secondary_memberships": int(
            memberships["membership_type"].eq("secondary").sum()
        ),
        "attached_singletons": int(
            memberships["membership_type"].eq("attached_singleton").sum()
        ),
        "children_with_multiple_families": int(child_parent_counts.gt(1).sum()),
        "cross_broad_attachments": int(
            (~secondary["same_broad_cluster"].astype(bool)).sum()
        ),
        "families_requiring_expansion_review": int(
            summaries["expansion_review_required"].astype(bool).sum()
        ),
        "interpretation": (
            "Strict family cores remain unchanged. Secondary memberships are "
            "diagnostic proposals supported by cosine proximity plus either a "
            "shared broader cluster or shared theme and process stage."
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
