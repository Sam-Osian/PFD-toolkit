#!/usr/bin/env python3
"""Archived: discover broad issue families from precise recurring sub-issues.

This is a model-free diagnostic. It does not alter child-group membership and
does not assign isolated occurrences. Existing occurrence embeddings are
averaged into child-group centroids, then clustered at broader cosine
similarities with average linkage.
"""

from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
from sklearn.cluster import AgglomerativeClustering


DISCOVERY_VERSION = "automatic-issue-family-discovery-v1"
DEFAULT_SIMILARITIES = (0.78, 0.80, 0.82, 0.84, 0.86)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--occurrences-csv", required=True, type=Path)
    parser.add_argument("--embeddings-npy", required=True, type=Path)
    parser.add_argument("--assignments-csv", required=True, type=Path)
    parser.add_argument("--groups-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--similarities",
        default=",".join(map(str, DEFAULT_SIMILARITIES)),
        help="Comma-separated family-level cosine similarities to compare.",
    )
    parser.add_argument("--selected-similarity", type=float, default=0.82)
    parser.add_argument("--minimum-child-groups", type=int, default=2)
    parser.add_argument("--review-family-limit", type=int, default=40)
    return parser.parse_args()


def parse_similarities(value: str) -> list[float]:
    similarities = sorted(
        {float(item.strip()) for item in value.split(",") if item.strip()}
    )
    if not similarities or any(not 0.0 < item < 1.0 for item in similarities):
        raise ValueError("Similarities must contain values between zero and one")
    return similarities


def unit_normalize(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm <= 1e-12:
        raise ValueError("Cannot normalize a zero vector")
    return np.asarray(vector / norm, dtype=np.float32)


def group_columns(
    assignments: pd.DataFrame, groups: pd.DataFrame
) -> tuple[str, str, str]:
    if "refined_group_id" in assignments and "refined_group_id" in groups:
        return "refined_group_id", "refinement_status", "refined_recurring"
    if "relational_group_id" in assignments and "relational_group_id" in groups:
        return "relational_group_id", "recurrence_status", "recurring"
    raise ValueError("Assignments and groups lack a shared supported group identifier")


def stable_family_id(child_group_ids: Iterable[str]) -> str:
    membership = "\n".join(sorted(map(str, child_group_ids)))
    digest = hashlib.sha256(membership.encode("utf-8")).hexdigest()[:16]
    return f"fam_{digest}"


def split_pipe_values(values: Iterable[Any]) -> list[str]:
    result: list[str] = []
    for value in values:
        result.extend(
            item.strip()
            for item in str(value).split("|")
            if item.strip() and item.strip().casefold() != "nan"
        )
    return result


def dominant_values(values: Iterable[Any], limit: int = 3) -> str:
    counts = Counter(split_pipe_values(values))
    return "|".join(
        item for item, _ in sorted(counts.items(), key=lambda pair: (-pair[1], pair[0]))[:limit]
    )


def build_child_features(
    occurrences: pd.DataFrame,
    embeddings: np.ndarray,
    assignments: pd.DataFrame,
    groups: pd.DataFrame,
) -> tuple[pd.DataFrame, np.ndarray, pd.DataFrame]:
    if len(occurrences) != len(embeddings):
        raise ValueError("Occurrence and embedding row counts do not match")
    if occurrences["issue_id"].astype(str).duplicated().any():
        raise ValueError("Occurrence issue IDs must be unique")
    group_id_column, status_column, recurring_value = group_columns(assignments, groups)
    recurring_ids = set(
        groups.loc[
            groups[status_column].astype(str).eq(recurring_value), group_id_column
        ].astype(str)
    )
    membership = assignments[
        assignments[group_id_column].astype(str).isin(recurring_ids)
    ].copy()
    positions = pd.Series(
        np.arange(len(occurrences)),
        index=occurrences["issue_id"].astype(str),
    )
    if not set(membership["issue_id"].astype(str)).issubset(set(positions.index)):
        raise ValueError("Some group assignments are absent from the embedding occurrence file")

    occurrence_columns = [
        column
        for column in (
            "issue_id",
            "report_key",
            "canonical_issue",
            "issue_themes",
            "process_stage",
            "service_sectors",
        )
        if column in occurrences
    ]
    members = membership.merge(
        occurrences[occurrence_columns],
        on=[column for column in ("issue_id", "report_key") if column in occurrence_columns],
        how="left",
        validate="one_to_one",
    )
    group_lookup = groups.set_index(group_id_column).to_dict("index")
    feature_rows: list[dict[str, Any]] = []
    centroids: list[np.ndarray] = []
    for child_group_id, local in members.groupby(group_id_column, sort=True):
        indices = positions.loc[local["issue_id"].astype(str)].astype(int).to_numpy()
        centroid = unit_normalize(
            np.asarray(embeddings[indices], dtype=np.float32).mean(axis=0)
        )
        metadata = group_lookup[str(child_group_id)]
        feature_rows.append(
            {
                "child_group_id": str(child_group_id),
                "occurrence_count": len(local),
                "report_count": int(local["report_key"].astype(str).nunique()),
                "prototype_canonical_issue": str(
                    metadata.get("prototype_canonical_issue", "")
                ),
                "dominant_themes": dominant_values(local.get("issue_themes", [])),
                "dominant_process_stages": dominant_values(
                    local.get("process_stage", [])
                ),
                "dominant_sectors": dominant_values(local.get("service_sectors", [])),
            }
        )
        centroids.append(centroid)
    return pd.DataFrame(feature_rows), np.stack(centroids), members.rename(
        columns={group_id_column: "child_group_id"}
    )


def cluster_labels(centroids: np.ndarray, similarity: float) -> np.ndarray:
    if not 0.0 < similarity < 1.0:
        raise ValueError("Similarity must be between zero and one")
    if len(centroids) == 1:
        return np.zeros(1, dtype=int)
    return AgglomerativeClustering(
        n_clusters=None,
        distance_threshold=1.0 - similarity,
        metric="cosine",
        linkage="average",
    ).fit_predict(centroids)


def threshold_metrics(labels: np.ndarray, similarity: float) -> dict[str, Any]:
    sizes = pd.Series(labels).value_counts()
    families = sizes[sizes.ge(2)]
    return {
        "similarity": similarity,
        "families": len(families),
        "grouped_child_groups": int(families.sum()),
        "singleton_child_groups": int(sizes.eq(1).sum()),
        "median_family_children": (
            float(families.median()) if not families.empty else 0.0
        ),
        "p90_family_children": (
            float(families.quantile(0.90)) if not families.empty else 0.0
        ),
        "largest_family_children": int(families.max()) if not families.empty else 0,
    }


def build_family_outputs(
    child_features: pd.DataFrame,
    centroids: np.ndarray,
    members: pd.DataFrame,
    *,
    similarity: float,
    minimum_child_groups: int,
) -> tuple[pd.DataFrame, pd.DataFrame, np.ndarray]:
    labels = cluster_labels(centroids, similarity)
    assignment_rows: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    family_centroid_lookup: dict[str, np.ndarray] = {}
    for cluster_label in sorted(set(labels)):
        positions = np.where(labels == cluster_label)[0]
        if len(positions) < minimum_child_groups:
            continue
        children = child_features.iloc[positions].copy()
        child_ids = children["child_group_id"].astype(str).tolist()
        family_id = stable_family_id(child_ids)
        local_centroids = centroids[positions]
        family_centroid = unit_normalize(local_centroids.mean(axis=0))
        centroid_similarities = local_centroids @ family_centroid
        medoid_position = int(np.argmax(centroid_similarities))
        pair_similarities = local_centroids @ local_centroids.T
        triangle = pair_similarities[np.triu_indices(len(positions), 1)]
        family_members = members[
            members["child_group_id"].astype(str).isin(child_ids)
        ]
        family_report_count = int(
            family_members["report_key"].astype(str).nunique()
        )
        largest_child_reports = int(children["report_count"].max())
        ordered_positions = np.argsort(-centroid_similarities)
        examples = children.iloc[ordered_positions[:5]][
            "prototype_canonical_issue"
        ].astype(str)
        questionable = children.iloc[np.argsort(centroid_similarities)[:3]]
        dominant_themes = dominant_values(children["dominant_themes"])
        dominant_stages = dominant_values(children["dominant_process_stages"])
        label_parts = [
            value.replace("_", " ")
            for value in (
                dominant_themes.split("|")[0] if dominant_themes else "",
                dominant_stages.split("|")[0] if dominant_stages else "",
            )
            if value
        ]
        summary_rows.append(
            {
                "family_id": family_id,
                "family_label_hint": " — ".join(label_parts),
                "family_prototype": children.iloc[medoid_position][
                    "prototype_canonical_issue"
                ],
                "child_group_count": len(children),
                "occurrence_count": int(children["occurrence_count"].sum()),
                "report_count": family_report_count,
                "largest_child_report_count": largest_child_reports,
                "report_amplification": (
                    family_report_count / largest_child_reports
                    if largest_child_reports
                    else 0.0
                ),
                "minimum_child_similarity": float(triangle.min()),
                "median_child_similarity": float(np.median(triangle)),
                "dominant_themes": dominant_themes,
                "dominant_process_stages": dominant_stages,
                "dominant_sectors": dominant_values(children["dominant_sectors"]),
                "representative_subissues": " | ".join(examples),
                "questionable_child_groups": "|".join(
                    questionable["child_group_id"].astype(str)
                ),
            }
        )
        for local_position, (_, child) in enumerate(children.iterrows()):
            assignment_rows.append(
                {
                    "family_id": family_id,
                    "child_group_id": child["child_group_id"],
                    "child_report_count": child["report_count"],
                    "child_occurrence_count": child["occurrence_count"],
                    "similarity_to_family_centroid": float(
                        centroid_similarities[local_position]
                    ),
                    "prototype_canonical_issue": child[
                        "prototype_canonical_issue"
                    ],
                    "dominant_themes": child["dominant_themes"],
                    "dominant_process_stages": child["dominant_process_stages"],
                }
            )
        family_centroid_lookup[family_id] = family_centroid
    summaries = pd.DataFrame(summary_rows).sort_values(
        ["report_count", "family_id"], ascending=[False, True]
    ).reset_index(drop=True)
    summaries["centroid_row"] = np.arange(len(summaries))
    assignments_output = pd.DataFrame(assignment_rows)
    if not assignments_output.empty:
        family_order = {
            family_id: ordinal
            for ordinal, family_id in enumerate(summaries["family_id"], start=1)
        }
        assignments_output["family_rank"] = assignments_output["family_id"].map(
            family_order
        )
        assignments_output = assignments_output.sort_values(
            ["family_rank", "similarity_to_family_centroid"],
            ascending=[True, False],
        ).drop(columns="family_rank")
    ordered_family_centroids = np.stack(
        [family_centroid_lookup[family_id] for family_id in summaries["family_id"]]
    )
    return summaries, assignments_output, ordered_family_centroids


def review_markdown(
    summaries: pd.DataFrame,
    assignments: pd.DataFrame,
    *,
    limit: int,
    similarity: float,
) -> str:
    lines = [
        "# Automatic issue-family discovery review",
        "",
        (
            f"Selected child-centroid similarity: {similarity:.3f}. "
            "These are diagnostic families, not validated publication groups."
        ),
        "",
    ]
    for _, family in summaries.head(max(limit, 0)).iterrows():
        family_id = str(family["family_id"])
        children = assignments[assignments["family_id"].eq(family_id)]
        lines.extend(
            [
                f"## {family_id}: {family['family_label_hint']}",
                "",
                f"Prototype: {family['family_prototype']}",
                "",
                (
                    f"Reports: {family['report_count']}; child groups: "
                    f"{family['child_group_count']}; largest child: "
                    f"{family['largest_child_report_count']}; amplification: "
                    f"{family['report_amplification']:.1f}×."
                ),
                "",
            ]
        )
        for _, child in children.iterrows():
            lines.append(
                f"- `{child['child_group_id']}` "
                f"({child['child_report_count']} reports; "
                f"family similarity {child['similarity_to_family_centroid']:.3f}) "
                f"— {child['prototype_canonical_issue']}"
            )
        lines.append("")
    return "\n".join(lines)


def run_discovery(
    occurrences: pd.DataFrame,
    embeddings: np.ndarray,
    assignments: pd.DataFrame,
    groups: pd.DataFrame,
    *,
    similarities: list[float],
    selected_similarity: float,
    minimum_child_groups: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    child_features, centroids, members = build_child_features(
        occurrences, embeddings, assignments, groups
    )
    sweep_rows = [
        threshold_metrics(cluster_labels(centroids, similarity), similarity)
        for similarity in similarities
    ]
    summaries, family_assignments, family_centroids = build_family_outputs(
        child_features,
        centroids,
        members,
        similarity=selected_similarity,
        minimum_child_groups=minimum_child_groups,
    )
    return (
        child_features,
        pd.DataFrame(sweep_rows),
        summaries,
        family_assignments,
        family_centroids,
    )


def main() -> None:
    args = parse_args()
    similarities = parse_similarities(args.similarities)
    if args.selected_similarity not in similarities:
        similarities = sorted({*similarities, args.selected_similarity})
    if args.minimum_child_groups < 2:
        raise ValueError("Minimum child groups must be at least two")
    occurrences = pd.read_csv(args.occurrences_csv).fillna("")
    embeddings = np.load(args.embeddings_npy, mmap_mode="r")
    assignments = pd.read_csv(args.assignments_csv).fillna("")
    groups = pd.read_csv(args.groups_csv).fillna("")
    outputs = run_discovery(
        occurrences,
        embeddings,
        assignments,
        groups,
        similarities=similarities,
        selected_similarity=args.selected_similarity,
        minimum_child_groups=args.minimum_child_groups,
    )
    child_features, sweep, summaries, family_assignments, family_centroids = outputs
    args.output_dir.mkdir(parents=True, exist_ok=True)
    child_features.to_csv(args.output_dir / "01_subissue_features.csv", index=False)
    sweep.to_csv(args.output_dir / "02_threshold_sweep.csv", index=False)
    summaries.to_csv(args.output_dir / "03_family_summary.csv", index=False)
    family_assignments.to_csv(
        args.output_dir / "04_family_child_assignments.csv", index=False
    )
    np.save(args.output_dir / "05_family_centroids.npy", family_centroids)
    (args.output_dir / "review_packet.md").write_text(
        review_markdown(
            summaries,
            family_assignments,
            limit=args.review_family_limit,
            similarity=args.selected_similarity,
        ),
        encoding="utf-8",
    )
    manifest = {
        "discovery_version": DISCOVERY_VERSION,
        "occurrences_csv": str(args.occurrences_csv),
        "embeddings_npy": str(args.embeddings_npy),
        "assignments_csv": str(args.assignments_csv),
        "groups_csv": str(args.groups_csv),
        "selected_similarity": args.selected_similarity,
        "similarities_compared": similarities,
        "minimum_child_groups": args.minimum_child_groups,
        "input_recurring_child_groups": len(child_features),
        "discovered_families": len(summaries),
        "grouped_child_groups": int(summaries["child_group_count"].sum()),
        "ungrouped_child_groups": int(
            len(child_features) - summaries["child_group_count"].sum()
        ),
        "interpretation": (
            "Families are automatic average-linkage groupings of precise recurring "
            "child centroids. Report counts are deduplicated unions. No isolated "
            "occurrences are attached and no family is publication-validated."
        ),
    }
    (args.output_dir / "manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
