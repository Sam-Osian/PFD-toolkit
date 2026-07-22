#!/usr/bin/env python3
"""Build reproducible coherence and missed-link audit queues for an issue index."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors


FACETS = (
    "failure_state",
    "process_stage",
    "communication_direction",
    "responsible_actor_role",
    "service_sectors",
    "issue_themes",
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build stratified review queues from an existing issue-index run."
    )
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument("--group-review-size", type=int, default=80)
    parser.add_argument("--boundary-sample-size", type=int, default=20)
    parser.add_argument("--facet-risk-sample-size", type=int, default=20)
    parser.add_argument("--large-report-count", type=int, default=8)
    parser.add_argument("--missed-link-review-size", type=int, default=70)
    parser.add_argument("--missed-link-min-similarity", type=float, default=0.78)
    parser.add_argument(
        "--exclude-group-ids-csv",
        type=Path,
        default=None,
        help="Optional CSV containing subissue_id values to exclude from group sampling.",
    )
    parser.add_argument(
        "--exclude-group-members-csv",
        type=Path,
        default=None,
        help=(
            "Optional CSV containing issue_id values; any group containing one of "
            "those reviewed members is excluded from sampling."
        ),
    )
    parser.add_argument("--seed", type=int, default=20260722)
    return parser.parse_args()


def group_diagnostics(indexed: pd.DataFrame, subissues: pd.DataFrame) -> pd.DataFrame:
    recurring = subissues[subissues["recurrence_status"].eq("recurring")].copy()
    members = indexed[indexed["recurrence_status"].eq("recurring")].copy()
    rows: list[dict[str, object]] = []
    for group in recurring.to_dict("records"):
        group_id = group["subissue_id"]
        subset = members[members["subissue_id"].eq(group_id)]
        row = dict(group)
        for facet in FACETS:
            row[f"unique_{facet}"] = int(subset[facet].nunique())
        row["facet_risk_score"] = sum(
            max(0, int(row[f"unique_{facet}"]) - 1) for facet in FACETS
        )
        rows.append(row)
    return pd.DataFrame(rows)


def select_group_review(
    groups: pd.DataFrame,
    *,
    review_size: int,
    boundary_size: int,
    risk_size: int,
    large_report_count: int,
    seed: int,
) -> pd.DataFrame:
    if groups.empty or review_size <= 0:
        return groups.iloc[0:0].copy()
    selected: dict[str, dict[str, object]] = {}
    reasons: dict[str, set[str]] = {}

    def add(frame: pd.DataFrame, reason: str) -> None:
        for row in frame.to_dict("records"):
            group_id = str(row["subissue_id"])
            selected[group_id] = row
            reasons.setdefault(group_id, set()).add(reason)

    add(
        groups[groups["report_count"].ge(large_report_count)].sort_values(
            ["report_count", "issue_count", "subissue_id"],
            ascending=[False, False, True],
        ),
        "large_group",
    )
    remaining = groups[~groups["subissue_id"].isin(selected)]
    boundary = remaining[remaining["report_count"].eq(3)]
    if len(boundary) > boundary_size:
        boundary = boundary.sample(boundary_size, random_state=seed)
    add(boundary.sort_values("subissue_id"), "recurrence_boundary")
    remaining = groups[~groups["subissue_id"].isin(selected)]
    add(
        remaining.sort_values(
            ["facet_risk_score", "median_centroid_similarity", "subissue_id"],
            ascending=[False, True, True],
        ).head(risk_size),
        "facet_conflict_risk",
    )
    remaining = groups[~groups["subissue_id"].isin(selected)]
    slots = max(0, review_size - len(selected))
    if slots:
        fill = remaining.sample(min(slots, len(remaining)), random_state=seed + 1)
        add(fill.sort_values("subissue_id"), "random_fill")
    queue = pd.DataFrame(selected.values()).head(review_size).copy()
    queue.insert(
        3,
        "review_reason",
        queue["subissue_id"].map(lambda value: " | ".join(sorted(reasons[str(value)]))),
    )
    queue["audit_decision"] = ""
    queue["incorrect_member_issue_ids"] = ""
    queue["proposed_split"] = ""
    queue["audit_notes"] = ""
    return queue


def build_missed_link_candidates(
    indexed: pd.DataFrame,
    embeddings: np.ndarray,
    *,
    minimum_similarity: float,
    review_size: int,
) -> pd.DataFrame:
    if len(indexed) != len(embeddings):
        raise ValueError("Embedding rows do not match indexed occurrence rows")
    eligible = ~indexed["recurrence_status"].eq("recurring")
    positions = np.flatnonzero(eligible.to_numpy())
    if not len(positions) or review_size <= 0:
        return pd.DataFrame()
    neighbours = min(16, len(indexed))
    model = NearestNeighbors(n_neighbors=neighbours, metric="cosine", n_jobs=-1)
    model.fit(embeddings)
    distances, indices = model.kneighbors(embeddings[positions])
    pairs: dict[tuple[str, str], dict[str, object]] = {}
    for source_position, row_distances, row_indices in zip(
        positions, distances, indices, strict=False
    ):
        source = indexed.iloc[source_position]
        for distance, target_position in zip(row_distances, row_indices, strict=False):
            target_position = int(target_position)
            target = indexed.iloc[target_position]
            similarity = 1.0 - float(distance)
            if (
                target_position == source_position
                or source["report_key"] == target["report_key"]
                or target["recurrence_status"] == "recurring"
                or similarity < minimum_similarity
            ):
                continue
            pair_key = tuple(sorted((str(source["issue_id"]), str(target["issue_id"]))))
            if pair_key in pairs:
                break
            left, right = (source, target)
            if str(left["issue_id"]) > str(right["issue_id"]):
                left, right = right, left
            pairs[pair_key] = {
                "left_issue_id": left["issue_id"],
                "right_issue_id": right["issue_id"],
                "cosine_similarity": similarity,
                "left_recurrence_status": left["recurrence_status"],
                "right_recurrence_status": right["recurrence_status"],
                "left_subissue_id": left["subissue_id"],
                "right_subissue_id": right["subissue_id"],
                "left_report_key": left["report_key"],
                "right_report_key": right["report_key"],
                "left_issue": left["canonical_issue"],
                "right_issue": right["canonical_issue"],
                "left_actor": left["responsible_actor_role"],
                "right_actor": right["responsible_actor_role"],
                "left_object": left["issue_object"],
                "right_object": right["issue_object"],
                "left_failure_state": left["failure_state"],
                "right_failure_state": right["failure_state"],
                "left_process_stage": left["process_stage"],
                "right_process_stage": right["process_stage"],
                "left_direction": left["communication_direction"],
                "right_direction": right["communication_direction"],
            }
            break
    candidates = pd.DataFrame(pairs.values())
    if candidates.empty:
        return candidates
    candidates = candidates.sort_values(
        ["cosine_similarity", "left_issue_id", "right_issue_id"],
        ascending=[False, True, True],
    )
    strata = (
        (
            "above_edge_threshold",
            candidates[candidates["cosine_similarity"].ge(0.84)],
            25,
        ),
        (
            "near_edge_threshold",
            candidates[
                candidates["cosine_similarity"].between(0.82, 0.84, inclusive="left")
            ],
            25,
        ),
        (
            "lower_similarity_control",
            candidates[candidates["cosine_similarity"].lt(0.82)],
            20,
        ),
    )
    selected: list[pd.DataFrame] = []
    seen: set[tuple[str, str]] = set()
    for reason, frame, size in strata:
        chosen = frame.head(size).copy()
        chosen.insert(0, "review_reason", reason)
        selected.append(chosen)
        seen.update(
            zip(chosen["left_issue_id"], chosen["right_issue_id"], strict=False)
        )
    queue = pd.concat(selected, ignore_index=True) if selected else candidates.iloc[0:0]
    if len(queue) < review_size:
        remainder = (
            candidates[
                ~candidates.apply(
                    lambda row: (row["left_issue_id"], row["right_issue_id"]) in seen,
                    axis=1,
                )
            ]
            .head(review_size - len(queue))
            .copy()
        )
        remainder.insert(0, "review_reason", "similarity_fill")
        queue = pd.concat([queue, remainder], ignore_index=True)
    queue = queue.head(review_size).copy()
    queue["same_recurring_issue"] = ""
    queue["preferred_action"] = ""
    queue["audit_notes"] = ""
    return queue


def write_group_packet(path: Path, queue: pd.DataFrame, members: pd.DataFrame) -> None:
    lines = ["# Recurring issue coherence audit", ""]
    for group in queue.to_dict("records"):
        group_id = group["subissue_id"]
        lines.extend(
            [
                f"## {group_id}",
                "",
                (
                    f"Reason: {group['review_reason']}; reports: {group['report_count']}; "
                    f"issues: {group['issue_count']}; cohesion: "
                    f"{float(group['median_centroid_similarity']):.4f}; "
                    f"facet risk: {group['facet_risk_score']}"
                ),
                "",
            ]
        )
        subset = members[members["subissue_id"].eq(group_id)]
        for row in subset.to_dict("records"):
            lines.append(
                f"- `{row['issue_id']}` — {row['canonical_issue']} "
                f"[actor={row['responsible_actor_role']}; object={row['issue_object']}; "
                f"failure={row['failure_state']}; stage={row['process_stage']}; "
                f"direction={row['communication_direction']}]"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def main() -> None:
    args = parse_args()
    run_dir = args.run_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else run_dir / "07_quality_audit"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    indexed = pd.read_csv(run_dir / "03_occurrences_indexed.csv").fillna("")
    subissues = pd.read_csv(run_dir / "03_subissues.csv").fillna("")
    embeddings = np.load(run_dir / "02_issue_embeddings.npy")
    groups = group_diagnostics(indexed, subissues)
    recurring_groups_total = int(len(groups))
    excluded_group_count = 0
    if args.exclude_group_ids_csv:
        exclusions = pd.read_csv(args.exclude_group_ids_csv).fillna("")
        if "subissue_id" not in exclusions.columns:
            raise ValueError("Exclusion CSV must contain a subissue_id column")
        excluded_ids = set(exclusions["subissue_id"].astype(str))
        excluded_group_count = int(groups["subissue_id"].isin(excluded_ids).sum())
        groups = groups[~groups["subissue_id"].isin(excluded_ids)].copy()
    if args.exclude_group_members_csv:
        exclusions = pd.read_csv(args.exclude_group_members_csv).fillna("")
        if "issue_id" not in exclusions.columns:
            raise ValueError("Member exclusion CSV must contain an issue_id column")
        reviewed_issue_ids = set(exclusions["issue_id"].astype(str))
        member_group_ids = set(
            indexed.loc[indexed["issue_id"].isin(reviewed_issue_ids), "subissue_id"]
            .astype(str)
            .tolist()
        )
        newly_excluded = groups["subissue_id"].isin(member_group_ids)
        excluded_group_count += int(newly_excluded.sum())
        groups = groups[~newly_excluded].copy()
    group_queue = select_group_review(
        groups,
        review_size=args.group_review_size,
        boundary_size=args.boundary_sample_size,
        risk_size=args.facet_risk_sample_size,
        large_report_count=args.large_report_count,
        seed=args.seed,
    )
    selected_members = indexed[indexed["subissue_id"].isin(group_queue["subissue_id"])]
    missed_links = build_missed_link_candidates(
        indexed,
        embeddings,
        minimum_similarity=args.missed_link_min_similarity,
        review_size=args.missed_link_review_size,
    )
    group_queue.to_csv(output_dir / "group_review_queue.csv", index=False)
    selected_members.to_csv(output_dir / "group_review_members.csv", index=False)
    missed_links.to_csv(output_dir / "missed_link_review_queue.csv", index=False)
    write_group_packet(
        output_dir / "group_review_packet.md", group_queue, selected_members
    )
    metrics = {
        "recurring_groups_total": recurring_groups_total,
        "groups_eligible_for_sampling": int(len(groups)),
        "groups_excluded_from_sampling": excluded_group_count,
        "groups_selected": int(len(group_queue)),
        "group_review_reason_counts": group_queue["review_reason"]
        .value_counts()
        .to_dict(),
        "group_members_selected": int(len(selected_members)),
        "missed_link_pairs_selected": int(len(missed_links)),
        "missed_link_reason_counts": (
            missed_links["review_reason"].value_counts().to_dict()
            if "review_reason" in missed_links.columns
            else {}
        ),
        "seed": args.seed,
    }
    (output_dir / "audit_manifest.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
