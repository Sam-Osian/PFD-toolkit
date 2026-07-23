#!/usr/bin/env python3
"""Create durable issue-type identities from one snapshot of clustered issues."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

import build_issue_index as pipeline


REGISTRY_VERSION = "issue-registry-v1"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--groups-csv", type=Path, default=None)
    parser.add_argument("--assignments-csv", type=Path, default=None)
    parser.add_argument(
        "--registry-dir",
        type=Path,
        default=None,
        help="Defaults to <run-dir>/05_issue_registry.",
    )
    parser.add_argument(
        "--previous-registry-dir",
        type=Path,
        default=None,
        help=(
            "Registry from the previous snapshot. If omitted, an existing --registry-dir "
            "is updated in place."
        ),
    )
    parser.add_argument("--match-containment", type=float, default=0.50)
    parser.add_argument("--minimum-overlap", type=int, default=2)
    return parser.parse_args()


def normalize_snapshot_frames(
    groups: pd.DataFrame, assignments: pd.DataFrame
) -> tuple[pd.DataFrame, pd.DataFrame]:
    groups = groups.copy().fillna("")
    assignments = assignments.copy().fillna("")
    group_id = "subissue_id" if "subissue_id" in groups.columns else "final_group_id"
    assignment_group_id = (
        "subissue_id" if "subissue_id" in assignments.columns else "final_group_id"
    )
    required_groups = {group_id, "recurrence_status", "report_count", "issue_count"}
    required_assignments = {assignment_group_id, "issue_id"}
    if missing := required_groups - set(groups.columns):
        raise ValueError(f"Group CSV is missing columns: {sorted(missing)}")
    if missing := required_assignments - set(assignments.columns):
        raise ValueError(f"Assignment CSV is missing columns: {sorted(missing)}")
    groups = groups.rename(columns={group_id: "cluster_snapshot_id"})
    assignments = assignments.rename(
        columns={assignment_group_id: "cluster_snapshot_id"}
    )
    if groups["cluster_snapshot_id"].duplicated().any():
        raise ValueError("Cluster snapshot IDs must be unique")
    if assignments["issue_id"].duplicated().any():
        raise ValueError("Every occurrence must have at most one cluster assignment")
    unknown = set(assignments["cluster_snapshot_id"]) - set(
        groups["cluster_snapshot_id"]
    )
    if unknown:
        raise ValueError(f"Assignments reference {len(unknown)} unknown clusters")
    if "recurrence_strength" not in groups.columns:
        groups["recurrence_strength"] = groups["report_count"].map(
            lambda value: pipeline.recurrence_strength(int(value))
        )
    return groups, assignments


def load_previous_registry(
    path: Path | None,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    empty_types = pd.DataFrame()
    empty_members = pd.DataFrame(columns=["issue_type_id", "issue_id"])
    empty_lineage = pd.DataFrame()
    if path is None:
        return empty_types, empty_members, empty_lineage
    types_path = path / "issue_types.csv"
    members_path = path / "issue_type_memberships.csv"
    if not types_path.exists() or not members_path.exists():
        return empty_types, empty_members, empty_lineage
    types = pd.read_csv(types_path).fillna("")
    members = pd.read_csv(members_path).fillna("")
    lineage_path = path / "issue_type_lineage.csv"
    lineage = (
        pd.read_csv(lineage_path).fillna("")
        if lineage_path.exists()
        else empty_lineage
    )
    return types, members, lineage


def member_change(previous: set[str], current: set[str]) -> str:
    if previous == current:
        return "unchanged"
    if previous < current:
        return "expanded"
    if current < previous:
        return "contracted"
    return "reconfigured"


def build_registry(
    groups: pd.DataFrame,
    assignments: pd.DataFrame,
    previous_types: pd.DataFrame,
    previous_memberships: pd.DataFrame,
    *,
    match_containment: float,
    minimum_overlap: int,
    generated_at: str,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    groups, assignments = normalize_snapshot_frames(groups, assignments)
    current_members = {
        cluster_id: set(frame["issue_id"].astype(str))
        for cluster_id, frame in assignments.groupby("cluster_snapshot_id")
    }
    all_previous_rows = (
        previous_types.set_index("issue_type_id").to_dict("index")
        if not previous_types.empty
        else {}
    )
    active_previous = (
        previous_types[
            ~previous_types.get(
                "registry_status", pd.Series("", index=previous_types.index)
            ).eq("inactive")
        ]
        if not previous_types.empty
        else previous_types
    )
    previous_rows = (
        active_previous.set_index("issue_type_id").to_dict("index")
        if not active_previous.empty
        else {}
    )
    previous_members = {
        issue_type_id: set(frame["issue_id"].astype(str))
        for issue_type_id, frame in previous_memberships.groupby("issue_type_id")
    }
    previous_types_by_issue: dict[str, set[str]] = {}
    for issue_type_id, old_members in previous_members.items():
        for issue_id in old_members:
            previous_types_by_issue.setdefault(issue_id, set()).add(issue_type_id)
    previous_cluster_types = {
        pipeline.clean_text(row.get("current_cluster_snapshot_id")): issue_type_id
        for issue_type_id, row in previous_rows.items()
        if pipeline.clean_text(row.get("current_cluster_snapshot_id"))
    }
    matches: dict[str, list[dict[str, Any]]] = {}
    for cluster_id, members in current_members.items():
        candidates: list[dict[str, Any]] = []
        overlapping_types = {
            issue_type_id
            for issue_id in members
            for issue_type_id in previous_types_by_issue.get(issue_id, set())
        }
        for issue_type_id in overlapping_types:
            old_members = previous_members[issue_type_id]
            overlap = len(members & old_members)
            exact_snapshot = previous_cluster_types.get(cluster_id) == issue_type_id
            if overlap < minimum_overlap and not exact_snapshot:
                continue
            containment = overlap / max(1, min(len(members), len(old_members)))
            if containment < match_containment and not exact_snapshot:
                continue
            candidates.append(
                {
                    "issue_type_id": issue_type_id,
                    "overlap": overlap,
                    "containment": containment,
                    "jaccard": overlap / max(1, len(members | old_members)),
                }
            )
        matches[cluster_id] = sorted(
            candidates,
            key=lambda row: (
                -row["containment"],
                -row["jaccard"],
                -row["overlap"],
                row["issue_type_id"],
            ),
        )

    group_lookup = groups.set_index("cluster_snapshot_id").to_dict("index")
    claimed_previous: set[str] = set()
    registry_rows: list[dict[str, Any]] = []
    membership_rows: list[dict[str, Any]] = []
    lineage_rows: list[dict[str, Any]] = []
    cluster_to_type: dict[str, str] = {}
    ordered_clusters = groups.sort_values(
        ["report_count", "issue_count", "cluster_snapshot_id"],
        ascending=[False, False, True],
    )["cluster_snapshot_id"].astype(str)
    for cluster_id in ordered_clusters:
        members = current_members.get(cluster_id, set())
        candidates = matches.get(cluster_id, [])
        continuation = next(
            (
                candidate
                for candidate in candidates
                if candidate["issue_type_id"] not in claimed_previous
            ),
            None,
        )
        if continuation:
            issue_type_id = continuation["issue_type_id"]
            claimed_previous.add(issue_type_id)
            previous = previous_rows.get(issue_type_id, {})
            old_members = previous_members.get(issue_type_id, set())
            change = member_change(old_members, members)
            relationship = "continued" if change == "unchanged" else change
            curation_status = pipeline.clean_text(
                previous.get("curation_status")
            ) or "unreviewed"
            publication_status = pipeline.clean_text(
                previous.get("publication_status")
            ) or "not_published"
            if change != "unchanged":
                if curation_status == "human_validated":
                    curation_status = "needs_review"
                if publication_status == "published":
                    publication_status = "review_required"
            created_at = pipeline.clean_text(previous.get("created_at")) or generated_at
            previous_cluster_id = pipeline.clean_text(
                previous.get("current_cluster_snapshot_id")
            )
            if previous_cluster_id != cluster_id or change != "unchanged":
                lineage_rows.append(
                    {
                        "from_issue_type_id": issue_type_id,
                        "to_issue_type_id": issue_type_id,
                        "from_cluster_snapshot_id": previous_cluster_id,
                        "to_cluster_snapshot_id": cluster_id,
                        "relationship": relationship,
                        "overlap_issue_count": continuation["overlap"],
                        "containment": round(continuation["containment"], 6),
                        "created_at": generated_at,
                    }
                )
        else:
            issue_type_id = (
                f"itype_{pipeline.stable_hash(REGISTRY_VERSION, *sorted(members))}"
            )
            created_at = generated_at
            curation_status = "unreviewed"
            publication_status = "not_published"
            split_source = candidates[0] if candidates else None
            lineage_rows.append(
                {
                    "from_issue_type_id": (
                        split_source["issue_type_id"] if split_source else ""
                    ),
                    "to_issue_type_id": issue_type_id,
                    "from_cluster_snapshot_id": pipeline.clean_text(
                        previous_rows.get(
                            split_source["issue_type_id"], {}
                        ).get("current_cluster_snapshot_id")
                    )
                    if split_source
                    else "",
                    "to_cluster_snapshot_id": cluster_id,
                    "relationship": "split_from" if split_source else "created",
                    "overlap_issue_count": split_source["overlap"] if split_source else 0,
                    "containment": (
                        round(split_source["containment"], 6) if split_source else 0.0
                    ),
                    "created_at": generated_at,
                }
            )
        cluster_to_type[cluster_id] = issue_type_id
        group = group_lookup[cluster_id]
        registry_rows.append(
            {
                "issue_type_id": issue_type_id,
                "current_cluster_snapshot_id": cluster_id,
                "registry_status": "active",
                "curation_status": curation_status,
                "publication_status": publication_status,
                "label": pipeline.clean_text(group.get("label")),
                "description": pipeline.clean_text(group.get("description")),
                "recurrence_status": group["recurrence_status"],
                "recurrence_strength": group["recurrence_strength"],
                "report_count": int(group["report_count"]),
                "issue_count": int(group["issue_count"]),
                "created_at": created_at,
                "updated_at": generated_at,
            }
        )
        for issue_id in sorted(members):
            membership_rows.append(
                {
                    "issue_type_id": issue_type_id,
                    "cluster_snapshot_id": cluster_id,
                    "issue_id": issue_id,
                    "snapshot_at": generated_at,
                }
            )
        for merged in candidates:
            if merged["issue_type_id"] == issue_type_id:
                continue
            lineage_rows.append(
                {
                    "from_issue_type_id": merged["issue_type_id"],
                    "to_issue_type_id": issue_type_id,
                    "from_cluster_snapshot_id": pipeline.clean_text(
                        previous_rows.get(merged["issue_type_id"], {}).get(
                            "current_cluster_snapshot_id"
                        )
                    ),
                    "to_cluster_snapshot_id": cluster_id,
                    "relationship": "merged_from",
                    "overlap_issue_count": merged["overlap"],
                    "containment": round(merged["containment"], 6),
                    "created_at": generated_at,
                }
            )

    current_type_ids = {row["issue_type_id"] for row in registry_rows}
    for issue_type_id, previous in all_previous_rows.items():
        if issue_type_id in current_type_ids:
            continue
        registry_rows.append(
            {
                **previous,
                "issue_type_id": issue_type_id,
                "current_cluster_snapshot_id": "",
                "registry_status": "inactive",
                "updated_at": generated_at,
            }
        )
    registry = pd.DataFrame(registry_rows).sort_values(
        ["registry_status", "report_count", "issue_type_id"],
        ascending=[True, False, True],
        ignore_index=True,
    )
    memberships = pd.DataFrame(membership_rows)
    lineage = pd.DataFrame(lineage_rows)
    registered_assignments = assignments.copy()
    registered_assignments.insert(
        1,
        "issue_type_id",
        registered_assignments["cluster_snapshot_id"].map(cluster_to_type),
    )
    return registry, memberships, lineage, registered_assignments


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.match_containment <= 1.0:
        raise ValueError("--match-containment must be between 0 and 1")
    if args.minimum_overlap < 1:
        raise ValueError("--minimum-overlap must be at least 1")
    run_dir = args.run_dir.expanduser().resolve()
    groups_path = args.groups_csv or run_dir / "03_subissues.csv"
    assignments_path = args.assignments_csv or run_dir / "03_issue_assignments.csv"
    registry_dir = (
        args.registry_dir.expanduser().resolve()
        if args.registry_dir
        else run_dir / "05_issue_registry"
    )
    previous_dir = (
        args.previous_registry_dir.expanduser().resolve()
        if args.previous_registry_dir
        else registry_dir
    )
    groups = pd.read_csv(groups_path)
    assignments = pd.read_csv(assignments_path)
    previous_types, previous_memberships, previous_lineage = load_previous_registry(
        previous_dir
    )
    generated_at = datetime.now(timezone.utc).isoformat()
    registry, memberships, lineage, registered = build_registry(
        groups,
        assignments,
        previous_types,
        previous_memberships,
        match_containment=args.match_containment,
        minimum_overlap=args.minimum_overlap,
        generated_at=generated_at,
    )
    if not previous_lineage.empty:
        lineage = pd.concat([previous_lineage, lineage], ignore_index=True).drop_duplicates(
            subset=[
                "from_issue_type_id",
                "to_issue_type_id",
                "from_cluster_snapshot_id",
                "to_cluster_snapshot_id",
                "relationship",
            ],
            keep="first",
        )
    registry_dir.mkdir(parents=True, exist_ok=True)
    registry.to_csv(registry_dir / "issue_types.csv", index=False)
    memberships.to_csv(registry_dir / "issue_type_memberships.csv", index=False)
    lineage.to_csv(registry_dir / "issue_type_lineage.csv", index=False)
    registered.to_csv(registry_dir / "registered_assignments.csv", index=False)
    metrics = {
        "registry_version": REGISTRY_VERSION,
        "active_issue_types": int(registry["registry_status"].eq("active").sum()),
        "inactive_issue_types": int(registry["registry_status"].eq("inactive").sum()),
        "registered_occurrences": len(registered),
        "lineage_events_written": len(lineage),
        "generated_at": generated_at,
    }
    (registry_dir / "registry_metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
