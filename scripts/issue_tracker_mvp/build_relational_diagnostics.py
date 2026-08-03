#!/usr/bin/env python3
"""Build deterministic review queues for relational issue-linkage errors.

The queues are diagnostic evidence, not automatic merge or exclusion decisions.
They use artifacts already produced by the relational pipeline and make no LLM
calls.
"""

from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import json
import math
from pathlib import Path
from typing import Any

import pandas as pd

import refine_relational_groups as refinement
import run_relational_linkage_experiment as linkage


DIAGNOSTIC_VERSION = "relational-diagnostics-v1"


def require_columns(frame: pd.DataFrame, columns: set[str], label: str) -> None:
    missing = sorted(columns - set(frame.columns))
    if missing:
        raise ValueError(f"{label} lacks required columns: {', '.join(missing)}")


def load_csv(path: Path) -> pd.DataFrame:
    return pd.read_csv(path).fillna("")


def ordered_group_pair(left: str, right: str) -> tuple[str, str]:
    return (left, right) if left < right else (right, left)


def conflict_counts(
    left: pd.DataFrame,
    right: pd.DataFrame,
) -> Counter[str]:
    counts: Counter[str] = Counter()
    for _, left_row in left.iterrows():
        for _, right_row in right.iterrows():
            reason = linkage.guarded_relation_conflict(left_row, right_row)
            if reason:
                counts[reason] += 1
    return counts


def candidate_veto_counts(frame: pd.DataFrame) -> Counter[str]:
    counts: Counter[str] = Counter()
    for reasons in frame.get("compatibility_reasons", pd.Series(dtype=str)):
        for item in str(reasons).split(" | "):
            if item.startswith("guarded_veto:"):
                counts[item.removeprefix("guarded_veto:")] += 1
    return counts


def _edge_group_columns(
    edges: pd.DataFrame,
    issue_to_group: dict[str, str],
) -> pd.DataFrame:
    result = edges.copy()
    result["left_group"] = result["left_issue_id"].map(issue_to_group).fillna("")
    result["right_group"] = result["right_issue_id"].map(issue_to_group).fillna("")
    result = result[
        result["left_group"].ne("")
        & result["right_group"].ne("")
        & result["left_group"].ne(result["right_group"])
    ].copy()
    result["group_a"] = result.apply(
        lambda row: ordered_group_pair(row["left_group"], row["right_group"])[0],
        axis=1,
    )
    result["group_b"] = result.apply(
        lambda row: ordered_group_pair(row["left_group"], row["right_group"])[1],
        axis=1,
    )
    result["issue_a"] = result.apply(
        lambda row: (
            row["left_issue_id"]
            if row["left_group"] == row["group_a"]
            else row["right_issue_id"]
        ),
        axis=1,
    )
    result["issue_b"] = result.apply(
        lambda row: (
            row["right_issue_id"]
            if row["right_group"] == row["group_b"]
            else row["left_issue_id"]
        ),
        axis=1,
    )
    return result


def build_duplicate_group_queue(
    occurrences: pd.DataFrame,
    assignments: pd.DataFrame,
    groups: pd.DataFrame,
    candidates: pd.DataFrame,
    accepted_edges: pd.DataFrame,
    reports: pd.DataFrame | None = None,
    *,
    minimum_cross_edges: int = 2,
    consolidation_coverage: float = 0.65,
) -> pd.DataFrame:
    """Rank separated recurring groups with strong cross-group edge evidence."""
    recurring = groups[groups["recurrence_status"].eq("recurring")].copy()
    recurring_ids = set(recurring["relational_group_id"].astype(str))
    recurring_assignments = assignments[
        assignments["relational_group_id"].isin(recurring_ids)
    ].copy()
    issue_to_group = dict(
        zip(
            recurring_assignments["issue_id"].astype(str),
            recurring_assignments["relational_group_id"].astype(str),
        )
    )
    cross_candidates = _edge_group_columns(candidates, issue_to_group)
    cross_accepted = _edge_group_columns(accepted_edges, issue_to_group)
    if cross_accepted.empty:
        return pd.DataFrame()

    metadata = recurring.set_index("relational_group_id").to_dict("index")
    members = recurring_assignments.merge(
        occurrences,
        on=["issue_id", "report_key"],
        how="left",
        suffixes=("", "_occurrence"),
    )
    if reports is not None and not reports.empty:
        report_columns = [
            column
            for column in ["report_key", "report_id", "report_url", "report_date"]
            if column in reports
        ]
        members = members.merge(reports[report_columns], on="report_key", how="left")
    members_by_issue = members.set_index("issue_id", drop=False)
    members = members.set_index("relational_group_id", drop=False)
    rows: list[dict[str, Any]] = []
    for (group_a, group_b), edges in cross_accepted.groupby(
        ["group_a", "group_b"], sort=False
    ):
        if len(edges) < minimum_cross_edges:
            continue
        candidate_local = cross_candidates[
            cross_candidates["group_a"].eq(group_a)
            & cross_candidates["group_b"].eq(group_b)
        ]
        member_a = members.loc[[group_a]]
        member_b = members.loc[[group_b]]
        supported_a = edges["issue_a"].nunique()
        supported_b = edges["issue_b"].nunique()
        size_a = len(member_a)
        size_b = len(member_b)
        coverage_a = supported_a / max(1, size_a)
        coverage_b = supported_b / max(1, size_b)
        smaller_coverage = coverage_a if size_a <= size_b else coverage_b
        conflicts = conflict_counts(member_a, member_b)
        candidate_vetoes = candidate_veto_counts(candidate_local)
        if smaller_coverage < consolidation_coverage:
            blocked_by = "insufficient_cross_edge_coverage"
        elif conflicts:
            blocked_by = "all_pairs_conflict_veto"
        else:
            blocked_by = "medoid_score_or_merge_order_not_observable"
        median_score = float(edges["adjusted_similarity"].median())
        distinctive_supported_a: set[str] = set()
        distinctive_supported_b: set[str] = set()
        high_overlap_supported_a: set[str] = set()
        high_overlap_supported_b: set[str] = set()
        for edge_row in edges.itertuples(index=False):
            tokens_a = linkage.consolidation_object_tokens(
                members_by_issue.loc[edge_row.issue_a].get("issue_object", "")
            )
            tokens_b = linkage.consolidation_object_tokens(
                members_by_issue.loc[edge_row.issue_b].get("issue_object", "")
            )
            intersection = tokens_a & tokens_b
            union = tokens_a | tokens_b
            if intersection:
                distinctive_supported_a.add(str(edge_row.issue_a))
                distinctive_supported_b.add(str(edge_row.issue_b))
            if intersection and len(intersection) / len(union) >= 0.50:
                high_overlap_supported_a.add(str(edge_row.issue_a))
                high_overlap_supported_b.add(str(edge_row.issue_b))
        representative_edge = edges.sort_values(
            "adjusted_similarity", ascending=False
        ).iloc[0]
        representative_a = members_by_issue.loc[representative_edge["issue_a"]]
        representative_b = members_by_issue.loc[representative_edge["issue_b"]]
        evidence_score = (
            smaller_coverage
            * median_score
            * (1.0 + min(math.log1p(len(edges)) / 10.0, 0.5))
        )
        rows.append(
            {
                "group_a": group_a,
                "group_b": group_b,
                "group_a_label": metadata[group_a]["prototype_canonical_issue"],
                "group_b_label": metadata[group_b]["prototype_canonical_issue"],
                "group_a_occurrences": size_a,
                "group_b_occurrences": size_b,
                "group_a_reports": metadata[group_a]["report_count"],
                "group_b_reports": metadata[group_b]["report_count"],
                "candidate_pair_count": len(candidate_local),
                "accepted_cross_edges": len(edges),
                "supported_members_a": supported_a,
                "supported_members_b": supported_b,
                "member_coverage_a": coverage_a,
                "member_coverage_b": coverage_b,
                "smaller_group_coverage": smaller_coverage,
                "maximum_cross_edge_score": float(edges["adjusted_similarity"].max()),
                "median_cross_edge_score": median_score,
                "median_canonical_similarity": float(
                    edges.get("canonical_similarity", pd.Series([0.0])).median()
                ),
                "median_relation_similarity": float(
                    edges.get("relation_similarity", pd.Series([0.0])).median()
                ),
                "median_action_object_similarity": float(
                    edges.get(
                        "action_object_similarity", pd.Series([0.0])
                    ).median()
                ),
                "minimum_action_object_similarity": float(
                    edges.get(
                        "action_object_similarity", pd.Series([0.0])
                    ).min()
                ),
                "distinctive_object_coverage_a": (
                    len(distinctive_supported_a) / max(1, size_a)
                ),
                "distinctive_object_coverage_b": (
                    len(distinctive_supported_b) / max(1, size_b)
                ),
                "high_overlap_object_coverage_a": (
                    len(high_overlap_supported_a) / max(1, size_a)
                ),
                "high_overlap_object_coverage_b": (
                    len(high_overlap_supported_b) / max(1, size_b)
                ),
                "representative_issue_id_a": representative_edge["issue_a"],
                "representative_issue_id_b": representative_edge["issue_b"],
                "representative_issue_a": representative_a.get("canonical_issue", ""),
                "representative_issue_b": representative_b.get("canonical_issue", ""),
                "representative_report_url_a": representative_a.get("report_url", ""),
                "representative_report_url_b": representative_b.get("report_url", ""),
                "all_pair_conflict_count": sum(conflicts.values()),
                "all_pair_conflict_fraction": (
                    sum(conflicts.values()) / max(1, size_a * size_b)
                ),
                "all_pair_conflict_reasons": json.dumps(dict(conflicts), sort_keys=True),
                "candidate_veto_reasons": json.dumps(
                    dict(candidate_vetoes), sort_keys=True
                ),
                "likely_consolidation_blocker": blocked_by,
                "diagnostic_score": evidence_score,
                "review_decision": "",
                "review_notes": "",
            }
        )
    result = pd.DataFrame(rows)
    if result.empty:
        return result
    return result.sort_values(
        ["diagnostic_score", "accepted_cross_edges", "group_a", "group_b"],
        ascending=[False, False, True, True],
    ).reset_index(drop=True)


def articulation_points(nodes: set[str], adjacency: dict[str, set[str]]) -> set[str]:
    """Return articulation points for one undirected graph."""
    discovery: dict[str, int] = {}
    low: dict[str, int] = {}
    parent: dict[str, str | None] = {}
    found: set[str] = set()
    clock = 0

    def visit(node: str) -> None:
        nonlocal clock
        discovery[node] = low[node] = clock
        clock += 1
        children = 0
        for neighbour in adjacency.get(node, set()) & nodes:
            if neighbour not in discovery:
                parent[neighbour] = node
                children += 1
                visit(neighbour)
                low[node] = min(low[node], low[neighbour])
                if parent.get(node) is None and children > 1:
                    found.add(node)
                if parent.get(node) is not None and low[neighbour] >= discovery[node]:
                    found.add(node)
            elif neighbour != parent.get(node):
                low[node] = min(low[node], discovery[neighbour])

    for node in sorted(nodes):
        if node not in discovery:
            parent[node] = None
            visit(node)
    return found


def build_contamination_queues(
    occurrences: pd.DataFrame,
    assignments: pd.DataFrame,
    groups: pd.DataFrame,
    accepted_edges: pd.DataFrame,
    reports: pd.DataFrame | None = None,
    *,
    low_prototype_similarity: float = 0.82,
    sparse_support_fraction: float = 0.20,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Rank possible outlier/bridge members and the groups containing them."""
    recurring_groups = groups[groups["recurrence_status"].eq("recurring")].copy()
    recurring_ids = set(recurring_groups["relational_group_id"].astype(str))
    members = assignments[
        assignments["relational_group_id"].isin(recurring_ids)
    ].merge(occurrences, on=["issue_id", "report_key"], how="left")
    if reports is not None and not reports.empty:
        report_columns = [
            column
            for column in ["report_key", "report_id", "report_url", "report_date"]
            if column in reports
        ]
        members = members.merge(reports[report_columns], on="report_key", how="left")

    issue_to_group = dict(zip(members["issue_id"], members["relational_group_id"]))
    internal_adjacency: dict[str, set[str]] = defaultdict(set)
    internal_scores: dict[str, list[float]] = defaultdict(list)
    external_scores: dict[str, list[tuple[float, str, str]]] = defaultdict(list)
    for edge in accepted_edges.itertuples(index=False):
        left_group = issue_to_group.get(str(edge.left_issue_id), "")
        right_group = issue_to_group.get(str(edge.right_issue_id), "")
        if not left_group or not right_group:
            continue
        score = float(edge.adjusted_similarity)
        if left_group == right_group:
            internal_adjacency[str(edge.left_issue_id)].add(str(edge.right_issue_id))
            internal_adjacency[str(edge.right_issue_id)].add(str(edge.left_issue_id))
            internal_scores[str(edge.left_issue_id)].append(score)
            internal_scores[str(edge.right_issue_id)].append(score)
        else:
            external_scores[str(edge.left_issue_id)].append(
                (score, right_group, str(edge.right_issue_id))
            )
            external_scores[str(edge.right_issue_id)].append(
                (score, left_group, str(edge.left_issue_id))
            )

    group_metadata = recurring_groups.set_index("relational_group_id").to_dict("index")
    occurrence_by_issue = members.set_index("issue_id", drop=False)
    member_rows: list[dict[str, Any]] = []
    group_rows: list[dict[str, Any]] = []
    for group_id, local in members.groupby("relational_group_id", sort=False):
        node_ids = set(local["issue_id"].astype(str))
        articulation = articulation_points(node_ids, internal_adjacency)
        prototype_id = str(group_metadata[group_id]["prototype_issue_id"])
        prototype_row = occurrence_by_issue.loc[prototype_id]
        prototype_tokens = refinement.refinement_object_tokens(
            prototype_row.get("issue_object", "")
        )
        conflicts_by_member: Counter[str] = Counter()
        local_records = list(local.iterrows())
        for position, (_, left) in enumerate(local_records):
            for _, right in local_records[position + 1 :]:
                if linkage.guarded_relation_conflict(left, right):
                    conflicts_by_member[str(left["issue_id"])] += 1
                    conflicts_by_member[str(right["issue_id"])] += 1

        local_member_rows: list[dict[str, Any]] = []
        size = len(local)
        for _, member in local.iterrows():
            issue_id = str(member["issue_id"])
            similarity = float(member.get("prototype_similarity", 0.0))
            degree = len(internal_adjacency.get(issue_id, set()) & node_ids)
            support_fraction = degree / max(1, size - 1)
            best_internal = max(internal_scores.get(issue_id, [-1.0]))
            external = external_scores.get(issue_id, [])
            best_external = max(external, default=(-1.0, "", ""))
            object_tokens = refinement.refinement_object_tokens(
                member.get("issue_object", "")
            )
            reasons: list[str] = []
            if conflicts_by_member[issue_id]:
                reasons.append("conflicts_with_group_member")
            if issue_id in articulation:
                reasons.append("accepted_edge_bridge")
            if refinement.is_compound_relation(member):
                reasons.append("compound_relation")
            if similarity < low_prototype_similarity:
                reasons.append("low_prototype_similarity")
            if size >= 5 and support_fraction < sparse_support_fraction:
                reasons.append("sparse_internal_edge_support")
            if (
                prototype_tokens
                and object_tokens
                and not (prototype_tokens & object_tokens)
            ):
                reasons.append("distinct_object_vocabulary")
            if best_external[0] > best_internal + 0.02:
                reasons.append("stronger_external_group_edge")
            weights = {
                "conflicts_with_group_member": 4.0,
                "accepted_edge_bridge": 2.0,
                "compound_relation": 2.0,
                "low_prototype_similarity": 1.0,
                "sparse_internal_edge_support": 1.0,
                "distinct_object_vocabulary": 1.0,
                "stronger_external_group_edge": 1.0,
            }
            risk = sum(weights[reason] for reason in reasons)
            row = {
                "relational_group_id": group_id,
                "group_label": group_metadata[group_id]["prototype_canonical_issue"],
                "issue_id": issue_id,
                "report_key": member["report_key"],
                "report_id": member.get("report_id", ""),
                "report_url": member.get("report_url", ""),
                "canonical_issue": member.get("canonical_issue", ""),
                "failed_action": member.get("failed_action", ""),
                "issue_object": member.get("issue_object", ""),
                "failure_state": member.get("failure_state", ""),
                "evidence_quote": member.get("evidence_quote", ""),
                "prototype_similarity": similarity,
                "internal_accepted_degree": degree,
                "internal_support_fraction": support_fraction,
                "best_internal_edge_score": best_internal,
                "best_external_edge_score": best_external[0],
                "best_external_group_id": best_external[1],
                "best_external_issue_id": best_external[2],
                "incompatible_group_members": conflicts_by_member[issue_id],
                "diagnostic_reasons": " | ".join(reasons),
                "diagnostic_score": risk,
                "review_decision": "",
                "review_notes": "",
            }
            local_member_rows.append(row)
            if reasons:
                member_rows.append(row)

        flagged = [row for row in local_member_rows if row["diagnostic_score"] > 0]
        reason_counts = Counter(
            reason
            for row in flagged
            for reason in row["diagnostic_reasons"].split(" | ")
            if reason
        )
        prototype_is_articulation = prototype_id in articulation
        has_low_fit_member = any(
            row["prototype_similarity"] < low_prototype_similarity
            for row in local_member_rows
        )
        compound_bridges = [
            row
            for row in local_member_rows
            if "compound_relation" in row["diagnostic_reasons"]
            and "accepted_edge_bridge" in row["diagnostic_reasons"]
        ]
        compound_distinct = [
            row
            for row in local_member_rows
            if "compound_relation" in row["diagnostic_reasons"]
            and "distinct_object_vocabulary" in row["diagnostic_reasons"]
        ]
        strong_external_low_fit = [
            row
            for row in local_member_rows
            if "stronger_external_group_edge" in row["diagnostic_reasons"]
            and "low_prototype_similarity" in row["diagnostic_reasons"]
        ]
        high_priority_reasons: list[str] = []
        priority_bonus = 0.0
        if conflicts_by_member:
            high_priority_reasons.append("within_group_relational_conflict")
            priority_bonus += 4.0
        if prototype_is_articulation and has_low_fit_member:
            high_priority_reasons.append("prototype_bridge_with_low_fit_member")
            priority_bonus += 4.0
        if compound_bridges:
            high_priority_reasons.append("compound_bridge_member")
            priority_bonus += 4.0
        if len(compound_distinct) >= 2:
            high_priority_reasons.append("multiple_compound_distinct_members")
            priority_bonus += 2.0
        if strong_external_low_fit:
            high_priority_reasons.append("low_fit_member_prefers_external_group")
            priority_bonus += 2.0
        group_score = priority_bonus + (
            max((row["diagnostic_score"] for row in flagged), default=0.0)
            + len(flagged) / max(1, size)
            + max(0.0, 0.84 - float(local["prototype_similarity"].min())) * 10.0
        )
        if flagged:
            group_rows.append(
                {
                    "relational_group_id": group_id,
                    "group_label": group_metadata[group_id]["prototype_canonical_issue"],
                    "occurrence_count": size,
                    "report_count": group_metadata[group_id]["report_count"],
                    "flagged_member_count": len(flagged),
                    "flagged_member_fraction": len(flagged) / max(1, size),
                    "minimum_prototype_similarity": float(
                        local["prototype_similarity"].min()
                    ),
                    "accepted_edge_count": sum(
                        len(internal_adjacency.get(issue_id, set()) & node_ids)
                        for issue_id in node_ids
                    )
                    // 2,
                    "articulation_member_count": len(articulation),
                    "prototype_is_articulation": prototype_is_articulation,
                    "high_priority_reasons": " | ".join(high_priority_reasons),
                    "diagnostic_reason_counts": json.dumps(
                        dict(reason_counts), sort_keys=True
                    ),
                    "diagnostic_score": group_score,
                    "review_decision": "",
                    "incorrect_member_issue_ids": "",
                    "split_description": "",
                    "review_notes": "",
                }
            )

    member_queue = pd.DataFrame(member_rows)
    group_queue = pd.DataFrame(group_rows)
    if not member_queue.empty:
        member_queue = member_queue.sort_values(
            ["diagnostic_score", "prototype_similarity", "relational_group_id"],
            ascending=[False, True, True],
        ).reset_index(drop=True)
    if not group_queue.empty:
        group_queue = group_queue.sort_values(
            ["diagnostic_score", "flagged_member_fraction", "relational_group_id"],
            ascending=[False, False, True],
        ).reset_index(drop=True)
    return group_queue, member_queue


def write_review_packet(
    path: Path,
    duplicates: pd.DataFrame,
    contaminated: pd.DataFrame,
    members: pd.DataFrame,
    *,
    limit: int = 50,
) -> None:
    lines = [
        "# Relational linkage diagnostic review",
        "",
        "These are deterministic review leads, not automatic corrections.",
        "",
        "## Possible duplicate recurring issues",
        "",
    ]
    for row in duplicates.head(limit).itertuples(index=False):
        lines.extend(
            [
                f"### {row.group_a} ↔ {row.group_b}",
                "",
                f"- A: {row.group_a_label}",
                f"- B: {row.group_b_label}",
                (
                    f"- Evidence: {row.accepted_cross_edges} accepted cross-edges; "
                    f"smaller-group coverage {row.smaller_group_coverage:.1%}; "
                    f"median score {row.median_cross_edge_score:.3f}"
                ),
                f"- Current blocker: {row.likely_consolidation_blocker}",
                f"- Conflict reasons: {row.all_pair_conflict_reasons}",
                f"- Example A: {row.representative_issue_a}",
                f"- Example B: {row.representative_issue_b}",
                "",
            ]
        )
    lines.extend(["## Possible contaminated recurring groups", ""])
    for group in contaminated.head(limit).itertuples(index=False):
        lines.extend(
            [
                f"### {group.relational_group_id}",
                "",
                f"Prototype: {group.group_label}",
                "",
                (
                    f"Flagged members: {group.flagged_member_count}/"
                    f"{group.occurrence_count}; reasons: "
                    f"{group.diagnostic_reason_counts}"
                ),
                "",
            ]
        )
        local = members[
            members["relational_group_id"].eq(group.relational_group_id)
        ].head(10)
        for member in local.itertuples(index=False):
            lines.append(
                f"- `{member.issue_id}` ({member.diagnostic_reasons}): "
                f"{member.canonical_issue}"
            )
        lines.append("")
    path.write_text("\n".join(lines), encoding="utf-8")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--artifact-dir", required=True, type=Path)
    parser.add_argument("--reports-csv", type=Path)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--minimum-cross-edges", type=int, default=2)
    parser.add_argument("--consolidation-coverage", type=float, default=0.65)
    parser.add_argument("--low-prototype-similarity", type=float, default=0.82)
    parser.add_argument("--sparse-support-fraction", type=float, default=0.20)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    artifact_dir = args.artifact_dir
    output_dir = args.output_dir or artifact_dir / "diagnostics"
    occurrences = load_csv(artifact_dir / "01_linkage_quality_gate.csv")
    assignments = load_csv(artifact_dir / "04_group_assignments.csv")
    groups = load_csv(artifact_dir / "05_relational_groups.csv")
    candidates = load_csv(artifact_dir / "02_candidate_pairs.csv")
    accepted_edges = load_csv(artifact_dir / "03_accepted_edges.csv")
    reports = load_csv(args.reports_csv) if args.reports_csv else None

    require_columns(
        occurrences,
        {"issue_id", "report_key", "canonical_issue", "failed_action", "issue_object"},
        "occurrences",
    )
    require_columns(
        assignments,
        {"issue_id", "report_key", "relational_group_id", "prototype_similarity"},
        "assignments",
    )
    require_columns(
        groups,
        {
            "relational_group_id",
            "recurrence_status",
            "prototype_issue_id",
            "prototype_canonical_issue",
            "report_count",
        },
        "groups",
    )
    require_columns(
        candidates,
        {"left_issue_id", "right_issue_id", "adjusted_similarity"},
        "candidate pairs",
    )
    require_columns(
        accepted_edges,
        {"left_issue_id", "right_issue_id", "adjusted_similarity"},
        "accepted edges",
    )

    duplicates = build_duplicate_group_queue(
        occurrences,
        assignments,
        groups,
        candidates,
        accepted_edges,
        reports,
        minimum_cross_edges=args.minimum_cross_edges,
        consolidation_coverage=args.consolidation_coverage,
    )
    contaminated, member_risk = build_contamination_queues(
        occurrences,
        assignments,
        groups,
        accepted_edges,
        reports,
        low_prototype_similarity=args.low_prototype_similarity,
        sparse_support_fraction=args.sparse_support_fraction,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    duplicates.to_csv(output_dir / "possible_duplicate_groups.csv", index=False)
    contaminated.to_csv(output_dir / "possible_contaminated_groups.csv", index=False)
    member_risk.to_csv(output_dir / "possible_contaminated_members.csv", index=False)
    write_review_packet(
        output_dir / "review_packet.md", duplicates, contaminated, member_risk
    )
    manifest = {
        "diagnostic_version": DIAGNOSTIC_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "artifact_dir": str(artifact_dir),
        "reports_csv": str(args.reports_csv) if args.reports_csv else "",
        "parameters": {
            "minimum_cross_edges": args.minimum_cross_edges,
            "consolidation_coverage": args.consolidation_coverage,
            "low_prototype_similarity": args.low_prototype_similarity,
            "sparse_support_fraction": args.sparse_support_fraction,
        },
        "outputs": {
            "possible_duplicate_groups": len(duplicates),
            "possible_contaminated_groups": len(contaminated),
            "possible_contaminated_members": len(member_risk),
        },
        "limitations": [
            "Queues are prioritisation aids and do not constitute validation decisions.",
            "Historical artifacts do not persist directed retrieval ranks.",
            "Pairs absent from the candidate table cannot be assigned an exact retrieval-stage rejection reason.",
        ],
    }
    (output_dir / "diagnostic_manifest.json").write_text(
        json.dumps(manifest, indent=2), encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
