#!/usr/bin/env python3
"""Split broad relational groups into operationally coherent candidate subtypes."""

from __future__ import annotations

import argparse
from collections import Counter
from dataclasses import dataclass
import json
import math
from pathlib import Path
import re
from typing import Any

import numpy as np
import pandas as pd

import run_relational_linkage_experiment as linkage


REFINEMENT_VERSION = "relational-group-refinement-v1"
REFINEMENT_BRIDGE_TOKENS = {
    "clinical",
    "department",
    "emergency",
    "fall",
    "falls",
    "health",
    "hospital",
    "incident",
    "medical",
    "mental",
    "note",
    "notes",
    "patient",
    "patients",
    "pedestrian",
    "pedestrians",
    "provider",
    "resident",
    "residents",
    "staff",
}
ASSESSMENT_TERMS = {
    "assessment",
    "assessments",
    "evaluation",
    "review",
    "screen",
    "screening",
    "test",
}
OBSERVATION_TERMS = {
    "monitoring",
    "observation",
    "observations",
    "oversight",
}
COMPOUND_SPLIT_RE = re.compile(r"\s+(?:and|or)\s+|[,;]")


@dataclass
class RefinedCluster:
    members: set[int]
    status: str
    source_group_id: str
    reason: str
    anchor: int | None = None


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--normalized-csv", required=True, type=Path)
    parser.add_argument("--assignments-csv", required=True, type=Path)
    parser.add_argument("--groups-csv", required=True, type=Path)
    parser.add_argument("--edges-csv", required=True, type=Path)
    parser.add_argument("--embeddings-npy", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--minimum-edge-score", type=float, default=0.84)
    parser.add_argument("--minimum-semantic-object-similarity", type=float, default=0.90)
    parser.add_argument("--minimum-assignment-margin", type=float, default=0.03)
    parser.add_argument("--maximum-incompatibility-fraction", type=float, default=0.25)
    parser.add_argument("--min-recurring-reports", type=int, default=3)
    parser.add_argument("--assessment-csv", type=Path)
    parser.add_argument("--assessment-members-csv", type=Path)
    return parser.parse_args()


def refinement_object_tokens(value: Any) -> set[str]:
    return linkage.discriminative_object_tokens(value) - REFINEMENT_BRIDGE_TOKENS


def is_compound_relation(row: pd.Series) -> bool:
    action = linkage.key(row.get("failed_action"))
    if COMPOUND_SPLIT_RE.search(action):
        return True
    object_value = linkage.key(row.get("issue_object"))
    clauses = [
        refinement_object_tokens(clause)
        for clause in COMPOUND_SPLIT_RE.split(object_value)
    ]
    clauses = [tokens for tokens in clauses if tokens]
    return (
        len(clauses) >= 2
        and any(
            not (left & right)
            for position, left in enumerate(clauses)
            for right in clauses[position + 1 :]
        )
    )


def action_families_compatible(left: pd.Series, right: pd.Series) -> bool:
    left_family = linkage.action_family(left.get("failed_action"))
    right_family = linkage.action_family(right.get("failed_action"))
    if not left_family or not right_family or left_family == right_family:
        return True
    pair = frozenset((left_family, right_family))
    object_words = (
        linkage.object_tokens(left.get("issue_object"))
        | linkage.object_tokens(right.get("issue_object"))
    )
    if pair == frozenset(("assessment", "execution")):
        return bool(object_words & ASSESSMENT_TERMS)
    if pair == frozenset(("monitoring", "execution")):
        return bool(object_words & OBSERVATION_TERMS)
    if refinement_object_tokens(left.get("issue_object")) & refinement_object_tokens(
        right.get("issue_object")
    ):
        # Material verb contradictions have already been rejected by the
        # guarded cannot-link rules. Shared specific objects can tolerate
        # residual normalization variants such as raise/submit or provide/ensure.
        return True
    return False


def failure_families_compatible(left: pd.Series, right: pd.Series) -> bool:
    # High-contrast failure states are already handled by
    # ``guarded_relation_conflict``. Absence, incompleteness, and inadequacy
    # may legitimately describe the same operational obligation.
    return True


def actor_families_compatible(left: pd.Series, right: pd.Series) -> bool:
    left_actor = linkage.actor_family(left.get("responsible_actor_role"))
    right_actor = linkage.actor_family(right.get("responsible_actor_role"))
    if not left_actor or not right_actor or left_actor == right_actor:
        return True
    service_side = {
        "multi_organisation",
        "practitioner",
        "provider_service",
        "team",
    }
    return left_actor in service_side and right_actor in service_side


def operationally_compatible(
    left: pd.Series,
    right: pd.Series,
    *,
    action_object_similarity: float,
    minimum_semantic_object_similarity: float,
    allow_compound: bool = False,
) -> tuple[bool, str]:
    if not allow_compound and (
        is_compound_relation(left) or is_compound_relation(right)
    ):
        return False, "compound_relation"
    conflict = linkage.guarded_relation_conflict(left, right)
    if conflict:
        return False, conflict
    if not actor_families_compatible(left, right):
        return False, "different_actor_family"
    if not action_families_compatible(left, right):
        return False, "different_action_subtype"
    if not failure_families_compatible(left, right):
        return False, "different_failure_family"
    left_tokens = refinement_object_tokens(left.get("issue_object"))
    right_tokens = refinement_object_tokens(right.get("issue_object"))
    if left_tokens and right_tokens:
        if left_tokens & right_tokens:
            return True, ""
        if action_object_similarity >= minimum_semantic_object_similarity:
            return True, ""
        return False, "different_object_subtype"
    if not left_tokens and not right_tokens:
        if (
            linkage.generic_object_signature(left.get("issue_object"))
            == linkage.generic_object_signature(right.get("issue_object"))
        ):
            return True, ""
        return False, "different_generic_object"
    return True, "underspecified_object"


def edge_lookup(edges: pd.DataFrame, index_by_issue: dict[str, int]) -> dict[tuple[int, int], dict[str, float]]:
    lookup: dict[tuple[int, int], dict[str, float]] = {}
    for _, edge in edges.iterrows():
        left = index_by_issue.get(str(edge["left_issue_id"]))
        right = index_by_issue.get(str(edge["right_issue_id"]))
        if left is None or right is None:
            continue
        lookup[(min(left, right), max(left, right))] = {
            "score": float(edge["adjusted_similarity"]),
            "action_object_similarity": float(edge["action_object_similarity"]),
        }
    return lookup


def pair_edge(
    lookup: dict[tuple[int, int], dict[str, float]], left: int, right: int
) -> dict[str, float] | None:
    return lookup.get((min(left, right), max(left, right)))


def pair_operationally_compatible(
    frame: pd.DataFrame,
    lookup: dict[tuple[int, int], dict[str, float]],
    left: int,
    right: int,
    *,
    minimum_semantic_object_similarity: float,
    require_edge: bool,
    allow_compound: bool = False,
) -> tuple[bool, str]:
    edge = pair_edge(lookup, left, right)
    if require_edge and edge is None:
        return False, "no_direct_edge"
    action_object_similarity = (
        edge["action_object_similarity"] if edge is not None else -1.0
    )
    return operationally_compatible(
        frame.iloc[left],
        frame.iloc[right],
        action_object_similarity=action_object_similarity,
        minimum_semantic_object_similarity=minimum_semantic_object_similarity,
        allow_compound=allow_compound,
    )


def incompatibility_fraction(
    frame: pd.DataFrame,
    members: list[int],
    lookup: dict[tuple[int, int], dict[str, float]],
    *,
    minimum_semantic_object_similarity: float,
) -> tuple[float, Counter[str]]:
    if len(members) < 2:
        return 0.0, Counter()
    incompatible = 0
    reasons: Counter[str] = Counter()
    possible = len(members) * (len(members) - 1) // 2
    for position, left in enumerate(members):
        for right in members[position + 1 :]:
            compatible, reason = pair_operationally_compatible(
                frame,
                lookup,
                left,
                right,
                minimum_semantic_object_similarity=minimum_semantic_object_similarity,
                require_edge=False,
                allow_compound=True,
            )
            if not compatible or reason == "underspecified_object":
                incompatible += 1
                reasons[reason or "incompatible"] += 1
    return incompatible / possible, reasons


def supported_anchor_core(
    frame: pd.DataFrame,
    available: set[int],
    anchor: int,
    lookup: dict[tuple[int, int], dict[str, float]],
    *,
    minimum_edge_score: float,
    minimum_semantic_object_similarity: float,
) -> set[int]:
    core = {anchor}
    candidates: list[tuple[float, int]] = []
    for member in available - {anchor}:
        edge = pair_edge(lookup, anchor, member)
        if edge is None or edge["score"] < minimum_edge_score:
            continue
        compatible, reason = pair_operationally_compatible(
            frame,
            lookup,
            anchor,
            member,
            minimum_semantic_object_similarity=minimum_semantic_object_similarity,
            require_edge=True,
        )
        if compatible and reason != "underspecified_object":
            candidates.append((edge["score"], member))
    for _, member in sorted(candidates, reverse=True):
        compatibility = [
            pair_operationally_compatible(
                frame,
                lookup,
                member,
                existing,
                minimum_semantic_object_similarity=minimum_semantic_object_similarity,
                require_edge=False,
            )
            for existing in core
        ]
        if all(value for value, _ in compatibility) and not any(
            reason == "underspecified_object" for _, reason in compatibility
        ):
            core.add(member)
    return core


def best_core(
    frame: pd.DataFrame,
    available: set[int],
    anchors: set[int],
    lookup: dict[tuple[int, int], dict[str, float]],
    *,
    minimum_edge_score: float,
    minimum_semantic_object_similarity: float,
    min_recurring_reports: int,
) -> tuple[set[int], int | None]:
    candidates: list[tuple[int, int, float, int, set[int]]] = []
    for anchor in anchors & available:
        core = supported_anchor_core(
            frame,
            available,
            anchor,
            lookup,
            minimum_edge_score=minimum_edge_score,
            minimum_semantic_object_similarity=minimum_semantic_object_similarity,
        )
        reports = frame.iloc[sorted(core)]["report_key"].astype(str).nunique()
        if reports < min_recurring_reports:
            continue
        scores = [
            pair_edge(lookup, anchor, member)["score"]
            for member in core - {anchor}
            if pair_edge(lookup, anchor, member) is not None
        ]
        mean_score = float(np.mean(scores)) if scores else 0.0
        candidates.append((reports, len(core), mean_score, -anchor, core))
    if not candidates:
        return set(), None
    _, _, _, negative_anchor, core = max(candidates, key=lambda item: item[:4])
    return core, -negative_anchor


def attach_unambiguous_members(
    frame: pd.DataFrame,
    cores: list[RefinedCluster],
    remaining: set[int],
    lookup: dict[tuple[int, int], dict[str, float]],
    *,
    minimum_edge_score: float,
    minimum_semantic_object_similarity: float,
    minimum_assignment_margin: float,
) -> set[int]:
    unresolved: set[int] = set()
    for member in sorted(remaining):
        if is_compound_relation(frame.iloc[member]):
            unresolved.add(member)
            continue
        options: list[tuple[float, int]] = []
        for ordinal, core in enumerate(cores):
            assert core.anchor is not None
            edge = pair_edge(lookup, member, core.anchor)
            if edge is None or edge["score"] < minimum_edge_score:
                continue
            compatibility = [
                pair_operationally_compatible(
                    frame,
                    lookup,
                    member,
                    existing,
                    minimum_semantic_object_similarity=minimum_semantic_object_similarity,
                    require_edge=False,
                )
                for existing in core.members
            ]
            if not all(compatible for compatible, _ in compatibility):
                continue
            # An underspecified object may form a homogeneous generic group,
            # but it must not be silently absorbed into a specific subtype.
            if any(reason == "underspecified_object" for _, reason in compatibility):
                continue
            options.append((edge["score"], ordinal))
        options.sort(reverse=True)
        if not options:
            unresolved.add(member)
            continue
        if len(options) > 1 and options[0][0] - options[1][0] < minimum_assignment_margin:
            unresolved.add(member)
            continue
        cores[options[0][1]].members.add(member)
    return unresolved


def refine_group(
    frame: pd.DataFrame,
    members: list[int],
    source_group_id: str,
    lookup: dict[tuple[int, int], dict[str, float]],
    *,
    minimum_edge_score: float,
    minimum_semantic_object_similarity: float,
    minimum_assignment_margin: float,
    maximum_incompatibility_fraction: float,
    min_recurring_reports: int,
) -> tuple[list[RefinedCluster], dict[str, Any]]:
    fraction, reasons = incompatibility_fraction(
        frame,
        members,
        lookup,
        minimum_semantic_object_similarity=minimum_semantic_object_similarity,
    )
    reports = frame.iloc[members]["report_key"].astype(str).nunique()
    compounds = {member for member in members if is_compound_relation(frame.iloc[member])}
    provenance = {
        "source_group_id": source_group_id,
        "source_members": len(members),
        "source_reports": int(reports),
        "incompatibility_fraction": fraction,
        "compound_members": len(compounds),
        "incompatibility_reasons": " | ".join(
            f"{reason}:{count}" for reason, count in reasons.most_common()
        ),
    }
    if (
        reports >= min_recurring_reports
        and fraction <= maximum_incompatibility_fraction
    ):
        status = "refined_recurring"
        reason = "operationally_homogeneous"
        return [
            RefinedCluster(set(members), status, source_group_id, reason)
        ], provenance

    available = set(members) - compounds
    specific_anchors = {
        member
        for member in available
        if refinement_object_tokens(frame.iloc[member].get("issue_object"))
    }
    cores: list[RefinedCluster] = []
    while True:
        core, anchor = best_core(
            frame,
            available,
            specific_anchors,
            lookup,
            minimum_edge_score=minimum_edge_score,
            minimum_semantic_object_similarity=minimum_semantic_object_similarity,
            min_recurring_reports=min_recurring_reports,
        )
        if not core or anchor is None:
            break
        cores.append(
            RefinedCluster(
                core,
                "refined_recurring",
                source_group_id,
                "specific_operational_core",
                anchor,
            )
        )
        available -= core
        specific_anchors -= core

    if not cores:
        generic_available = {
            member
            for member in available
            if not refinement_object_tokens(frame.iloc[member].get("issue_object"))
        }
        generic_core, anchor = best_core(
            frame,
            generic_available,
            generic_available,
            lookup,
            minimum_edge_score=minimum_edge_score,
            minimum_semantic_object_similarity=minimum_semantic_object_similarity,
            min_recurring_reports=min_recurring_reports,
        )
        if generic_core and anchor is not None:
            cores.append(
                RefinedCluster(
                    generic_core,
                    "refined_recurring",
                    source_group_id,
                    "homogeneous_generic_operational_core",
                    anchor,
                )
            )
            available -= generic_core

    unresolved = attach_unambiguous_members(
        frame,
        cores,
        available | compounds,
        lookup,
        minimum_edge_score=minimum_edge_score,
        minimum_semantic_object_similarity=minimum_semantic_object_similarity,
        minimum_assignment_margin=minimum_assignment_margin,
    )
    if unresolved:
        unresolved_reports = (
            frame.iloc[sorted(unresolved)]["report_key"].astype(str).nunique()
        )
        cores.append(
            RefinedCluster(
                unresolved,
                (
                    "review_required"
                    if unresolved_reports >= min_recurring_reports
                    else "isolated_or_pair"
                ),
                source_group_id,
                "ambiguous_or_compound_members",
            )
        )
    provenance["refined_cores"] = sum(
        cluster.status == "refined_recurring" for cluster in cores
    )
    provenance["review_members"] = sum(
        len(cluster.members) for cluster in cores if cluster.status == "review_required"
    )
    return cores, provenance


def invariant_value(values: list[str], *, fallback: str) -> str:
    cleaned = [value for value in values if value]
    unique = sorted(set(cleaned))
    return unique[0] if len(unique) == 1 else fallback


def structured_group_label(frame: pd.DataFrame, members: set[int]) -> str:
    local = frame.iloc[sorted(members)]
    actor = invariant_value(
        [linkage.actor_family(value) for value in local["responsible_actor_role"]],
        fallback="mixed actors",
    )
    action = invariant_value(
        [linkage.action_family(value) for value in local["failed_action"]],
        fallback="related actions",
    )
    failure = invariant_value(
        [linkage.failure_family(value) for value in local["failure_state"]],
        fallback="mixed failure",
    )
    token_sets = [
        refinement_object_tokens(value) for value in local["issue_object"]
    ]
    shared = set.intersection(*token_sets) if token_sets and all(token_sets) else set()
    if shared:
        object_label = " ".join(sorted(shared))
    else:
        signatures = [
            linkage.generic_object_signature(value) for value in local["issue_object"]
        ]
        object_label = invariant_value(signatures, fallback="related objects")
    direction = invariant_value(
        [
            linkage.direction_family(value)
            for value in local.get(
                "communication_direction", pd.Series([""] * len(local))
            )
            if linkage.direction_family(value) not in {"", "not applicable"}
        ],
        fallback="",
    )
    parts = [actor, failure, action, object_label]
    if direction:
        parts.append(direction)
    return " — ".join(part.replace("_", " ") for part in parts)


def build_outputs(
    frame: pd.DataFrame,
    embeddings: np.ndarray,
    clusters: list[RefinedCluster],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    ordered = sorted(
        clusters,
        key=lambda cluster: (
            cluster.status != "refined_recurring",
            -len(cluster.members),
            min(cluster.members),
        ),
    )
    assignments: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    for ordinal, cluster in enumerate(ordered, 1):
        members = sorted(cluster.members)
        local = embeddings[members]
        similarities = local @ local.T
        medoid_position = int(np.argmax(similarities.mean(axis=1)))
        medoid_index = members[medoid_position]
        group_id = f"ref_{ordinal:05d}"
        reports = frame.iloc[members]["report_key"].astype(str).nunique()
        summaries.append(
            {
                "refined_group_id": group_id,
                "refinement_status": cluster.status,
                "source_group_id": cluster.source_group_id,
                "refinement_reason": cluster.reason,
                "occurrence_count": len(members),
                "report_count": int(reports),
                "structured_group_label": structured_group_label(
                    frame, cluster.members
                ),
                "prototype_issue_id": frame.iloc[medoid_index]["issue_id"],
                "prototype_canonical_issue": frame.iloc[medoid_index][
                    "canonical_issue"
                ],
                "minimum_pair_similarity": (
                    float(similarities[np.triu_indices(len(members), 1)].min())
                    if len(members) > 1
                    else 1.0
                ),
                "median_pair_similarity": (
                    float(np.median(similarities[np.triu_indices(len(members), 1)]))
                    if len(members) > 1
                    else 1.0
                ),
            }
        )
        for member in members:
            assignments.append(
                {
                    "issue_id": frame.iloc[member]["issue_id"],
                    "report_key": frame.iloc[member]["report_key"],
                    "source_group_id": cluster.source_group_id,
                    "refined_group_id": group_id,
                    "refinement_status": cluster.status,
                    "refinement_reason": cluster.reason,
                    "prototype_similarity": float(
                        embeddings[member] @ embeddings[medoid_index]
                    ),
                }
            )
    return pd.DataFrame(assignments), pd.DataFrame(summaries)


def score_assessment(
    assignments: pd.DataFrame,
    assessment: pd.DataFrame,
    assessment_members: pd.DataFrame,
) -> dict[str, Any]:
    auto = assignments[
        assignments["refinement_status"].eq("refined_recurring")
    ].set_index("issue_id")["refined_group_id"].to_dict()
    reviewed_ids = set(assessment_members["issue_id"].astype(str))
    expected_positive: set[tuple[str, str]] = set()
    expected_negative: set[tuple[str, str]] = set()
    correct_ids: set[str] = set()
    incorrect_ids: set[str] = set()
    resolved_groups = 0
    for _, decision in assessment.iterrows():
        source_group = str(decision["relational_group_id"])
        members = set(
            assessment_members.loc[
                assessment_members["relational_group_id"].astype(str).eq(source_group),
                "issue_id",
            ].astype(str)
        )
        wrong = {
            value
            for value in str(decision.get("incorrect_member_issue_ids", "")).split("|")
            if value
        }
        correct = members - wrong
        correct_ids.update(correct)
        incorrect_ids.update(wrong)
        expected_positive.update(
            tuple(sorted((left, right)))
            for position, left in enumerate(sorted(correct))
            for right in sorted(correct)[position + 1 :]
        )
        expected_negative.update(
            tuple(sorted((left, right))) for left in correct for right in wrong
        )
        correct_groups = {auto[item] for item in correct if item in auto}
        wrong_groups = {auto[item] for item in wrong if item in auto}
        decision_key = linkage.key(decision.get("audit_decision"))
        if decision_key == "reject":
            resolved = not any(
                sum(auto.get(item) == group_id for item in members) >= 3
                for group_id in {auto[item] for item in members if item in auto}
            )
        else:
            resolved = (
                len(correct_groups) == 1
                and all(item in auto for item in correct)
                and not (correct_groups & wrong_groups)
            )
        resolved_groups += int(resolved)

    def same_auto_group(pair: tuple[str, str]) -> bool:
        left, right = pair
        return left in auto and right in auto and auto[left] == auto[right]

    positive_retained = sum(same_auto_group(pair) for pair in expected_positive)
    negative_separated = sum(
        not same_auto_group(pair) for pair in expected_negative
    )
    return {
        "reviewed_groups": len(assessment),
        "reviewed_issue_ids": len(reviewed_ids),
        "groups_fully_resolved": resolved_groups,
        "group_resolution_rate": resolved_groups / max(1, len(assessment)),
        "correct_members_auto_retained": len(correct_ids & set(auto)),
        "correct_members": len(correct_ids),
        "correct_member_auto_retention": len(correct_ids & set(auto))
        / max(1, len(correct_ids)),
        "incorrect_members_auto_retained": len(incorrect_ids & set(auto)),
        "incorrect_members": len(incorrect_ids),
        "expected_same_pairs": len(expected_positive),
        "expected_same_pairs_retained": positive_retained,
        "expected_same_pair_retention": positive_retained
        / max(1, len(expected_positive)),
        "expected_different_pairs": len(expected_negative),
        "expected_different_pairs_separated": negative_separated,
        "expected_different_pair_separation": negative_separated
        / max(1, len(expected_negative)),
    }


def main() -> None:
    args = parse_args()
    for value, name in (
        (args.minimum_edge_score, "--minimum-edge-score"),
        (args.minimum_semantic_object_similarity, "--minimum-semantic-object-similarity"),
        (args.minimum_assignment_margin, "--minimum-assignment-margin"),
        (args.maximum_incompatibility_fraction, "--maximum-incompatibility-fraction"),
    ):
        if not 0.0 <= value <= 1.0:
            raise ValueError(f"{name} must be between 0 and 1")
    if bool(args.assessment_csv) != bool(args.assessment_members_csv):
        raise ValueError(
            "--assessment-csv and --assessment-members-csv must be supplied together"
        )

    normalized = pd.read_csv(args.normalized_csv).fillna("")
    if "linkage_eligible" in normalized:
        normalized = normalized[
            normalized["linkage_eligible"].astype(str).str.lower().isin(("true", "1"))
        ].copy()
    normalized = normalized.reset_index(drop=True)
    normalized["_embedding_index"] = np.arange(len(normalized))
    embeddings = np.load(args.embeddings_npy)
    if len(embeddings) != len(normalized):
        raise ValueError(
            f"Embeddings contain {len(embeddings)} rows but normalized input has "
            f"{len(normalized)} eligible rows"
        )
    original_assignments = pd.read_csv(args.assignments_csv).fillna("")
    original_groups = pd.read_csv(args.groups_csv).fillna("")
    edges = pd.read_csv(args.edges_csv).fillna("")
    frame = original_assignments.merge(
        normalized,
        on=["issue_id", "report_key"],
        how="left",
        validate="one_to_one",
    ).sort_values("_embedding_index")
    if frame["_embedding_index"].isna().any():
        raise ValueError("Some assigned occurrences were not found in normalized input")
    frame = frame.reset_index(drop=True)
    embedding_positions = frame["_embedding_index"].astype(int).to_numpy()
    frame_embeddings = embeddings[embedding_positions]
    index_by_issue = {
        str(issue_id): index for index, issue_id in enumerate(frame["issue_id"])
    }
    lookup = edge_lookup(edges, index_by_issue)

    clusters: list[RefinedCluster] = []
    provenance: list[dict[str, Any]] = []
    for source_group_id, local in frame.groupby("relational_group_id", sort=False):
        members = local.index.tolist()
        refined, details = refine_group(
            frame,
            members,
            str(source_group_id),
            lookup,
            minimum_edge_score=args.minimum_edge_score,
            minimum_semantic_object_similarity=args.minimum_semantic_object_similarity,
            minimum_assignment_margin=args.minimum_assignment_margin,
            maximum_incompatibility_fraction=args.maximum_incompatibility_fraction,
            min_recurring_reports=args.min_recurring_reports,
        )
        clusters.extend(refined)
        provenance.append(details)

    assignments, summaries = build_outputs(frame, frame_embeddings, clusters)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    assignments.to_csv(args.output_dir / "01_refined_assignments.csv", index=False)
    summaries.to_csv(args.output_dir / "02_refined_groups.csv", index=False)
    pd.DataFrame(provenance).to_csv(
        args.output_dir / "03_refinement_provenance.csv", index=False
    )
    metrics: dict[str, Any] = {
        "refinement_version": REFINEMENT_VERSION,
        "source_occurrences": len(frame),
        "source_groups": original_groups.shape[0],
        "source_recurring_groups": int(
            original_groups["recurrence_status"].eq("recurring").sum()
        ),
        "refined_groups": len(summaries),
        "refined_recurring_groups": int(
            summaries["refinement_status"].eq("refined_recurring").sum()
        ),
        "review_required_groups": int(
            summaries["refinement_status"].eq("review_required").sum()
        ),
        "refined_recurring_occurrences": int(
            summaries.loc[
                summaries["refinement_status"].eq("refined_recurring"),
                "occurrence_count",
            ].sum()
        ),
        "review_required_occurrences": int(
            summaries.loc[
                summaries["refinement_status"].eq("review_required"),
                "occurrence_count",
            ].sum()
        ),
        "parameters": {
            "minimum_edge_score": args.minimum_edge_score,
            "minimum_semantic_object_similarity": (
                args.minimum_semantic_object_similarity
            ),
            "minimum_assignment_margin": args.minimum_assignment_margin,
            "maximum_incompatibility_fraction": (
                args.maximum_incompatibility_fraction
            ),
            "min_recurring_reports": args.min_recurring_reports,
        },
    }
    if args.assessment_csv:
        assessment = pd.read_csv(args.assessment_csv).fillna("")
        assessment_members = pd.read_csv(args.assessment_members_csv).fillna("")
        metrics["manual_assessment"] = score_assessment(
            assignments, assessment, assessment_members
        )
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
