from __future__ import annotations

import sys
from pathlib import Path

import pandas as pd
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "issue_tracker_mvp"
sys.path.insert(0, str(SCRIPT_DIR))

import build_relational_diagnostics as diagnostics  # noqa: E402
import evaluate_relational_regressions as regression  # noqa: E402
import trace_relational_case as tracing  # noqa: E402
import run_relational_linkage_experiment as linkage  # noqa: E402


def fixture_frames():
    occurrences = pd.DataFrame(
        [
            {
                "issue_id": issue_id,
                "report_key": f"rpt_{issue_id}",
                "canonical_issue": canonical,
                "failed_action": action,
                "issue_object": object_value,
                "failure_state": "omitted",
                "communication_direction": direction,
                "responsible_actor_role": "provider organisation",
                "counterparty_role": "patient",
                "evidence_quote": canonical,
                "normalization_status": "success",
            }
            for issue_id, canonical, action, object_value, direction in [
                ("a1", "Ambulance response was delayed", "provide", "ambulance response", "professional to patient"),
                ("a2", "Ambulance response time was missed", "meet", "ambulance response time", "professional to patient"),
                ("a3", "Ambulance arrival was late", "provide", "ambulance arrival", "professional to patient"),
                ("b1", "The ambulance was dispatched late", "dispatch", "ambulance", "patient family to professional"),
                ("b2", "Ambulance target time was missed", "meet", "ambulance target time", "professional to patient"),
                ("b3", "The ambulance response was late", "provide", "ambulance response", "professional to patient"),
                ("c1", "Staffing was inadequate", "maintain", "staffing", "not stated"),
                ("c2", "Doctors and physical space were inadequate", "maintain", "doctors and physical space", "not stated"),
                ("c3", "Clinical space was inadequate", "maintain", "clinical space", "not stated"),
            ]
        ]
    )
    assignments = pd.DataFrame(
        [
            {
                "issue_id": issue_id,
                "report_key": f"rpt_{issue_id}",
                "relational_group_id": group_id,
                "recurrence_status": "recurring",
                "prototype_similarity": similarity,
            }
            for issue_id, group_id, similarity in [
                ("a1", "rel_a", 1.0),
                ("a2", "rel_a", 0.91),
                ("a3", "rel_a", 0.90),
                ("b1", "rel_b", 1.0),
                ("b2", "rel_b", 0.91),
                ("b3", "rel_b", 0.90),
                ("c1", "rel_c", 1.0),
                ("c2", "rel_c", 0.79),
                ("c3", "rel_c", 0.90),
            ]
        ]
    )
    groups = pd.DataFrame(
        [
            {
                "relational_group_id": group_id,
                "recurrence_status": "recurring",
                "occurrence_count": 3,
                "report_count": 3,
                "prototype_issue_id": prototype,
                "prototype_canonical_issue": label,
            }
            for group_id, prototype, label in [
                ("rel_a", "a1", "Delayed ambulance response"),
                ("rel_b", "b1", "Delayed ambulance dispatch"),
                ("rel_c", "c1", "Inadequate staffing"),
            ]
        ]
    )
    edge_rows = [
        ("a1", "a2", 0.92),
        ("a1", "a3", 0.91),
        ("b1", "b2", 0.92),
        ("b1", "b3", 0.91),
        ("c1", "c2", 0.90),
        ("c2", "c3", 0.89),
        ("a1", "b1", 0.93),
        ("a2", "b2", 0.92),
        ("a3", "b3", 0.91),
    ]
    accepted = pd.DataFrame(
        [
            {
                "left_issue_id": left,
                "right_issue_id": right,
                "adjusted_similarity": score,
                "compatibility_reasons": "",
            }
            for left, right, score in edge_rows
        ]
    )
    candidates = accepted.copy()
    candidates.loc[len(candidates)] = {
        "left_issue_id": "a2",
        "right_issue_id": "b1",
        "adjusted_similarity": -1.0,
        "compatibility_reasons": "guarded_veto:opposite_communication_direction",
    }
    return occurrences, assignments, groups, candidates, accepted


def test_duplicate_queue_traces_all_pairs_consolidation_veto():
    occurrences, assignments, groups, candidates, accepted = fixture_frames()

    result = diagnostics.build_duplicate_group_queue(
        occurrences,
        assignments,
        groups,
        candidates,
        accepted,
        minimum_cross_edges=2,
    )

    row = result.iloc[0]
    assert (row["group_a"], row["group_b"]) == ("rel_a", "rel_b")
    assert row["accepted_cross_edges"] == 3
    assert row["smaller_group_coverage"] == 1.0
    assert row["likely_consolidation_blocker"] == "all_pairs_conflict_veto"
    assert "opposite_communication_direction" in row["all_pair_conflict_reasons"]


def test_contamination_queue_flags_compound_bridge_member():
    occurrences, assignments, groups, _, accepted = fixture_frames()

    group_queue, member_queue = diagnostics.build_contamination_queues(
        occurrences, assignments, groups, accepted
    )

    member = member_queue[member_queue["issue_id"].eq("c2")].iloc[0]
    assert "accepted_edge_bridge" in member["diagnostic_reasons"]
    assert "compound_relation" in member["diagnostic_reasons"]
    assert "low_prototype_similarity" in member["diagnostic_reasons"]
    assert "rel_c" in set(group_queue["relational_group_id"])


def test_trace_exposes_cross_group_accepted_edges_and_missing_ids():
    occurrences, assignments, groups, candidates, accepted = fixture_frames()

    result = tracing.build_trace(
        occurrences,
        assignments,
        groups,
        candidates,
        accepted,
        group_ids=["rel_a"],
        issue_ids=["missing"],
    )

    assert result["missing_issue_ids"] == ["missing"]
    assert {row["issue_id"] for row in result["assignments"]} == {"a1", "a2", "a3"}
    assert any(
        row["crosses_final_group_boundary"] for row in result["accepted_edges"]
    )


def consolidation_embeddings(count: int) -> tuple[np.ndarray, list[float]]:
    weights = [0.10, 0.55, 0.25, 0.10]
    vector = np.asarray(
        [value for weight in weights for value in (weight**0.5, 0.0)],
        dtype=np.float32,
    )
    return np.vstack([vector] * count), weights


def test_tolerant_consolidation_allows_bounded_non_medoid_conflicts():
    frame = pd.DataFrame(
        [
            {
                "report_key": f"r{index}",
                "responsible_actor_role": "provider organisation",
                "counterparty_role": "patient",
                "failed_action": "inform",
                "issue_object": "discharge instructions",
                "communication_direction": direction,
                "failure_state": "inadequate",
            }
            for index, direction in enumerate(
                [
                    "professional_to_patient",
                    "professional_to_patient",
                    "professional_to_patient",
                    "patient_family_to_professional",
                ]
            )
        ]
    )
    embeddings, weights = consolidation_embeddings(4)
    edges = [
        linkage.PairScore(0, 2, 0.95, 0.0, 0.95, ()),
        linkage.PairScore(1, 2, 0.95, 0.0, 0.95, ()),
    ]

    strict, _ = linkage.consolidate_guarded_groups(
        frame,
        embeddings,
        [[0, 1], [2, 3]],
        edges,
        weights=weights,
        minimum_score=0.85,
        minimum_coverage=0.5,
        minimum_relation_similarity=0.8,
        minimum_action_object_similarity=0.8,
        minimum_specific_object_similarity=0.88,
        maximum_positive_adjustment=0.03,
    )
    tolerant, provenance = linkage.consolidate_guarded_groups(
        frame,
        embeddings,
        [[0, 1], [2, 3]],
        edges,
        weights=weights,
        minimum_score=0.85,
        minimum_coverage=0.5,
        maximum_conflict_fraction=0.5,
        minimum_distinctive_object_coverage=0.5,
        minimum_relation_similarity=0.8,
        minimum_action_object_similarity=0.8,
        minimum_specific_object_similarity=0.88,
        maximum_positive_adjustment=0.03,
    )

    assert strict == [[0, 1], [2, 3]]
    assert tolerant == [[0, 1, 2, 3]]
    assert provenance[0]["conflict_fraction"] == 0.5
    assert provenance[0]["distinctive_object_coverage"] == 0.5


def test_distinctive_object_guard_blocks_generic_cross_group_bridge():
    frame = pd.DataFrame(
        [
            {
                "report_key": f"r{index}",
                "responsible_actor_role": "provider organisation",
                "counterparty_role": "patient",
                "failed_action": "conduct",
                "issue_object": "risk assessment",
                "communication_direction": "not_applicable",
                "failure_state": "omitted",
            }
            for index in range(4)
        ]
    )
    embeddings, weights = consolidation_embeddings(4)
    edges = [
        linkage.PairScore(0, 2, 0.95, 0.0, 0.95, ()),
        linkage.PairScore(1, 3, 0.95, 0.0, 0.95, ()),
    ]

    groups, provenance = linkage.consolidate_guarded_groups(
        frame,
        embeddings,
        [[0, 1], [2, 3]],
        edges,
        weights=weights,
        minimum_score=0.85,
        minimum_coverage=0.5,
        maximum_conflict_fraction=0.5,
        minimum_distinctive_object_coverage=0.5,
        minimum_relation_similarity=0.8,
        minimum_action_object_similarity=0.8,
        minimum_specific_object_similarity=0.88,
        maximum_positive_adjustment=0.03,
    )

    assert groups == [[0, 1], [2, 3]]
    assert provenance == []


def test_regression_evaluator_uses_stable_memberships_not_group_ids():
    baseline_assignments = pd.DataFrame(
        [
            {"issue_id": issue_id, "relational_group_id": group_id}
            for issue_id, group_id in [
                ("a1", "old_a"),
                ("a2", "old_a"),
                ("b1", "old_b"),
                ("b2", "old_b"),
                ("c1", "old_c"),
                ("c2", "old_c"),
            ]
        ]
    )
    baseline_groups = pd.DataFrame(
        [
            {"relational_group_id": "old_a", "prototype_issue_id": "a1"},
            {"relational_group_id": "old_b", "prototype_issue_id": "b1"},
            {"relational_group_id": "old_c", "prototype_issue_id": "c1"},
        ]
    )
    candidate_assignments = pd.DataFrame(
        [
            {"issue_id": issue_id, "relational_group_id": group_id}
            for issue_id, group_id in [
                ("a1", "new_merged"),
                ("a2", "new_merged"),
                ("b1", "new_merged"),
                ("b2", "new_merged"),
                ("c1", "new_c"),
                ("c2", "new_c"),
            ]
        ]
    )
    cases = {
        "cases": [
            {
                "case_id": "merge",
                "expectation": "merge",
                "review_status": "test",
                "group_ids": ["old_a", "old_b"],
                "minimum_member_coverage_each": 1.0,
            },
            {
                "case_id": "preserve",
                "expectation": "preserve_group",
                "review_status": "test",
                "group_ids": ["old_c"],
                "minimum_member_coverage": 1.0,
            },
        ]
    }

    result = regression.evaluate_cases(
        cases,
        baseline_assignments,
        baseline_groups,
        candidate_assignments,
    ).set_index("case_id")

    assert bool(result.loc["merge", "passed"])
    assert bool(result.loc["preserve", "passed"])


def test_spot_check_regressions_exclude_contaminated_shared_cores():
    frame = pd.DataFrame(
        [
            {
                "rank": 1,
                "group_a": "a",
                "group_b": "b",
                "decision": "same_whole_group",
            },
            {
                "rank": 2,
                "group_a": "c",
                "group_b": "d",
                "decision": "related_but_distinct",
            },
            {
                "rank": 3,
                "group_a": "e",
                "group_b": "f",
                "decision": "same_core_contaminated",
            },
        ]
    )

    cases = regression.cases_from_duplicate_spot_check(frame)

    assert [case["expectation"] for case in cases] == ["merge", "separate"]
    assert cases[0]["minimum_member_coverage_each"] == 0.75
