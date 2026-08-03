from __future__ import annotations

# Archived regression coverage for the superseded automatic-parent experiment.

import sys
from pathlib import Path

import pandas as pd
import pytest


ARCHIVE_DIR = Path(__file__).resolve().parents[1]
ACTIVE_DIR = ARCHIVE_DIR.parents[1]
sys.path.insert(0, str(ACTIVE_DIR))
sys.path.insert(0, str(ARCHIVE_DIR))

import adjudicate_issue_family_recall_pilot as family_recall_adjudication  # noqa: E402
import build_issue_family_hierarchy as family_hierarchy  # noqa: E402
import build_issue_family_recall_pilot as family_recall  # noqa: E402


def test_family_recall_rules_retrieve_records_without_treating_mentions_as_truth():
    frame = pd.DataFrame(
        [
            {
                "canonical_issue": "Clinical records were incomplete.",
                "issue_object": "clinical records",
                "evidence_quote": "The records contained gaps.",
                "process_stage": "information_record_management",
                "issue_themes": "records_information",
            },
            {
                "canonical_issue": "The clinician failed to provide treatment.",
                "issue_object": "treatment",
                "evidence_quote": "The treatment was not provided.",
                "process_stage": "treatment_care_delivery",
                "issue_themes": "clinical_assessment_care",
            },
        ]
    )
    family = next(
        item for item in family_recall.FAMILIES if item.family_id == "records_information"
    )

    strict, rule = family_recall.family_masks(frame, family)

    assert strict.tolist() == [True, False]
    assert rule.tolist() == [True, False]


def test_family_recall_adjudication_requires_one_decision_per_issue():
    candidates = pd.DataFrame([{"issue_id": "iss_1"}, {"issue_id": "iss_2"}])
    payload = {
        "decisions": [
            {
                "issue_id": issue_id,
                "supports_family": True,
                "confidence": "high",
                "reason": "none",
            }
            for issue_id in candidates["issue_id"]
        ]
    }

    decisions = family_recall_adjudication.validate_payload(
        payload, "records_information:0001", candidates
    )

    assert [decision["issue_id"] for decision in decisions] == ["iss_1", "iss_2"]
    assert all(decision["supports_family"] for decision in decisions)


def test_family_hierarchy_threshold_requires_calibrated_precision():
    calibration = pd.DataFrame(
        {
            "family_similarity": [0.95, 0.93, 0.91, 0.70],
            "supports_family": [True, True, True, False],
        }
    )

    result = family_hierarchy.choose_threshold(
        calibration,
        minimum_precision=0.90,
        minimum_rows=3,
    )

    assert result["threshold"] == pytest.approx(0.91)
    assert result["selected_rows"] == 3
    assert result["precision"] == 1.0


def test_family_hierarchy_preserves_nonexclusive_parent_membership():
    base = {
        "issue_id": "iss_shared",
        "report_key": "rpt_1",
        "refined_group_id": "refined_1",
        "refinement_status": "refined_recurring",
        "rule_selected": True,
        "semantic_only": False,
        "family_similarity": 0.91,
        "canonical_issue": "Records were not transferred during discharge.",
        "issue_object": "medical records",
        "evidence_quote": "No records accompanied the discharge.",
        "process_stage": "information_record_management",
        "issue_themes": "records_information|communication_handover",
    }
    pool = pd.DataFrame(
        [
            {
                **base,
                "family_id": "records_information",
                "family_label": "Medical and care-record failures",
            },
            {
                **base,
                "family_id": "discharge_transitions",
                "family_label": "Unsafe or incomplete discharge and care-transition processes",
            },
        ]
    )
    adjudication_frame = pd.DataFrame(
        [
            {
                "family_id": family_id,
                "issue_id": "iss_shared",
                "supports_family": True,
                "confidence": "high",
                "reason": "none",
            }
            for family_id in ["records_information", "discharge_transitions"]
        ]
    )

    assignments, _, child_groups, issue_parents, _ = family_hierarchy.build_hierarchy(
        pool,
        adjudication_frame,
        minimum_precision=0.90,
        minimum_calibration_rows=8,
        calibration_fraction=0.70,
        seed=1,
    )

    assert assignments["is_parent_member"].all()
    assert issue_parents.iloc[0]["parent_family_count"] == 2
    assert set(child_groups["family_id"]) == {
        "records_information",
        "discharge_transitions",
    }
