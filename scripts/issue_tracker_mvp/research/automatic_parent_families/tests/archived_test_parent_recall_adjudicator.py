from __future__ import annotations

# Archived regression coverage for the superseded automatic-parent experiment.

import sys
from pathlib import Path

import pandas as pd


ARCHIVE_DIR = Path(__file__).resolve().parents[1]
ACTIVE_DIR = ARCHIVE_DIR.parents[1]
sys.path.insert(0, str(ACTIVE_DIR))
sys.path.insert(0, str(ARCHIVE_DIR))

import adjudicate_parent_recall_candidates as adjudicator  # noqa: E402


def test_validate_payload_requires_all_ids():
    local = pd.DataFrame([{"audit_id": "a"}, {"audit_id": "b"}])
    payload = {"decisions": [{"audit_id": "a", "supports_parent": True, "confidence": "high", "reason": "yes"}]}

    try:
        adjudicator.validate_payload(payload, "batch", local)
    except ValueError as exc:
        assert "Missing 1 decisions" in str(exc)
    else:
        raise AssertionError("Incomplete result should fail")


def test_validate_payload_recovers_unique_truncated_id():
    local = pd.DataFrame([{"audit_id": "recall_1234567890abcdef"}])
    payload = {
        "decisions": [{
            "audit_id": "recall_1234567890abcde",
            "supports_parent": True,
            "confidence": "high",
            "reason": "same parent",
        }]
    }

    result = adjudicator.validate_payload(payload, "batch", local)

    assert result[0]["audit_id"] == "recall_1234567890abcdef"


def test_validate_payload_recovers_unique_truncated_id_with_substitution():
    local = pd.DataFrame([
        {"audit_id": "recall_1234567890abcdef"},
        {"audit_id": "recall_fedcba0987654321"},
    ])
    payload = {
        "decisions": [
            {
                "audit_id": "recall_1234567890ac",
                "supports_parent": True,
                "confidence": "high",
                "reason": "same parent",
            },
            {
                "audit_id": "recall_fedcba0987654321",
                "supports_parent": False,
                "confidence": "high",
                "reason": "different parent",
            },
        ]
    }

    result = adjudicator.validate_payload(payload, "batch", local)

    assert result[0]["audit_id"] == "recall_1234567890abcdef"


def test_recover_audit_id_does_not_choose_tied_near_match():
    expected = ["recall_1234567890abcdef", "recall_1234567890abcdee"]

    result = adjudicator.recover_audit_id("recall_1234567890abcde0", expected)

    assert result == "recall_1234567890abcde0"


def test_benchmark_scores_only_high_confidence_accepts():
    output = pd.DataFrame(
        [
            {"audit_id": "a", "parent_assignment_status": "auto_accept"},
            {"audit_id": "b", "parent_assignment_status": "review"},
            {"audit_id": "c", "parent_assignment_status": "reject"},
        ]
    )
    truth = [
        {"audit_id": "a", "decision": "yes"},
        {"audit_id": "b", "decision": "yes"},
        {"audit_id": "c", "decision": "no"},
    ]

    result = adjudicator.benchmark_metrics(output, truth, 0.9)

    assert result["auto_accept_precision"] == 1.0
    assert result["auto_accept_recall"] == 0.5
    assert result["precision_gate_passed"] is True
