from __future__ import annotations

# Archived regression coverage for the superseded automatic-parent experiment.

import sys
from pathlib import Path

import pandas as pd


ARCHIVE_DIR = Path(__file__).resolve().parents[1]
ACTIVE_DIR = ARCHIVE_DIR.parents[1]
sys.path.insert(0, str(ACTIVE_DIR))
sys.path.insert(0, str(ARCHIVE_DIR))

import adjudicate_overlapping_family_attachments as adjudicator  # noqa: E402


def test_deterministic_veto_preserves_ambulance_direction():
    target = "The care provider failed to call an ambulance promptly."
    candidate = "The ambulance service provided a delayed response and attendance."

    assert adjudicator.deterministic_veto(target, candidate) == "wrong_direction"


def test_deterministic_veto_separates_performing_and_recording_observations():
    target = "Staff failed to record patient observations in the chart."
    candidate = "Staff failed to carry out physiological observations."

    assert adjudicator.deterministic_veto(target, candidate) == "wrong_action"


def test_validate_payload_requires_every_attachment_once():
    local = pd.DataFrame([{"attachment_id": "att_1"}, {"attachment_id": "att_2"}])
    payload = {
        "decisions": [
            {
                "attachment_id": "att_1",
                "supports_family": True,
                "confidence": "high",
                "reason": "shared_issue",
                "rationale": "Same parent concern.",
            }
        ]
    }

    try:
        adjudicator.validate_payload(payload, "batch", local)
    except ValueError as exc:
        assert "Missing 1 decisions" in str(exc)
    else:
        raise AssertionError("Incomplete payload should fail")


def test_benchmark_gate_measures_retained_precision_and_weight():
    output = pd.DataFrame(
        [
            {"attachment_id": "att_1", "adjudication_status": "accept", "child_report_count": 9, "decision_source": "model"},
            {"attachment_id": "att_2", "adjudication_status": "reject", "child_report_count": 1, "decision_source": "deterministic_veto"},
        ]
    )
    benchmark = pd.DataFrame([{"audit_id": "att_1"}, {"audit_id": "att_2"}])
    truth = [{"audit_id": "att_1", "decision": "yes"}, {"audit_id": "att_2", "decision": "no"}]

    metrics = adjudicator.benchmark_metrics(
        output, benchmark, truth, minimum_precision=0.9, minimum_weighted_precision=0.9
    )

    assert metrics["accepted_precision"] == 1.0
    assert metrics["accepted_report_weighted_precision"] == 1.0
    assert metrics["precision_gate_passed"] is True


def test_verifier_rejection_overrides_first_pass_accept(tmp_path):
    output = pd.DataFrame(
        [{
            "attachment_id": "att_1", "decision_source": "model",
            "adjudication_status": "accept", "supports_family": True,
            "confidence": "high", "reason": "shared_issue", "rationale": "",
        }]
    )
    checkpoint = tmp_path / "verify.jsonl"
    checkpoint.write_text(
        '{"batch_id":"b1","status":"completed","decisions":['
        '{"attachment_id":"att_1","supports_family":false,"confidence":"high",'
        '"reason":"wrong_action","rationale":"Different obligation."}]}\n',
        encoding="utf-8",
    )

    verified = adjudicator.apply_verification(output, checkpoint)

    assert verified.iloc[0]["adjudication_status"] == "reject"
    assert verified.iloc[0]["decision_source"] == "model_verifier_reject"
