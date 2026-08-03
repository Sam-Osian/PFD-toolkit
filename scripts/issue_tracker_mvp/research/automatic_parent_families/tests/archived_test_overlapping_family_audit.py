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

import build_overlapping_family_audit as audit  # noqa: E402
import score_overlapping_family_audit as scoring  # noqa: E402


def test_group_balanced_sample_caps_each_family():
    frame = pd.DataFrame(
        [
            {"family_id": family_id, "child_group_id": f"{family_id}_{index}"}
            for family_id in ("fam_a", "fam_b", "fam_c")
            for index in range(5)
        ]
    )

    sampled = audit.group_balanced_sample(
        frame, 6, maximum_per_family=2, seed=1
    )

    assert len(sampled) == 6
    assert sampled["family_id"].value_counts().max() == 2


def test_audit_scorer_compares_flagged_and_unflagged_strata():
    frame = pd.DataFrame(
        [
            {
                "audit_id": "att_1",
                "audit_stratum": "risk_flagged",
                "supports_secondary_family": "yes",
                "child_report_count": 4,
                "target_family_id": "fam_a",
            },
            {
                "audit_id": "att_2",
                "audit_stratum": "risk_flagged",
                "supports_secondary_family": "no",
                "child_report_count": 1,
                "target_family_id": "fam_a",
            },
            {
                "audit_id": "att_3",
                "audit_stratum": "unflagged_random",
                "supports_secondary_family": "uncertain",
                "child_report_count": 2,
                "target_family_id": "fam_b",
            },
        ]
    )

    result = scoring.score(frame)

    assert result["strata"]["risk_flagged"][
        "attachment_precision_among_decided"
    ] == 0.5
    assert result["strata"]["risk_flagged"][
        "report_weighted_precision_among_decided"
    ] == 0.8
    assert result["strata"]["unflagged_random"]["uncertain"] == 1
    assert result["overall"]["attachment_precision_among_decided"] == 0.5


def test_audit_scorer_rejects_incomplete_decisions():
    frame = pd.DataFrame(
        [
            {
                "audit_id": "att_1",
                "audit_stratum": "risk_flagged",
                "supports_secondary_family": "",
                "child_report_count": 1,
                "target_family_id": "fam_a",
            }
        ]
    )

    with pytest.raises(ValueError, match="requires yes/no/uncertain"):
        scoring.score(frame)


def test_decision_payload_requires_exact_audit_coverage():
    frame = pd.DataFrame(
        [
            {"audit_id": "att_1", "supports_secondary_family": ""},
            {"audit_id": "att_2", "supports_secondary_family": ""},
        ]
    )

    with pytest.raises(ValueError, match="cover the audit exactly"):
        scoring.apply_decisions(
            frame, [{"audit_id": "att_1", "decision": "yes"}]
        )
