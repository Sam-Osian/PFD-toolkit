from __future__ import annotations

# Archived regression coverage for the superseded automatic-parent experiment.

import sys
from pathlib import Path

import pandas as pd


ARCHIVE_DIR = Path(__file__).resolve().parents[1]
ACTIVE_DIR = ARCHIVE_DIR.parents[1]
sys.path.insert(0, str(ACTIVE_DIR))
sys.path.insert(0, str(ARCHIVE_DIR))

import build_direct_parent_assignments as builder  # noqa: E402


def test_build_assignments_unions_core_and_direct_reports():
    summaries = pd.DataFrame([{"family_id": "f", "family_label_hint": "label", "family_prototype": "prototype"}])
    memberships = pd.DataFrame([{"family_id": "f", "child_group_id": "g", "membership_type": "core"}])
    groups = pd.DataFrame([{"report_key": "r1", "issue_id": "i1", "relational_group_id": "g"}])
    adjudication = pd.DataFrame(
        [{
            "family_id": "f", "report_key": "r2", "issue_id": "i2",
            "representative_group_id": "g2", "confidence": "high",
            "reason": "same parent", "parent_assignment_status": "auto_accept",
        }]
    )

    output, summary, metrics = builder.build_assignments(
        summaries, memberships, groups, adjudication
    )

    assert set(output["report_key"]) == {"r1", "r2"}
    assert summary.iloc[0]["total_report_count"] == 2
    assert summary.iloc[0]["direct_retrieval_report_count"] == 1
    assert metrics["parent_report_assignments"] == 2
