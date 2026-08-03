from __future__ import annotations

# Archived regression coverage for the superseded automatic-parent experiment.

import sys
from pathlib import Path

import pandas as pd


ARCHIVE_DIR = Path(__file__).resolve().parents[1]
ACTIVE_DIR = ARCHIVE_DIR.parents[1]
sys.path.insert(0, str(ACTIVE_DIR))
sys.path.insert(0, str(ARCHIVE_DIR))

import score_parent_recall_audit as scoring  # noqa: E402


def test_apply_and_trace_distinguishes_recurring_isolated_and_unextracted():
    queue = pd.DataFrame(
        [
            {"audit_id": "a", "report_key": "r1"},
            {"audit_id": "b", "report_key": "r2"},
            {"audit_id": "c", "report_key": "r3"},
        ]
    )
    assignments = pd.DataFrame(
        [
            {"issue_id": "i1", "report_key": "r1", "relational_group_id": "g1", "recurrence_status": "recurring"},
            {"issue_id": "i2", "report_key": "r2", "relational_group_id": "g2", "recurrence_status": "isolated_or_pair"},
        ]
    )
    payload = [
        {"audit_id": "a", "decision": "yes", "matched_issue_id": "i1"},
        {"audit_id": "b", "decision": "yes", "matched_issue_id": "i2"},
        {"audit_id": "c", "decision": "yes", "matched_issue_id": ""},
    ]

    result = scoring.apply_and_trace(queue, assignments, payload)

    assert result["miss_stage"].tolist() == ["other_recurring_child", "isolated_or_pair", "not_extracted"]
