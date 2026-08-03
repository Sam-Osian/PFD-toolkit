from __future__ import annotations

# Archived regression coverage for the superseded automatic-parent experiment.

import sys
from pathlib import Path

import numpy as np
import pandas as pd


ARCHIVE_DIR = Path(__file__).resolve().parents[1]
ACTIVE_DIR = ARCHIVE_DIR.parents[1]
sys.path.insert(0, str(ACTIVE_DIR))
sys.path.insert(0, str(ARCHIVE_DIR))

import discover_issue_families as discovery  # noqa: E402


def test_family_discovery_groups_children_and_deduplicates_reports():
    occurrences = pd.DataFrame(
        [
            {
                "issue_id": "iss_1",
                "report_key": "rpt_1",
                "canonical_issue": "Clinical records were incomplete.",
                "issue_themes": "records_information",
                "process_stage": "information_record_management",
                "service_sectors": "healthcare",
            },
            {
                "issue_id": "iss_2",
                "report_key": "rpt_2",
                "canonical_issue": "Medical notes were missing.",
                "issue_themes": "records_information",
                "process_stage": "information_record_management",
                "service_sectors": "healthcare",
            },
            {
                "issue_id": "iss_3",
                "report_key": "rpt_2",
                "canonical_issue": "Medical records were inaccurate.",
                "issue_themes": "records_information",
                "process_stage": "information_record_management",
                "service_sectors": "healthcare",
            },
            {
                "issue_id": "iss_4",
                "report_key": "rpt_3",
                "canonical_issue": "Staffing levels were insufficient.",
                "issue_themes": "staffing_capacity",
                "process_stage": "workforce_management",
                "service_sectors": "healthcare",
            },
        ]
    )
    embeddings = np.asarray(
        [
            [1.00, 0.00, 0.00],
            [0.99, 0.05, 0.00],
            [0.98, 0.10, 0.00],
            [0.00, 1.00, 0.00],
        ],
        dtype=np.float32,
    )
    embeddings /= np.linalg.norm(embeddings, axis=1, keepdims=True)
    assignments = pd.DataFrame(
        [
            {
                "issue_id": "iss_1",
                "report_key": "rpt_1",
                "relational_group_id": "rel_records_a",
                "recurrence_status": "recurring",
            },
            {
                "issue_id": "iss_2",
                "report_key": "rpt_2",
                "relational_group_id": "rel_records_a",
                "recurrence_status": "recurring",
            },
            {
                "issue_id": "iss_3",
                "report_key": "rpt_2",
                "relational_group_id": "rel_records_b",
                "recurrence_status": "recurring",
            },
            {
                "issue_id": "iss_4",
                "report_key": "rpt_3",
                "relational_group_id": "rel_staffing",
                "recurrence_status": "recurring",
            },
        ]
    )
    groups = pd.DataFrame(
        [
            {
                "relational_group_id": "rel_records_a",
                "recurrence_status": "recurring",
                "prototype_canonical_issue": "Clinical records were incomplete.",
            },
            {
                "relational_group_id": "rel_records_b",
                "recurrence_status": "recurring",
                "prototype_canonical_issue": "Medical records were inaccurate.",
            },
            {
                "relational_group_id": "rel_staffing",
                "recurrence_status": "recurring",
                "prototype_canonical_issue": "Staffing levels were insufficient.",
            },
        ]
    )

    _, _, summaries, family_assignments, family_centroids = discovery.run_discovery(
        occurrences,
        embeddings,
        assignments,
        groups,
        similarities=[0.90],
        selected_similarity=0.90,
        minimum_child_groups=2,
    )

    assert len(summaries) == 1
    family = summaries.iloc[0]
    assert family["child_group_count"] == 2
    assert family["report_count"] == 2
    assert family["largest_child_report_count"] == 2
    assert set(family_assignments["child_group_id"]) == {
        "rel_records_a",
        "rel_records_b",
    }
    assert summaries.iloc[0]["centroid_row"] == 0
    assert family_centroids.shape == (1, 3)


def test_stable_family_id_is_independent_of_child_order():
    assert discovery.stable_family_id(["rel_b", "rel_a"]) == (
        discovery.stable_family_id(["rel_a", "rel_b"])
    )


def test_threshold_sweep_reports_family_sizes():
    labels = np.asarray([0, 0, 0, 1, 2, 2])

    metrics = discovery.threshold_metrics(labels, 0.82)

    assert metrics["families"] == 2
    assert metrics["grouped_child_groups"] == 5
    assert metrics["singleton_child_groups"] == 1
    assert metrics["largest_family_children"] == 3
