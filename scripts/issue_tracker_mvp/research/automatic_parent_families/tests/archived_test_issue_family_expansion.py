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

import expand_issue_families as expansion  # noqa: E402


def feature_frame() -> pd.DataFrame:
    return pd.DataFrame(
        [
            {
                "child_group_id": "records_a",
                "report_count": 3,
                "occurrence_count": 3,
                "prototype_canonical_issue": "Clinical records were incomplete.",
                "dominant_themes": "records_information",
                "dominant_process_stages": "information_record_management",
            },
            {
                "child_group_id": "records_b",
                "report_count": 3,
                "occurrence_count": 3,
                "prototype_canonical_issue": "Medical notes were missing.",
                "dominant_themes": "records_information",
                "dominant_process_stages": "information_record_management",
            },
            {
                "child_group_id": "mixed_records",
                "report_count": 3,
                "occurrence_count": 3,
                "prototype_canonical_issue": "Observations were not recorded.",
                "dominant_themes": "records_information|clinical_assessment_care",
                "dominant_process_stages": "information_record_management|monitoring_observation",
            },
            {
                "child_group_id": "monitoring",
                "report_count": 3,
                "occurrence_count": 3,
                "prototype_canonical_issue": "Observations were not performed.",
                "dominant_themes": "clinical_assessment_care",
                "dominant_process_stages": "monitoring_observation",
            },
        ]
    )


def test_secondary_membership_requires_similarity_gap_and_structural_evidence():
    features = feature_frame()
    centroids = np.asarray(
        [
            [1.00, 0.00, 0.00],
            [0.99, 0.05, 0.00],
            [0.92, 0.39, 0.00],
            [0.80, 0.60, 0.00],
        ],
        dtype=np.float32,
    )
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    strict = np.asarray([0, 0, 1, 1])
    broad = np.asarray([0, 0, 1, 1])
    cores, owners = expansion.build_cores(
        features,
        centroids,
        strict,
        broad,
        minimum_core_children=2,
    )

    memberships = expansion.attachment_rows(
        features,
        centroids,
        strict,
        broad,
        cores,
        owners,
        minimum_similarity=0.90,
        maximum_primary_gap=0.08,
        maximum_secondary_families=2,
    )

    mixed = memberships[memberships["child_group_id"].eq("mixed_records")]
    assert set(mixed["membership_type"]) == {"core", "secondary"}
    secondary = mixed[mixed["membership_type"].eq("secondary")].iloc[0]
    assert secondary["attachment_evidence"] == "shared_theme_and_stage"
    assert secondary["shared_themes"] == "records_information"
    assert secondary["shared_process_stages"] == "information_record_management"


def test_secondary_membership_is_limited_per_child():
    features = feature_frame()
    centroids = np.asarray(
        [[1.0, 0.0], [0.99, 0.02], [0.98, 0.04], [0.97, 0.06]],
        dtype=np.float32,
    )
    centroids /= np.linalg.norm(centroids, axis=1, keepdims=True)
    strict = np.asarray([0, 0, 1, 1])
    broad = np.asarray([0, 0, 0, 0])
    cores, owners = expansion.build_cores(
        features,
        centroids,
        strict,
        broad,
        minimum_core_children=2,
    )

    memberships = expansion.attachment_rows(
        features,
        centroids,
        strict,
        broad,
        cores,
        owners,
        minimum_similarity=0.90,
        maximum_primary_gap=0.10,
        maximum_secondary_families=1,
    )

    assert memberships.groupby("child_group_id").size().max() == 2


def test_expanded_family_report_count_is_a_deduplicated_union():
    features = feature_frame().iloc[:3].copy()
    members = pd.DataFrame(
        [
            {"child_group_id": "records_a", "report_key": "rpt_1"},
            {"child_group_id": "records_b", "report_key": "rpt_2"},
            {"child_group_id": "mixed_records", "report_key": "rpt_2"},
            {"child_group_id": "mixed_records", "report_key": "rpt_3"},
        ]
    )
    cores = [
        {
            "family_id": "fam_records",
            "centroid": np.asarray([1.0, 0.0], dtype=np.float32),
            "medoid_position": 0,
        }
    ]
    memberships = pd.DataFrame(
        [
            {"family_id": "fam_records", "child_group_id": "records_a", "membership_type": "core", "similarity_to_family_centroid": 0.98},
            {"family_id": "fam_records", "child_group_id": "records_b", "membership_type": "core", "similarity_to_family_centroid": 0.97},
            {"family_id": "fam_records", "child_group_id": "mixed_records", "membership_type": "secondary", "similarity_to_family_centroid": 0.92},
        ]
    ).assign(
        primary_family_id="fam_records",
        primary_similarity=0.98,
        similarity_gap_from_primary=0.0,
        same_broad_cluster=True,
        shared_themes="records_information",
        shared_process_stages="information_record_management",
        attachment_evidence="test",
    )

    summaries, _, _ = expansion.family_outputs(
        features, members, cores, memberships
    )

    assert summaries.iloc[0]["core_report_count"] == 2
    assert summaries.iloc[0]["expanded_report_count"] == 3
    assert summaries.iloc[0]["added_report_count"] == 1
    assert summaries.iloc[0]["family_label_hint"] == (
        "records information — information record management"
    )
