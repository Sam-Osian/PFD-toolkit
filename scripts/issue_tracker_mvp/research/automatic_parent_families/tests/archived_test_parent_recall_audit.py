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

import build_parent_recall_audit as audit  # noqa: E402


def test_parse_parent_spec():
    spec = audit.parse_parent_spec("fam_a::Records::theme_a,theme_b")

    assert spec.family_id == "fam_a"
    assert spec.review_name == "Records"
    assert spec.source_theme_columns == ("theme_a", "theme_b")


def test_parse_parent_spec_rejects_incomplete_value():
    with pytest.raises(ValueError, match="must be"):
        audit.parse_parent_spec("fam_a::Records")


def test_choose_strata_excludes_duplicate_reports():
    frame = pd.DataFrame(
        [
            {"report_key": f"r{i}", "family_similarity": 1 - i / 100, "source_theme_selected": i % 2 == 0}
            for i in range(20)
        ]
    )

    result = audit.choose_strata(
        frame, high_semantic=4, source_theme=4, random_source_theme=0,
        minimum_random_similarity=0.0,
        semantic_boundary=4, seed=1
    )

    assert len(result) == 12
    assert result["report_key"].nunique() == 12
    assert result["recall_stratum"].value_counts().to_dict() == {
        "high_semantic": 4,
        "source_theme": 4,
        "semantic_boundary": 4,
    }


def test_random_source_theme_samples_beyond_semantic_top():
    frame = pd.DataFrame(
        [
            {"report_key": f"r{i}", "family_similarity": 1 - i / 100, "source_theme_selected": True}
            for i in range(30)
        ]
    )

    result = audit.choose_strata(
        frame, high_semantic=2, source_theme=0, random_source_theme=5,
        minimum_random_similarity=0.0,
        semantic_boundary=0, seed=3
    )

    assert result["recall_stratum"].value_counts().to_dict() == {
        "random_source_theme": 5,
        "high_semantic": 2,
    }
    assert not set(result.loc[result["recall_stratum"].eq("random_source_theme"), "report_key"]) & {"r0", "r1"}


def test_stable_audit_id_is_parent_specific():
    assert audit.stable_audit_id("fam_a", "report") == audit.stable_audit_id("fam_a", "report")
    assert audit.stable_audit_id("fam_a", "report") != audit.stable_audit_id("fam_b", "report")
