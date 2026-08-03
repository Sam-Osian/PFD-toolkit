from __future__ import annotations

import json
import sys
from pathlib import Path

import pytest


SCRIPT_DIR = Path(__file__).resolve().parents[1] / "scripts" / "issue_tracker_mvp"
sys.path.insert(0, str(SCRIPT_DIR))

import topic_taxonomy  # noqa: E402


def test_locked_topic_taxonomy_has_agreed_catalogue():
    taxonomy = topic_taxonomy.load_topic_taxonomy()

    assert taxonomy.version == "1.0.0"
    assert taxonomy.status == "locked"
    assert taxonomy.assignment_mode == "multi_label"
    assert len(taxonomy.topics) == 40
    assert {topic.facet for topic in taxonomy.topics} == topic_taxonomy.FACETS
    assert sum(topic.facet == "concern" for topic in taxonomy.topics) == 27
    assert sum(topic.facet == "setting" for topic in taxonomy.topics) == 6
    assert sum(topic.facet == "circumstance_condition" for topic in taxonomy.topics) == 3
    assert sum(topic.facet == "population" for topic in taxonomy.topics) == 4


def test_locked_topic_taxonomy_preserves_requested_labels_and_split():
    topics = topic_taxonomy.load_topic_taxonomy().by_id

    assert topics["training_competence_supervision"].label == "Training, competence and supervision"
    assert topics["learning_disability"].label == "Learning disability"
    assert topics["autism"].label == "Autism"
    assert "learning_disability_autism" not in topics


def test_locked_topic_taxonomy_is_non_hierarchical():
    payload = json.loads(topic_taxonomy.DEFAULT_TAXONOMY_PATH.read_text(encoding="utf-8"))

    assert all("parent_id" not in topic and "children" not in topic for topic in payload["topics"])
    assert "not parents of extracted issues" in payload["relationship_to_issues"]


def test_loader_rejects_topic_count_mismatch(tmp_path: Path):
    payload = json.loads(topic_taxonomy.DEFAULT_TAXONOMY_PATH.read_text(encoding="utf-8"))
    payload["topic_count"] = 41
    path = tmp_path / "invalid_topics.json"
    path.write_text(json.dumps(payload), encoding="utf-8")

    with pytest.raises(ValueError, match="topic_count"):
        topic_taxonomy.load_topic_taxonomy(path)
