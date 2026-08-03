#!/usr/bin/env python3
"""Load and validate the locked PFD Toolkit report-topic taxonomy."""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any


DEFAULT_TAXONOMY_PATH = Path(__file__).with_name("locked_topics_v1.json")
TAXONOMY_ID = "pfd-toolkit-topics"
FACETS = frozenset({"concern", "setting", "circumstance_condition", "population"})
TOPIC_ID_PATTERN = re.compile(r"^[a-z][a-z0-9_]*$")


@dataclass(frozen=True)
class Topic:
    id: str
    label: str
    facet: str
    description: str


@dataclass(frozen=True)
class TopicTaxonomy:
    taxonomy_id: str
    version: str
    status: str
    locked_on: str
    assignment_mode: str
    discovery_policy: str
    relationship_to_issues: str
    topics: tuple[Topic, ...]

    @property
    def by_id(self) -> dict[str, Topic]:
        return {topic.id: topic for topic in self.topics}


def _required_text(payload: dict[str, Any], key: str) -> str:
    value = str(payload.get(key, "")).strip()
    if not value:
        raise ValueError(f"Topic taxonomy requires non-empty {key!r}")
    return value


def load_topic_taxonomy(path: Path = DEFAULT_TAXONOMY_PATH) -> TopicTaxonomy:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if payload.get("taxonomy_id") != TAXONOMY_ID:
        raise ValueError(f"Unexpected topic taxonomy ID: {payload.get('taxonomy_id')!r}")
    if payload.get("status") != "locked":
        raise ValueError("Topic taxonomy must have locked status")
    if payload.get("assignment_mode") != "multi_label":
        raise ValueError("Topic taxonomy must use multi-label assignment")

    raw_topics = payload.get("topics")
    if not isinstance(raw_topics, list) or not raw_topics:
        raise ValueError("Topic taxonomy requires a non-empty topics list")
    if payload.get("topic_count") != len(raw_topics):
        raise ValueError("Declared topic_count does not match topics list")

    topics: list[Topic] = []
    seen_ids: set[str] = set()
    seen_labels: set[str] = set()
    for raw_topic in raw_topics:
        if not isinstance(raw_topic, dict):
            raise ValueError("Every topic must be an object")
        if "parent_id" in raw_topic or "children" in raw_topic:
            raise ValueError("Locked topics must not define hierarchy")
        topic_id = _required_text(raw_topic, "id")
        label = _required_text(raw_topic, "label")
        facet = _required_text(raw_topic, "facet")
        description = _required_text(raw_topic, "description")
        if not TOPIC_ID_PATTERN.fullmatch(topic_id):
            raise ValueError(f"Invalid topic ID: {topic_id!r}")
        if topic_id in seen_ids:
            raise ValueError(f"Duplicate topic ID: {topic_id}")
        normalized_label = label.casefold()
        if normalized_label in seen_labels:
            raise ValueError(f"Duplicate topic label: {label}")
        if facet not in FACETS:
            raise ValueError(f"Unknown topic facet for {topic_id}: {facet}")
        seen_ids.add(topic_id)
        seen_labels.add(normalized_label)
        topics.append(Topic(topic_id, label, facet, description))

    return TopicTaxonomy(
        taxonomy_id=TAXONOMY_ID,
        version=_required_text(payload, "version"),
        status="locked",
        locked_on=_required_text(payload, "locked_on"),
        assignment_mode="multi_label",
        discovery_policy=_required_text(payload, "discovery_policy"),
        relationship_to_issues=_required_text(payload, "relationship_to_issues"),
        topics=tuple(topics),
    )
