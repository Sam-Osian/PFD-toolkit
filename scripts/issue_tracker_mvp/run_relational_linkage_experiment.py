#!/usr/bin/env python3
"""Evaluate recurrence using relational issue representations.

This is deliberately separate from ``build_issue_index.py``. It consumes v2
normalized occurrences and writes experimental candidates without changing a
completed production run.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from itertools import combinations
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from sklearn.neighbors import NearestNeighbors

import build_issue_index as pipeline


EXPERIMENT_VERSION = "relational-linkage-v3"
# Preserve the fingerprint used by the completed v1 embedding caches. Scoring
# and grouping changes do not alter the four embedded views.
EMBEDDING_FINGERPRINT_VERSION = "relational-linkage-v1"
GENERIC_VALUES = {
    "",
    "action described in evidence",
    "issue described in evidence",
    "issue",
    "problem",
    "service",
    "system",
}
EMPTY_RELATIONAL_VALUES = {"", "not stated", "not applicable", "unclear", "other review"}
GENERIC_ACTOR_ROLES = {
    "",
    "not stated",
    "individual practitioner",
    "provider organisation",
    "provider organisations",
    "team",
    "multi organisation",
    "multi organisations",
    "multi-organisation",
    "multi-organisations",
    "government or public authority",
}
GENERIC_ACTIONS = {
    "provide",
    "maintain",
    "implement",
    "ensure",
    "manage",
    "conduct",
    "perform",
    "complete",
    "review",
    "consider",
}
ACTION_FAMILIES = {
    "assess": "assessment",
    "evaluate": "assessment",
    "review": "assessment",
    "screen": "assessment",
    "conduct": "execution",
    "perform": "execution",
    "undertake": "execution",
    "carry": "execution",
    "complete": "execution",
    "update": "update",
    "contact": "communication",
    "communicate": "communication",
    "inform": "communication",
    "notify": "communication",
    "send": "communication",
    "share": "communication",
    "respond": "response",
    "reply": "response",
    "follow": "follow_up",
    "monitor": "monitoring",
    "observe": "monitoring",
    "supervise": "monitoring",
    "check": "monitoring",
    "provide": "provision",
    "deliver": "provision",
    "offer": "provision",
    "supply": "provision",
    "arrange": "provision",
    "maintain": "maintenance",
    "repair": "maintenance",
    "inspect": "inspection",
    "implement": "implementation",
    "enforce": "implementation",
    "comply": "implementation",
    "escalate": "escalation",
    "refer": "referral",
    "record": "recording",
    "document": "recording",
    "coordinate": "coordination",
    "train": "training",
    "staff": "staffing",
}
OBJECT_STOPWORDS = {
    "a", "an", "and", "for", "in", "of", "on", "the", "to", "with",
}
GENERIC_OBJECT_TOKENS = {
    "action", "actions", "assessment", "assessments", "call", "calls", "care",
    "communication", "communications", "condition", "documentation", "information",
    "guidance", "guideline", "guidelines", "management", "medication", "medications",
    "mental", "national",
    "observation", "observations", "appointment", "appointments",
    "issue", "issues", "need", "needs", "patient", "patients", "plan", "planning",
    "policy", "policies", "procedure", "procedures", "protocol", "protocols",
    "provision", "provisions", "referral", "referrals", "review", "reviews",
    "risk", "risks",
    "process", "report", "reports", "record", "records", "response", "responses",
    "result", "results", "service", "services", "staff", "system", "systems",
    "training",
}

# Context words can identify a subject area without identifying the corrective
# obligation. They must not, by themselves, justify merging established groups
# (for example road markings with road studs).
CONSOLIDATION_CONTEXT_TOKENS = {
    "clinical",
    "department",
    "health",
    "hospital",
    "medical",
    "organisation",
    "patient",
    "provider",
    "road",
    "roadway",
    "safe",
    "safety",
    "service",
}

FAILURE_FAMILIES = {
    "omitted": "absence",
    "delayed": "timeliness",
    "excessive": "timeliness",
    "incomplete": "quality",
    "inadequate": "quality",
    "inconsistent": "quality",
    "unverified": "quality",
    "incorrect": "incorrect",
    "unsafe": "unsafe",
    "non compliant": "non_compliance",
    "unavailable": "availability",
    "inaccessible": "availability",
    "ambiguous": "ambiguity",
    "uncoordinated": "coordination",
}


@dataclass(frozen=True)
class PairScore:
    left: int
    right: int
    embedding_similarity: float
    compatibility_adjustment: float
    adjusted_similarity: float
    reasons: tuple[str, ...]
    canonical_similarity: float = 0.0
    relation_similarity: float = 0.0
    action_object_similarity: float = 0.0
    roles_similarity: float = 0.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run a conservative relational-linkage recurrence experiment."
    )
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument(
        "--embedding-cache-dir",
        type=Path,
        default=None,
        help="Optional shared cache directory for threshold comparisons.",
    )
    parser.add_argument(
        "--embedding-model",
        default="Qwen/Qwen3-Embedding-8B",
    )
    parser.add_argument("--allow-model-download", action="store_true")
    parser.add_argument("--embedding-batch-size", type=int, default=48)
    parser.add_argument("--retrieval-k", type=int, default=80)
    parser.add_argument("--mutual-top-k", type=int, default=40)
    parser.add_argument("--minimum-retrieval-similarity", type=float, default=0.68)
    parser.add_argument("--minimum-edge-score", type=float, default=0.82)
    parser.add_argument("--minimum-group-score", type=float, default=0.80)
    parser.add_argument(
        "--scoring-mode",
        choices=("baseline", "guarded"),
        default="baseline",
        help="Use the frozen v1 arithmetic or guarded full-corpus scoring.",
    )
    parser.add_argument(
        "--minimum-relation-similarity",
        type=float,
        default=0.78,
        help="Guarded-mode floor for the full relational embedding view.",
    )
    parser.add_argument(
        "--minimum-action-object-similarity",
        type=float,
        default=0.78,
        help="Guarded-mode floor for the action-object embedding view.",
    )
    parser.add_argument(
        "--minimum-specific-object-similarity",
        type=float,
        default=0.86,
        help="Guarded fallback when discriminative object tokens do not overlap.",
    )
    parser.add_argument(
        "--maximum-positive-adjustment",
        type=float,
        default=0.03,
        help="Maximum cumulative positive bonus in guarded mode.",
    )
    parser.add_argument(
        "--grouping-mode",
        choices=("prototype", "constrained_density"),
        default="prototype",
    )
    parser.add_argument("--minimum-edge-density", type=float, default=0.60)
    parser.add_argument("--minimum-member-coverage", type=float, default=0.34)
    parser.add_argument("--min-recurring-reports", type=int, default=3)
    parser.add_argument(
        "--consolidate-compatible-groups",
        action=argparse.BooleanOptionalAction,
        default=False,
        help="Merge strongly compatible guarded seed groups after clustering.",
    )
    parser.add_argument(
        "--minimum-consolidation-coverage",
        type=float,
        default=0.50,
        help="Fraction of the smaller group needing cross-group edge support.",
    )
    parser.add_argument(
        "--maximum-consolidation-conflict-fraction",
        type=float,
        default=0.15,
        help=(
            "Maximum fraction of all cross-member pairs that may trigger a "
            "guarded conflict. Zero preserves the strict all-pairs veto."
        ),
    )
    parser.add_argument(
        "--minimum-consolidation-distinctive-object-coverage",
        type=float,
        default=0.45,
        help=(
            "Minimum fraction required on both groups for accepted cross-edges "
            "with substantially shared non-context object tokens. Zero disables "
            "this guard."
        ),
    )
    parser.add_argument(
        "--minimum-consolidation-object-jaccard",
        type=float,
        default=0.50,
        help="Minimum non-context object-token Jaccard for distinctive support.",
    )
    parser.add_argument(
        "--audit-run-dir",
        type=Path,
        default=None,
        help="Optional completed run containing the manual validation CSVs.",
    )
    parser.add_argument(
        "--canonical-weight",
        type=float,
        default=0.10,
        help="Weight of the readable canonical sentence.",
    )
    parser.add_argument(
        "--relation-weight",
        type=float,
        default=0.55,
        help="Weight of actor-action-object-counterparty plus failure/direction.",
    )
    parser.add_argument("--action-object-weight", type=float, default=0.25)
    parser.add_argument("--roles-weight", type=float, default=0.10)
    return parser.parse_args()


def readable(value: Any) -> str:
    return pipeline.clean_text(value).replace("_", " ").replace("|", ", ")


def key(value: Any) -> str:
    return pipeline.normalised_key(value).replace("_", " ")


def linkage_quality(row: pd.Series) -> tuple[bool, str]:
    if key(row.get("normalization_status")) != "success":
        return False, "normalization_not_successful"
    action = key(row.get("failed_action"))
    issue_object = key(row.get("issue_object"))
    if action in GENERIC_VALUES:
        return False, "missing_or_generic_failed_action"
    if issue_object in GENERIC_VALUES:
        return False, "missing_or_generic_issue_object"
    # Direction remains usable relational evidence when the source does not
    # support a narrower counterparty. Keep such rows linkable but separately
    # flagged by the normalization review queue.
    return True, "eligible"


def embedding_views(row: pd.Series) -> tuple[str, str, str, str]:
    actor = readable(row.get("responsible_actor_role"))
    action = readable(row.get("failed_action"))
    issue_object = readable(row.get("issue_object"))
    counterparty = readable(row.get("counterparty_role"))
    failure = readable(row.get("failure_state"))
    stage = readable(row.get("process_stage"))
    direction = readable(row.get("communication_direction"))
    relation = (
        f"Actor: {actor}. Failed action: {action}. Object: {issue_object}. "
        f"Counterparty: {counterparty}. Failure: {failure}. "
        f"Stage: {stage}. Direction: {direction}."
    )
    action_object = f"Failed action: {action}. Object: {issue_object}."
    roles = f"Actor: {actor}. Counterparty: {counterparty}. Direction: {direction}."
    return readable(row.get("canonical_issue")), relation, action_object, roles


def embedding_fingerprint(
    frame: pd.DataFrame, model_name: str, weights: list[float]
) -> str:
    values = {
        "version": EMBEDDING_FINGERPRINT_VERSION,
        "model": model_name,
        "weights": weights,
        "rows": [
            [pipeline.clean_text(row.get("issue_id")), *embedding_views(row)]
            for _, row in frame.iterrows()
        ],
    }
    return hashlib.sha256(
        json.dumps(values, ensure_ascii=False, sort_keys=True).encode("utf-8")
    ).hexdigest()


def encode_weighted_views(
    frame: pd.DataFrame,
    *,
    model_name: str,
    allow_model_download: bool,
    batch_size: int,
    weights: list[float],
    output_dir: Path,
) -> np.ndarray:
    fingerprint = embedding_fingerprint(frame, model_name, weights)
    array_path = output_dir / "relational_embeddings.npy"
    metadata_path = output_dir / "relational_embeddings.meta.json"
    if array_path.exists() and metadata_path.exists():
        metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        if metadata.get("fingerprint") == fingerprint:
            cached = np.load(array_path)
            if cached.shape[0] == len(frame):
                return np.asarray(cached, dtype=np.float32)
    views = [embedding_views(row) for _, row in frame.iterrows()]
    flattened = [text for row_views in views for text in row_views]
    encoded = pipeline.encode_embeddings(
        flattened,
        model_name,
        allow_model_download=allow_model_download,
        batch_size=batch_size,
    ).reshape(len(frame), 4, -1)
    if all(weight > 0 for weight in weights):
        for position, weight in enumerate(weights):
            encoded[:, position, :] *= math.sqrt(weight)
        combined = encoded.reshape(len(frame), -1)
    else:
        blocks = [
            encoded[:, position, :] * math.sqrt(weight)
            for position, weight in enumerate(weights)
            if weight > 0
        ]
        combined = np.concatenate(blocks, axis=1)
    combined /= np.maximum(np.linalg.norm(combined, axis=1, keepdims=True), 1e-12)
    combined = np.asarray(combined, dtype=np.float32)
    np.save(array_path, combined)
    metadata_path.write_text(
        json.dumps(
            {
                "fingerprint": fingerprint,
                "model": model_name,
                "weights": weights,
                "rows": len(frame),
                "dimensions": int(combined.shape[1]),
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return combined


def first_word(value: Any) -> str:
    words = key(value).split()
    return words[0] if words else ""


def action_family(value: Any) -> str:
    first = first_word(value)
    return ACTION_FAMILIES.get(first, first)


def object_tokens(value: Any) -> set[str]:
    return {
        token
        for token in key(value).split()
        if len(token) > 1 and token not in OBJECT_STOPWORDS
    }


def discriminative_object_tokens(value: Any) -> set[str]:
    return {
        token[:-1] if token.endswith("s") and len(token) > 4 else token
        for token in object_tokens(value) - GENERIC_OBJECT_TOKENS
    }


def consolidation_object_tokens(value: Any) -> set[str]:
    """Return object tokens specific enough to support a group merge."""
    return discriminative_object_tokens(value) - CONSOLIDATION_CONTEXT_TOKENS


def generic_object_signature(value: Any) -> str:
    """Return the broad concept of an otherwise underspecified object."""
    tokens = object_tokens(value)
    concepts = (
        ("risk", {"risk", "risks"}),
        ("observation", {"observation", "observations"}),
        ("appointment", {"appointment", "appointments"}),
        ("record", {"record", "records", "documentation"}),
        ("communication", {"communication", "communications", "information"}),
        ("care", {"care"}),
        ("policy", {"policy", "policies", "procedure", "procedures"}),
        ("training", {"training", "guidance", "guideline", "guidelines"}),
    )
    for name, members in concepts:
        if tokens & members:
            return name
    return " ".join(sorted(tokens))


def actor_family(value: Any) -> str:
    value_key = key(value)
    if value_key in EMPTY_RELATIONAL_VALUES:
        return ""
    if "practitioner" in value_key or value_key in {"doctor", "clinician"}:
        return "practitioner"
    if "manufacturer" in value_key or "supplier" in value_key:
        return "manufacturer_supplier"
    if "regulator" in value_key or "inspector" in value_key:
        return "regulator"
    if "employer" in value_key or "operator" in value_key:
        return "employer_operator"
    if any(
        term in value_key
        for term in ("local authority", "highway authority", "public authority", "government")
    ):
        return "public_authority"
    if "multi" in value_key and "organisation" in value_key:
        return "multi_organisation"
    if "team" in value_key:
        return "team"
    if any(
        term in value_key
        for term in (
            "provider organisation",
            "provider organisations",
            "healthcare provider",
            "mental health service",
            "ambulance service",
        )
    ):
        return "provider_service"
    return value_key


def counterparty_family(value: Any) -> str:
    value_key = key(value)
    if value_key in EMPTY_RELATIONAL_VALUES:
        return ""
    if value_key in {"patient", "patients", "individual", "individuals"}:
        return "patient"
    if any(term in value_key for term in ("family", "carer", "parent")):
        return "family_carer"
    if value_key.startswith("gp") or "general practitioner" in value_key:
        return "gp"
    if any(
        term in value_key
        for term in ("staff", "clinician", "doctor", "consultant", "nurse", "officer")
    ):
        return "professional_staff"
    if any(term in value_key for term in ("public", "road user", "pedestrian", "motorist")):
        return "public"
    if any(term in value_key for term in ("organisation", "service", "team", "clinic")):
        return "organisation"
    return value_key


def direction_family(value: Any) -> str:
    value_key = key(value)
    if value_key in EMPTY_RELATIONAL_VALUES:
        return ""
    if value_key in {"within team", "between teams same organisation"}:
        return "internal_professional"
    return value_key


def failure_family(value: Any) -> str:
    value_key = key(value)
    if value_key in EMPTY_RELATIONAL_VALUES:
        return ""
    return FAILURE_FAMILIES.get(value_key, value_key)


def is_communication_relation(row: pd.Series) -> bool:
    return (
        action_family(row.get("failed_action"))
        in {"communication", "response", "referral", "escalation", "handover"}
        or direction_family(row.get("communication_direction"))
        not in {"", "not applicable"}
    )


def ambulance_relation_direction(row: pd.Series) -> str:
    """Distinguish summoning an ambulance from the service responding."""
    object_key = key(row.get("issue_object"))
    canonical_key = key(row.get("canonical_issue"))
    if "ambulance" not in object_key and "ambulance" not in canonical_key:
        return ""
    action_key = key(row.get("failed_action"))
    if (
        first_word(action_key) in {"call", "request", "summon"}
        or "decide on ambulance call" in action_key
        or "contact emergency service" in action_key
    ):
        return "summon_ambulance"
    if first_word(action_key) in {"respond", "dispatch", "attend", "arrive"}:
        return "ambulance_service_response"
    return ""


def view_similarities(
    embeddings: np.ndarray,
    left: int,
    right: int,
    weights: list[float],
) -> tuple[float, float, float, float]:
    """Recover per-view cosine similarities from weighted concatenation."""
    if len(weights) != 4 or any(weight <= 0 for weight in weights):
        raise ValueError("Per-view scoring requires four positive embedding weights")
    if embeddings.shape[1] % 4:
        raise ValueError("Combined embedding dimensions are not divisible into four views")
    width = embeddings.shape[1] // 4
    values: list[float] = []
    for position, weight in enumerate(weights):
        start = position * width
        stop = start + width
        weighted_dot = float(
            embeddings[left, start:stop] @ embeddings[right, start:stop]
        )
        values.append(max(-1.0, min(1.0, weighted_dot / weight)))
    return values[0], values[1], values[2], values[3]


def guarded_relation_conflict(left: pd.Series, right: pd.Series) -> str:
    """Return a cannot-link reason for material relational disagreement."""
    baseline = hard_relation_conflict(left, right)
    # Guarded view-specific scoring replaces the broad v1
    # incompatible-action/object rule, which was too sensitive to verb choice.
    if baseline and baseline != "incompatible_action_and_object":
        return baseline
    left_failure = failure_family(left.get("failure_state"))
    right_failure = failure_family(right.get("failure_state"))
    high_contrast_failure_pairs = {
        frozenset(("absence", "timeliness")),
        frozenset(("absence", "incorrect")),
        frozenset(("absence", "unsafe")),
        frozenset(("timeliness", "incorrect")),
        frozenset(("timeliness", "unsafe")),
        frozenset(("incorrect", "unsafe")),
    }
    if (
        left_failure
        and right_failure
        and frozenset((left_failure, right_failure)) in high_contrast_failure_pairs
    ):
        return "incompatible_failure_state"
    left_action_family = action_family(left.get("failed_action"))
    right_action_family = action_family(right.get("failed_action"))
    incompatible_action_pairs = {
        frozenset(("recording", "execution")),
        frozenset(("recording", "assessment")),
        frozenset(("recording", "provision")),
        frozenset(("recording", "update")),
        frozenset(("recording", "implementation")),
        frozenset(("update", "execution")),
        frozenset(("update", "assessment")),
        frozenset(("update", "provision")),
        frozenset(("provision", "execution")),
        frozenset(("provision", "assessment")),
    }
    if (
        left_action_family
        and right_action_family
        and frozenset((left_action_family, right_action_family))
        in incompatible_action_pairs
    ):
        return "incompatible_action_family"
    left_ambulance_direction = ambulance_relation_direction(left)
    right_ambulance_direction = ambulance_relation_direction(right)
    if (
        left_ambulance_direction
        and right_ambulance_direction
        and left_ambulance_direction != right_ambulance_direction
    ):
        return "incompatible_ambulance_direction"
    left_direction = direction_family(left.get("communication_direction"))
    right_direction = direction_family(right.get("communication_direction"))
    if (
        is_communication_relation(left)
        or is_communication_relation(right)
    ) and left_direction and right_direction and left_direction != right_direction:
        return "incompatible_communication_direction"
    left_counterparty = counterparty_family(left.get("counterparty_role"))
    right_counterparty = counterparty_family(right.get("counterparty_role"))
    if (
        is_communication_relation(left)
        or is_communication_relation(right)
    ) and left_counterparty and right_counterparty and left_counterparty != right_counterparty:
        return "incompatible_counterparty"
    return ""


def guarded_object_conflict(
    left: pd.Series,
    right: pd.Series,
    *,
    action_object_similarity: float,
    minimum_specific_object_similarity: float,
) -> str:
    left_tokens = discriminative_object_tokens(left.get("issue_object"))
    right_tokens = discriminative_object_tokens(right.get("issue_object"))
    if not left_tokens and not right_tokens:
        # A semantically generic head (for example "risk assessment") must
        # not become a bridge between otherwise different concerns. Permit
        # automatic linkage only when its operational relation is exact.
        if (
            generic_object_signature(left.get("issue_object"))
            != generic_object_signature(right.get("issue_object"))
            or action_family(left.get("failed_action"))
            != action_family(right.get("failed_action"))
            or actor_family(left.get("responsible_actor_role"))
            != actor_family(right.get("responsible_actor_role"))
            or failure_family(left.get("failure_state"))
            != failure_family(right.get("failure_state"))
        ):
            return "underspecified_generic_object"
        if action_object_similarity < minimum_specific_object_similarity:
            return "generic_object_below_similarity_floor"
        return ""
    if not left_tokens or not right_tokens:
        if action_object_similarity < minimum_specific_object_similarity:
            return "generic_object_below_similarity_floor"
        return ""
    if left_tokens & right_tokens:
        return ""
    if action_object_similarity < minimum_specific_object_similarity:
        return "disjoint_specific_objects"
    return ""


def guarded_adjustment(
    left: pd.Series,
    right: pd.Series,
    *,
    maximum_positive_adjustment: float,
) -> tuple[float, tuple[str, ...]]:
    """Use facets as bounded corroboration rather than additive identity."""
    adjustment = 0.0
    reasons: list[str] = []
    left_action = key(left.get("failed_action"))
    right_action = key(right.get("failed_action"))
    if (
        left_action == right_action
        and left_action not in GENERIC_ACTIONS
        and left_action not in GENERIC_VALUES
    ):
        adjustment += 0.015
        reasons.append("specific_same_action:+0.015")
    elif action_family(left_action) != action_family(right_action):
        adjustment -= 0.05
        reasons.append("different_action_family:-0.05")

    left_object = discriminative_object_tokens(left.get("issue_object"))
    right_object = discriminative_object_tokens(right.get("issue_object"))
    union = left_object | right_object
    overlap = len(left_object & right_object) / len(union) if union else 0.0
    if overlap >= 0.5:
        adjustment += 0.02
        reasons.append("specific_object_overlap_high:+0.02")
    elif overlap == 0:
        adjustment -= 0.04
        reasons.append("specific_object_overlap_none:-0.04")

    left_actor_key = key(left.get("responsible_actor_role"))
    right_actor_key = key(right.get("responsible_actor_role"))
    if (
        left_actor_key == right_actor_key
        and left_actor_key not in GENERIC_ACTOR_ROLES
        and left_actor_key not in EMPTY_RELATIONAL_VALUES
    ):
        adjustment += 0.01
        reasons.append("specific_same_actor:+0.01")

    for field, extractor, bonus, penalty in (
        ("counterparty", counterparty_family, 0.01, -0.06),
        ("direction", direction_family, 0.01, -0.08),
        ("failure", failure_family, 0.01, -0.05),
    ):
        left_value = extractor(
            left.get(
                {
                    "counterparty": "counterparty_role",
                    "direction": "communication_direction",
                    "failure": "failure_state",
                }[field]
            )
        )
        right_value = extractor(
            right.get(
                {
                    "counterparty": "counterparty_role",
                    "direction": "communication_direction",
                    "failure": "failure_state",
                }[field]
            )
        )
        if left_value and right_value and left_value == right_value:
            adjustment += bonus
            reasons.append(f"same_{field}:+{bonus:g}")
        elif left_value and right_value:
            adjustment += penalty
            reasons.append(f"different_{field}:{penalty:g}")

    adjustment = max(-0.20, min(maximum_positive_adjustment, adjustment))
    if adjustment == maximum_positive_adjustment:
        reasons.append(f"positive_adjustment_capped:{maximum_positive_adjustment:g}")
    return adjustment, tuple(reasons)


def categorical_adjustment(
    left: pd.Series, right: pd.Series
) -> tuple[float, tuple[str, ...]]:
    adjustment = 0.0
    reasons: list[str] = []
    left_action = key(left.get("failed_action"))
    right_action = key(right.get("failed_action"))
    if left_action == right_action:
        adjustment += 0.04
        reasons.append("same_action:+0.04")
    elif action_family(left_action) == action_family(right_action):
        adjustment += 0.02
        reasons.append("same_action_family:+0.02")
    else:
        adjustment -= 0.05
        reasons.append("different_action_family:-0.05")

    left_object = object_tokens(left.get("issue_object"))
    right_object = object_tokens(right.get("issue_object"))
    union = left_object | right_object
    overlap = len(left_object & right_object) / len(union) if union else 0.0
    shared = left_object & right_object
    left_discriminative = discriminative_object_tokens(left.get("issue_object"))
    right_discriminative = discriminative_object_tokens(right.get("issue_object"))
    if (
        shared
        and shared <= GENERIC_OBJECT_TOKENS
        and left_discriminative
        and right_discriminative
        and not (left_discriminative & right_discriminative)
    ):
        adjustment -= 0.08
        reasons.append("object_shared_terms_generic_only:-0.08")
    elif overlap >= 0.5:
        adjustment += 0.035
        reasons.append("object_overlap_high:+0.035")
    elif overlap == 0:
        adjustment -= 0.04
        reasons.append("object_overlap_none:-0.04")

    for field, same_bonus, mismatch_penalty in (
        ("responsible_actor_role", 0.02, -0.025),
        ("counterparty_role", 0.025, -0.05),
        ("communication_direction", 0.02, -0.055),
        ("process_stage", 0.015, -0.025),
        ("failure_state", 0.01, -0.015),
    ):
        left_value = key(left.get(field))
        right_value = key(right.get(field))
        if left_value == right_value and left_value not in EMPTY_RELATIONAL_VALUES:
            adjustment += same_bonus
            reasons.append(f"same_{field}:+{same_bonus:g}")
        elif (
            left_value not in EMPTY_RELATIONAL_VALUES
            and right_value not in EMPTY_RELATIONAL_VALUES
        ):
            adjustment += mismatch_penalty
            reasons.append(f"different_{field}:{mismatch_penalty:g}")
    return adjustment, tuple(reasons)


def score_candidates(
    frame: pd.DataFrame,
    embeddings: np.ndarray,
    *,
    retrieval_k: int,
    minimum_similarity: float,
    weights: list[float] | None = None,
    scoring_mode: str = "baseline",
    minimum_relation_similarity: float = 0.80,
    minimum_action_object_similarity: float = 0.82,
    minimum_specific_object_similarity: float = 0.88,
    maximum_positive_adjustment: float = 0.03,
) -> list[PairScore]:
    count = len(frame)
    if count < 2:
        return []
    neighbours = min(count, max(2, retrieval_k + 1))
    index = NearestNeighbors(n_neighbors=neighbours, metric="cosine", n_jobs=-1)
    index.fit(embeddings)
    distances, indices = index.kneighbors(embeddings)
    report_keys = frame["report_key"].astype(str).tolist()
    directed: list[dict[int, float]] = [dict() for _ in range(count)]
    for left in range(count):
        for distance, raw_right in zip(distances[left], indices[left], strict=False):
            right = int(raw_right)
            similarity = 1.0 - float(distance)
            if (
                right == left
                or report_keys[left] == report_keys[right]
                or similarity < minimum_similarity
            ):
                continue
            directed[left][right] = similarity
    pairs: list[PairScore] = []
    for left in range(count):
        for right, similarity in directed[left].items():
            if right <= left or left not in directed[right]:
                continue
            similarity = min(similarity, directed[right][left])
            canonical_similarity = relation_similarity = 0.0
            action_object_similarity = roles_similarity = 0.0
            if scoring_mode == "guarded":
                if weights is None:
                    raise ValueError("Guarded scoring requires embedding view weights")
                (
                    canonical_similarity,
                    relation_similarity,
                    action_object_similarity,
                    roles_similarity,
                ) = view_similarities(embeddings, left, right, weights)
                conflict = guarded_relation_conflict(
                    frame.iloc[left], frame.iloc[right]
                )
                if not conflict and relation_similarity < minimum_relation_similarity:
                    conflict = "relation_view_below_floor"
                if (
                    not conflict
                    and action_object_similarity < minimum_action_object_similarity
                ):
                    conflict = "action_object_view_below_floor"
                if not conflict:
                    conflict = guarded_object_conflict(
                        frame.iloc[left],
                        frame.iloc[right],
                        action_object_similarity=action_object_similarity,
                        minimum_specific_object_similarity=(
                            minimum_specific_object_similarity
                        ),
                    )
                if conflict:
                    adjustment = -1.0
                    reasons = (f"guarded_veto:{conflict}",)
                    adjusted = -1.0
                else:
                    adjustment, reasons = guarded_adjustment(
                        frame.iloc[left],
                        frame.iloc[right],
                        maximum_positive_adjustment=maximum_positive_adjustment,
                    )
                    adjusted = max(-1.0, min(0.999999, similarity + adjustment))
            else:
                adjustment, reasons = categorical_adjustment(
                    frame.iloc[left], frame.iloc[right]
                )
                adjusted = max(-1.0, min(1.0, similarity + adjustment))
            pairs.append(
                PairScore(
                    left=left,
                    right=right,
                    embedding_similarity=similarity,
                    compatibility_adjustment=adjustment,
                    adjusted_similarity=adjusted,
                    reasons=reasons,
                    canonical_similarity=canonical_similarity,
                    relation_similarity=relation_similarity,
                    action_object_similarity=action_object_similarity,
                    roles_similarity=roles_similarity,
                )
            )
    return pairs


def mutual_edges(
    pairs: list[PairScore], count: int, top_k: int, minimum_score: float
) -> list[PairScore]:
    ranked: list[list[PairScore]] = [[] for _ in range(count)]
    for pair in pairs:
        if pair.adjusted_similarity < minimum_score:
            continue
        ranked[pair.left].append(pair)
        ranked[pair.right].append(pair)
    selected: list[set[tuple[int, int]]] = []
    for node, candidates in enumerate(ranked):
        candidates.sort(key=lambda item: item.adjusted_similarity, reverse=True)
        selected.append(
            {
                (min(item.left, item.right), max(item.left, item.right))
                for item in candidates[:top_k]
                if node in {item.left, item.right}
            }
        )
    return [
        pair
        for pair in pairs
        if pair.adjusted_similarity >= minimum_score
        and (pair.left, pair.right) in selected[pair.left]
        and (pair.left, pair.right) in selected[pair.right]
    ]


def complete_link_groups(
    frame: pd.DataFrame,
    edges: list[PairScore],
    *,
    minimum_group_score: float,
) -> list[list[int]]:
    """Merge only when every cross-pair is supported and compatible."""
    clusters: dict[int, set[int]] = {index: {index} for index in range(len(frame))}
    owner = list(range(len(frame)))
    edge_scores = {
        (min(edge.left, edge.right), max(edge.left, edge.right)): edge.adjusted_similarity
        for edge in edges
    }
    report_keys = frame["report_key"].astype(str).tolist()
    for edge in sorted(edges, key=lambda item: item.adjusted_similarity, reverse=True):
        left_owner = owner[edge.left]
        right_owner = owner[edge.right]
        if left_owner == right_owner:
            continue
        left_members = clusters[left_owner]
        right_members = clusters[right_owner]
        reports = [report_keys[index] for index in left_members | right_members]
        if len(reports) != len(set(reports)):
            continue
        cross_scores = [
            edge_scores.get((min(left, right), max(left, right)))
            for left in left_members
            for right in right_members
        ]
        if any(score is None or score < minimum_group_score for score in cross_scores):
            continue
        merged = left_members | right_members
        clusters[left_owner] = merged
        del clusters[right_owner]
        for member in merged:
            owner[member] = left_owner
    return [sorted(members) for members in clusters.values()]


def hard_relation_conflict(left: pd.Series, right: pd.Series) -> str:
    left_direction = key(left.get("communication_direction"))
    right_direction = key(right.get("communication_direction"))
    opposite_directions = (
        {"professional to patient", "patient family to professional"},
        {"service to public", "public to service"},
    )
    if any({left_direction, right_direction} == pair for pair in opposite_directions):
        return "opposite_communication_direction"
    left_actor = key(left.get("responsible_actor_role"))
    right_actor = key(right.get("responsible_actor_role"))
    left_counterparty = key(left.get("counterparty_role"))
    right_counterparty = key(right.get("counterparty_role"))
    if (
        left_actor not in EMPTY_RELATIONAL_VALUES
        and right_actor not in EMPTY_RELATIONAL_VALUES
        and left_counterparty not in EMPTY_RELATIONAL_VALUES
        and right_counterparty not in EMPTY_RELATIONAL_VALUES
        and left_actor == right_counterparty
        and right_actor == left_counterparty
    ):
        return "actor_counterparty_reversal"
    left_first = first_word(left.get("failed_action"))
    right_first = first_word(right.get("failed_action"))
    left_family = ACTION_FAMILIES.get(left_first)
    right_family = ACTION_FAMILIES.get(right_first)
    if (
        left_family
        and right_family
        and left_family != right_family
        and not (object_tokens(left.get("issue_object")) & object_tokens(right.get("issue_object")))
    ):
        return "incompatible_action_and_object"
    left_object = object_tokens(left.get("issue_object"))
    right_object = object_tokens(right.get("issue_object"))
    shared = left_object & right_object
    left_discriminative = discriminative_object_tokens(left.get("issue_object"))
    right_discriminative = discriminative_object_tokens(right.get("issue_object"))
    if (
        action_family(left.get("failed_action")) == action_family(right.get("failed_action"))
        and shared
        and shared <= GENERIC_OBJECT_TOKENS
        and left_discriminative
        and right_discriminative
        and not (left_discriminative & right_discriminative)
    ):
        return "object_entity_mismatch"
    return ""


def prototype_anchored_groups(
    frame: pd.DataFrame,
    edges: list[PairScore],
    *,
    minimum_group_score: float,
    scoring_mode: str = "baseline",
    allow_same_report_members: bool = False,
) -> list[list[int]]:
    """Grow groups around a member directly linked to every other member.

    Unlike connected components, an arbitrary chain is insufficient. Unlike
    complete-link clustering, two non-prototype paraphrases need not be close
    when both have a strong relation-preserving link to the prototype.
    """
    clusters: dict[int, set[int]] = {index: {index} for index in range(len(frame))}
    owner = list(range(len(frame)))
    edge_scores = {
        (min(edge.left, edge.right), max(edge.left, edge.right)): edge.adjusted_similarity
        for edge in edges
    }
    report_keys = frame["report_key"].astype(str).tolist()
    for edge in sorted(edges, key=lambda item: item.adjusted_similarity, reverse=True):
        left_owner = owner[edge.left]
        right_owner = owner[edge.right]
        if left_owner == right_owner:
            continue
        merged = clusters[left_owner] | clusters[right_owner]
        reports = [report_keys[index] for index in merged]
        if not allow_same_report_members and len(reports) != len(set(reports)):
            continue
        if any(
            (
                guarded_relation_conflict(frame.iloc[left], frame.iloc[right])
                if scoring_mode == "guarded"
                else hard_relation_conflict(frame.iloc[left], frame.iloc[right])
            )
            for position, left in enumerate(sorted(merged))
            for right in sorted(merged)[position + 1 :]
        ):
            continue
        prototypes = [
            candidate
            for candidate in merged
            if discriminative_object_tokens(frame.iloc[candidate].get("issue_object"))
            and all(
                other == candidate
                or edge_scores.get(
                    (min(candidate, other), max(candidate, other)), -1.0
                )
                >= minimum_group_score
                for other in merged
            )
        ]
        if not prototypes:
            continue
        clusters[left_owner] = merged
        del clusters[right_owner]
        for member in merged:
            owner[member] = left_owner
    return [sorted(members) for members in clusters.values()]


def expand_seeded_groups(
    frame: pd.DataFrame,
    seed_groups: list[list[int]],
    expansion_edges: list[PairScore],
    *,
    minimum_group_score: float,
    scoring_mode: str = "baseline",
    allow_same_report_members: bool = False,
) -> list[list[int]]:
    """Attach singletons to high-confidence seeded groups through a prototype."""
    clusters: dict[int, set[int]] = {
        ordinal: set(members) for ordinal, members in enumerate(seed_groups)
    }
    owner = {
        member: ordinal
        for ordinal, members in clusters.items()
        for member in members
    }
    scores = {
        (min(edge.left, edge.right), max(edge.left, edge.right)): edge.adjusted_similarity
        for edge in expansion_edges
    }
    report_keys = frame["report_key"].astype(str).tolist()
    changed = True
    while changed:
        changed = False
        for edge in sorted(
            expansion_edges, key=lambda item: item.adjusted_similarity, reverse=True
        ):
            left_owner = owner[edge.left]
            right_owner = owner[edge.right]
            if left_owner == right_owner:
                continue
            left_members = clusters[left_owner]
            right_members = clusters[right_owner]
            if len(left_members) == 1 and len(right_members) >= 2:
                singleton_owner, group_owner = left_owner, right_owner
            elif len(right_members) == 1 and len(left_members) >= 2:
                singleton_owner, group_owner = right_owner, left_owner
            else:
                continue
            singleton = next(iter(clusters[singleton_owner]))
            group = clusters[group_owner]
            if (
                not allow_same_report_members
                and report_keys[singleton] in {report_keys[member] for member in group}
            ):
                continue
            if any(
                (
                    guarded_relation_conflict(
                        frame.iloc[singleton], frame.iloc[member]
                    )
                    if scoring_mode == "guarded"
                    else hard_relation_conflict(
                        frame.iloc[singleton], frame.iloc[member]
                    )
                )
                for member in group
            ):
                continue
            prototypes = [
                candidate
                for candidate in group
                if discriminative_object_tokens(
                    frame.iloc[candidate].get("issue_object")
                )
                and all(
                    other == candidate
                    or scores.get(
                        (min(candidate, other), max(candidate, other)), -1.0
                    )
                    >= minimum_group_score
                    for other in group
                )
                and scores.get(
                    (min(candidate, singleton), max(candidate, singleton)), -1.0
                )
                >= minimum_group_score
            ]
            if not prototypes:
                continue
            group.add(singleton)
            del clusters[singleton_owner]
            owner[singleton] = group_owner
            changed = True
    return [sorted(members) for members in clusters.values()]


def guarded_direct_score(
    frame: pd.DataFrame,
    embeddings: np.ndarray,
    left: int,
    right: int,
    *,
    weights: list[float],
    minimum_relation_similarity: float,
    minimum_action_object_similarity: float,
    minimum_specific_object_similarity: float,
    maximum_positive_adjustment: float,
) -> tuple[float, str]:
    similarity = float(embeddings[left] @ embeddings[right])
    _, relation_similarity, action_object_similarity, _ = view_similarities(
        embeddings, left, right, weights
    )
    conflict = guarded_relation_conflict(frame.iloc[left], frame.iloc[right])
    if not conflict and relation_similarity < minimum_relation_similarity:
        conflict = "relation_view_below_floor"
    if not conflict and action_object_similarity < minimum_action_object_similarity:
        conflict = "action_object_view_below_floor"
    if not conflict:
        conflict = guarded_object_conflict(
            frame.iloc[left],
            frame.iloc[right],
            action_object_similarity=action_object_similarity,
            minimum_specific_object_similarity=minimum_specific_object_similarity,
        )
    if conflict:
        return -1.0, conflict
    adjustment, _ = guarded_adjustment(
        frame.iloc[left],
        frame.iloc[right],
        maximum_positive_adjustment=maximum_positive_adjustment,
    )
    return max(-1.0, min(0.999999, similarity + adjustment)), ""


def group_medoid(embeddings: np.ndarray, members: set[int]) -> int:
    ordered = sorted(members)
    local = embeddings[ordered]
    similarities = local @ local.T
    return ordered[int(np.argmax(similarities.mean(axis=1)))]


def consolidate_guarded_groups(
    frame: pd.DataFrame,
    embeddings: np.ndarray,
    groups: list[list[int]],
    edges: list[PairScore],
    *,
    weights: list[float],
    minimum_score: float,
    minimum_coverage: float,
    maximum_conflict_fraction: float = 0.0,
    minimum_distinctive_object_coverage: float = 0.0,
    minimum_object_jaccard: float = 0.0,
    minimum_cluster_reports: int = 0,
    minimum_relation_similarity: float,
    minimum_action_object_similarity: float,
    minimum_specific_object_similarity: float,
    maximum_positive_adjustment: float,
) -> tuple[list[list[int]], list[dict[str, Any]]]:
    """Merge compatible seed groups without a mutual-top-k size ceiling.

    Precise mutual edges still establish support. A merge additionally requires
    a guarded direct match between the current group medoids and cross-edge
    coverage across at least half of the smaller group. Multiple occurrences
    from one report may coexist, but recurrence remains based on distinct
    report keys in ``build_outputs``.
    """
    clusters: dict[int, set[int]] = {
        ordinal: set(members) for ordinal, members in enumerate(groups)
    }
    owner = {
        member: ordinal
        for ordinal, members in clusters.items()
        for member in members
    }
    adjacency: list[set[int]] = [set() for _ in range(len(frame))]
    for edge in edges:
        if edge.adjusted_similarity < minimum_score:
            continue
        adjacency[edge.left].add(edge.right)
        adjacency[edge.right].add(edge.left)
    ordered_edges = sorted(
        edges, key=lambda item: item.adjusted_similarity, reverse=True
    )
    provenance: list[dict[str, Any]] = []
    changed = True
    while changed:
        changed = False
        considered: set[tuple[int, int]] = set()
        for edge in ordered_edges:
            left_owner = owner[edge.left]
            right_owner = owner[edge.right]
            if left_owner == right_owner:
                continue
            pair = (min(left_owner, right_owner), max(left_owner, right_owner))
            if pair in considered:
                continue
            considered.add(pair)
            left_members = clusters[left_owner]
            right_members = clusters[right_owner]
            if minimum_cluster_reports > 0 and (
                frame.iloc[list(left_members)]["report_key"].astype(str).nunique()
                < minimum_cluster_reports
                or frame.iloc[list(right_members)]["report_key"].astype(str).nunique()
                < minimum_cluster_reports
            ):
                continue
            smaller, larger = (
                (left_members, right_members)
                if len(left_members) <= len(right_members)
                else (right_members, left_members)
            )
            supported = sum(bool(adjacency[member] & larger) for member in smaller)
            coverage = supported / max(1, len(smaller))
            if coverage < minimum_coverage:
                continue
            conflict_reasons = Counter(
                reason
                for left in left_members
                for right in right_members
                if (reason := guarded_relation_conflict(
                    frame.iloc[left], frame.iloc[right]
                ))
            )
            cross_pair_count = len(left_members) * len(right_members)
            conflict_fraction = sum(conflict_reasons.values()) / max(
                1, cross_pair_count
            )
            if conflict_fraction > maximum_conflict_fraction:
                continue
            def distinctively_supported(member: int, other: set[int]) -> bool:
                member_tokens = consolidation_object_tokens(
                    frame.iloc[member].get("issue_object")
                )
                for neighbour in adjacency[member] & other:
                    neighbour_tokens = consolidation_object_tokens(
                        frame.iloc[neighbour].get("issue_object")
                    )
                    intersection = member_tokens & neighbour_tokens
                    union = member_tokens | neighbour_tokens
                    jaccard = len(intersection) / len(union) if union else 0.0
                    if intersection and jaccard >= minimum_object_jaccard:
                        return True
                return False

            distinctive_supported_left = sum(
                distinctively_supported(member, right_members)
                for member in left_members
            )
            distinctive_supported_right = sum(
                distinctively_supported(member, left_members)
                for member in right_members
            )
            distinctive_coverage_left = distinctive_supported_left / max(
                1, len(left_members)
            )
            distinctive_coverage_right = distinctive_supported_right / max(
                1, len(right_members)
            )
            distinctive_coverage = min(
                distinctive_coverage_left, distinctive_coverage_right
            )
            if distinctive_coverage < minimum_distinctive_object_coverage:
                continue
            left_medoid = group_medoid(embeddings, left_members)
            right_medoid = group_medoid(embeddings, right_members)
            score, conflict = guarded_direct_score(
                frame,
                embeddings,
                left_medoid,
                right_medoid,
                weights=weights,
                minimum_relation_similarity=minimum_relation_similarity,
                minimum_action_object_similarity=minimum_action_object_similarity,
                minimum_specific_object_similarity=minimum_specific_object_similarity,
                maximum_positive_adjustment=maximum_positive_adjustment,
            )
            if conflict or score < minimum_score:
                continue
            merged = left_members | right_members
            clusters[left_owner] = merged
            del clusters[right_owner]
            for member in merged:
                owner[member] = left_owner
            provenance.append(
                {
                    "left_seed_group": left_owner,
                    "right_seed_group": right_owner,
                    "left_size": len(left_members),
                    "right_size": len(right_members),
                    "cross_edge_coverage": coverage,
                    "cross_pair_count": cross_pair_count,
                    "conflict_count": sum(conflict_reasons.values()),
                    "conflict_fraction": conflict_fraction,
                    "conflict_reasons": json.dumps(
                        dict(conflict_reasons), sort_keys=True
                    ),
                    "distinctive_object_coverage": distinctive_coverage,
                    "distinctive_object_coverage_left": (
                        distinctive_coverage_left
                    ),
                    "distinctive_object_coverage_right": (
                        distinctive_coverage_right
                    ),
                    "medoid_score": score,
                }
            )
            changed = True
    return [sorted(members) for members in clusters.values()], provenance


def constrained_density_groups(
    frame: pd.DataFrame,
    seed_edges: list[PairScore],
    eligible_edges: list[PairScore],
    *,
    minimum_group_score: float,
    minimum_edge_density: float,
    minimum_member_coverage: float,
) -> list[list[int]]:
    """Cluster a sparse positive graph subject to relational cannot-link rules.

    A qualifying high-confidence edge must seed each non-singleton cluster.
    Thereafter clusters may merge only when the combined graph remains dense,
    every member has adequate internal support, reports remain distinct, and no
    member pair violates a hard relational constraint.
    """
    clusters: dict[int, set[int]] = {index: {index} for index in range(len(frame))}
    owner = list(range(len(frame)))
    seed_pairs = {
        (min(edge.left, edge.right), max(edge.left, edge.right))
        for edge in seed_edges
    }
    scores = {
        (min(edge.left, edge.right), max(edge.left, edge.right)): edge.adjusted_similarity
        for edge in eligible_edges
        if edge.adjusted_similarity >= minimum_group_score
    }
    report_keys = frame["report_key"].astype(str).tolist()

    def merge_allowed(left_members: set[int], right_members: set[int]) -> bool:
        merged = left_members | right_members
        ordered = sorted(merged)
        reports = [report_keys[index] for index in ordered]
        if len(reports) != len(set(reports)):
            return False
        if len(left_members) == len(right_members) == 1:
            pair = (min(ordered), max(ordered))
            if pair not in seed_pairs:
                return False
        if len(left_members) > 1 and len(right_members) > 1:
            cross_support = sum(
                (min(left, right), max(left, right)) in scores
                for left in left_members
                for right in right_members
            )
            if cross_support < 2:
                return False
        if any(
            hard_relation_conflict(frame.iloc[left], frame.iloc[right])
            for position, left in enumerate(ordered)
            for right in ordered[position + 1 :]
        ):
            return False
        possible = len(ordered) * (len(ordered) - 1) // 2
        present = sum(
            (left, right) in scores
            for position, left in enumerate(ordered)
            for right in ordered[position + 1 :]
        )
        if possible and present / possible < minimum_edge_density:
            return False
        required_degree = max(
            1, math.ceil(minimum_member_coverage * (len(ordered) - 1))
        )
        return all(
            sum(
                (min(member, other), max(member, other)) in scores
                for other in ordered
                if other != member
            )
            >= required_degree
            for member in ordered
        )

    changed = True
    ordered_edges = sorted(
        eligible_edges, key=lambda item: item.adjusted_similarity, reverse=True
    )
    while changed:
        changed = False
        for edge in ordered_edges:
            left_owner = owner[edge.left]
            right_owner = owner[edge.right]
            if left_owner == right_owner:
                continue
            left_members = clusters[left_owner]
            right_members = clusters[right_owner]
            if not merge_allowed(left_members, right_members):
                continue
            merged = left_members | right_members
            clusters[left_owner] = merged
            del clusters[right_owner]
            for member in merged:
                owner[member] = left_owner
            changed = True
    return [sorted(members) for members in clusters.values()]


def pair_frame(frame: pd.DataFrame, pairs: list[PairScore]) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    for pair in pairs:
        left = frame.iloc[pair.left]
        right = frame.iloc[pair.right]
        rows.append(
            {
                "left_issue_id": left["issue_id"],
                "right_issue_id": right["issue_id"],
                "left_report_key": left["report_key"],
                "right_report_key": right["report_key"],
                "embedding_similarity": pair.embedding_similarity,
                "canonical_similarity": pair.canonical_similarity,
                "relation_similarity": pair.relation_similarity,
                "action_object_similarity": pair.action_object_similarity,
                "roles_similarity": pair.roles_similarity,
                "compatibility_adjustment": pair.compatibility_adjustment,
                "adjusted_similarity": pair.adjusted_similarity,
                "compatibility_reasons": " | ".join(pair.reasons),
                "left_canonical_issue": left["canonical_issue"],
                "right_canonical_issue": right["canonical_issue"],
            }
        )
    return pd.DataFrame(rows)


def build_outputs(
    frame: pd.DataFrame,
    embeddings: np.ndarray,
    groups: list[list[int]],
    *,
    min_recurring_reports: int,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    assignments: list[dict[str, Any]] = []
    summaries: list[dict[str, Any]] = []
    ordered = sorted(groups, key=lambda members: (-len(members), members[0]))
    for ordinal, members in enumerate(ordered, 1):
        local = embeddings[members]
        similarities = local @ local.T
        medoid_position = int(np.argmax(similarities.mean(axis=1)))
        medoid_index = members[medoid_position]
        reports = frame.iloc[members]["report_key"].astype(str).nunique()
        recurring = reports >= min_recurring_reports
        group_id = f"rel_{ordinal:05d}"
        summaries.append(
            {
                "relational_group_id": group_id,
                "recurrence_status": "recurring" if recurring else "isolated_or_pair",
                "occurrence_count": len(members),
                "report_count": int(reports),
                "prototype_issue_id": frame.iloc[medoid_index]["issue_id"],
                "prototype_canonical_issue": frame.iloc[medoid_index]["canonical_issue"],
                "minimum_pair_similarity": (
                    float(similarities[np.triu_indices(len(members), 1)].min())
                    if len(members) > 1
                    else 1.0
                ),
                "median_pair_similarity": (
                    float(np.median(similarities[np.triu_indices(len(members), 1)]))
                    if len(members) > 1
                    else 1.0
                ),
            }
        )
        for member in members:
            assignments.append(
                {
                    "issue_id": frame.iloc[member]["issue_id"],
                    "report_key": frame.iloc[member]["report_key"],
                    "relational_group_id": group_id,
                    "recurrence_status": "recurring" if recurring else "isolated_or_pair",
                    "prototype_similarity": float(
                        embeddings[member] @ embeddings[medoid_index]
                    ),
                }
            )
    return pd.DataFrame(assignments), pd.DataFrame(summaries)


def score_audit(
    assignments: pd.DataFrame,
    accepted_edges: pd.DataFrame,
    audit_run_dir: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    decisions = pd.read_csv(
        audit_run_dir / "09_validation" / "random_group_audit.csv"
    ).fillna("")
    members = pd.read_csv(
        audit_run_dir / "07_quality_audit_random" / "group_review_members.csv"
    ).fillna("")
    links = pd.read_csv(
        audit_run_dir / "09_validation" / "missed_link_audit.csv"
    ).fillna("")
    recurring = assignments[
        assignments["recurrence_status"] == "recurring"
    ].set_index("issue_id")["relational_group_id"].to_dict()
    accepted_pairs = {
        tuple(sorted((str(row["left_issue_id"]), str(row["right_issue_id"]))))
        for _, row in accepted_edges.iterrows()
    }
    rows: list[dict[str, Any]] = []
    expected_positive_pairs: set[tuple[str, str]] = set()
    expected_negative_pairs: set[tuple[str, str]] = set()
    for _, decision in decisions.iterrows():
        source_group = pipeline.clean_text(decision["subissue_id"])
        issue_ids = set(
            members.loc[members["subissue_id"] == source_group, "issue_id"].astype(str)
        )
        present = issue_ids & set(recurring)
        new_groups = {recurring[issue_id] for issue_id in present}
        audit_decision = key(decision["audit_decision"])
        incorrect = {
            value.strip()
            for value in pipeline.clean_text(
                decision.get("incorrect_member_issue_ids")
            ).replace("|", ",").split(",")
            if value.strip()
        }
        if audit_decision == "accept":
            expected_positive_pairs.update(
                tuple(sorted(pair)) for pair in combinations(issue_ids, 2)
            )
            passed = len(present) == len(issue_ids) and len(new_groups) == 1
            criterion = "all reviewed members remain one recurring group"
        elif audit_decision == "partial":
            correct = issue_ids - incorrect
            expected_positive_pairs.update(
                tuple(sorted(pair)) for pair in combinations(correct, 2)
            )
            expected_negative_pairs.update(
                tuple(sorted((correct_id, incorrect_id)))
                for correct_id in correct
                for incorrect_id in incorrect
            )
            correct_groups = {recurring[item] for item in correct if item in recurring}
            core_group = next(iter(correct_groups)) if len(correct_groups) == 1 else ""
            passed = (
                len(correct & set(recurring)) == len(correct)
                and bool(core_group)
                and all(recurring.get(item) != core_group for item in incorrect)
            )
            criterion = "correct core retained and incorrect members detached"
        elif audit_decision == "reject":
            expected_negative_pairs.update(
                tuple(sorted(pair)) for pair in combinations(issue_ids, 2)
            )
            passed = len(present) < 2 or len(new_groups) == len(present)
            criterion = "reviewed members do not recur together"
        else:
            passed = len(new_groups) != 1
            criterion = "previous umbrella group no longer remains intact"
        rows.append(
            {
                "audit_type": "group",
                "case_id": source_group,
                "human_decision": audit_decision,
                "passed": passed,
                "criterion": criterion,
                "reviewed_issue_count": len(issue_ids),
                "recurring_issue_count": len(present),
                "new_group_count": len(new_groups),
            }
        )
    for _, link in links.iterrows():
        left = pipeline.clean_text(link["left_issue_id"])
        right = pipeline.clean_text(link["right_issue_id"])
        expected_same = key(link["same_recurring_issue"]) == "yes"
        both_present = left in recurring and right in recurring
        observed_same = (
            tuple(sorted((left, right))) in accepted_pairs
            or (both_present and recurring[left] == recurring[right])
        )
        rows.append(
            {
                "audit_type": "pair",
                "case_id": f"{left}|{right}",
                "human_decision": "same" if expected_same else "different",
                "passed": observed_same if expected_same else not observed_same,
                "criterion": "pair recurrence agrees with manual review",
                "reviewed_issue_count": 2,
                "recurring_issue_count": int(left in recurring) + int(right in recurring),
                "new_group_count": (
                    len({recurring[item] for item in (left, right) if item in recurring})
                ),
            }
        )
    result = pd.DataFrame(rows)
    positive_retained = sum(
        left in recurring
        and right in recurring
        and recurring[left] == recurring[right]
        for left, right in expected_positive_pairs
    )
    negative_separated = sum(
        not (
            left in recurring
            and right in recurring
            and recurring[left] == recurring[right]
        )
        for left, right in expected_negative_pairs
    )
    by_type = {
        audit_type: {
            "cases": len(group),
            "passed": int(group["passed"].sum()),
            "pass_rate": float(group["passed"].mean()),
        }
        for audit_type, group in result.groupby("audit_type")
    }
    return result, {
        "cases": len(result),
        "passed": int(result["passed"].sum()),
        "pass_rate": float(result["passed"].mean()) if len(result) else 0.0,
        "by_type": by_type,
        "relational_pairwise": {
            "expected_same_pairs": len(expected_positive_pairs),
            "expected_same_retained": int(positive_retained),
            "expected_same_recall": (
                positive_retained / len(expected_positive_pairs)
                if expected_positive_pairs
                else 0.0
            ),
            "expected_different_pairs": len(expected_negative_pairs),
            "expected_different_separated": int(negative_separated),
            "expected_different_specificity": (
                negative_separated / len(expected_negative_pairs)
                if expected_negative_pairs
                else 0.0
            ),
        },
    }


def main() -> None:
    args = parse_args()
    if args.retrieval_k < args.mutual_top_k:
        raise ValueError("--retrieval-k must be at least --mutual-top-k")
    if not 0.0 < args.minimum_edge_density <= 1.0:
        raise ValueError("--minimum-edge-density must be in (0, 1]")
    if not 0.0 < args.minimum_member_coverage <= 1.0:
        raise ValueError("--minimum-member-coverage must be in (0, 1]")
    if not 0.0 < args.minimum_consolidation_coverage <= 1.0:
        raise ValueError("--minimum-consolidation-coverage must be in (0, 1]")
    if not 0.0 <= args.maximum_consolidation_conflict_fraction <= 1.0:
        raise ValueError(
            "--maximum-consolidation-conflict-fraction must be between 0 and 1"
        )
    if not 0.0 <= args.minimum_consolidation_distinctive_object_coverage <= 1.0:
        raise ValueError(
            "--minimum-consolidation-distinctive-object-coverage must be between 0 and 1"
        )
    if not 0.0 <= args.minimum_consolidation_object_jaccard <= 1.0:
        raise ValueError(
            "--minimum-consolidation-object-jaccard must be between 0 and 1"
        )
    if not 0.0 <= args.maximum_positive_adjustment <= 0.20:
        raise ValueError("--maximum-positive-adjustment must be between 0 and 0.20")
    weights = [
        args.canonical_weight,
        args.relation_weight,
        args.action_object_weight,
        args.roles_weight,
    ]
    if any(weight < 0 for weight in weights) or not math.isclose(sum(weights), 1.0):
        raise ValueError("Embedding view weights must be non-negative and sum to 1")
    source = pd.read_csv(args.input_csv).fillna("")
    required = {
        "issue_id",
        "report_key",
        "canonical_issue",
        "responsible_actor_role",
        "failed_action",
        "issue_object",
        "counterparty_role",
        "normalization_status",
    }
    missing = required - set(source.columns)
    if missing:
        raise ValueError(
            "Input must be relational normalization v2; missing "
            f"columns: {sorted(missing)}"
        )
    quality = source.apply(linkage_quality, axis=1)
    source["linkage_eligible"] = [eligible for eligible, _ in quality]
    source["linkage_quality_reason"] = [reason for _, reason in quality]
    selected_mask = source["normalization_status"].astype(str).ne("not_selected")
    frame = source[source["linkage_eligible"]].reset_index(drop=True)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    embedding_cache_dir = args.embedding_cache_dir or args.output_dir
    embedding_cache_dir.mkdir(parents=True, exist_ok=True)
    source.to_csv(args.output_dir / "01_linkage_quality_gate.csv", index=False)
    if frame.empty:
        raise ValueError("No occurrences passed the relational linkage quality gate")
    embeddings = encode_weighted_views(
        frame,
        model_name=args.embedding_model,
        allow_model_download=args.allow_model_download,
        batch_size=args.embedding_batch_size,
        weights=weights,
        output_dir=embedding_cache_dir,
    )
    candidates = score_candidates(
        frame,
        embeddings,
        retrieval_k=args.retrieval_k,
        minimum_similarity=args.minimum_retrieval_similarity,
        weights=weights,
        scoring_mode=args.scoring_mode,
        minimum_relation_similarity=args.minimum_relation_similarity,
        minimum_action_object_similarity=args.minimum_action_object_similarity,
        minimum_specific_object_similarity=args.minimum_specific_object_similarity,
        maximum_positive_adjustment=args.maximum_positive_adjustment,
    )
    edges = mutual_edges(
        candidates,
        len(frame),
        args.mutual_top_k,
        args.minimum_edge_score,
    )
    expansion_edges = mutual_edges(
        candidates,
        len(frame),
        args.mutual_top_k,
        args.minimum_group_score,
    )
    if args.grouping_mode == "prototype":
        seed_groups = prototype_anchored_groups(
            frame,
            edges,
            minimum_group_score=args.minimum_edge_score,
            scoring_mode=args.scoring_mode,
            allow_same_report_members=args.scoring_mode == "guarded",
        )
        groups = expand_seeded_groups(
            frame,
            seed_groups,
            expansion_edges,
            minimum_group_score=args.minimum_group_score,
            scoring_mode=args.scoring_mode,
            allow_same_report_members=args.scoring_mode == "guarded",
        )
    else:
        groups = constrained_density_groups(
            frame,
            edges,
            expansion_edges,
            minimum_group_score=args.minimum_group_score,
            minimum_edge_density=args.minimum_edge_density,
            minimum_member_coverage=args.minimum_member_coverage,
        )
    consolidation_provenance: list[dict[str, Any]] = []
    if args.consolidate_compatible_groups:
        if args.scoring_mode != "guarded":
            raise ValueError(
                "--consolidate-compatible-groups requires --scoring-mode guarded"
            )
        groups, strict_provenance = consolidate_guarded_groups(
            frame,
            embeddings,
            groups,
            expansion_edges,
            weights=weights,
            minimum_score=args.minimum_group_score,
            minimum_coverage=args.minimum_consolidation_coverage,
            minimum_relation_similarity=args.minimum_relation_similarity,
            minimum_action_object_similarity=args.minimum_action_object_similarity,
            minimum_specific_object_similarity=args.minimum_specific_object_similarity,
            maximum_positive_adjustment=args.maximum_positive_adjustment,
        )
        consolidation_provenance.extend(
            {**row, "consolidation_phase": "strict"}
            for row in strict_provenance
        )
        if (
            args.maximum_consolidation_conflict_fraction > 0.0
            or args.minimum_consolidation_distinctive_object_coverage > 0.0
        ):
            groups, tolerant_provenance = consolidate_guarded_groups(
                frame,
                embeddings,
                groups,
                expansion_edges,
                weights=weights,
                minimum_score=args.minimum_group_score,
                minimum_coverage=args.minimum_consolidation_coverage,
                maximum_conflict_fraction=(
                    args.maximum_consolidation_conflict_fraction
                ),
                minimum_distinctive_object_coverage=(
                    args.minimum_consolidation_distinctive_object_coverage
                ),
                minimum_object_jaccard=args.minimum_consolidation_object_jaccard,
                minimum_cluster_reports=args.min_recurring_reports,
                minimum_relation_similarity=args.minimum_relation_similarity,
                minimum_action_object_similarity=args.minimum_action_object_similarity,
                minimum_specific_object_similarity=(
                    args.minimum_specific_object_similarity
                ),
                maximum_positive_adjustment=args.maximum_positive_adjustment,
            )
            consolidation_provenance.extend(
                {**row, "consolidation_phase": "tolerant_additive"}
                for row in tolerant_provenance
            )
    assignments, summaries = build_outputs(
        frame,
        embeddings,
        groups,
        min_recurring_reports=args.min_recurring_reports,
    )
    candidate_output = pair_frame(frame, candidates)
    edge_output = pair_frame(frame, edges)
    candidate_output.to_csv(
        args.output_dir / "02_candidate_pairs.csv", index=False
    )
    edge_output.to_csv(
        args.output_dir / "03_accepted_edges.csv", index=False
    )
    expansion_output = pair_frame(frame, expansion_edges)
    expansion_output.to_csv(
        args.output_dir / "03_expansion_edges.csv", index=False
    )
    pd.DataFrame(consolidation_provenance).to_csv(
        args.output_dir / "03_group_consolidation.csv", index=False
    )
    assignments.to_csv(args.output_dir / "04_group_assignments.csv", index=False)
    summaries.to_csv(args.output_dir / "05_relational_groups.csv", index=False)
    recurring = summaries[summaries["recurrence_status"] == "recurring"]
    metrics = {
        "experiment_version": EXPERIMENT_VERSION,
        "created_at": datetime.now(timezone.utc).isoformat(),
        "input_csv": str(args.input_csv),
        "occurrences_input": len(source),
        "occurrences_selected": int(selected_mask.sum()),
        "occurrences_eligible": len(frame),
        "quality_gate_exclusions": dict(
            Counter(
                source.loc[
                    selected_mask & ~source["linkage_eligible"],
                    "linkage_quality_reason",
                ]
            )
        ),
        "candidate_pairs": len(candidates),
        "accepted_mutual_edges": len(edges),
        "eligible_expansion_edges": len(expansion_edges),
        "groups_total": len(summaries),
        "recurring_groups": len(recurring),
        "recurring_occurrences": int(
            recurring["occurrence_count"].sum() if not recurring.empty else 0
        ),
        "reports_with_recurring_issue": int(
            assignments.loc[
                assignments["recurrence_status"] == "recurring", "report_key"
            ].nunique()
        ),
        "parameters": {
            "embedding_model": args.embedding_model,
            "view_weights": weights,
            "retrieval_k": args.retrieval_k,
            "mutual_top_k": args.mutual_top_k,
            "minimum_retrieval_similarity": args.minimum_retrieval_similarity,
            "minimum_edge_score": args.minimum_edge_score,
            "minimum_group_score": args.minimum_group_score,
            "scoring_mode": args.scoring_mode,
            "minimum_relation_similarity": args.minimum_relation_similarity,
            "minimum_action_object_similarity": (
                args.minimum_action_object_similarity
            ),
            "minimum_specific_object_similarity": (
                args.minimum_specific_object_similarity
            ),
            "maximum_positive_adjustment": args.maximum_positive_adjustment,
            "grouping_mode": args.grouping_mode,
            "minimum_edge_density": args.minimum_edge_density,
            "minimum_member_coverage": args.minimum_member_coverage,
            "min_recurring_reports": args.min_recurring_reports,
            "consolidate_compatible_groups": args.consolidate_compatible_groups,
            "minimum_consolidation_coverage": (
                args.minimum_consolidation_coverage
            ),
            "maximum_consolidation_conflict_fraction": (
                args.maximum_consolidation_conflict_fraction
            ),
            "minimum_consolidation_distinctive_object_coverage": (
                args.minimum_consolidation_distinctive_object_coverage
            ),
            "minimum_consolidation_object_jaccard": (
                args.minimum_consolidation_object_jaccard
            ),
        },
        "group_consolidations": len(consolidation_provenance),
        "group_consolidations_by_phase": dict(
            Counter(
                row.get("consolidation_phase", "unspecified")
                for row in consolidation_provenance
            )
        ),
    }
    if args.audit_run_dir:
        audit_results, audit_metrics = score_audit(
            assignments, expansion_output, args.audit_run_dir
        )
        audit_results.to_csv(args.output_dir / "06_audit_results.csv", index=False)
        metrics["manual_audit"] = audit_metrics
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
