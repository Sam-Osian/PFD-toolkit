#!/usr/bin/env python3
"""Risk-gate, adjudicate, repair, and label recurring issue groups."""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
from tqdm import tqdm

import build_issue_index as pipeline


DECISIONS = {"accept", "exclude", "split", "reject"}
RISK_GATE_VERSION = "targeted-risk-v2"
GENERIC_OBJECT_TERMS = {
    "assessment",
    "communication",
    "documentation",
    "information",
    "records",
    "record keeping",
    "referral",
    "resources",
    "response",
    "staff training",
    "training",
}
UNINFORMATIVE_VALUES = {"", "not stated", "not_stated", "not applicable", "not_applicable"}
OBJECT_STOPWORDS = {
    "and",
    "care",
    "clinical",
    "communication",
    "documentation",
    "for",
    "information",
    "management",
    "of",
    "patient",
    "process",
    "provider",
    "record",
    "recording",
    "records",
    "response",
    "service",
    "staff",
    "system",
    "the",
    "to",
}
DOCUMENTATION_STAGES = {"information_record_management"}
ACTION_STAGE_FAMILIES = {
    "documentation": DOCUMENTATION_STAGES,
    "escalation": {"escalation"},
    "referral": {"referral"},
    "follow_up": {"follow_up"},
    "training": {"training_delivery"},
    "performance": {
        "assessment",
        "emergency_response",
        "medication_administration",
        "monitoring_observation",
        "professional_practice",
        "treatment_care_delivery",
    },
}
ACTION_PATTERNS = {
    "documentation": re.compile(r"\b(record|recorded|recording|document|documented|notes?|charts?)\b"),
    "performance": re.compile(
        r"\b(perform|performed|carry out|carried out|conduct|conducted|undertake|undertaken|continue|continued)\b"
    ),
    "initiate": re.compile(r"\b(refer\w*|referral|escalat\w*|request(?:ed)? review)\b"),
    "respond": re.compile(r"\b(respond|response|act(?:ed)? on|chase|process(?:ed|ing)?|attend)\b"),
    "obtain": re.compile(r"\b(obtain|seek|consult|liaise)\b"),
    "communicate": re.compile(
        r"\b(communicat\w*|inform\w*|notif\w*|hand(?: |-)?over|bring to .*attention)\b"
    ),
}
CONFLICTING_ACTION_PAIRS = {
    frozenset(("documentation", "performance")),
    frozenset(("initiate", "respond")),
    frozenset(("obtain", "communicate")),
}

ADJUDICATION_SYSTEM_PROMPT = """You adjudicate proposed recurring issue groups from UK Prevention of Future Deaths reports.

A recurring issue must describe the same directional failure in at least three distinct reports. Shared vocabulary or subject matter alone is insufficient. Treat these as materially different unless the concrete obligation is the same:
- performing an action versus documenting it;
- initiating escalation/referral versus responding to it;
- obtaining information/advice versus communicating existing information/advice;
- communication to different recipients where responsibility changes;
- generic training, records, resources, response, referral, or communication with different objects.

Choose exactly one decision:
- accept: every member supports one precise recurring issue;
- exclude: one coherent core remains after removing explicit outliers;
- split: members form two or more directionally distinct groups;
- reject: no precise three-report recurring core is established.

Never invent issue IDs. For split, partitions must contain every source issue ID exactly once. A partition may contain one or two reports and therefore become isolated or emerging. Return JSON only."""

ADJUDICATION_SCHEMA = {
    "type": "object",
    "properties": {
        "decision": {"type": "string", "enum": sorted(DECISIONS)},
        "rationale": {"type": "string"},
        "exclude_issue_ids": {"type": "array", "items": {"type": "string"}},
        "partitions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "label_hint": {"type": "string"},
                    "issue_ids": {"type": "array", "items": {"type": "string"}},
                },
                "required": ["label_hint", "issue_ids"],
            },
        },
    },
    "required": ["decision", "rationale", "exclude_issue_ids", "partitions"],
}

LABEL_SYSTEM_PROMPT = """Create a concise label for one validated recurring issue group.
The label must foreground responsible actor (when supported), failure/action, and object.
Preserve material direction and qualifiers. Do not add facts or medicalise non-medical issues.
Use at most 12 words. The description may use at most 35 words. Return JSON only."""

LABEL_SCHEMA = {
    "type": "object",
    "properties": {
        "label": {"type": "string"},
        "description": {"type": "string"},
    },
    "required": ["label", "description"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--run-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=None)
    parser.add_argument(
        "--stage", choices=["risk", "adjudicate", "repair", "label", "all"], default="all"
    )
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--model", default="gemma4:26b")
    parser.add_argument("--request-timeout", type=int, default=360)
    parser.add_argument("--ollama-num-ctx", type=int, default=16384)
    parser.add_argument("--adjudication-retries", type=int, default=2)
    parser.add_argument("--risk-threshold", type=int, default=3)
    parser.add_argument("--low-cohesion", type=float, default=0.955)
    parser.add_argument("--min-recurring-reports", type=int, default=3)
    parser.add_argument("--max-groups", type=int, default=0)
    parser.add_argument(
        "--label-groups",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate post-repair labels during label/all stages.",
    )
    parser.add_argument(
        "--label-changed-only",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Reuse source labels for unchanged groups and label only changed recurring outputs.",
    )
    return parser.parse_args()


def clean_key(value: object) -> str:
    return re.sub(r"[^a-z0-9]+", " ", pipeline.clean_text(value).casefold()).strip()


def pipe_values(value: object) -> list[str]:
    return [item.strip() for item in pipeline.clean_text(value).split("|") if item.strip()]


def substantive_values(frame: pd.DataFrame, column: str) -> set[str]:
    values = {
        pipeline.clean_text(item).casefold()
        for value in frame.get(column, pd.Series(dtype=str))
        for item in pipe_values(value)
    }
    return values - UNINFORMATIVE_VALUES


def object_anchor_fraction(objects: pd.Series) -> float:
    token_sets = [
        set(clean_key(value).split()) - OBJECT_STOPWORDS
        for value in objects
        if pipeline.clean_text(value)
    ]
    if not token_sets:
        return 0.0
    counts = Counter(token for tokens in token_sets for token in tokens)
    return max(counts.values(), default=0) / len(token_sets)


def is_generic_object(value: object) -> bool:
    key = clean_key(value)
    if not key:
        return True
    words = key.split()
    return key in GENERIC_OBJECT_TERMS or (
        len(words) <= 4 and any(term in key for term in GENERIC_OBJECT_TERMS)
    )


def stage_families(stages: set[str]) -> set[str]:
    families = {
        family
        for family, values in ACTION_STAGE_FAMILIES.items()
        if stages.intersection(values)
    }
    return families


def action_families(members: pd.DataFrame) -> set[str]:
    text = " | ".join(
        clean_key(value)
        for column in ("canonical_issue", "issue_object")
        for value in members.get(column, pd.Series(dtype=str))
    )
    return {family for family, pattern in ACTION_PATTERNS.items() if pattern.search(text)}


def risk_features(
    group: pd.Series, members: pd.DataFrame, *, low_cohesion: float
) -> dict[str, Any]:
    directions = substantive_values(members, "communication_direction")
    stages = substantive_values(members, "process_stage")
    failures = substantive_values(members, "failure_state")
    actors = substantive_values(members, "responsible_actor_role")
    objects = members.get("issue_object", pd.Series(dtype=str)).map(pipeline.clean_text)
    generic_fraction = float(objects.map(is_generic_object).mean()) if len(objects) else 1.0
    anchor_fraction = object_anchor_fraction(objects)
    families = stage_families(stages)
    semantic_actions = action_families(members)
    documentation_conflict = "documentation" in families and len(families) > 1
    direction_conflict = len(directions) > 1
    lifecycle_conflict = documentation_conflict or len(
        families.intersection({"escalation", "referral", "follow_up", "training", "performance"})
    ) > 1
    semantic_action_conflict = any(pair.issubset(semantic_actions) for pair in CONFLICTING_ACTION_PAIRS)
    object_failure_conflict = len(failures) > 1 and (
        generic_fraction >= 0.5 or anchor_fraction < 0.6
    )
    score = 0
    reasons: list[str] = []

    def add(points: int, reason: str) -> None:
        nonlocal score
        score += points
        reasons.append(reason)

    if generic_fraction >= 0.5:
        add(1, "generic_object")
    if anchor_fraction < 0.6:
        add(1, "mixed_object")
    if direction_conflict:
        add(2, "mixed_direction")
    if lifecycle_conflict:
        add(2, "mixed_action_stage")
    if semantic_action_conflict:
        add(2, "mixed_semantic_action")
    if len(failures) > 1:
        add(1, "mixed_failure_state")
    if len(actors) > 2:
        add(1, "mixed_actor")
    if int(group.get("report_count", 0)) == 3:
        add(1, "recurrence_boundary")
    cohesion = float(group.get("median_centroid_similarity", 1.0) or 1.0)
    if cohesion < low_cohesion:
        add(1, "low_cohesion")
    return {
        "risk_score": score,
        "risk_reasons": " | ".join(reasons),
        "generic_object_fraction": round(generic_fraction, 4),
        "object_anchor_fraction": round(anchor_fraction, 4),
        "unique_direction": len(directions),
        "unique_process_stage": len(stages),
        "unique_failure_state": len(failures),
        "unique_actor": len(actors),
        "action_stage_families": " | ".join(sorted(families)),
        "semantic_action_families": " | ".join(sorted(semantic_actions)),
        "direction_conflict": direction_conflict,
        "lifecycle_conflict": lifecycle_conflict,
        "semantic_action_conflict": semantic_action_conflict,
        "object_failure_conflict": object_failure_conflict,
    }


def build_risk_gate(
    groups: pd.DataFrame,
    indexed: pd.DataFrame,
    *,
    risk_threshold: int,
    low_cohesion: float,
) -> pd.DataFrame:
    rows: list[dict[str, Any]] = []
    recurring = groups[groups["recurrence_status"].eq("recurring")]
    for group in recurring.to_dict("records"):
        members = indexed[indexed["subissue_id"].eq(group["subissue_id"])]
        features = risk_features(pd.Series(group), members, low_cohesion=low_cohesion)
        features["risk_gate_version"] = RISK_GATE_VERSION
        features["low_cohesion_threshold"] = low_cohesion
        structural_conflict = bool(
            features["lifecycle_conflict"]
            or features["semantic_action_conflict"]
            or features["object_failure_conflict"]
            or (
                features["direction_conflict"]
                and (
                    features["generic_object_fraction"] >= 0.5
                    or features["object_anchor_fraction"] < 0.6
                    or int(group.get("report_count", 0)) == 3
                )
            )
        )
        flagged = bool(
            features["risk_score"] >= risk_threshold
            or structural_conflict
            or float(group.get("median_centroid_similarity", 1.0) or 1.0) < low_cohesion
        )
        features["adjudication_priority"] = (
            "high"
            if structural_conflict
            else "medium"
            if flagged
            else "monitor"
            if int(group.get("report_count", 0)) == 3
            else "routine"
        )
        features["flagged_for_adjudication"] = flagged
        rows.append({**group, **features})
    return pd.DataFrame(rows).sort_values(
        ["flagged_for_adjudication", "risk_score", "report_count", "subissue_id"],
        ascending=[False, False, False, True],
        ignore_index=True,
    )


def member_payload(members: pd.DataFrame) -> list[dict[str, str]]:
    columns = (
        "issue_id",
        "report_key",
        "canonical_issue",
        "responsible_actor_role",
        "issue_object",
        "failure_state",
        "process_stage",
        "communication_direction",
    )
    return [
        {column: pipeline.clean_text(row.get(column)) for column in columns}
        for row in members.to_dict("records")
    ]


def correct_opaque_issue_id(value: str, valid_ids: set[str]) -> str:
    if value in valid_ids:
        return value

    def edit_distance(left: str, right: str) -> int:
        previous = list(range(len(right) + 1))
        for left_index, left_character in enumerate(left, start=1):
            current = [left_index]
            for right_index, right_character in enumerate(right, start=1):
                current.append(
                    min(
                        current[-1] + 1,
                        previous[right_index] + 1,
                        previous[right_index - 1]
                        + (left_character != right_character),
                    )
                )
            previous = current
        return previous[-1]

    candidates = [
        candidate
        for candidate in valid_ids
        if abs(len(candidate) - len(value)) <= 2
        and edit_distance(candidate, value) <= 2
    ]
    return candidates[0] if len(candidates) == 1 else value


def validate_decision(
    payload: dict[str, Any], source_group_id: str, members: pd.DataFrame
) -> dict[str, Any]:
    issue_ids = set(members["issue_id"].astype(str))
    decision = pipeline.clean_text(payload.get("decision")).casefold()
    if decision not in DECISIONS:
        raise ValueError(f"Unsupported decision for {source_group_id}: {decision}")
    exclusions = [
        correct_opaque_issue_id(pipeline.clean_text(value), issue_ids)
        for value in payload.get("exclude_issue_ids", [])
    ]
    if len(exclusions) != len(set(exclusions)) or not set(exclusions).issubset(issue_ids):
        raise ValueError(f"Invalid exclusion IDs for {source_group_id}")
    partitions: list[dict[str, Any]] = []
    for part in payload.get("partitions", []):
        ids = [
            correct_opaque_issue_id(pipeline.clean_text(value), issue_ids)
            for value in part.get("issue_ids", [])
        ]
        if not ids:
            raise ValueError(f"Empty split partition for {source_group_id}")
        partitions.append(
            {"label_hint": pipeline.clean_text(part.get("label_hint")), "issue_ids": ids}
        )
    if decision in {"accept", "reject"}:
        exclusions = []
        partitions = []
    if decision == "exclude" and not exclusions:
        raise ValueError(f"Exclude decision is malformed for {source_group_id}")
    if decision == "exclude":
        partitions = []
    if decision == "split":
        flattened = [issue_id for part in partitions for issue_id in part["issue_ids"]]
        if len(flattened) != len(set(flattened)) or not set(flattened).issubset(issue_ids):
            raise ValueError(
                f"Split contains duplicate or unknown issue IDs for {source_group_id}"
            )
        missing = sorted(issue_ids - set(flattened))
        partitions.extend(
            {"label_hint": "Outlier retained separately", "issue_ids": [issue_id]}
            for issue_id in missing
        )
        if len(partitions) < 2:
            raise ValueError(f"Split requires at least two partitions for {source_group_id}")
        exclusions = []
    return {
        "source_group_id": source_group_id,
        "decision": decision,
        "rationale": pipeline.clean_text(payload.get("rationale")),
        "exclude_issue_ids": exclusions,
        "partitions": partitions,
        "status": "completed",
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


def completed_records(path: Path) -> dict[str, dict[str, Any]]:
    return {
        key: value
        for key, value in pipeline.read_keyed_jsonl(path, "source_group_id").items()
        if value.get("status") == "completed"
    }


def recover_checkpoint_payloads(
    checkpoint_path: Path, indexed: pd.DataFrame
) -> int:
    latest = pipeline.read_keyed_jsonl(checkpoint_path, "source_group_id")
    recovered = 0
    for group_id, record in latest.items():
        if record.get("status") == "completed" or not isinstance(
            record.get("raw_payload"), dict
        ):
            continue
        members = indexed[indexed["subissue_id"].eq(group_id)]
        if members.empty:
            continue
        try:
            corrected = validate_decision(record["raw_payload"], group_id, members)
        except ValueError:
            continue
        corrected["recovered_from_checkpoint_error"] = True
        pipeline.append_checkpoint(checkpoint_path, corrected)
        recovered += 1
    return recovered


def adjudicate_flagged_groups(
    risk: pd.DataFrame,
    indexed: pd.DataFrame,
    checkpoint_path: Path,
    *,
    host: str,
    model: str,
    timeout: int,
    num_ctx: int,
    max_groups: int,
    retries: int,
) -> dict[str, int]:
    recovered = recover_checkpoint_payloads(checkpoint_path, indexed)
    completed = completed_records(checkpoint_path)
    queue = risk[risk["flagged_for_adjudication"]].copy()
    if max_groups > 0:
        queue = queue.head(max_groups)
    if any(str(group_id) not in completed for group_id in queue["subissue_id"]):
        pipeline.ensure_ollama(host, model)
    attempted = 0
    failed = 0
    for group in tqdm(queue.to_dict("records"), desc="Adjudicating groups", unit="group"):
        group_id = str(group["subissue_id"])
        if group_id in completed:
            continue
        members = indexed[indexed["subissue_id"].eq(group_id)]
        prompt = json.dumps(
            {
                "source_group_id": group_id,
                "risk_reasons": group["risk_reasons"],
                "report_count": int(group["report_count"]),
                "members": member_payload(members),
            },
            indent=2,
        )
        attempted += 1
        payload: dict[str, Any] = {}
        raw = ""
        last_error: Exception | None = None
        for retry in range(retries + 1):
            retry_prompt = prompt
            if last_error is not None:
                retry_prompt += (
                    "\n\nYour previous response was invalid. Correct it without changing "
                    f"the source issue IDs. Validation error: {last_error}. "
                    f"Previous JSON: {json.dumps(payload)}"
                )
            try:
                payload, raw = pipeline.ollama_json(
                    host=host,
                    model=model,
                    system_prompt=ADJUDICATION_SYSTEM_PROMPT,
                    user_prompt=retry_prompt,
                    schema=ADJUDICATION_SCHEMA,
                    timeout=timeout,
                    num_predict=2200,
                    num_ctx=num_ctx,
                )
                record = validate_decision(payload, group_id, members)
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        else:
            failed += 1
            record = {
                "source_group_id": group_id,
                "status": "error",
                "error": str(last_error),
                "raw_payload": payload,
                "raw_response": raw,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
        pipeline.append_checkpoint(checkpoint_path, record)
    return {
        "attempted": attempted,
        "failed": failed,
        "recovered_from_checkpoint": recovered,
        "completed": len(completed_records(checkpoint_path)),
    }


def centroid_details(
    members: list[int], embeddings: np.ndarray
) -> tuple[dict[int, float], int, float]:
    local = embeddings[members]
    centroid = local.mean(axis=0)
    centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
    similarities = local @ centroid
    return (
        {member: float(score) for member, score in zip(members, similarities, strict=False)},
        members[int(np.argmax(similarities))],
        float(np.median(similarities)),
    )


def repair_units_for_decision(
    issue_ids: list[str], decision: dict[str, Any]
) -> list[tuple[str, list[str], str]]:
    action = decision["decision"]
    if action == "accept":
        return [("accepted", issue_ids, "")]
    if action == "reject":
        return [("rejected_singleton", [issue_id], "") for issue_id in issue_ids]
    if action == "exclude":
        excluded = set(decision["exclude_issue_ids"])
        core = [issue_id for issue_id in issue_ids if issue_id not in excluded]
        units = [("excluded_core", core, "")] if core else []
        units.extend(("excluded_singleton", [issue_id], "") for issue_id in sorted(excluded))
        return units
    return [
        ("split", list(part["issue_ids"]), pipeline.clean_text(part.get("label_hint")))
        for part in decision["partitions"]
    ]


def apply_adjudication(
    occurrences: pd.DataFrame,
    embeddings: np.ndarray,
    groups: pd.DataFrame,
    assignments: pd.DataFrame,
    risk: pd.DataFrame,
    decisions: dict[str, dict[str, Any]],
    *,
    min_recurring_reports: int,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    issue_index = {value: index for index, value in enumerate(occurrences["issue_id"].astype(str))}
    flagged = set(risk.loc[risk["flagged_for_adjudication"], "subissue_id"].astype(str))
    missing = flagged - set(decisions)
    if missing:
        raise ValueError(f"Adjudication is incomplete for {len(missing)} flagged groups")
    risk_lookup = risk.set_index("subissue_id").to_dict("index")
    group_rows: list[dict[str, Any]] = []
    assignment_rows: list[dict[str, Any]] = []
    provenance_rows: list[dict[str, Any]] = []
    for source in groups.to_dict("records"):
        source_id = str(source["subissue_id"])
        issue_ids = assignments.loc[assignments["subissue_id"].eq(source_id), "issue_id"].astype(str).tolist()
        if not issue_ids:
            continue
        # Checkpoints are append-only and may contain decisions produced under
        # an earlier gate. Only the currently flagged set may affect this repair.
        decision = decisions.get(source_id) if source_id in flagged else None
        if decision:
            units = repair_units_for_decision(issue_ids, decision)
        else:
            units = [("not_flagged", issue_ids, "")]
        for action, unit_ids, label_hint in units:
            member_positions = [issue_index[issue_id] for issue_id in unit_ids]
            subset = occurrences.iloc[member_positions]
            report_count = int(subset["report_key"].nunique())
            status = pipeline.recurrence_status(report_count, min_recurring_reports)
            strength = pipeline.recurrence_strength(
                report_count, min_recurring_reports
            )
            changed = action not in {"not_flagged", "accepted"}
            final_id = f"adj_{pipeline.stable_hash(*sorted(unit_ids))}" if changed else source_id
            scores, representative, cohesion = centroid_details(member_positions, embeddings)
            representative_issue = pipeline.clean_text(occurrences.iloc[representative]["canonical_issue"])
            source_label = pipeline.clean_text(source.get("label"))
            source_description = pipeline.clean_text(source.get("description"))
            initial_label = (
                source_label
                if action in {"not_flagged", "accepted"} and source_label
                else label_hint or representative_issue.rstrip(".")
            )
            initial_description = (
                source_description if action in {"not_flagged", "accepted"} else ""
            )
            group_rows.append(
                {
                    "final_group_id": final_id,
                    "source_group_id": source_id,
                    "adjudication_action": action,
                    "adjudication_status": (
                        "machine_proposed" if decision else "not_adjudicated"
                    ),
                    "publication_status": "not_published",
                    "adjudication_rationale": pipeline.clean_text(decision.get("rationale")) if decision else "",
                    "label_hint": label_hint,
                    "label": initial_label,
                    "description": initial_description,
                    "recurrence_status": status,
                    "recurrence_strength": strength,
                    "report_count": report_count,
                    "issue_count": len(unit_ids),
                    "median_centroid_similarity": cohesion,
                    "representative_issue_id": occurrences.iloc[representative]["issue_id"],
                    "failure_mode": pipeline.dominant_value(subset, "failure_state"),
                    "subject_domain": pipeline.dominant_array_value(subset, "issue_themes"),
                    "process_stage": pipeline.dominant_value(subset, "process_stage"),
                    "responsible_actor_role": pipeline.dominant_value(subset, "responsible_actor_role"),
                    "issue_object": pipeline.dominant_value(subset, "issue_object"),
                    "sample_issues": " | ".join(subset["canonical_issue"].drop_duplicates().head(8)),
                }
            )
            for position in member_positions:
                assignment_rows.append(
                    {
                        "final_group_id": final_id,
                        "source_group_id": source_id,
                        "adjudication_action": action,
                        "adjudication_status": (
                            "machine_proposed" if decision else "not_adjudicated"
                        ),
                        "issue_id": occurrences.iloc[position]["issue_id"],
                        "report_key": occurrences.iloc[position]["report_key"],
                        "recurrence_status": status,
                        "recurrence_strength": strength,
                        "assignment_similarity": scores[position],
                    }
                )
            provenance_rows.append(
                {
                    "source_group_id": source_id,
                    "final_group_id": final_id,
                    "flagged": source_id in flagged,
                    "risk_score": risk_lookup.get(source_id, {}).get("risk_score", ""),
                    "risk_reasons": risk_lookup.get(source_id, {}).get("risk_reasons", ""),
                    "decision": decision.get("decision") if decision else "not_flagged",
                    "action": action,
                    "source_issue_count": len(issue_ids),
                    "final_issue_count": len(unit_ids),
                    "final_report_count": report_count,
                    "final_recurrence_status": status,
                    "final_recurrence_strength": strength,
                    "rationale": pipeline.clean_text(decision.get("rationale")) if decision else "",
                }
            )
    final_groups = pd.DataFrame(group_rows).sort_values(
        ["recurrence_status", "report_count", "final_group_id"],
        ascending=[True, False, True],
        ignore_index=True,
    )
    final_assignments = pd.DataFrame(assignment_rows)
    provenance = pd.DataFrame(provenance_rows)
    metrics = {
        "source_groups": int(len(groups)),
        "source_recurring_groups": int(groups["recurrence_status"].eq("recurring").sum()),
        "flagged_recurring_groups": len(flagged),
        "decision_counts": dict(Counter(value["decision"] for value in decisions.values())),
        "final_groups": int(len(final_groups)),
        "final_recurring_groups": int(final_groups["recurrence_status"].eq("recurring").sum()),
        "final_emerging_groups": int(final_groups["recurrence_status"].eq("emerging").sum()),
        "final_isolated_groups": int(final_groups["recurrence_status"].eq("isolated").sum()),
    }
    return final_groups, final_assignments, provenance, metrics


def label_final_groups(
    groups: pd.DataFrame,
    assignments: pd.DataFrame,
    occurrences: pd.DataFrame,
    checkpoint_path: Path,
    *,
    host: str,
    model: str,
    timeout: int,
    num_ctx: int,
    max_groups: int,
    use_llm: bool,
    changed_only: bool,
) -> pd.DataFrame:
    if use_llm:
        pipeline.ensure_ollama(host, model)
    completed = {
        key: value
        for key, value in pipeline.read_keyed_jsonl(checkpoint_path, "final_group_id").items()
        if value.get("status") == "completed"
    }
    recurring_ids = groups.loc[groups["recurrence_status"].eq("recurring"), "final_group_id"].tolist()
    if changed_only:
        recurring_ids = groups.loc[
            groups["recurrence_status"].eq("recurring")
            & ~groups["adjudication_action"].isin(["not_flagged", "accepted"]),
            "final_group_id",
        ].tolist()
    if max_groups > 0:
        recurring_ids = recurring_ids[:max_groups]
    occurrence_lookup = occurrences.set_index("issue_id")
    for group_id in tqdm(recurring_ids, desc="Labelling final groups", unit="group"):
        if group_id in completed:
            continue
        issue_ids = assignments.loc[assignments["final_group_id"].eq(group_id), "issue_id"].tolist()
        members = occurrence_lookup.loc[issue_ids].reset_index()
        if use_llm:
            payload, _ = pipeline.ollama_json(
                host=host,
                model=model,
                system_prompt=LABEL_SYSTEM_PROMPT,
                user_prompt=json.dumps({"members": member_payload(members)}, indent=2),
                schema=LABEL_SCHEMA,
                timeout=timeout,
                num_predict=260,
                num_ctx=num_ctx,
            )
            label = pipeline.trim_words(payload.get("label"), 12).strip(" .")
            description = pipeline.trim_words(payload.get("description"), 35)
        else:
            source = groups.loc[groups["final_group_id"].eq(group_id)].iloc[0]
            label = pipeline.clean_text(source["label"])
            description = pipeline.clean_text(source["description"])
        record = {
            "final_group_id": group_id,
            "label": label,
            "description": description,
            "status": "completed",
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
        pipeline.append_checkpoint(checkpoint_path, record)
        completed[group_id] = record
    labelled = groups.copy()
    labelled["label"] = labelled.apply(
        lambda row: completed.get(str(row["final_group_id"]), {}).get("label", row["label"]), axis=1
    )
    labelled["description"] = labelled.apply(
        lambda row: completed.get(str(row["final_group_id"]), {}).get(
            "description", row["description"]
        ),
        axis=1,
    )
    return labelled


def load_run(
    run_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame, np.ndarray]:
    occurrences = pd.read_csv(run_dir / "02_embedding_occurrences.csv").fillna("")
    groups = pd.read_csv(run_dir / "03_subissues.csv").fillna("")
    assignments = pd.read_csv(run_dir / "03_issue_assignments.csv").fillna("")
    indexed = pd.read_csv(run_dir / "03_occurrences_indexed.csv").fillna("")
    embeddings = np.load(run_dir / "02_issue_embeddings.npy")
    if len(occurrences) != len(embeddings):
        raise ValueError("Occurrence and embedding row counts do not match")
    return occurrences, groups, assignments, indexed, embeddings


def compatible_audit_frames(
    occurrences: pd.DataFrame,
    groups: pd.DataFrame,
    assignments: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    subissues = groups.rename(columns={"final_group_id": "subissue_id"}).copy()
    assignment_columns = [
        "issue_id",
        "final_group_id",
        "assignment_similarity",
        "recurrence_status",
    ]
    membership = assignments[assignment_columns].rename(
        columns={"final_group_id": "subissue_id"}
    )
    indexed = occurrences.merge(membership, on="issue_id", how="left", validate="one_to_one")
    indexed["subissue_id"] = indexed["subissue_id"].fillna("")
    indexed["recurrence_status"] = indexed["recurrence_status"].fillna("ungrouped")
    indexed["assignment_similarity"] = indexed["assignment_similarity"].fillna(0.0)
    return subissues, indexed


def main() -> None:
    args = parse_args()
    if args.risk_threshold < 0:
        raise ValueError("--risk-threshold must be non-negative")
    if args.adjudication_retries < 0:
        raise ValueError("--adjudication-retries must be non-negative")
    if args.min_recurring_reports < 2:
        raise ValueError("--min-recurring-reports must be at least 2")
    run_dir = args.run_dir.expanduser().resolve()
    output_dir = (
        args.output_dir.expanduser().resolve()
        if args.output_dir
        else run_dir / "08_targeted_adjudication"
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    occurrences, groups, assignments, indexed, embeddings = load_run(run_dir)
    risk_path = output_dir / "01_group_risk_scores.csv"
    if args.stage in {"risk", "all"} or not risk_path.exists():
        risk = build_risk_gate(
            groups,
            indexed,
            risk_threshold=args.risk_threshold,
            low_cohesion=args.low_cohesion,
        )
        risk.to_csv(risk_path, index=False)
    else:
        risk = pd.read_csv(risk_path).fillna("")
    checkpoint_path = output_dir / "02_adjudication_checkpoint.jsonl"
    if args.stage in {"adjudicate", "all"}:
        adjudication_metrics = adjudicate_flagged_groups(
            risk,
            indexed,
            checkpoint_path,
            host=args.ollama_host,
            model=args.model,
            timeout=args.request_timeout,
            num_ctx=args.ollama_num_ctx,
            max_groups=args.max_groups,
            retries=args.adjudication_retries,
        )
        (output_dir / "02_adjudication_metrics.json").write_text(
            json.dumps(adjudication_metrics, indent=2), encoding="utf-8"
        )
    if args.stage in {"repair", "all"}:
        decisions = completed_records(checkpoint_path)
        final_groups, final_assignments, provenance, metrics = apply_adjudication(
            occurrences,
            embeddings,
            groups,
            assignments,
            risk,
            decisions,
            min_recurring_reports=args.min_recurring_reports,
        )
        final_groups.to_csv(output_dir / "03_repaired_groups.csv", index=False)
        final_assignments.to_csv(output_dir / "03_repaired_assignments.csv", index=False)
        final_groups.to_csv(output_dir / "03_proposed_groups.csv", index=False)
        final_assignments.to_csv(
            output_dir / "03_proposed_assignments.csv", index=False
        )
        audit_groups, audit_indexed = compatible_audit_frames(
            occurrences, final_groups, final_assignments
        )
        audit_groups.to_csv(output_dir / "03_repaired_subissues.csv", index=False)
        audit_groups.to_csv(output_dir / "03_proposed_subissues.csv", index=False)
        audit_indexed.to_csv(
            output_dir / "03_repaired_occurrences_indexed.csv", index=False
        )
        audit_indexed.to_csv(
            output_dir / "03_proposed_occurrences_indexed.csv", index=False
        )
        provenance.to_csv(output_dir / "03_adjudication_provenance.csv", index=False)
        (output_dir / "03_repair_metrics.json").write_text(
            json.dumps(metrics, indent=2), encoding="utf-8"
        )
    if args.stage in {"label", "all"}:
        repaired_groups = pd.read_csv(output_dir / "03_repaired_groups.csv").fillna("")
        repaired_assignments = pd.read_csv(output_dir / "03_repaired_assignments.csv").fillna("")
        labelled = label_final_groups(
            repaired_groups,
            repaired_assignments,
            occurrences,
            output_dir / "04_label_checkpoint.jsonl",
            host=args.ollama_host,
            model=args.model,
            timeout=args.request_timeout,
            num_ctx=args.ollama_num_ctx,
            max_groups=args.max_groups,
            use_llm=args.label_groups,
            changed_only=args.label_changed_only,
        )
        labelled.to_csv(output_dir / "04_final_groups.csv", index=False)
        labelled.to_csv(output_dir / "04_proposed_groups.csv", index=False)
        final_subissues, final_indexed = compatible_audit_frames(
            occurrences, labelled, repaired_assignments
        )
        final_subissues.to_csv(output_dir / "04_final_subissues.csv", index=False)
        final_subissues.to_csv(
            output_dir / "04_proposed_subissues.csv", index=False
        )
        final_indexed.to_csv(
            output_dir / "04_final_occurrences_indexed.csv", index=False
        )
        final_indexed.to_csv(
            output_dir / "04_proposed_occurrences_indexed.csv", index=False
        )
    config = {
        "run_dir": str(run_dir),
        "stage": args.stage,
        "risk_threshold": args.risk_threshold,
        "low_cohesion": args.low_cohesion,
        "min_recurring_reports": args.min_recurring_reports,
        "model": args.model,
        "label_groups": args.label_groups,
        "label_changed_only": args.label_changed_only,
        "output_mode": "non_destructive_machine_proposal",
        "risk_gate_version": RISK_GATE_VERSION,
        "generated_at": datetime.now(timezone.utc).isoformat(),
    }
    (output_dir / "adjudication_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )
    flagged = int(risk["flagged_for_adjudication"].astype(bool).sum())
    print(json.dumps({"recurring_groups": len(risk), "flagged_groups": flagged}, indent=2))


if __name__ == "__main__":
    main()
