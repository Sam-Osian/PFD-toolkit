#!/usr/bin/env python3
"""Calibrate broad-aware recurring-group validation against a manual audit."""

from __future__ import annotations

import argparse
import json
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

import build_issue_index as pipeline


VALIDATOR_VERSION = "label-entailment-calibration-v1"
GROUP_QUALITIES = {"coherent", "incoherent", "uncertain"}
GRANULARITIES = {"broad_issue_family", "operational_issue", "specific_subtype"}
CONFLICT_TYPES = {
    "none",
    "different_issue",
    "opposite_direction",
    "incompatible_actor_responsibility",
    "incompatible_action",
    "incompatible_object",
    "insufficient_evidence",
}

SYSTEM_PROMPT = """You validate proposed recurring issue groups extracted from UK
Prevention of Future Deaths reports.

The objective is a useful generalized issue label supported by every retained
occurrence. Members do NOT need to describe one narrow operational subtype. A broad
issue family is valid when all members genuinely entail it. For example, failures at
different stages of one psychiatric-referral pathway may share a broad pathway
label. Broadness alone is never a defect.

Apply these criteria:
1. Label fidelity: each retained occurrence genuinely entails the proposed label.
2. Directional consistency: actor responsibility, failed action, object, recipient,
   and direction must remain compatible with the label. Do not merge patient
   disengagement with a service failing to contact a patient.
3. Useful granularity: choose broad_issue_family, operational_issue, or
   specific_subtype. A broad label must still identify a concrete shared obligation,
   process, hazard, resource, or system problem. Do not rescue an incoherent group
   with a vacuous label such as "problems with care".
4. Evidence discipline: use only the supplied issue statement, relation fields, and
   evidence. Do not infer facts merely from shared vocabulary.

Set group_quality to:
- coherent: one useful label is supported after, at most, removing explicit outliers;
- incoherent: no useful common issue remains;
- uncertain: the supplied evidence is insufficient to decide.

For every member, state whether it supports the proposed label. Mark only genuine
non-entailment as unsupported; variation in specificity is allowed. Keep opaque issue
IDs exactly unchanged. Use a concise label, no more than 14 words, and a description
of no more than 40 words. Return JSON only."""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "group_quality": {"type": "string", "enum": sorted(GROUP_QUALITIES)},
        "granularity": {"type": "string", "enum": sorted(GRANULARITIES)},
        "label": {"type": "string"},
        "description": {"type": "string"},
        "rationale": {"type": "string"},
        "members": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "issue_id": {"type": "string"},
                    "supports_label": {"type": "boolean"},
                    "conflict_type": {
                        "type": "string",
                        "enum": sorted(CONFLICT_TYPES),
                    },
                    "rationale": {"type": "string"},
                },
                "required": [
                    "issue_id",
                    "supports_label",
                    "conflict_type",
                    "rationale",
                ],
            },
        },
    },
    "required": [
        "group_quality",
        "granularity",
        "label",
        "description",
        "rationale",
        "members",
    ],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--groups-csv", type=Path, required=True)
    parser.add_argument("--members-csv", type=Path, required=True)
    parser.add_argument("--benchmark-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--model", default="gemma4:26b")
    parser.add_argument("--request-timeout", type=int, default=360)
    parser.add_argument("--ollama-num-ctx", type=int, default=16384)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-groups", type=int, default=0)
    return parser.parse_args()


def member_payload(members: pd.DataFrame) -> list[dict[str, str]]:
    columns = (
        "issue_id",
        "report_key",
        "canonical_issue",
        "evidence_quote",
        "responsible_actor_role",
        "failed_action",
        "issue_object",
        "counterparty_role",
        "failure_state",
        "process_stage",
        "communication_direction",
    )
    return [
        {column: pipeline.clean_text(row.get(column)) for column in columns}
        for row in members.to_dict("records")
    ]


def validate_payload(
    payload: dict[str, Any],
    group_id: str,
    members: pd.DataFrame,
) -> dict[str, Any]:
    quality = pipeline.clean_text(payload.get("group_quality")).casefold()
    granularity = pipeline.clean_text(payload.get("granularity")).casefold()
    if quality not in GROUP_QUALITIES:
        raise ValueError(f"Invalid group_quality for {group_id}: {quality}")
    if granularity not in GRANULARITIES:
        raise ValueError(f"Invalid granularity for {group_id}: {granularity}")
    label = pipeline.trim_words(payload.get("label"), 14).strip(" .")
    description = pipeline.trim_words(payload.get("description"), 40)
    if not label:
        raise ValueError(f"Missing label for {group_id}")

    expected_ids = members["issue_id"].astype(str).tolist()
    judgments: dict[str, dict[str, Any]] = {}
    for judgment in payload.get("members", []):
        issue_id = pipeline.clean_text(judgment.get("issue_id"))
        conflict = pipeline.clean_text(judgment.get("conflict_type")).casefold()
        if issue_id in judgments:
            raise ValueError(f"Duplicate member {issue_id} for {group_id}")
        if issue_id not in expected_ids:
            raise ValueError(f"Unknown member {issue_id} for {group_id}")
        if conflict not in CONFLICT_TYPES:
            raise ValueError(f"Invalid conflict type for {issue_id}: {conflict}")
        supports = bool(judgment.get("supports_label"))
        if supports and conflict != "none":
            raise ValueError(f"Supporting member {issue_id} has conflict {conflict}")
        if not supports and conflict == "none":
            raise ValueError(f"Unsupported member {issue_id} has no conflict")
        judgments[issue_id] = {
            "issue_id": issue_id,
            "supports_label": supports,
            "conflict_type": conflict,
            "rationale": pipeline.clean_text(judgment.get("rationale")),
        }
    missing = set(expected_ids) - set(judgments)
    if missing:
        raise ValueError(f"Missing {len(missing)} member judgments for {group_id}")

    unsupported = [
        issue_id for issue_id in expected_ids if not judgments[issue_id]["supports_label"]
    ]
    retained = members[~members["issue_id"].astype(str).isin(unsupported)]
    retained_reports = int(retained["report_key"].astype(str).nunique())
    if quality == "coherent" and not unsupported:
        decision = "accept"
    elif quality == "coherent" and retained_reports >= 3:
        decision = "exclude"
    elif quality == "incoherent":
        decision = "reject"
    else:
        decision = "review"
    return {
        "relational_group_id": group_id,
        "status": "completed",
        "validator_version": VALIDATOR_VERSION,
        "group_quality": quality,
        "decision": decision,
        "granularity": granularity,
        "label": label,
        "description": description,
        "rationale": pipeline.clean_text(payload.get("rationale")),
        "source_occurrence_count": len(expected_ids),
        "retained_occurrence_count": len(retained),
        "retained_report_count": retained_reports,
        "unsupported_issue_ids": unsupported,
        "members": [judgments[issue_id] for issue_id in expected_ids],
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


def run_validation(
    groups: pd.DataFrame,
    members: pd.DataFrame,
    checkpoint_path: Path,
    *,
    host: str,
    model: str,
    timeout: int,
    num_ctx: int,
    retries: int,
    max_groups: int,
) -> dict[str, int]:
    completed = {
        key: value
        for key, value in pipeline.read_keyed_jsonl(
            checkpoint_path, "relational_group_id"
        ).items()
        if value.get("status") == "completed"
    }
    queue = groups.copy()
    if max_groups > 0:
        queue = queue.head(max_groups)
    if any(str(value) not in completed for value in queue["relational_group_id"]):
        pipeline.ensure_ollama(host, model)

    attempted = failed = 0
    for group in tqdm(queue.to_dict("records"), desc="Validating groups", unit="group"):
        group_id = str(group["relational_group_id"])
        if group_id in completed:
            continue
        local = members[members["relational_group_id"].astype(str).eq(group_id)]
        prompt = json.dumps(
            {
                "relational_group_id": group_id,
                "source_prototype": pipeline.clean_text(
                    group.get("prototype_canonical_issue")
                ),
                "report_count": int(group.get("report_count", 0)),
                "members": member_payload(local),
            },
            indent=2,
        )
        attempted += 1
        last_error: Exception | None = None
        raw = ""
        payload: dict[str, Any] = {}
        for _ in range(retries + 1):
            retry_prompt = prompt
            if last_error is not None:
                retry_prompt += (
                    "\n\nYour previous JSON failed validation. Correct it while "
                    "preserving every supplied issue ID exactly once. Error: "
                    f"{last_error}. Previous JSON: {json.dumps(payload)}"
                )
            try:
                payload, raw = pipeline.ollama_json(
                    host=host,
                    model=model,
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=retry_prompt,
                    schema=RESPONSE_SCHEMA,
                    timeout=timeout,
                    num_predict=2600,
                    num_ctx=num_ctx,
                )
                record = validate_payload(payload, group_id, local)
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        else:
            failed += 1
            record = {
                "relational_group_id": group_id,
                "status": "error",
                "error": str(last_error),
                "raw_payload": payload,
                "raw_response": raw,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
        pipeline.append_checkpoint(checkpoint_path, record)
    latest = pipeline.read_keyed_jsonl(checkpoint_path, "relational_group_id")
    return {
        "attempted": attempted,
        "failed": failed,
        "completed": sum(value.get("status") == "completed" for value in latest.values()),
    }


def split_ids(value: object) -> set[str]:
    text = pipeline.clean_text(value)
    return {item.strip() for item in text.split(";") if item.strip()}


def build_outputs(
    groups: pd.DataFrame,
    checkpoint_path: Path,
    benchmark: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any]]:
    records = [
        value
        for value in pipeline.read_keyed_jsonl(
            checkpoint_path, "relational_group_id"
        ).values()
        if value.get("status") == "completed"
    ]
    group_rows: list[dict[str, Any]] = []
    member_rows: list[dict[str, Any]] = []
    for record in records:
        group_rows.append(
            {
                key: value
                for key, value in record.items()
                if key not in {"members", "unsupported_issue_ids"}
            }
            | {
                "unsupported_issue_ids": ";".join(
                    record.get("unsupported_issue_ids", [])
                )
            }
        )
        for member in record["members"]:
            member_rows.append(
                {
                    "relational_group_id": record["relational_group_id"],
                    **member,
                }
            )
    result = pd.DataFrame(group_rows)
    member_result = pd.DataFrame(member_rows)
    expected = benchmark[
        [
            "relational_group_id",
            "label_decision",
            "granularity",
            "unsupported_member_issue_ids",
        ]
    ].rename(
        columns={
            "label_decision": "expected_label_decision",
            "granularity": "expected_granularity",
            "unsupported_member_issue_ids": "expected_unsupported_issue_ids",
        }
    )
    comparison = expected.merge(result, on="relational_group_id", how="left")
    comparison["expected_unsupported_set"] = comparison[
        "expected_unsupported_issue_ids"
    ].map(split_ids)
    comparison["predicted_unsupported_set"] = comparison[
        "unsupported_issue_ids"
    ].map(split_ids)
    comparison["support_set_exact"] = comparison.apply(
        lambda row: row["expected_unsupported_set"]
        == row["predicted_unsupported_set"],
        axis=1,
    )
    comparison["granularity_exact"] = (
        comparison["expected_granularity"] == comparison["granularity"]
    )
    comparison["expected_decision"] = comparison[
        "expected_label_decision"
    ].replace({"partial": "exclude"})
    comparison["decision_exact"] = (
        comparison["expected_decision"] == comparison["decision"]
    )

    expected_outliers = set().union(*comparison["expected_unsupported_set"])
    predicted_outliers = set().union(*comparison["predicted_unsupported_set"])
    true_positive = len(expected_outliers & predicted_outliers)
    outlier_precision = (
        true_positive / len(predicted_outliers) if predicted_outliers else 0.0
    )
    outlier_recall = (
        true_positive / len(expected_outliers) if expected_outliers else 0.0
    )
    metrics = {
        "validator_version": VALIDATOR_VERSION,
        "benchmark_groups": len(benchmark),
        "completed_groups": len(result),
        "decision_counts": dict(Counter(result.get("decision", []))),
        "exact_group_support_sets": int(comparison["support_set_exact"].sum()),
        "group_support_set_accuracy": float(comparison["support_set_exact"].mean()),
        "exact_decisions": int(comparison["decision_exact"].sum()),
        "decision_accuracy": float(comparison["decision_exact"].mean()),
        "exact_granularity": int(comparison["granularity_exact"].sum()),
        "granularity_accuracy": float(comparison["granularity_exact"].mean()),
        "expected_outlier_count": len(expected_outliers),
        "predicted_outlier_count": len(predicted_outliers),
        "outlier_true_positives": true_positive,
        "outlier_precision": outlier_precision,
        "outlier_recall": outlier_recall,
        "false_exclusion_issue_ids": sorted(predicted_outliers - expected_outliers),
        "missed_outlier_issue_ids": sorted(expected_outliers - predicted_outliers),
    }
    disagreements = comparison[
        ~comparison["support_set_exact"]
        | ~comparison["decision_exact"]
        | ~comparison["granularity_exact"]
    ].copy()
    for column in ("expected_unsupported_set", "predicted_unsupported_set"):
        comparison[column] = comparison[column].map(
            lambda values: ";".join(sorted(values))
        )
        disagreements[column] = disagreements[column].map(
            lambda values: ";".join(sorted(values))
        )
    return result, member_result, disagreements, metrics


def main() -> None:
    args = parse_args()
    if args.retries < 0:
        raise ValueError("--retries must be non-negative")
    groups = pd.read_csv(args.groups_csv).fillna("")
    members = pd.read_csv(args.members_csv).fillna("")
    benchmark = pd.read_csv(args.benchmark_csv).fillna("")
    required_groups = set(benchmark["relational_group_id"].astype(str))
    groups = groups[
        groups["relational_group_id"].astype(str).isin(required_groups)
    ].copy()
    members = members[
        members["relational_group_id"].astype(str).isin(required_groups)
    ].copy()
    if set(groups["relational_group_id"].astype(str)) != required_groups:
        raise ValueError("Groups CSV does not cover the complete benchmark")
    if members["issue_id"].astype(str).duplicated().any():
        raise ValueError("Benchmark members contain duplicate issue IDs")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "01_validation_checkpoint.jsonl"
    run_metrics = run_validation(
        groups,
        members,
        checkpoint_path,
        host=args.ollama_host,
        model=args.model,
        timeout=args.request_timeout,
        num_ctx=args.ollama_num_ctx,
        retries=args.retries,
        max_groups=args.max_groups,
    )
    result, member_result, disagreements, metrics = build_outputs(
        groups, checkpoint_path, benchmark
    )
    metrics["run"] = run_metrics
    result.to_csv(args.output_dir / "02_group_validation.csv", index=False)
    member_result.to_csv(args.output_dir / "03_member_validation.csv", index=False)
    disagreements.to_csv(args.output_dir / "04_disagreements.csv", index=False)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
