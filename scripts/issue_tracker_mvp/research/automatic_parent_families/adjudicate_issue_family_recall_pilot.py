#!/usr/bin/env python3
"""Archived: adjudicate issue-family candidates and quantify fragmentation."""

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
from build_issue_family_recall_pilot import FAMILIES, PILOT_VERSION


ADJUDICATION_VERSION = "issue-family-entailment-v2"
CONFIDENCE_VALUES = {"high", "medium", "low"}
REASON_VALUES = {
    "none",
    "different_issue",
    "wrong_direction",
    "explicit_exclusion",
    "insufficient_evidence",
}

SYSTEM_PROMPT = """You decide whether individual concerns from UK Prevention of
Future Deaths reports genuinely belong to a specified broad issue family.

The family is intentionally broader than a precise operational recurring group.
Accept different subtypes when each concern entails the family definition. Reject a
candidate when shared vocabulary is incidental, actor/action direction contradicts
the family, or the evidence concerns an explicit exclusion.

Important:
- Judge the service or system concern, not the person's diagnosis or outcome.
- Preserve direction. Patient non-engagement is not a service failure to follow up.
- A concern may validly belong to more than one broad family.
- Do not require identical actors, settings, or operational stages.
- Use only the supplied statement, relation fields, and evidence.
- Keep every opaque issue ID exactly unchanged and return one decision per candidate.

Return JSON only."""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "issue_id": {"type": "string"},
                    "supports_family": {"type": "boolean"},
                    "confidence": {
                        "type": "string",
                        "enum": sorted(CONFIDENCE_VALUES),
                    },
                    "reason": {
                        "type": "string",
                        "enum": sorted(REASON_VALUES),
                    },
                },
                "required": [
                    "issue_id",
                    "supports_family",
                    "confidence",
                    "reason",
                ],
            },
        }
    },
    "required": ["decisions"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--sample-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--model", default="gemma4:26b")
    parser.add_argument("--request-timeout", type=int, default=360)
    parser.add_argument("--ollama-num-ctx", type=int, default=16384)
    parser.add_argument("--batch-size", type=int, default=20)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=0)
    return parser.parse_args()


def candidate_payload(frame: pd.DataFrame) -> list[dict[str, str]]:
    columns = (
        "issue_id",
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
    payload: list[dict[str, str]] = []
    for row in frame.to_dict("records"):
        item = {
            column: pipeline.clean_text(row.get(column))
            for column in columns
        }
        item["evidence_quote"] = pipeline.trim_words(item["evidence_quote"], 55)
        payload.append(item)
    return payload


def batches(sample: pd.DataFrame, batch_size: int) -> list[tuple[str, pd.DataFrame]]:
    result: list[tuple[str, pd.DataFrame]] = []
    for family_id, local in sample.groupby("family_id", sort=True):
        local = local.sort_values("sample_order")
        for batch_index, start in enumerate(range(0, len(local), batch_size), start=1):
            result.append(
                (
                    f"{family_id}:{batch_index:04d}",
                    local.iloc[start : start + batch_size].copy(),
                )
            )
    return result


def validate_payload(
    payload: dict[str, Any],
    batch_id: str,
    candidates: pd.DataFrame,
) -> list[dict[str, Any]]:
    expected_ids = candidates["issue_id"].astype(str).tolist()
    decisions: dict[str, dict[str, Any]] = {}
    for raw in payload.get("decisions", []):
        issue_id = pipeline.clean_text(raw.get("issue_id"))
        confidence = pipeline.clean_text(raw.get("confidence")).casefold()
        if issue_id not in expected_ids:
            raise ValueError(f"Unknown issue ID in {batch_id}: {issue_id}")
        if issue_id in decisions:
            raise ValueError(f"Duplicate issue ID in {batch_id}: {issue_id}")
        if confidence not in CONFIDENCE_VALUES:
            raise ValueError(f"Invalid confidence for {issue_id}: {confidence}")
        reason = pipeline.clean_text(raw.get("reason")).casefold()
        if reason not in REASON_VALUES:
            raise ValueError(f"Invalid reason for {issue_id}: {reason}")
        supports = bool(raw.get("supports_family"))
        # Membership is the calibrated outcome. Some otherwise valid constrained
        # responses describe the evidence basis in ``reason`` even when membership
        # is true. Canonicalise that secondary field rather than discarding the
        # binary judgment.
        if supports:
            reason = "none"
        if not supports and reason == "none":
            reason = "insufficient_evidence"
        decisions[issue_id] = {
            "issue_id": issue_id,
            "supports_family": supports,
            "confidence": confidence,
            "reason": reason,
        }
    missing = set(expected_ids) - set(decisions)
    if missing:
        raise ValueError(f"Missing {len(missing)} decisions in {batch_id}")
    return [decisions[issue_id] for issue_id in expected_ids]


def run(
    sample: pd.DataFrame,
    checkpoint_path: Path,
    *,
    host: str,
    model: str,
    timeout: int,
    num_ctx: int,
    batch_size: int,
    retries: int,
    max_batches: int,
) -> dict[str, int]:
    family_lookup = {family.family_id: family for family in FAMILIES}
    queue = batches(sample, batch_size)
    batch_lookup = {batch_id: local for batch_id, local in queue}
    latest = pipeline.read_keyed_jsonl(checkpoint_path, "batch_id")
    recovered = 0
    for batch_id, record in latest.items():
        if record.get("status") != "error" or not isinstance(
            record.get("raw_payload"), dict
        ):
            continue
        local = batch_lookup.get(batch_id)
        if local is None:
            continue
        try:
            decisions = validate_payload(record["raw_payload"], batch_id, local)
        except ValueError:
            continue
        pipeline.append_checkpoint(
            checkpoint_path,
            {
                "batch_id": batch_id,
                "family_id": str(local.iloc[0]["family_id"]),
                "status": "completed",
                "adjudication_version": ADJUDICATION_VERSION,
                "decisions": decisions,
                "recovered_from_checkpoint_error": True,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        recovered += 1
    completed = {
        key: value
        for key, value in pipeline.read_keyed_jsonl(checkpoint_path, "batch_id").items()
        if value.get("status") == "completed"
    }
    if max_batches > 0:
        queue = queue[:max_batches]
    if any(batch_id not in completed for batch_id, _ in queue):
        pipeline.ensure_ollama(host, model)
    attempted = failed = 0
    for batch_id, local in tqdm(queue, desc="Adjudicating family recall", unit="batch"):
        if batch_id in completed:
            continue
        family_id = str(local.iloc[0]["family_id"])
        family = family_lookup[family_id]
        prompt = json.dumps(
            {
                "family_id": family.family_id,
                "family_label": family.label,
                "definition": family.definition,
                "inclusions": family.inclusions,
                "exclusions": family.exclusions,
                "candidates": candidate_payload(local),
            },
            indent=2,
        )
        attempted += 1
        payload: dict[str, Any] = {}
        raw = ""
        last_error: Exception | None = None
        for _ in range(retries + 1):
            retry_prompt = prompt
            if last_error is not None:
                retry_prompt += (
                    "\n\nCorrect your previous invalid JSON. Preserve every supplied "
                    f"issue ID exactly once. Error: {last_error}. "
                    f"Previous JSON: {json.dumps(payload)}"
                )
            try:
                payload, raw = pipeline.ollama_json(
                    host=host,
                    model=model,
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=retry_prompt,
                    schema=RESPONSE_SCHEMA,
                    timeout=timeout,
                    num_predict=1400,
                    num_ctx=num_ctx,
                )
                decisions = validate_payload(payload, batch_id, local)
                record = {
                    "batch_id": batch_id,
                    "family_id": family_id,
                    "status": "completed",
                    "adjudication_version": ADJUDICATION_VERSION,
                    "decisions": decisions,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
                break
            except Exception as exc:  # noqa: BLE001
                last_error = exc
        else:
            failed += 1
            record = {
                "batch_id": batch_id,
                "family_id": family_id,
                "status": "error",
                "error": str(last_error),
                "raw_payload": payload,
                "raw_response": raw,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
        pipeline.append_checkpoint(checkpoint_path, record)
    latest = pipeline.read_keyed_jsonl(checkpoint_path, "batch_id")
    return {
        "queued_batches": len(queue),
        "attempted_batches": attempted,
        "failed_batches": failed,
        "recovered_batches": recovered,
        "completed_batches": sum(
            value.get("status") == "completed" for value in latest.values()
        ),
    }


def build_outputs(
    sample: pd.DataFrame,
    checkpoint_path: Path,
) -> tuple[pd.DataFrame, dict[str, Any]]:
    latest = pipeline.read_keyed_jsonl(checkpoint_path, "batch_id")
    rows: list[dict[str, Any]] = []
    for record in latest.values():
        if record.get("status") != "completed":
            continue
        rows.extend(
            {
                "family_id": record["family_id"],
                "batch_id": record["batch_id"],
                **decision,
            }
            for decision in record["decisions"]
        )
    decisions = pd.DataFrame(rows)
    output = sample.merge(
        decisions,
        on=["family_id", "issue_id"],
        how="left",
        validate="one_to_one",
    )
    completed = output["supports_family"].notna()
    evaluated = output[completed].copy()
    evaluated["supports_family"] = evaluated["supports_family"].astype(bool)
    family_metrics: dict[str, Any] = {}
    for family_id, local in evaluated.groupby("family_id"):
        supported = local[local["supports_family"]]
        recurring = local["refinement_status"].eq("refined_recurring")
        supported_recurring = local[local["supports_family"] & recurring]
        outside = supported[
            ~supported["refinement_status"].eq("refined_recurring")
        ]
        supported_reports = set(supported["report_key"].astype(str))
        supported_recurring_reports = set(
            supported_recurring["report_key"].astype(str)
        )
        supported_review = supported[
            supported["refinement_status"].eq("review_required")
        ]
        supported_isolated = supported[
            ~supported["refinement_status"].isin(
                ["refined_recurring", "review_required"]
            )
        ]
        by_stratum: dict[str, Any] = {}
        for stratum, stratum_rows in local.groupby("recall_stratum"):
            stratum_supported = stratum_rows["supports_family"].astype(bool)
            by_stratum[str(stratum)] = {
                "evaluated": len(stratum_rows),
                "supported": int(stratum_supported.sum()),
                "support_rate": float(stratum_supported.mean()),
                "supported_reports": int(
                    stratum_rows.loc[stratum_supported, "report_key"].nunique()
                ),
            }
        family_metrics[str(family_id)] = {
            "evaluated": len(local),
            "supported": len(supported),
            "support_rate": float(local["supports_family"].mean()),
            "supported_reports": len(supported_reports),
            "supported_already_in_any_recurring_group": len(supported_recurring),
            "supported_reports_already_in_any_recurring_group": len(
                supported_recurring_reports
            ),
            "sample_supported_report_capture_fraction": (
                len(supported_recurring_reports) / len(supported_reports)
                if supported_reports
                else 0.0
            ),
            "supported_outside_any_recurring_group": len(outside),
            "supported_reports_outside_any_recurring_group": int(
                outside["report_key"].nunique()
            ),
            "supported_stranded_in_review": len(supported_review),
            "supported_stranded_in_review_reports": int(
                supported_review["report_key"].nunique()
            ),
            "supported_stranded_isolated_or_unassigned": len(
                supported_isolated
            ),
            "supported_stranded_isolated_or_unassigned_reports": int(
                supported_isolated["report_key"].nunique()
            ),
            "supported_operational_child_groups": int(
                supported_recurring.loc[
                    supported_recurring["refined_group_id"].astype(str).ne(""),
                    "refined_group_id",
                ].nunique()
            ),
            "unsupported": int((~local["supports_family"]).sum()),
            "supported_by_stratum": by_stratum,
            "exclusion_reasons": dict(
                Counter(
                    reason
                    for reason in local.loc[
                        ~local["supports_family"], "reason"
                    ].astype(str)
                    if reason
                )
            ),
        }
    metrics = {
        "pilot_version": PILOT_VERSION,
        "adjudication_version": ADJUDICATION_VERSION,
        "sample_rows": len(sample),
        "evaluated_rows": len(evaluated),
        "families": family_metrics,
        "interpretation": (
            "The sample is stratified and group-balanced. Support rates diagnose "
            "fragmentation but are not corpus prevalence or unbiased recall estimates. "
            "A supported case already in a recurring group may still be in a different "
            "operational child group; this pilot tests the need for a parent-family layer."
        ),
    }
    return output, metrics


def main() -> None:
    args = parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive")
    if args.retries < 0:
        raise ValueError("--retries must be non-negative")
    sample = pd.read_csv(args.sample_csv).fillna("")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint_path = args.output_dir / "03_family_adjudication_checkpoint.jsonl"
    run_metrics = run(
        sample,
        checkpoint_path,
        host=args.ollama_host,
        model=args.model,
        timeout=args.request_timeout,
        num_ctx=args.ollama_num_ctx,
        batch_size=args.batch_size,
        retries=args.retries,
        max_batches=args.max_batches,
    )
    output, metrics = build_outputs(sample, checkpoint_path)
    metrics["run"] = run_metrics
    output.to_csv(args.output_dir / "04_family_recall_adjudication.csv", index=False)
    (args.output_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2), encoding="utf-8"
    )
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
