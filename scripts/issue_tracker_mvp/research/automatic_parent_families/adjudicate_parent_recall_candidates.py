#!/usr/bin/env python3
"""Archived: adjudicate direct report-to-parent candidates against tight cores."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

import build_issue_index as pipeline


ADJUDICATOR_VERSION = "direct-parent-recall-adjudicator-v1"
CONFIDENCES = {"high", "medium", "low"}

SYSTEM_PROMPT = """You decide whether a UK Prevention of Future Deaths report
mentions the same broad issue parent demonstrated by supplied tight-core examples.

The core examples define the parent. The short label is only a hint. Accept
different settings and subtypes when the report entails the same concrete class of
concern. Reject shared vocabulary where the failed object or obligation differs.
Judge whether the report mentions the parent at all, not whether it belongs
exclusively. Use the original concern text to verify extracted statements. Do not
broaden a healthcare or care-process core to unrelated police, employment,
construction, commercial, or incident-governance records merely because the word
"record" or "information" appears.

Keep every opaque audit_id unchanged and return exactly one decision per candidate
as JSON only. Keep each reason to at most 12 words."""

SCHEMA = {
    "type": "object",
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "audit_id": {"type": "string"},
                    "supports_parent": {"type": "boolean"},
                    "confidence": {"type": "string", "enum": sorted(CONFIDENCES)},
                    "reason": {"type": "string", "maxLength": 100},
                },
                "required": ["audit_id", "supports_parent", "confidence", "reason"],
            },
        }
    },
    "required": ["decisions"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--candidates-csv", required=True, type=Path)
    parser.add_argument("--memberships-csv", required=True, type=Path)
    parser.add_argument("--family-summary-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--benchmark-decisions-json", type=Path)
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--model", default="gemma4:26b")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--core-examples", type=int, default=12)
    parser.add_argument("--request-timeout", type=int, default=360)
    parser.add_argument("--ollama-num-ctx", type=int, default=16384)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--minimum-precision", type=float, default=0.90)
    return parser.parse_args()


def core_lookup(memberships: pd.DataFrame, count: int) -> dict[str, list[str]]:
    lookup: dict[str, list[str]] = {}
    core = memberships[memberships["membership_type"].eq("core")]
    for family_id, local in core.groupby("family_id"):
        lookup[str(family_id)] = (
            local.sort_values("similarity_to_family_centroid", ascending=False)
            .head(count)["prototype_canonical_issue"]
            .astype(str)
            .tolist()
        )
    return lookup


def batches(frame: pd.DataFrame, batch_size: int) -> list[tuple[str, pd.DataFrame]]:
    output: list[tuple[str, pd.DataFrame]] = []
    for family_id, local in frame.groupby("family_id", sort=True):
        for index, start in enumerate(range(0, len(local), batch_size), start=1):
            output.append((f"{family_id}:{index:05d}", local.iloc[start : start + batch_size].copy()))
    return output


def prompt_payload(
    local: pd.DataFrame,
    examples: dict[str, list[str]],
    summaries: dict[str, dict[str, Any]],
) -> dict[str, Any]:
    family_id = str(local.iloc[0]["family_id"])
    summary = summaries[family_id]
    return {
        "target_parent": {
            "family_id": family_id,
            "label_hint": str(summary.get("family_label_hint", "")),
            "prototype": str(summary.get("family_prototype", "")),
            "tight_core_examples": examples[family_id],
        },
        "candidates": [
            {
                "audit_id": str(row["audit_id"]),
                "extracted_issues": str(row.get("report_extracted_issues", "")),
                "original_concerns": pipeline.trim_words(row.get("concerns", ""), 450),
            }
            for row in local.to_dict("records")
        ],
    }


def recover_audit_id(audit_id: str, expected: list[str]) -> str:
    """Recover a uniquely identifiable opaque ID after a small copy error."""
    if audit_id in expected or len(audit_id) < 18:
        return audit_id

    prefix_matches = [
        value for value in expected
        if value.startswith(audit_id) or audit_id.startswith(value)
    ]
    if len(prefix_matches) == 1:
        return prefix_matches[0]

    def edit_distance(left: str, right: str) -> int:
        previous = list(range(len(right) + 1))
        for left_index, left_character in enumerate(left, start=1):
            current = [left_index]
            for right_index, right_character in enumerate(right, start=1):
                current.append(min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_character != right_character),
                ))
            previous = current
        return previous[-1]

    ranked = sorted((edit_distance(audit_id, value), value) for value in expected)
    if ranked and ranked[0][0] <= 4 and (len(ranked) == 1 or ranked[1][0] > ranked[0][0]):
        return ranked[0][1]
    return audit_id


def validate_payload(payload: dict[str, Any], batch_id: str, local: pd.DataFrame) -> list[dict[str, Any]]:
    expected = local["audit_id"].astype(str).tolist()
    decisions: dict[str, dict[str, Any]] = {}
    for raw in payload.get("decisions", []):
        audit_id = pipeline.clean_text(raw.get("audit_id"))
        if audit_id not in expected:
            # Local models occasionally truncate or substitute a few characters
            # while copying opaque IDs. Recovery is limited to one clear match
            # among the eight IDs in this batch.
            audit_id = recover_audit_id(audit_id, expected)
        confidence = pipeline.clean_text(raw.get("confidence")).casefold()
        if audit_id not in expected or audit_id in decisions:
            raise ValueError(f"Unknown or duplicate audit ID in {batch_id}: {audit_id}")
        if confidence not in CONFIDENCES:
            raise ValueError(f"Invalid confidence for {audit_id}: {confidence}")
        decisions[audit_id] = {
            "audit_id": audit_id,
            "supports_parent": bool(raw.get("supports_parent")),
            "confidence": confidence,
            "reason": pipeline.clean_text(raw.get("reason")),
        }
    if missing := set(expected) - set(decisions):
        raise ValueError(f"Missing {len(missing)} decisions in {batch_id}")
    return [decisions[value] for value in expected]


def run(
    candidates: pd.DataFrame,
    memberships: pd.DataFrame,
    summaries: pd.DataFrame,
    checkpoint: Path,
    *,
    host: str,
    model: str,
    batch_size: int,
    core_examples: int,
    timeout: int,
    num_ctx: int,
    retries: int,
    max_batches: int,
) -> dict[str, int]:
    examples = core_lookup(memberships, core_examples)
    summary_lookup = summaries.set_index("family_id").to_dict("index")
    work = batches(candidates, batch_size)
    work_lookup = {batch_id: local for batch_id, local in work}
    if max_batches > 0:
        work = work[:max_batches]
    recovered = 0
    for batch_id, record in pipeline.read_keyed_jsonl(checkpoint, "batch_id").items():
        if record.get("status") != "error" or not isinstance(record.get("raw_payload"), dict):
            continue
        local = work_lookup.get(batch_id)
        if local is None:
            continue
        try:
            decisions = validate_payload(record["raw_payload"], batch_id, local)
        except ValueError:
            continue
        pipeline.append_checkpoint(
            checkpoint,
            {
                "batch_id": batch_id, "status": "completed",
                "adjudicator_version": ADJUDICATOR_VERSION,
                "decisions": decisions, "recovered_from_error": True,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            },
        )
        recovered += 1
    completed = {
        key: value for key, value in pipeline.read_keyed_jsonl(checkpoint, "batch_id").items()
        if value.get("status") == "completed"
    }
    if any(batch_id not in completed for batch_id, _ in work):
        pipeline.ensure_ollama(host, model)
    attempted = failed = 0
    for batch_id, local in tqdm(work, desc="Adjudicating direct parent recall", unit="batch"):
        if batch_id in completed:
            continue
        attempted += 1
        base_prompt = json.dumps(prompt_payload(local, examples, summary_lookup), ensure_ascii=False, indent=2)
        payload: dict[str, Any] = {}
        raw = ""
        error: Exception | None = None
        for _ in range(retries + 1):
            prompt = base_prompt
            if error is not None:
                prompt += f"\nCorrect the invalid JSON. Error: {error}. Previous: {json.dumps(payload)}"
            try:
                payload, raw = pipeline.ollama_json(
                    host=host, model=model, system_prompt=SYSTEM_PROMPT,
                    user_prompt=prompt, schema=SCHEMA, timeout=timeout,
                    num_predict=max(600, batch_size * 60), num_ctx=num_ctx,
                )
                decisions = validate_payload(payload, batch_id, local)
                record = {
                    "batch_id": batch_id, "status": "completed",
                    "adjudicator_version": ADJUDICATOR_VERSION,
                    "decisions": decisions,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
                break
            except Exception as exc:  # noqa: BLE001
                error = exc
        else:
            failed += 1
            record = {
                "batch_id": batch_id, "status": "error", "error": str(error),
                "raw_payload": payload, "raw_response": raw,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
        pipeline.append_checkpoint(checkpoint, record)
    return {
        "queued_batches": len(work), "attempted_batches": attempted,
        "failed_batches": failed, "recovered_batches": recovered,
    }


def build_output(candidates: pd.DataFrame, checkpoint: Path) -> pd.DataFrame:
    decisions: list[dict[str, Any]] = []
    for record in pipeline.read_keyed_jsonl(checkpoint, "batch_id").values():
        if record.get("status") == "completed":
            decisions.extend(record["decisions"])
    result = pd.DataFrame(decisions)
    output = candidates.merge(result, on="audit_id", how="left", validate="one_to_one", suffixes=("", "_model"))
    output["parent_assignment_status"] = output.apply(
        lambda row: "pending" if pd.isna(row.get("supports_parent_model")) else (
            "auto_accept" if bool(row["supports_parent_model"]) and row["confidence"] == "high" else (
                "review" if bool(row["supports_parent_model"]) else "reject"
            )
        ),
        axis=1,
    )
    return output


def benchmark_metrics(
    output: pd.DataFrame,
    decisions_payload: list[dict[str, Any]],
    minimum_precision: float,
) -> dict[str, Any]:
    truth = {str(row["audit_id"]): str(row["decision"]).casefold() for row in decisions_payload}
    evaluated = output[output["audit_id"].isin(truth)].copy()
    evaluated["truth"] = evaluated["audit_id"].map(truth)
    accepted = evaluated[evaluated["parent_assignment_status"].eq("auto_accept")]
    true_accepted = accepted[accepted["truth"].eq("yes")]
    positives = int(evaluated["truth"].eq("yes").sum())
    precision = len(true_accepted) / len(accepted) if len(accepted) else 0.0
    return {
        "rows": len(evaluated),
        "truth_positives": positives,
        "auto_accepted": len(accepted),
        "correct_auto_accepted": len(true_accepted),
        "auto_accept_precision": precision,
        "auto_accept_recall": len(true_accepted) / positives if positives else 0.0,
        "false_accept_ids": accepted.loc[accepted["truth"].eq("no"), "audit_id"].tolist(),
        "precision_gate_passed": bool(len(accepted) > 0 and precision >= minimum_precision),
    }


def main() -> None:
    args = parse_args()
    candidates = pd.read_csv(args.candidates_csv).fillna("")
    memberships = pd.read_csv(args.memberships_csv).fillna("")
    summaries = pd.read_csv(args.family_summary_csv).fillna("")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "01_parent_adjudication_checkpoint.jsonl"
    run_metrics = run(
        candidates, memberships, summaries, checkpoint,
        host=args.ollama_host, model=args.model, batch_size=args.batch_size,
        core_examples=args.core_examples, timeout=args.request_timeout,
        num_ctx=args.ollama_num_ctx, retries=args.retries,
        max_batches=args.max_batches,
    )
    output = build_output(candidates, checkpoint)
    output.to_csv(args.output_dir / "02_parent_candidate_adjudication.csv", index=False)
    metrics: dict[str, Any] = {
        "adjudicator_version": ADJUDICATOR_VERSION,
        "rows": len(output),
        "statuses": output["parent_assignment_status"].value_counts().to_dict(),
        "run": run_metrics,
    }
    if args.benchmark_decisions_json:
        payload = json.loads(args.benchmark_decisions_json.read_text(encoding="utf-8"))
        metrics["benchmark"] = benchmark_metrics(output, payload, args.minimum_precision)
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
