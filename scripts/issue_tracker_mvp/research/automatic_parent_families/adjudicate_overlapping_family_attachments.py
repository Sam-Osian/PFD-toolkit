#!/usr/bin/env python3
"""Archived: adjudicate proposed non-exclusive issue-family attachments.

The tight family core is the definition.  A small set of deterministic vetoes
protects distinctions that embedding similarity repeatedly obscures; all other
proposals are judged against core examples by a structured local model call.
"""

from __future__ import annotations

import argparse
import json
import re
from collections import Counter
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd
from tqdm import tqdm

import build_issue_index as pipeline
from build_overlapping_family_audit import stable_audit_id


ADJUDICATOR_VERSION = "overlapping-family-attachment-adjudicator-v1"
CONFIDENCES = {"high", "medium", "low"}
REASONS = {
    "shared_issue",
    "different_issue",
    "wrong_action",
    "wrong_direction",
    "wrong_failure_state",
    "incidental_similarity",
    "insufficient_evidence",
}

SYSTEM_PROMPT = """You adjudicate proposed additional, non-exclusive memberships
between precise issue groups extracted from UK Prevention of Future Deaths reports
and broader issue families.

The target family is defined by its tight core examples, not by its short label or
embedding similarity. Accept a candidate when its concern genuinely belongs to the
same useful parent issue, even if it is a different subtype, actor, setting, or
process stage. Reject incidental vocabulary matches and preserve the obligation:
who must do what, to or for whom, in which direction, and how it failed.

Important distinctions:
- calling or summoning an ambulance is not ambulance response or attendance;
- performing observations is not recording or documenting observations;
- incomplete or inaccurate records are not unavailable or inaccessible records;
- staff capacity is not training delivery;
- a broad label cannot rescue a candidate unsupported by the core examples.

A candidate may validly remain in its primary family and also join the target. Use
only supplied evidence. Keep each opaque attachment_id unchanged and return JSON
only."""

VERIFIER_SYSTEM_PROMPT = """You are the conservative second-pass verifier for
proposed issue-family attachments from UK Prevention of Future Deaths reports.

The first pass has already found thematic similarity. Your job is to find semantic
overreach. The tight core examples—not the label hint—define the target family.
Accept only when you can state a concrete parent obligation that is both (a) entailed
by the candidate and (b) consistently demonstrated by the tight core. Reject when
the proposed parent would have to become vaguer than the core to include the
candidate.

Check especially whether the candidate changes the object of assessment, the action
being performed, the direction of responsibility, the failure state, or a specific
pathway/context represented throughout the core. Shared words such as assessment,
review, care, records, communication, or mental health are not sufficient. Different
subtypes remain acceptable when the same concrete obligation genuinely spans them.
A candidate may remain in its primary family and also join the target.

Use only supplied evidence. Keep every opaque attachment_id unchanged and return
JSON only."""

RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "decisions": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "attachment_id": {"type": "string"},
                    "supports_family": {"type": "boolean"},
                    "confidence": {"type": "string", "enum": sorted(CONFIDENCES)},
                    "reason": {"type": "string", "enum": sorted(REASONS)},
                    "rationale": {"type": "string"},
                },
                "required": [
                    "attachment_id",
                    "supports_family",
                    "confidence",
                    "reason",
                    "rationale",
                ],
            },
        }
    },
    "required": ["decisions"],
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--family-summary-csv", required=True, type=Path)
    parser.add_argument("--memberships-csv", required=True, type=Path)
    parser.add_argument("--relational-groups-csv", required=True, type=Path)
    parser.add_argument("--relational-occurrences-csv", required=True, type=Path)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--benchmark-csv", type=Path)
    parser.add_argument("--benchmark-decisions-json", type=Path)
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--model", default="gemma4:26b")
    parser.add_argument("--request-timeout", type=int, default=360)
    parser.add_argument("--ollama-num-ctx", type=int, default=16384)
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--core-examples", type=int, default=5)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--max-batches", type=int, default=0)
    parser.add_argument("--minimum-precision", type=float, default=0.90)
    parser.add_argument("--minimum-weighted-precision", type=float, default=0.90)
    return parser.parse_args()


def clean(value: object) -> str:
    return pipeline.clean_text(value)


def _has(text: str, pattern: str) -> bool:
    return re.search(pattern, text, flags=re.IGNORECASE) is not None


def deterministic_veto(target_text: str, candidate_text: str) -> str | None:
    """Return only high-confidence contradictions observed in the audit."""
    target = clean(target_text)
    candidate = clean(candidate_text)
    combined = f" {target} "

    ambulance_call = r"\b(call|called|calling|summon|summoned|request|requested)\b.{0,35}\bambulance\b"
    ambulance_response = r"\bambulance\b.{0,45}\b(response|respond|attendance|attend|arrival|arrive)"
    if (_has(combined, ambulance_call) and _has(candidate, ambulance_response)) or (
        _has(combined, ambulance_response) and _has(candidate, ambulance_call)
    ):
        return "wrong_direction"

    recording = r"\b(record|recorded|recording|document|documented|documentation|chart|charts)\b"
    observations = r"\b(observation|observations|monitoring)\b"
    performing = r"\b(perform|performed|carry out|carried out|undertake|undertaken|conduct|conducted)\b"
    target_records_observations = _has(combined, recording) and _has(combined, observations)
    candidate_records_observations = _has(candidate, recording) and _has(candidate, observations)
    target_performs_observations = _has(combined, performing) and _has(combined, observations) and not _has(combined, recording)
    candidate_performs_observations = _has(candidate, performing) and _has(candidate, observations) and not _has(candidate, recording)
    if (target_records_observations and candidate_performs_observations) or (
        target_performs_observations and candidate_records_observations
    ):
        return "wrong_action"

    records = r"\b(record|records|notes|documentation)\b"
    unavailable = r"\b(unavailable|inaccessible|access|obtain|provide|missing)\b"
    defective = r"\b(incomplete|inaccurate|inconsistent|poor|inadequate)\b"
    target_unavailable = _has(combined, records) and _has(combined, unavailable) and not _has(combined, defective)
    candidate_unavailable = _has(candidate, records) and _has(candidate, unavailable) and not _has(candidate, defective)
    target_defective = _has(combined, records) and _has(combined, defective) and not _has(combined, unavailable)
    candidate_defective = _has(candidate, records) and _has(candidate, defective) and not _has(candidate, unavailable)
    if (target_unavailable and candidate_defective) or (target_defective and candidate_unavailable):
        return "wrong_failure_state"

    training = r"\b(training|trained|competence|competency|induction)\b"
    staffing_capacity = r"\b(insufficient|shortage|lack|number|numbers|level|levels)\b.{0,35}\b(staff|workforce|personnel)\b"
    if _has(combined, training) and _has(candidate, staffing_capacity):
        return "different_issue"
    return None


def relation_lookup(groups: pd.DataFrame, occurrences: pd.DataFrame) -> dict[str, dict[str, str]]:
    columns = (
        "responsible_actor_role",
        "failed_action",
        "issue_object",
        "counterparty_role",
        "failure_state",
        "process_stage",
        "communication_direction",
    )
    occurrence_lookup = occurrences.set_index("issue_id").to_dict("index")
    output: dict[str, dict[str, str]] = {}
    for row in groups.to_dict("records"):
        values = occurrence_lookup.get(str(row.get("prototype_issue_id", "")), {})
        output[str(row["relational_group_id"])] = {
            column: clean(values.get(column)) for column in columns
        }
    return output


def build_queue(
    summaries: pd.DataFrame,
    memberships: pd.DataFrame,
    signatures: dict[str, dict[str, str]],
    *,
    core_examples: int,
) -> pd.DataFrame:
    summary_lookup = summaries.set_index("family_id").to_dict("index")
    core_lookup: dict[str, list[dict[str, Any]]] = {}
    core = memberships[memberships["membership_type"].eq("core")]
    for family_id, local in core.groupby("family_id"):
        selected = local.sort_values("similarity_to_family_centroid", ascending=False).head(core_examples)
        core_lookup[str(family_id)] = [
            {
                "child_group_id": str(row["child_group_id"]),
                "issue": clean(row["prototype_canonical_issue"]),
                "relation": signatures.get(str(row["child_group_id"]), {}),
            }
            for row in selected.to_dict("records")
        ]

    rows: list[dict[str, Any]] = []
    for row in memberships[memberships["membership_type"].ne("core")].to_dict("records"):
        family_id = str(row["family_id"])
        child_id = str(row["child_group_id"])
        primary_id = str(row.get("primary_family_id", ""))
        target = summary_lookup[family_id]
        primary = summary_lookup.get(primary_id, {})
        examples = core_lookup.get(family_id, [])
        target_text = " | ".join([clean(target.get("family_prototype"))] + [item["issue"] for item in examples])
        rows.append(
            {
                "attachment_id": stable_audit_id(family_id, child_id),
                "target_family_id": family_id,
                "target_family_label_hint": clean(target.get("family_label_hint")),
                "target_family_prototype": clean(target.get("family_prototype")),
                "target_core_examples": json.dumps(examples, ensure_ascii=False),
                "child_group_id": child_id,
                "child_report_count": int(row.get("child_report_count", 0)),
                "child_prototype": clean(row.get("prototype_canonical_issue")),
                "child_relation": json.dumps(signatures.get(child_id, {}), ensure_ascii=False),
                "primary_family_id": primary_id,
                "primary_family_label_hint": clean(primary.get("family_label_hint")),
                "primary_family_prototype": clean(primary.get("family_prototype")),
                "similarity_to_target": float(row.get("similarity_to_family_centroid", 0)),
                "similarity_to_primary": float(row.get("primary_similarity", 0)),
                "similarity_gap_from_primary": float(row.get("similarity_gap_from_primary", 0)),
                "attachment_evidence": clean(row.get("attachment_evidence")),
                "shared_themes": clean(row.get("shared_themes")),
                "shared_process_stages": clean(row.get("shared_process_stages")),
                "deterministic_veto": deterministic_veto(target_text, clean(row.get("prototype_canonical_issue"))) or "",
            }
        )
    return pd.DataFrame(rows).sort_values(["target_family_id", "attachment_id"]).reset_index(drop=True)


def batches(queue: pd.DataFrame, batch_size: int) -> list[tuple[str, pd.DataFrame]]:
    result: list[tuple[str, pd.DataFrame]] = []
    model_queue = queue[queue["deterministic_veto"].eq("")]
    for family_id, local in model_queue.groupby("target_family_id", sort=True):
        for index, start in enumerate(range(0, len(local), batch_size), start=1):
            result.append((f"{family_id}:{index:04d}", local.iloc[start : start + batch_size].copy()))
    return result


def model_payload(local: pd.DataFrame) -> dict[str, Any]:
    first = local.iloc[0]
    candidates = []
    for row in local.to_dict("records"):
        candidates.append(
            {
                "attachment_id": row["attachment_id"],
                "candidate_issue": row["child_prototype"],
                "candidate_relation": json.loads(row["child_relation"]),
                "primary_family": {
                    "label": row["primary_family_label_hint"],
                    "prototype": row["primary_family_prototype"],
                },
                "shared_themes": row["shared_themes"],
                "shared_process_stages": row["shared_process_stages"],
            }
        )
    return {
        "target_family": {
            "family_id": first["target_family_id"],
            "label_hint": first["target_family_label_hint"],
            "prototype": first["target_family_prototype"],
            "tight_core_examples": json.loads(first["target_core_examples"]),
        },
        "candidates": candidates,
    }


def validate_payload(payload: dict[str, Any], batch_id: str, local: pd.DataFrame) -> list[dict[str, Any]]:
    expected = local["attachment_id"].astype(str).tolist()
    decisions: dict[str, dict[str, Any]] = {}
    for raw in payload.get("decisions", []):
        attachment_id = clean(raw.get("attachment_id"))
        if attachment_id not in expected or attachment_id in decisions:
            raise ValueError(f"Unknown or duplicate attachment ID in {batch_id}: {attachment_id}")
        confidence = clean(raw.get("confidence")).casefold()
        reason = clean(raw.get("reason")).casefold()
        supports = bool(raw.get("supports_family"))
        if confidence not in CONFIDENCES or reason not in REASONS:
            raise ValueError(f"Invalid confidence or reason for {attachment_id}")
        if supports:
            reason = "shared_issue"
        elif reason == "shared_issue":
            reason = "insufficient_evidence"
        decisions[attachment_id] = {
            "attachment_id": attachment_id,
            "supports_family": supports,
            "confidence": confidence,
            "reason": reason,
            "rationale": clean(raw.get("rationale")),
        }
    if missing := set(expected) - set(decisions):
        raise ValueError(f"Missing {len(missing)} decisions in {batch_id}")
    return [decisions[item] for item in expected]


def run_model(
    queue: pd.DataFrame,
    checkpoint_path: Path,
    *,
    host: str,
    model: str,
    timeout: int,
    num_ctx: int,
    batch_size: int,
    retries: int,
    max_batches: int,
    system_prompt: str = SYSTEM_PROMPT,
) -> dict[str, int]:
    work = batches(queue, batch_size)
    if max_batches > 0:
        work = work[:max_batches]
    completed = {
        key: value for key, value in pipeline.read_keyed_jsonl(checkpoint_path, "batch_id").items()
        if value.get("status") == "completed"
    }
    if any(batch_id not in completed for batch_id, _ in work):
        pipeline.ensure_ollama(host, model)
    attempted = failed = 0
    for batch_id, local in tqdm(work, desc="Adjudicating attachments", unit="batch"):
        if batch_id in completed:
            continue
        attempted += 1
        prompt = json.dumps(model_payload(local), indent=2, ensure_ascii=False)
        payload: dict[str, Any] = {}
        raw = ""
        last_error: Exception | None = None
        for _ in range(retries + 1):
            retry_prompt = prompt
            if last_error is not None:
                retry_prompt += f"\nCorrect the invalid JSON. Error: {last_error}. Previous: {json.dumps(payload)}"
            try:
                payload, raw = pipeline.ollama_json(
                    host=host, model=model, system_prompt=system_prompt,
                    user_prompt=retry_prompt, schema=RESPONSE_SCHEMA,
                    timeout=timeout, num_predict=1600, num_ctx=num_ctx,
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
                last_error = exc
        else:
            failed += 1
            record = {
                "batch_id": batch_id, "status": "error", "error": str(last_error),
                "raw_payload": payload, "raw_response": raw,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
        pipeline.append_checkpoint(checkpoint_path, record)
    return {"queued_batches": len(work), "attempted_batches": attempted, "failed_batches": failed}


def checkpoint_decisions(checkpoint_path: Path) -> dict[str, dict[str, Any]]:
    rows: dict[str, dict[str, Any]] = {}
    for record in pipeline.read_keyed_jsonl(checkpoint_path, "batch_id").values():
        if record.get("status") != "completed":
            continue
        for decision in record["decisions"]:
            rows[str(decision["attachment_id"])] = decision
    return rows


def combine_decisions(queue: pd.DataFrame, checkpoint_path: Path) -> pd.DataFrame:
    model_lookup = checkpoint_decisions(checkpoint_path)
    rows: list[dict[str, Any]] = []
    for row in queue.to_dict("records"):
        veto = clean(row["deterministic_veto"])
        if veto:
            decision = {
                "supports_family": False, "confidence": "high", "reason": veto,
                "rationale": "Deterministic high-confidence relational contradiction.",
                "decision_source": "deterministic_veto",
            }
        elif row["attachment_id"] in model_lookup:
            decision = model_lookup[row["attachment_id"]] | {"decision_source": "model"}
            decision.pop("attachment_id", None)
        else:
            decision = {
                "supports_family": None, "confidence": "", "reason": "",
                "rationale": "", "decision_source": "pending",
            }
        rows.append(row | decision)
    output = pd.DataFrame(rows)
    output["adjudication_status"] = output.apply(
        lambda row: "pending" if row["supports_family"] is None else (
            "review" if row["confidence"] == "low" else ("accept" if row["supports_family"] else "reject")
        ), axis=1,
    )
    return output


def apply_verification(output: pd.DataFrame, checkpoint_path: Path) -> pd.DataFrame:
    """Require an independent verifier to agree with first-pass model accepts."""
    verified = checkpoint_decisions(checkpoint_path)
    rows: list[dict[str, Any]] = []
    for row in output.to_dict("records"):
        if row["decision_source"] != "model" or row["adjudication_status"] != "accept":
            rows.append(row | {"verifier_supports_family": "", "verifier_confidence": "", "verifier_reason": ""})
            continue
        decision = verified.get(str(row["attachment_id"]))
        if decision is None:
            rows.append(row | {
                "adjudication_status": "pending_verification",
                "verifier_supports_family": "", "verifier_confidence": "", "verifier_reason": "",
            })
            continue
        verifier_supports = bool(decision["supports_family"])
        confidence = str(decision["confidence"])
        status = "review" if confidence == "low" else ("accept" if verifier_supports else "reject")
        rows.append(row | {
            "supports_family": verifier_supports,
            "confidence": confidence,
            "reason": decision["reason"],
            "rationale": decision["rationale"],
            "decision_source": "model_consensus" if verifier_supports else "model_verifier_reject",
            "adjudication_status": status,
            "verifier_supports_family": verifier_supports,
            "verifier_confidence": confidence,
            "verifier_reason": decision["reason"],
        })
    return pd.DataFrame(rows)


def benchmark_metrics(
    output: pd.DataFrame,
    benchmark: pd.DataFrame,
    decisions_payload: list[dict[str, Any]],
    *,
    minimum_precision: float,
    minimum_weighted_precision: float,
) -> dict[str, Any]:
    truth = {str(row["audit_id"]): str(row["decision"]).casefold() for row in decisions_payload}
    audit_ids = benchmark["audit_id"].astype(str)
    evaluated = output[output["attachment_id"].isin(set(audit_ids))].copy()
    evaluated["truth"] = evaluated["attachment_id"].map(truth)
    evaluated = evaluated[evaluated["truth"].isin(["yes", "no"])]
    decided = evaluated[evaluated["adjudication_status"].isin(["accept", "reject"])]
    accepted = decided[decided["adjudication_status"].eq("accept")]
    true_accepts = accepted[accepted["truth"].eq("yes")]
    positive_weight = int(accepted["child_report_count"].sum())
    all_true = evaluated["truth"].eq("yes")
    precision = len(true_accepts) / len(accepted) if len(accepted) else 0.0
    weighted_precision = int(true_accepts["child_report_count"].sum()) / positive_weight if positive_weight else 0.0
    recall = len(true_accepts) / int(all_true.sum()) if all_true.sum() else 0.0
    metrics = {
        "benchmark_rows": len(evaluated), "decided_rows": len(decided),
        "accepted_rows": len(accepted), "true_accepted_rows": len(true_accepts),
        "accepted_precision": precision,
        "accepted_report_weighted_precision": weighted_precision,
        "accepted_recall": recall,
        "false_accept_ids": accepted.loc[accepted["truth"].eq("no"), "attachment_id"].tolist(),
        "decision_sources": dict(Counter(decided["decision_source"])),
    }
    metrics["precision_gate_passed"] = bool(
        len(accepted) > 0 and precision >= minimum_precision and weighted_precision >= minimum_weighted_precision
    )
    return metrics


def main() -> None:
    args = parse_args()
    if args.batch_size < 1 or args.core_examples < 1:
        raise ValueError("Batch size and core example count must be positive")
    summaries = pd.read_csv(args.family_summary_csv).fillna("")
    memberships = pd.read_csv(args.memberships_csv).fillna("")
    groups = pd.read_csv(args.relational_groups_csv).fillna("")
    occurrences = pd.read_csv(args.relational_occurrences_csv).fillna("")
    queue = build_queue(summaries, memberships, relation_lookup(groups, occurrences), core_examples=args.core_examples)
    if args.benchmark_csv:
        benchmark_ids = set(pd.read_csv(args.benchmark_csv)["audit_id"].astype(str))
        queue = queue[queue["attachment_id"].isin(benchmark_ids)].copy()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = args.output_dir / "01_adjudication_checkpoint.jsonl"
    run = run_model(
        queue, checkpoint, host=args.ollama_host, model=args.model,
        timeout=args.request_timeout, num_ctx=args.ollama_num_ctx,
        batch_size=args.batch_size, retries=args.retries, max_batches=args.max_batches,
    )
    output = combine_decisions(queue, checkpoint)
    verification_queue = queue[
        queue["attachment_id"].isin(
            set(output.loc[output["adjudication_status"].eq("accept"), "attachment_id"])
        )
    ].copy()
    verification_checkpoint = args.output_dir / "02_verification_checkpoint.jsonl"
    verification_run = run_model(
        verification_queue, verification_checkpoint, host=args.ollama_host,
        model=args.model, timeout=args.request_timeout, num_ctx=args.ollama_num_ctx,
        batch_size=args.batch_size, retries=args.retries, max_batches=args.max_batches,
        system_prompt=VERIFIER_SYSTEM_PROMPT,
    )
    output = apply_verification(output, verification_checkpoint)
    output.to_csv(args.output_dir / "02_attachment_adjudication.csv", index=False)
    metrics: dict[str, Any] = {
        "adjudicator_version": ADJUDICATOR_VERSION,
        "attachment_rows": len(output),
        "statuses": output["adjudication_status"].value_counts().to_dict(),
        "deterministic_vetoes": output["deterministic_veto"].value_counts().drop(labels=[""], errors="ignore").to_dict(),
        "run": run,
        "verification_run": verification_run,
    }
    if args.benchmark_csv and args.benchmark_decisions_json:
        benchmark = pd.read_csv(args.benchmark_csv).fillna("")
        decisions_payload = json.loads(args.benchmark_decisions_json.read_text(encoding="utf-8"))
        metrics["benchmark"] = benchmark_metrics(
            output, benchmark, decisions_payload,
            minimum_precision=args.minimum_precision,
            minimum_weighted_precision=args.minimum_weighted_precision,
        )
    (args.output_dir / "metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
