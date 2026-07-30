#!/usr/bin/env python3
"""Add relation-explicit canonical issues without re-extracting reports."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable

import pandas as pd
from tqdm import tqdm

import build_issue_index as pipeline


NORMALIZATION_VERSION = "issue-normalization-v2"
DEFAULT_SCHEMA_PATH = Path(__file__).with_name("issue_normalization_schema_v2.json")
ACTOR_TYPE_FALLBACKS = {
    "provider organisation": "provider organisation",
    "individual practitioner": "individual practitioner",
    "team": "team",
    "employer": "employer",
    "government or public authority": "government or public authority",
    "manufacturer": "manufacturer",
    "regulator": "regulator",
    "multi organisation": "multiple organisations",
    "person individual": "individual",
}

SYSTEM_PROMPT = """You normalize already-extracted safety issues from Prevention of Future Death reports.
Return only JSON matching the requested schema. Use British English.

This is a representational normalization task, not a new extraction task. Preserve the meaning,
failure direction, responsible party, and level of certainty in the supplied evidence. Do not add a
new concern, causal claim, actor, or fact. Do not medicalise neutral or non-medical concerns.
"""

USER_PROMPT = """Normalize every supplied issue. Return exactly one output for every issue_id, in the
same order, with no additions or omissions.

Derive fields in this order before writing canonical_issue:

1. responsible_actor_role: a short, generic role responsible for the failed action, condition, duty,
   system, or decision. Prefer roles such as "mental health service", "prison healthcare team",
   "local authority", "highway authority", "employer", "manufacturer", or "regulator". Remove names
   of people and organisations. Treat a non-"not_stated" input responsible_actor_type as supported
   schema information: if no narrower generic role is supported, use its readable generic label
   (for example "provider organisation", "government or public authority", or "individual
   practitioner"). Use "not stated" only when responsible_actor_type is "not_stated" and no actor is
   supported by responsible_actor_text or evidence_quote. The person exposed to harm is not
   automatically the responsible actor.
2. failed_action: a short, reusable verb phrase naming the action, decision, provision, or maintenance
   that failed, normally 1-8 words. Use the positive action rather than embedding the failure state:
   for example "contact", "share information with", "respond to", "assess", "monitor", "maintain",
   "provide", or "implement". Do not write vague phrases such as "manage issue" or "take action".
3. issue_object: a neutral, reusable noun phrase naming the target, content, system, condition, or
   duty affected by the failed action, normally 2-12 words. It may be information, a service, policy,
   system, equipment, environment, decision, duty, or a nominalised process where that is the natural
   name. Do not merely repeat failed_action. Examples include "risk assessment", "discharge information", "staffing
   levels", "bridge barrier design", "product safety warning", and "custody observation policy".
   Do not call every object a safeguard and do not assume a clinical setting.
4. counterparty_role: the generic recipient, target, or other party required to interpret the action
   and its direction, such as "GP", "ambulance service", "patient", "family", "employee", or "road
   users". Use "not stated" when there is no relevant or supported counterparty. Do not repeat the
   responsible actor.
5. canonical_issue: at most {canonical_max_words} words, stating who failed to do what, to or for
   whom where relevant, and what object was affected.
   Include process, communication direction, recipient, or setting only where it changes the issue
   type. When responsible_actor_role is not "not stated", canonical_issue MUST contain that exact
   generic role phrase (normally as its grammatical subject). If the actor is "not stated", use a
   clear passive formulation rather than inventing one.

Direction and agency rules:
- Never use "disengagement from services" to describe a service failing to contact, follow up,
  escalate, offer reasonable adjustments, or respond to help-seeking.
- Distinguish a person's voluntary refusal or non-attendance from a service's omission. Attribute
  the action to the person only where the evidence explicitly does so.
- For communication failures, identify the sending/receiving sides or recipient when supported.
- Do not turn uncertainty about responsibility into an accusation.
- State the problem, not a recommendation or desired remedy.

The input facets may be imperfect. Resolve them conservatively against evidence_quote; do not copy a
facet into the sentence when the quotation contradicts it.

Input issues:
{issues_json}
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Normalize v3 issue occurrences around actor, object, and failure direction."
    )
    parser.add_argument("--input-csv", required=True, type=Path)
    parser.add_argument(
        "--output-csv",
        type=Path,
        default=None,
        help="Defaults to 01_issue_occurrences_normalized.csv beside the input.",
    )
    parser.add_argument(
        "--selection-csv",
        type=Path,
        default=None,
        help="Optional pilot selection containing issue_id or report_key.",
    )
    parser.add_argument("--limit", type=int, default=0, help="Normalize only the first N selected rows.")
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--model", default="gemma4:26b")
    parser.add_argument("--batch-size", type=int, default=12)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--request-timeout", type=int, default=360)
    parser.add_argument("--num-predict", type=int, default=3200)
    parser.add_argument("--num-ctx", type=int, default=8192)
    parser.add_argument("--canonical-max-words", type=int, default=26)
    parser.add_argument(
        "--rebuild-output-only",
        action="store_true",
        help="Rebuild CSV metrics/review outputs from checkpoints without model calls.",
    )
    return parser.parse_args()


def row_payload(row: pd.Series) -> dict[str, str]:
    fields = (
        "issue_id",
        "evidence_quote",
        "canonical_issue",
        "failure_state",
        "process_stage",
        "communication_direction",
        "responsible_actor_type",
        "responsible_actor_text",
        "service_sectors",
        "issue_themes",
        "service_contexts",
        "populations_at_risk",
        "concern_status",
    )
    return {field: pipeline.clean_text(row.get(field)) for field in fields}


def batch_key(rows: Iterable[dict[str, str]]) -> str:
    issue_ids = [row["issue_id"] for row in rows]
    return f"nrm_{pipeline.stable_hash(*issue_ids)}"


def input_hash(rows: Iterable[dict[str, str]]) -> str:
    value = json.dumps(
        {
            "version": NORMALIZATION_VERSION,
            "system_prompt": SYSTEM_PROMPT,
            "user_prompt": USER_PROMPT,
            "rows": list(rows),
        },
        sort_keys=True,
        ensure_ascii=False,
    )
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def edit_distance(left: str, right: str) -> int:
    """Return Levenshtein distance for short model-returned identifiers."""
    if len(left) < len(right):
        left, right = right, left
    previous = list(range(len(right) + 1))
    for left_index, left_character in enumerate(left, 1):
        current = [left_index]
        for right_index, right_character in enumerate(right, 1):
            current.append(
                min(
                    current[-1] + 1,
                    previous[right_index] + 1,
                    previous[right_index - 1] + (left_character != right_character),
                )
            )
        previous = current
    return previous[-1]


def recover_issue_id(issue_id: str, expected_ids: list[str]) -> tuple[str, bool]:
    if issue_id in expected_ids:
        return issue_id, False
    distances = [(edit_distance(issue_id, expected), expected) for expected in expected_ids]
    minimum = min((distance for distance, _ in distances), default=999)
    matches = [expected for distance, expected in distances if distance == minimum]
    if issue_id and minimum <= 2 and len(matches) == 1:
        return matches[0], True
    raise ValueError(f"unexpected issue_id: {issue_id or '<missing>'}")


def normalize_result(raw: Any, expected_ids: list[str], max_words: int) -> list[dict[str, Any]]:
    if not isinstance(raw, list):
        raise ValueError("response did not contain an issues array")
    by_id: dict[str, dict[str, str]] = {}
    for item in raw:
        if not isinstance(item, dict):
            raise ValueError("normalization item was not an object")
        supplied_issue_id = pipeline.clean_text(item.get("issue_id"))
        issue_id, issue_id_recovered = recover_issue_id(supplied_issue_id, expected_ids)
        if issue_id in by_id:
            raise ValueError(f"duplicate issue_id: {issue_id}")
        actor = pipeline.trim_words(item.get("responsible_actor_role"), 12)
        action = pipeline.trim_words(item.get("failed_action"), 8)
        obj = pipeline.trim_words(item.get("issue_object"), 12)
        counterparty = pipeline.trim_words(item.get("counterparty_role"), 12)
        canonical = pipeline.trim_words(item.get("canonical_issue"), max_words)
        if not actor or not action or not obj or not counterparty or not canonical:
            raise ValueError(f"empty normalized field for {issue_id}")
        if pipeline.normalised_key(actor) != "not stated" and pipeline.normalised_key(
            actor
        ) not in pipeline.normalised_key(canonical):
            raise ValueError(
                f"canonical_issue does not contain responsible_actor_role for {issue_id}"
            )
        by_id[issue_id] = {
            "responsible_actor_role": actor,
            "failed_action": action,
            "issue_object": obj,
            "counterparty_role": counterparty,
            "canonical_issue": canonical,
            "linkage_statement": build_linkage_statement(
                actor, action, obj, counterparty
            ),
            "normalization_warnings": (
                [f"issue_id_recovered:{supplied_issue_id}"] if issue_id_recovered else []
            ),
        }
    missing = [issue_id for issue_id in expected_ids if issue_id not in by_id]
    if missing:
        raise ValueError(f"response omitted issue_ids: {missing}")
    return [{"issue_id": issue_id, **by_id[issue_id]} for issue_id in expected_ids]


def build_linkage_statement(
    actor: Any, action: Any, issue_object: Any, counterparty: Any
) -> str:
    """Build a stable relational representation for embedding and pair scoring."""
    return (
        f"Actor: {pipeline.clean_text(actor)}. "
        f"Failed action: {pipeline.clean_text(action)}. "
        f"Object: {pipeline.clean_text(issue_object)}. "
        f"Counterparty: {pipeline.clean_text(counterparty)}."
    )


def validate_source_actor(
    results: list[dict[str, Any]], rows: list[dict[str, str]]
) -> None:
    inputs = {row["issue_id"]: row for row in rows}
    for result in results:
        source = inputs[result["issue_id"]]
        actor_missing = pipeline.normalised_key(result["responsible_actor_role"]) == "not stated"
        actor_type = pipeline.normalised_key(source.get("responsible_actor_type"))
        actor_text = pipeline.normalised_key(source.get("responsible_actor_text"))
        if actor_missing and (actor_type not in {"", "not stated"} or actor_text):
            raise ValueError(
                f"responsible_actor_role discarded extracted actor information for {result['issue_id']}"
            )


def normalize_batch(
    rows: list[dict[str, str]], args: argparse.Namespace, schema: dict[str, Any]
) -> dict[str, Any]:
    key = batch_key(rows)
    fingerprint = input_hash(rows)
    prompt = USER_PROMPT.format(
        canonical_max_words=args.canonical_max_words,
        issues_json=json.dumps(rows, ensure_ascii=False, indent=2),
    )
    last_error = ""
    last_raw = ""
    for attempt in range(1, args.retries + 2):
        try:
            payload, last_raw = pipeline.ollama_json(
                host=args.ollama_host,
                model=args.model,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=(
                    prompt
                    if not last_error
                    else f"{prompt}\n\nYour previous response was rejected: {last_error}. Correct that error."
                ),
                schema=schema,
                timeout=args.request_timeout,
                num_predict=args.num_predict,
                num_ctx=args.num_ctx,
            )
            results = normalize_result(
                payload.get("issues"), [row["issue_id"] for row in rows], args.canonical_max_words
            )
            validate_source_actor(results, rows)
            return {
                "batch_key": key,
                "input_sha256": fingerprint,
                "status": "success",
                "attempts": attempt,
                "results": results,
                "completed_at": datetime.now(timezone.utc).isoformat(),
            }
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
    # One malformed item should not discard an otherwise useful batch. Retry each
    # item independently so the correction message identifies the exact failure.
    recovered: list[dict[str, Any]] = []
    recovery_errors: list[str] = []
    for row in rows:
        row_prompt = USER_PROMPT.format(
            canonical_max_words=args.canonical_max_words,
            issues_json=json.dumps([row], ensure_ascii=False, indent=2),
        )
        row_error = ""
        for _attempt in range(1, args.retries + 2):
            try:
                payload, last_raw = pipeline.ollama_json(
                    host=args.ollama_host,
                    model=args.model,
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=(
                        row_prompt
                        if not row_error
                        else f"{row_prompt}\n\nYour previous response was rejected: {row_error}. Correct that error."
                    ),
                    schema=schema,
                    timeout=args.request_timeout,
                    num_predict=args.num_predict,
                    num_ctx=args.num_ctx,
                )
                result = normalize_result(
                    payload.get("issues"), [row["issue_id"]], args.canonical_max_words
                )
                validate_source_actor(result, [row])
                recovered.extend(result)
                break
            except Exception as exc:  # noqa: BLE001
                row_error = f"{type(exc).__name__}: {exc}"
        else:
            recovery_errors.append(f"{row['issue_id']}: {row_error}")
    if not recovery_errors:
        return {
            "batch_key": key,
            "input_sha256": fingerprint,
            "status": "success",
            "attempts": args.retries + 1,
            "recovered_individually": True,
            "results": recovered,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
    return {
        "batch_key": key,
        "input_sha256": fingerprint,
        "status": "partial" if recovered else "failure",
        "attempts": args.retries + 1,
        "issue_ids": [row["issue_id"] for row in rows],
        "results": recovered,
        "failed_issue_ids": [error.split(":", 1)[0] for error in recovery_errors],
        "error": " | ".join(recovery_errors) or last_error,
        "raw_response": last_raw[:2000],
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


def load_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    records: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return records
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), 1):
        if not line.strip():
            continue
        try:
            record = json.loads(line)
            records[pipeline.clean_text(record.get("batch_key"))] = record
        except json.JSONDecodeError as exc:
            raise ValueError(f"Invalid checkpoint JSON on line {line_number}") from exc
    return records


def append_checkpoint(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()
        os.fsync(handle.fileno())


@contextmanager
def normalization_lock(path: Path):
    descriptor: int | None = None
    for _ in range(2):
        try:
            descriptor = os.open(path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError as exc:
            details = pipeline.clean_text(path.read_text(encoding="utf-8"))
            try:
                pid = int((json.loads(details) or {}).get("pid"))
                os.kill(pid, 0)
            except (ValueError, TypeError, json.JSONDecodeError, ProcessLookupError):
                path.unlink(missing_ok=True)
                continue
            except PermissionError:
                pass
            command_path = Path(f"/proc/{pid}/cmdline")
            try:
                command = command_path.read_bytes().replace(b"\0", b" ").decode(
                    "utf-8", errors="replace"
                )
            except OSError:
                command = ""
            if "normalize_issue_occurrences.py" not in command:
                path.unlink(missing_ok=True)
                continue
            raise RuntimeError(f"Another normalizer owns {path}. {details}") from exc
    if descriptor is None:
        raise RuntimeError(f"Could not acquire normalization lock: {path}")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(json.dumps({"pid": os.getpid(), "started_at": datetime.now(timezone.utc).isoformat()}))
        yield
    finally:
        path.unlink(missing_ok=True)


def select_rows(frame: pd.DataFrame, selection_path: Path | None, limit: int) -> pd.DataFrame:
    selected = frame
    if selection_path:
        selection = pd.read_csv(selection_path).fillna("")
        if "issue_id" in selection.columns:
            selected = selected[selected["issue_id"].isin(selection["issue_id"])]
        elif "report_key" in selection.columns:
            selected = selected[selected["report_key"].isin(selection["report_key"])]
        else:
            raise ValueError("Selection CSV must contain issue_id or report_key")
    if limit > 0:
        selected = selected.head(limit)
    return selected.copy()


def build_output(frame: pd.DataFrame, records: dict[str, dict[str, Any]]) -> pd.DataFrame:
    normalized: dict[str, dict[str, Any]] = {}
    for record in records.values():
        if record.get("status") not in {"success", "partial"}:
            continue
        for result in record.get("results") or []:
            normalized[pipeline.clean_text(result.get("issue_id"))] = result
    output = frame.copy()
    if "canonical_issue_original" not in output.columns:
        output.insert(
            output.columns.get_loc("canonical_issue") + 1,
            "canonical_issue_original",
            output["canonical_issue"],
        )
    output["responsible_actor_role"] = ""
    output["failed_action"] = ""
    output["issue_object"] = ""
    output["counterparty_role"] = ""
    output["linkage_statement"] = ""
    output["normalization_version"] = NORMALIZATION_VERSION
    output["normalization_status"] = "not_selected"
    output["normalization_warnings"] = ""
    for index, issue_id in output["issue_id"].items():
        result = normalized.get(pipeline.clean_text(issue_id))
        if not result:
            continue
        output.at[index, "canonical_issue"] = result["canonical_issue"]
        output.at[index, "responsible_actor_role"] = result["responsible_actor_role"]
        output.at[index, "failed_action"] = result["failed_action"]
        output.at[index, "issue_object"] = result["issue_object"]
        output.at[index, "counterparty_role"] = result["counterparty_role"]
        output.at[index, "linkage_statement"] = build_linkage_statement(
            result["responsible_actor_role"],
            result["failed_action"],
            result["issue_object"],
            result["counterparty_role"],
        )
        output.at[index, "normalization_status"] = "success"
        warnings: list[str] = list(result.get("normalization_warnings") or [])
        actor_key = pipeline.normalised_key(result["responsible_actor_role"])
        source_actor = (
            output.at[index, "responsible_actor_text"]
            if "responsible_actor_text" in output.columns
            else ""
        )
        actor_text = pipeline.normalised_key(source_actor)
        actor_type = pipeline.normalised_key(
            output.at[index, "responsible_actor_type"]
            if "responsible_actor_type" in output.columns
            else ""
        )
        if actor_key == "not stated" and actor_text:
            warnings.append("actor_not_stated_despite_source_actor")
        if actor_key == "not stated" and actor_type not in {"", "not stated"}:
            warnings.append("actor_not_stated_despite_actor_type")
        if len(pipeline.clean_text(result["issue_object"]).split()) == 1:
            warnings.append("single_word_object_review")
        if pipeline.normalised_key(result["failed_action"]) in {
            "action described in evidence",
            "manage",
            "take action",
        }:
            warnings.append("generic_failed_action_review")
        direction = pipeline.normalised_key(
            output.at[index, "communication_direction"]
            if "communication_direction" in output.columns
            else ""
        )
        if direction not in {"", "not applicable", "unclear"} and pipeline.normalised_key(
            result["counterparty_role"]
        ) == "not stated":
            warnings.append("directional_issue_without_counterparty_review")
        if "disengag" in pipeline.normalised_key(result["canonical_issue"]):
            warnings.append("agency_sensitive_disengagement_review")
        named_roles = {
            "h m prison service",
            "home office",
            "national probation service",
            "medical assessment unit",
            "accident and emergency department",
        }
        if actor_key in named_roles:
            warnings.append("actor_role_may_be_named_entity")
        output.at[index, "normalization_warnings"] = " | ".join(warnings)
    return output


def apply_original_fallbacks(
    output: pd.DataFrame, failed_issue_ids: set[str]
) -> pd.DataFrame:
    """Preserve failed rows with explicit, reviewable non-empty fallback fields."""
    if not failed_issue_ids:
        return output
    output = output.copy()
    for index in output.index[output["issue_id"].isin(failed_issue_ids)]:
        actor_type = pipeline.normalised_key(
            output.at[index, "responsible_actor_type"]
            if "responsible_actor_type" in output.columns
            else ""
        )
        actor = ACTOR_TYPE_FALLBACKS.get(actor_type, "not stated")
        output.at[index, "responsible_actor_role"] = actor
        output.at[index, "failed_action"] = "action described in evidence"
        output.at[index, "issue_object"] = "issue described in evidence"
        output.at[index, "counterparty_role"] = "not stated"
        output.at[index, "linkage_statement"] = build_linkage_statement(
            actor,
            "action described in evidence",
            "issue described in evidence",
            "not stated",
        )
        output.at[index, "normalization_status"] = "fallback_original"
        warnings = [
            item
            for item in pipeline.clean_text(
                output.at[index, "normalization_warnings"]
            ).split(" | ")
            if item
        ]
        warnings.extend(
            [
                "normalization_failed_original_preserved",
                "actor_role_deterministic_fallback",
                "failed_action_unspecified_fallback",
                "issue_object_unspecified_fallback",
            ]
        )
        output.at[index, "normalization_warnings"] = " | ".join(
            dict.fromkeys(warnings)
        )
    return output


def build_review_queue(output: pd.DataFrame) -> pd.DataFrame:
    return output[
        output["normalization_status"].isin({"failed", "fallback_original"})
        | output["normalization_warnings"].str.len().gt(0)
    ].copy()


def main() -> None:
    args = parse_args()
    if not 1 <= args.batch_size <= 16:
        raise ValueError("--batch-size must be between 1 and 16")
    if args.workers < 1:
        raise ValueError("--workers must be at least 1")
    frame = pd.read_csv(args.input_csv).fillna("")
    required = {"issue_id", "canonical_issue", "evidence_quote"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Input occurrence CSV is missing columns: {sorted(missing)}")
    if frame["issue_id"].duplicated().any():
        raise ValueError("Input issue_id values must be unique")
    selected = select_rows(frame, args.selection_csv, args.limit)
    output_path = args.output_csv or args.input_csv.with_name("01_issue_occurrences_normalized.csv")
    checkpoint_path = output_path.with_suffix(".checkpoint.jsonl")
    metrics_path = output_path.with_suffix(".metrics.json")
    review_path = output_path.with_name(f"{output_path.stem}_review_queue.csv")
    output_path.parent.mkdir(parents=True, exist_ok=True)
    schema = json.loads(DEFAULT_SCHEMA_PATH.read_text(encoding="utf-8"))
    payloads = [row_payload(row) for _, row in selected.iterrows()]
    batches = [payloads[start : start + args.batch_size] for start in range(0, len(payloads), args.batch_size)]
    records = load_checkpoint(checkpoint_path)
    pending = [
        batch for batch in batches
        if not (
            (record := records.get(batch_key(batch)))
            and record.get("status") == "success"
            and record.get("input_sha256") == input_hash(batch)
        )
    ]
    print(f"Selected {len(selected):,} issues in {len(batches):,} batches; {len(pending):,} pending.")
    if pending and not args.rebuild_output_only:
        pipeline.ensure_ollama(args.ollama_host, args.model)
        with normalization_lock(output_path.with_suffix(".lock")):
            with ThreadPoolExecutor(max_workers=args.workers) as executor:
                futures = {executor.submit(normalize_batch, batch, args, schema): batch for batch in pending}
                for future in tqdm(as_completed(futures), total=len(futures), desc="Normalizing issues", unit="batch"):
                    record = future.result()
                    append_checkpoint(checkpoint_path, record)
                    records[record["batch_key"]] = record
    output = build_output(frame, records)
    selected_ids = set(selected["issue_id"])
    failed_ids = selected_ids - set(output.loc[output["normalization_status"] == "success", "issue_id"])
    output = apply_original_fallbacks(output, failed_ids)
    output.to_csv(output_path, index=False)
    review = build_review_queue(output)
    review.to_csv(review_path, index=False)
    metrics = {
        "normalization_version": NORMALIZATION_VERSION,
        "input_csv": str(args.input_csv),
        "output_csv": str(output_path),
        "issues_total": len(frame),
        "issues_selected": len(selected),
        "issues_normalized": int((output["normalization_status"] == "success").sum()),
        "issues_failed": len(failed_ids),
        "issues_fallback_original": int(
            (output["normalization_status"] == "fallback_original").sum()
        ),
        "issues_not_selected": int((output["normalization_status"] == "not_selected").sum()),
        "issues_flagged_for_review": len(review),
        "batches_total": len(batches),
        "batches_succeeded": sum(
            records.get(batch_key(batch), {}).get("status") == "success"
            for batch in batches
        ),
        "batches_partial": sum(
            records.get(batch_key(batch), {}).get("status") == "partial"
            for batch in batches
        ),
        "checkpoint_successful_batches": sum(
            record.get("status") == "success" for record in records.values()
        ),
        "review_queue_csv": str(review_path),
    }
    metrics_path.write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    print(f"Wrote {output_path}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
