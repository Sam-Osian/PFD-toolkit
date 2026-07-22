#!/usr/bin/env python3
"""Build a resumable, UI-ready recurring-issue index from PFD reports.

The expensive stage makes one structured Ollama request per report. Everything
after that point is deterministic and can be rerun from the occurrence CSV.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import re
import sys
from collections import Counter
from concurrent.futures import ThreadPoolExecutor, as_completed
from contextlib import contextmanager
from dataclasses import asdict, dataclass
from datetime import datetime, timezone
from difflib import SequenceMatcher
from pathlib import Path
from typing import Any, Iterable

import numpy as np
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer
from sklearn.cluster import AgglomerativeClustering
from sklearn.neighbors import NearestNeighbors
from tqdm import tqdm


PIPELINE_VERSION = "issue-index-v3"
SCHEMA_VERSION = "issue-occurrence-v3"
DEFAULT_SCHEMA_PATH = Path(__file__).with_name("issue_schema_v3.json")

SYSTEM_PROMPT = """You extract substantive safety issues from Prevention of Future Death reports.
Return only JSON matching the requested schema. Use British English.

An issue is one distinct systemic concern, omission, failure, unsafe condition, or service gap.
Extract the concerns communicated by the coroner, including the underlying missing safeguard when
a concern is expressed only as a recommendation or proposed remedy.
Do not invent actors, causes, settings, or details that are not supported by the text.
Use not_stated when evidence does not support a controlled value.
The evidence_quote must be a short verbatim excerpt copied from the supplied report text.
Treat Concerns as primary. Also extract an explicitly stated reusable systemic failure from
Circumstances when it is presented as a finding, policy or system gap, repeated practice, or
clear omission. Do not infer an issue merely from chronology, an adverse outcome, or an isolated
individual act described without a wider safety concern.
Avoid biography, chronology, consequences, causal conclusions about the death, and duplicates.
Optimise for recall: extract every independently actionable failure or missing safeguard.
Split materially different failures. Do not split one failure into examples, causes, remedies,
consequences, or restatements, but do not collapse distinct safeguards into a broad theme.
"""

USER_PROMPT = """Extract at most {max_issues} distinct issues from this report.

For every issue return all fields. Arrays must use only allowed values and respect their limits.

Definitions:
- canonical_issue: reusable neutral failure formulation, at most {canonical_max_words} words.
- evidence_quote: a short exact quotation copied from the report; never paraphrase it.
- failure_state: how the action or object failed, not the action itself.
- process_stage: the direct operational activity that failed.
- service_sectors: broad sectors involved; at most two.
- issue_themes: cross-cutting safety themes; at most three.
- service_contexts: services or environments directly involved; at most two.
- populations_at_risk: people who could suffer harm, not automatically the deceased's role; at most two.
- responsible_actor_text: actor stated in the report, or empty string if not stated.
- concern_status: historical_failure for a past event only; current_system_gap for an unresolved
  deficiency; future_risk for a prospective hazard; recommendation_only when only a remedy is proposed.

Canonicalisation:
1. State the problem, not the remedy. Prefer "Medication changes were not communicated" over
   "Need for a communication system".
2. Include the failed object, failure state, and process. Include direction or context only when
   it changes the reusable issue type.
3. Remove names, organisations, dates, local codes, durations, consequences, and local locations.
4. Put essential distinctions such as discharge, handover, Mental Health Act, falls, or Category 2
   directly in canonical_issue.
5. Do not begin with "Need for", "Recommendation for", or "Risk of" when an underlying failure exists.
6. Do not combine different failure states, process stages, or communication directions.
7. Do not emit outcomes such as "contributed to death" as separate issues.

Consolidation:
- Compare candidates by failed object, failure state, process, communication direction, and essential
  distinctions in canonical_issue.
- Emit one issue when candidates are restatements or one merely adds an example, cause, remedy, or consequence.
- Keep medication supply separate from administration, sending separate from receiving, and missing
  assessment separate from delayed assessment.
- Keep separate named policies, systems, referrals, assessments, records, training duties, action plans,
  notification duties, and review duties when each could be corrected independently.
- Keep failures by different responsible actors separate unless the evidence describes one genuinely
  shared cross-organisational process.
- Do not replace a list of concrete failures with one umbrella issue such as "policies were inadequate"
  or "communication was poor".

Controlled fields:
- Use omitted when an action, service, system, policy, or safeguard was absent or not done.
- Use inadequate for insufficient or ineffective quality; unverified for lack of assurance or
  confirmation; ambiguous for unclear ownership, instructions, authority, or criteria; and
  uncoordinated where connected services or teams operate in silos.
- The process names the action: missing assessment is assessment+omitted; missing records are
  information_record_management+omitted.
- Sectors and themes are separate. Medication communication in prison can use healthcare and
  justice_custody sectors with medication and communication_handover themes.
- Use legal_judicial for courts and prosecution, defence_military for armed-services activity,
  agriculture_animal for farming and livestock, sport_leisure for organised sport, and
  commercial_retail or digital_online_services for vendors and online platforms.
- Use animal_care_control for domestic or companion-animal management, and
  risk_assessment_management for non-clinical risk assessment and control.
- student_in_education is only an enrolled school, college, or university learner. Use trainee for
  vocational, diving, aviation, or workplace trainees.
- Use participant for people taking part in sport or recreation when trainee does not apply.
- Use not_applicable only when context does not apply, not when unstated. Use not_stated when evidence
  is absent. Use other_review only when evidence is present but no vocabulary value fits.

Allowed values:
{enum_guide}

Output shape:
{{"issues": [{{
  "canonical_issue": "string",
  "evidence_quote": "verbatim excerpt",
  "failure_state": "enum",
  "process_stage": "enum",
  "service_sectors": ["enum"],
  "issue_themes": ["enum"],
  "service_contexts": ["enum"],
  "populations_at_risk": ["enum"],
  "communication_direction": "enum",
  "responsible_actor_type": "enum",
  "responsible_actor_text": "string or empty",
  "concern_status": "enum"
}}]}}

Report text:
{source_text}
"""

LABEL_SYSTEM_PROMPT = """You label a tightly grouped recurring safety issue.
Return JSON only. Use British English. Do not broaden the issue beyond the examples."""

LABEL_USER_PROMPT = """Name this recurring sub-issue using the examples.

Return:
{{"label": "specific label of 2-8 words", "description": "one precise sentence"}}

Examples:
{examples}
"""

SCALAR_ENUM_FIELDS = (
    "failure_state",
    "process_stage",
    "communication_direction",
    "responsible_actor_type",
    "concern_status",
)

ARRAY_ENUM_FIELDS = (
    "service_sectors",
    "issue_themes",
    "service_contexts",
    "populations_at_risk",
)

ENUM_FIELDS = SCALAR_ENUM_FIELDS + ARRAY_ENUM_FIELDS

TEXT_LIMITS = {
    "canonical_issue": 260,
    "evidence_quote": 700,
    "responsible_actor_text": 180,
}

THEME_LABELS = {
    "clinical_assessment_care": "Clinical assessment and care",
    "medication": "Medication safety",
    "mental_health_suicide": "Mental health and suicide prevention",
    "communication_handover": "Communication and handover",
    "records_information": "Records and information",
    "staffing_capacity": "Staffing and capacity",
    "training_competence": "Training and competence",
    "equipment_infrastructure": "Equipment and infrastructure",
    "digital_technology": "Digital technology",
    "emergency_response": "Emergency response",
    "safeguarding": "Safeguarding",
    "governance_learning": "Governance and learning",
    "policy_regulation": "Policy and regulation",
    "environmental_design": "Environmental design",
    "access_service_capacity": "Access and service capacity",
    "public_information_warning": "Public information and warnings",
    "risk_assessment_management": "Risk assessment and management",
    "animal_management": "Animal management",
    "other_review": "Other issues requiring review",
}


@dataclass(frozen=True)
class Config:
    pipeline_version: str
    schema_version: str
    input_csv: str
    run_dir: str
    stage: str
    subset_size: int
    random_seed: int
    ollama_host: str
    model: str
    embedding_model: str
    embedding_mode: str
    original_view_weight: float
    allow_model_download: bool
    max_issues: int
    continuation_issues: int
    continue_at_cap: bool
    canonical_max_words: int
    max_source_chars: int
    retries: int
    extract_num_predict: int
    extraction_workers: int
    ollama_num_ctx: int
    top_k: int
    edge_similarity: float
    split_similarity: float
    min_centroid_similarity: float
    min_recurring_reports: int
    label_subissues: bool


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build a one-pass, resumable recurring-issue index for PFD reports."
    )
    parser.add_argument("--input-csv", default="all_reports.csv")
    parser.add_argument("--output-dir", default="artifacts/issue_index_v3")
    parser.add_argument(
        "--run-dir",
        default=None,
        help="Use an existing run directory to resume or rebuild its deterministic index.",
    )
    parser.add_argument("--stage", choices=["extract", "index", "all"], default="all")
    parser.add_argument(
        "--occurrences-csv",
        default=None,
        help="Occurrence CSV for --stage index. Defaults to <run-dir>/01_issue_occurrences.csv.",
    )
    parser.add_argument("--subset-size", type=int, default=0, help="0 means the complete archive.")
    parser.add_argument("--random-seed", type=int, default=42)
    parser.add_argument("--ollama-host", default="http://localhost:11434")
    parser.add_argument("--model", default="gemma4:26b")
    parser.add_argument("--embedding-model", default="Qwen/Qwen3-Embedding-8B")
    parser.add_argument(
        "--embedding-mode",
        choices=["auto", "single", "dual"],
        default="auto",
        help=(
            "auto uses the configured dual-view blend when canonical_issue_original is present; "
            "single embeds canonical_issue only; dual requires the original column."
        ),
    )
    parser.add_argument(
        "--original-view-weight",
        type=float,
        default=0.44,
        help="Original-sentence weight in dual mode; the normalized view receives 1-weight.",
    )
    parser.add_argument(
        "--allow-model-download",
        action="store_true",
        help="Allow the embedding library to contact Hugging Face if the model is not cached.",
    )
    parser.add_argument("--max-issues", type=int, default=24)
    parser.add_argument("--continuation-issues", type=int, default=12)
    parser.add_argument(
        "--continue-at-cap",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Make one continuation call when the initial response reaches --max-issues.",
    )
    parser.add_argument("--canonical-max-words", type=int, default=22)
    parser.add_argument("--max-source-chars", type=int, default=12000)
    parser.add_argument("--retries", type=int, default=2)
    parser.add_argument("--extract-num-predict", type=int, default=6400)
    parser.add_argument(
        "--ollama-num-ctx",
        type=int,
        default=16384,
        help="Ollama context window. 16k covers the bounded source, prompt, and response.",
    )
    parser.add_argument(
        "--extraction-workers",
        type=int,
        default=1,
        help="Concurrent Ollama requests. Keep at 1 unless the local model server supports parallel requests.",
    )
    parser.add_argument("--request-timeout", type=int, default=360)
    parser.add_argument("--top-k", type=int, default=40)
    parser.add_argument("--edge-similarity", type=float, default=0.84)
    parser.add_argument(
        "--split-similarity",
        type=float,
        default=0.872,
        help="Average-linkage similarity used to split chained graph components.",
    )
    parser.add_argument("--min-centroid-similarity", type=float, default=0.83)
    parser.add_argument("--min-recurring-reports", type=int, default=3)
    parser.add_argument(
        "--label-subissues",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="Generate one offline LLM label per candidate sub-issue.",
    )
    parser.add_argument(
        "--retry-failures",
        action=argparse.BooleanOptionalAction,
        default=True,
        help="When resuming, retry reports whose latest checkpoint record failed.",
    )
    return parser.parse_args()


def clean_text(value: Any) -> str:
    if value is None:
        return ""
    if isinstance(value, float) and math.isnan(value):
        return ""
    return re.sub(r"\s+", " ", str(value)).strip()


def normalised_key(value: Any) -> str:
    return re.sub(r"[^a-z0-9]+", " ", clean_text(value).casefold()).strip()


def trim_words(value: Any, maximum: int) -> str:
    words = clean_text(value).strip(" \"'").split()
    return " ".join(words[:maximum])


def stable_hash(*values: Any, length: int = 16) -> str:
    payload = "\x1f".join(clean_text(value) for value in values)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()[:length]


def load_schema(path: Path = DEFAULT_SCHEMA_PATH) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def schema_enums(schema: dict[str, Any]) -> dict[str, set[str]]:
    properties = schema["properties"]["issues"]["items"]["properties"]
    output: dict[str, set[str]] = {}
    for field in SCALAR_ENUM_FIELDS:
        output[field] = set(properties[field]["enum"])
    for field in ARRAY_ENUM_FIELDS:
        output[field] = set(properties[field]["items"]["enum"])
    return output


def compact_enum_guide(enums: dict[str, set[str]]) -> str:
    return "\n".join(f"- {field}: {', '.join(sorted(values))}" for field, values in enums.items())


def parse_json_payload(raw: str) -> dict[str, Any]:
    value = clean_text(raw)
    value = re.sub(r"^```(?:json)?", "", value, flags=re.IGNORECASE).strip()
    value = re.sub(r"```$", "", value).strip()
    try:
        parsed = json.loads(value)
        return parsed if isinstance(parsed, dict) else {}
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", value, flags=re.DOTALL)
        if not match:
            return {}
        try:
            parsed = json.loads(match.group(0))
            return parsed if isinstance(parsed, dict) else {}
        except json.JSONDecodeError:
            return {}


def ollama_json(
    *,
    host: str,
    model: str,
    system_prompt: str,
    user_prompt: str,
    schema: dict[str, Any] | None,
    timeout: int,
    num_predict: int,
    num_ctx: int | None = None,
) -> tuple[dict[str, Any], str]:
    options = {"temperature": 0.0, "top_p": 1.0, "num_predict": num_predict}
    if num_ctx:
        options["num_ctx"] = num_ctx
    payload = {
        "model": model,
        "messages": [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt},
        ],
        "stream": False,
        "format": schema or "json",
        "think": False,
        "options": options,
    }
    response = requests.post(f"{host.rstrip('/')}/api/chat", json=payload, timeout=timeout)
    response.raise_for_status()
    raw = clean_text((response.json().get("message") or {}).get("content"))
    return parse_json_payload(raw), raw


def ensure_ollama(host: str, model: str) -> None:
    response = requests.get(f"{host.rstrip('/')}/api/tags", timeout=20)
    response.raise_for_status()
    names = {clean_text(item.get("name")) for item in response.json().get("models", [])}
    if model not in names:
        raise RuntimeError(f"Ollama model '{model}' is unavailable. Available models: {sorted(names)[:12]}")


def report_identity(row: pd.Series) -> str:
    return clean_text(row.get("url")) or clean_text(row.get("id"))


def stable_report_key(row: pd.Series) -> str:
    return f"rpt_{stable_hash(report_identity(row))}"


def load_reports(path: Path, subset_size: int, seed: int) -> pd.DataFrame:
    required = {"id", "url", "date", "coroner", "area", "circumstances", "concerns"}
    frame = pd.read_csv(path)
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Input CSV is missing columns: {sorted(missing)}")
    if "receiver" not in frame.columns:
        frame["receiver"] = ""
    for column in required | {"receiver"}:
        frame[column] = frame[column].map(clean_text)
    frame = frame[(frame["circumstances"].str.len() > 0) | (frame["concerns"].str.len() > 0)]
    frame = frame.drop_duplicates(subset=["url"], keep="first").reset_index(drop=True)
    if 0 < subset_size < len(frame):
        frame = frame.sample(n=subset_size, random_state=seed).reset_index(drop=True)
    frame["report_key"] = frame.apply(stable_report_key, axis=1)
    return frame


def build_source_text(report: pd.Series, max_chars: int) -> str:
    concerns = clean_text(report.get("concerns"))
    circumstances = clean_text(report.get("circumstances"))
    heading_budget = len("Concerns:\n\n\nCircumstances:\n")
    budget = max(500, max_chars - heading_budget)
    # Preserve the concerns section first; it is the coroner's explicit future-risk statement.
    concerns_part = concerns[:budget]
    remaining = max(0, budget - len(concerns_part))
    circumstances_part = circumstances[:remaining]
    return f"Concerns:\n{concerns_part}\n\nCircumstances:\n{circumstances_part}"


def source_span_status(span: str, section: str, report: pd.Series) -> tuple[bool, str]:
    if not span:
        return False, "missing"
    needle = normalised_key(span)
    if len(needle) < 12:
        return False, "too_short"
    sources = {
        "concerns": clean_text(report.get("concerns")),
        "circumstances": clean_text(report.get("circumstances")),
        "both": f"{clean_text(report.get('concerns'))} {clean_text(report.get('circumstances'))}",
        "unclear": f"{clean_text(report.get('concerns'))} {clean_text(report.get('circumstances'))}",
    }
    haystack = normalised_key(sources.get(section, sources["unclear"]))
    if needle in haystack:
        return True, "exact_normalised"
    all_text = normalised_key(sources["both"])
    if needle in all_text:
        return True, "found_other_section"
    return False, "not_found"


def locate_evidence_section(quote: str, report: pd.Series) -> tuple[bool, str, str]:
    if not quote:
        return False, "not_stated", "missing"
    needle = normalised_key(quote)
    if len(needle) < 12:
        return False, "not_stated", "too_short"
    in_concerns = needle in normalised_key(report.get("concerns"))
    in_circumstances = needle in normalised_key(report.get("circumstances"))
    if in_concerns and in_circumstances:
        return True, "both", "exact_normalised"
    if in_concerns:
        return True, "concerns", "exact_normalised"
    if in_circumstances:
        return True, "circumstances", "exact_normalised"
    return False, "not_stated", "not_found"


SPAN_STOPWORDS = {
    "a", "an", "and", "as", "at", "be", "been", "by", "for", "from", "had",
    "has", "have", "he", "her", "his", "i", "in", "is", "it", "not", "of",
    "on", "or", "she", "that", "the", "their", "there", "they", "this", "to",
    "was", "were", "with", "would",
}


def recover_source_span(issue: dict[str, Any], report: pd.Series) -> str:
    proposed_span = clean_text(
        issue.get("evidence_quote") or issue.get("source_span")
    )
    has_ellipsis = "..." in proposed_span or "…" in proposed_span
    source = f"{clean_text(report.get('concerns'))} {clean_text(report.get('circumstances'))}"
    sentence_chunks = [
        clean_text(chunk)
        for chunk in re.split(r"(?<=[.!?])\s+|\n+", source)
        if 20 <= len(clean_text(chunk)) <= TEXT_LIMITS["evidence_quote"]
    ]
    adjacent_chunks = [
        f"{left} {right}"
        for left, right in zip(sentence_chunks, sentence_chunks[1:])
        if len(f"{left} {right}") <= TEXT_LIMITS["evidence_quote"]
    ]
    chunks = list(dict.fromkeys([*sentence_chunks, *adjacent_chunks]))
    if not chunks:
        return ""
    query = " ".join([proposed_span, clean_text(issue.get("canonical_issue"))])
    query_key = normalised_key(query)
    proposed_key = normalised_key(proposed_span)
    proposed_tokens = {
        token
        for token in proposed_key.split()
        if token not in SPAN_STOPWORDS and len(token) > 2
    }
    query_tokens = {
        token for token in query_key.split() if token not in SPAN_STOPWORDS and len(token) > 2
    }
    best_chunk = ""
    best_score = 0.0
    best_shared = 0
    recoverable_chunk = ""
    recoverable_score = 0.0
    proposed_words = proposed_key.split()
    proposed_length = len(proposed_words)
    for chunk in chunks:
        chunk_key = normalised_key(chunk)
        chunk_words = chunk_key.split()
        chunk_tokens = {
            token for token in chunk_words if token not in SPAN_STOPWORDS and len(token) > 2
        }
        shared = len(query_tokens & chunk_tokens)
        overlap = shared / max(1, min(len(query_tokens), len(chunk_tokens)))
        similarity = SequenceMatcher(None, query_key, chunk_key).ratio()
        score = 0.7 * overlap + 0.3 * similarity
        if score > best_score:
            best_chunk, best_score, best_shared = chunk, score, shared
        proposed_shared = len(proposed_tokens & chunk_tokens)
        proposed_overlap = proposed_shared / max(
            1, min(len(proposed_tokens), len(chunk_tokens))
        )
        if shared >= 3 and proposed_overlap >= 0.75:
            window_similarities = [
                SequenceMatcher(None, proposed_key, chunk_key).ratio()
            ]
            for window_length in range(
                max(1, proposed_length - 1),
                min(len(chunk_words), proposed_length + 2) + 1,
            ):
                window_similarities.extend(
                    SequenceMatcher(
                        None,
                        proposed_key,
                        " ".join(chunk_words[start : start + window_length]),
                    ).ratio()
                    for start in range(len(chunk_words) - window_length + 1)
                )
            proposed_similarity = max(window_similarities)
            matching_blocks = SequenceMatcher(
                None, proposed_words, chunk_words
            ).get_matching_blocks()
            longest_block = max((block.size for block in matching_blocks), default=0)
            contiguous_coverage = longest_block / max(1, proposed_length)
            if (
                proposed_similarity >= 0.9
                or (contiguous_coverage >= 0.55 and longest_block >= 8)
            ) and score > recoverable_score:
                recoverable_chunk, recoverable_score = chunk, score
    if has_ellipsis and best_shared >= 4 and best_score >= 0.38:
        return best_chunk
    return recoverable_chunk


def validate_issue(
    raw: Any,
    *,
    report: pd.Series,
    enums: dict[str, set[str]],
    canonical_max_words: int,
) -> tuple[dict[str, Any] | None, list[str]]:
    if not isinstance(raw, dict):
        return None, ["issue_not_object"]
    warnings: list[str] = []
    issue: dict[str, Any] = {}
    for field, limit in TEXT_LIMITS.items():
        issue[field] = clean_text(raw.get(field))[:limit]
    if not issue["canonical_issue"] or not issue["evidence_quote"]:
        return None, ["missing_issue_text"]
    issue["canonical_issue"] = trim_words(issue["canonical_issue"], canonical_max_words)
    for field in SCALAR_ENUM_FIELDS:
        value = clean_text(raw.get(field)).casefold().replace("-", "_").replace(" ", "_")
        if value not in enums[field]:
            warnings.append(f"invalid_{field}:{value or 'missing'}")
            value = "not_stated" if "not_stated" in enums[field] else "other_review"
        issue[field] = value
    for field in ARRAY_ENUM_FIELDS:
        raw_values = raw.get(field)
        if not isinstance(raw_values, list):
            raw_values = [raw_values] if clean_text(raw_values) else []
        values: list[str] = []
        for raw_value in raw_values:
            value = clean_text(raw_value).casefold().replace("-", "_").replace(" ", "_")
            if value not in enums[field]:
                warnings.append(f"invalid_{field}:{value or 'missing'}")
                continue
            if value not in values:
                values.append(value)
        if not values:
            fallback = "not_stated" if "not_stated" in enums[field] else "other_review"
            values = [fallback]
        issue[field] = values
    ambiguity_text = normalised_key(issue["canonical_issue"])
    if issue["failure_state"] == "other_review" and re.search(
        r"\b(?:ambiguous|unclear|not clearly defined)\b", ambiguity_text
    ):
        issue["failure_state"] = "ambiguous"
        warnings.append("failure_state_inferred_ambiguous")
    valid_evidence, evidence_section, evidence_status = locate_evidence_section(
        issue["evidence_quote"], report
    )
    if not valid_evidence:
        recovered_span = recover_source_span(issue, report)
        if recovered_span:
            issue["evidence_quote"] = recovered_span
            valid_evidence, evidence_section, _ = locate_evidence_section(
                recovered_span, report
            )
            evidence_status = "recovered_exact_sentence"
            warnings.append("evidence_quote_recovered")
    issue["evidence_section"] = evidence_section
    issue["evidence_valid"] = valid_evidence
    issue["evidence_status"] = evidence_status
    if not valid_evidence:
        warnings.append(f"evidence_quote_{evidence_status}")
    for field in SCALAR_ENUM_FIELDS:
        if issue[field] == "other_review":
            warnings.append(f"review_{field}:other_review")
    for field in ARRAY_ENUM_FIELDS:
        if "other_review" in issue[field]:
            warnings.append(f"review_{field}:other_review")
    if re.match(r"^(?:need|recommendation|requirement|risk)\s+(?:for|of)\b", issue["canonical_issue"], re.I):
        warnings.append("canonical_not_failure_framed")
    entity_key = normalised_key(issue["responsible_actor_text"])
    if len(entity_key.split()) >= 2 and entity_key in normalised_key(issue["canonical_issue"]):
        warnings.append("canonical_contains_responsible_actor")
    return issue, warnings


def deduplicate_issues(issues: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    output: list[dict[str, Any]] = []
    for issue in issues:
        key = normalised_key(issue.get("canonical_issue"))
        if not key:
            continue
        protected = set(re.findall(r"\b(?:category\s+\d+|level\s+\d+|\d+)\b", key))
        duplicate = False
        for existing in output:
            existing_key = normalised_key(existing.get("canonical_issue"))
            existing_protected = set(
                re.findall(r"\b(?:category\s+\d+|level\s+\d+|\d+)\b", existing_key)
            )
            same_identity = (
                issue.get("failure_state") == existing.get("failure_state")
                and issue.get("process_stage") == existing.get("process_stage")
                and issue.get("communication_direction")
                == existing.get("communication_direction")
                and protected == existing_protected
            )
            if key == existing_key or (
                same_identity
                and SequenceMatcher(None, key, existing_key).ratio() >= 0.94
            ):
                duplicate = True
                break
        if duplicate:
            continue
        output.append(issue)
    return output


def extract_report(
    report: pd.Series,
    *,
    args: argparse.Namespace,
    schema: dict[str, Any],
    enums: dict[str, set[str]],
) -> dict[str, Any]:
    source_text = build_source_text(report, args.max_source_chars)
    prompt = USER_PROMPT.format(
        max_issues=args.max_issues,
        canonical_max_words=args.canonical_max_words,
        enum_guide=compact_enum_guide(enums),
        source_text=source_text,
    )
    last_error = ""
    last_raw = ""
    warnings: list[str] = []
    best_validated: list[dict[str, Any]] = []
    best_score = (-1, -1, -1)
    completed_attempt = 0
    for attempt in range(1, args.retries + 2):
        try:
            payload, last_raw = ollama_json(
                host=args.ollama_host,
                model=args.model,
                system_prompt=SYSTEM_PROMPT,
                user_prompt=prompt,
                schema=schema,
                timeout=args.request_timeout,
                num_predict=args.extract_num_predict,
                num_ctx=args.ollama_num_ctx,
            )
            raw_issues = payload.get("issues")
            if not isinstance(raw_issues, list):
                raise ValueError("response did not contain an issues array")
            validated: list[dict[str, Any]] = []
            warnings = []
            for raw_issue in raw_issues[: args.max_issues]:
                issue, issue_warnings = validate_issue(
                    raw_issue,
                    report=report,
                    enums=enums,
                    canonical_max_words=args.canonical_max_words,
                )
                if issue:
                    issue["_validation_warnings"] = list(dict.fromkeys(issue_warnings))
                    warnings.extend(issue_warnings)
                    validated.append(issue)
            validated = deduplicate_issues(validated)
            valid_count = sum(bool(issue["evidence_valid"]) for issue in validated)
            invalid_count = len(validated) - valid_count
            score = (
                int(bool(validated) and invalid_count == 0),
                valid_count,
                -invalid_count,
            )
            if score > best_score:
                best_validated, best_score = validated, score
            completed_attempt = attempt
            needs_evidence_retry = invalid_count > 0
            needs_empty_retry = bool(clean_text(report.get("concerns"))) and not validated
            if (needs_evidence_retry or needs_empty_retry) and attempt <= args.retries:
                feedback: list[str] = []
                if needs_evidence_retry:
                    feedback.append(
                        "Some evidence quotes were not exact report text. Copy every quote verbatim."
                    )
                if needs_empty_retry:
                    feedback.append(
                        "The Concerns section is non-empty. If it proposes only a remedy, extract "
                        "the underlying missing safeguard and mark recommendation_only."
                    )
                prompt = f"{prompt}\n\nCorrection required: {' '.join(feedback)}"
                continue
            if (
                getattr(args, "continue_at_cap", True)
                and len(raw_issues) >= args.max_issues
                and getattr(args, "continuation_issues", 0) > 0
            ):
                existing = [issue["canonical_issue"] for issue in validated]
                continuation_prompt = (
                    f"Extract at most {args.continuation_issues} additional distinct issues from "
                    "the same report. Do not repeat any existing issue. Return the same schema.\n\n"
                    f"Existing canonical issues:\n{json.dumps(existing, indent=2)}\n\n"
                    f"Allowed values:\n{compact_enum_guide(enums)}\n\nReport text:\n{source_text}"
                )
                extra_payload, extra_raw = ollama_json(
                    host=args.ollama_host,
                    model=args.model,
                    system_prompt=SYSTEM_PROMPT,
                    user_prompt=continuation_prompt,
                    schema=schema,
                    timeout=args.request_timeout,
                    num_predict=args.extract_num_predict,
                    num_ctx=args.ollama_num_ctx,
                )
                for raw_issue in (extra_payload.get("issues") or [])[
                    : args.continuation_issues
                ]:
                    issue, issue_warnings = validate_issue(
                        raw_issue,
                        report=report,
                        enums=enums,
                        canonical_max_words=args.canonical_max_words,
                    )
                    if issue:
                        issue["_validation_warnings"] = list(
                            dict.fromkeys(issue_warnings)
                        )
                        warnings.extend(issue_warnings)
                        validated.append(issue)
                validated = deduplicate_issues(validated)
                last_raw = json.dumps(
                    {"initial": last_raw, "continuation": extra_raw},
                    ensure_ascii=False,
                )
            final_valid = sum(bool(issue["evidence_valid"]) for issue in validated)
            final_invalid = len(validated) - final_valid
            final_score = (
                int(bool(validated) and final_invalid == 0),
                final_valid,
                -final_invalid,
            )
            if final_score >= best_score:
                best_validated, best_score = validated, final_score
            break
        except Exception as exc:  # noqa: BLE001
            last_error = f"{type(exc).__name__}: {exc}"
    if best_validated or completed_attempt:
        warnings = [
            warning
            for issue in best_validated
            for warning in issue.get("_validation_warnings", [])
        ]
        return {
            "report_key": report["report_key"],
            "report_id": report["id"],
            "status": "success",
            "attempt": completed_attempt,
            "issues": best_validated,
            "warnings": list(dict.fromkeys(warnings)),
            "error": "",
            "raw_response": last_raw,
            "completed_at": datetime.now(timezone.utc).isoformat(),
        }
    return {
        "report_key": report["report_key"],
        "report_id": report["id"],
        "status": "failed",
        "attempt": args.retries + 1,
        "issues": [],
        "warnings": warnings,
        "error": last_error,
        "raw_response": last_raw,
        "completed_at": datetime.now(timezone.utc).isoformat(),
    }


def read_checkpoint(path: Path) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return latest
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(
                    f"Warning: ignored malformed checkpoint line {line_number} in {path}.",
                    file=sys.stderr,
                )
                continue
            key = clean_text(
                record.get("report_key") or record.get("report_identity")
            )
            if key:
                latest[key] = record
    return latest


def read_keyed_jsonl(path: Path, key: str) -> dict[str, dict[str, Any]]:
    latest: dict[str, dict[str, Any]] = {}
    if not path.exists():
        return latest
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                record = json.loads(line)
            except json.JSONDecodeError:
                print(
                    f"Warning: ignored malformed checkpoint line {line_number} in {path}.",
                    file=sys.stderr,
                )
                continue
            value = clean_text(record.get(key))
            if value:
                latest[value] = record
    return latest


def append_checkpoint(path: Path, record: dict[str, Any]) -> None:
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")
        handle.flush()


@contextmanager
def extraction_lock(run_dir: Path) -> Iterable[None]:
    lock_path = run_dir / ".extraction.lock"
    descriptor: int | None = None
    for _ in range(2):
        try:
            descriptor = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
            break
        except FileExistsError as exc:
            details = clean_text(lock_path.read_text(encoding="utf-8"))
            try:
                pid = int((json.loads(details) or {}).get("pid"))
                os.kill(pid, 0)
            except (ValueError, TypeError, json.JSONDecodeError, ProcessLookupError):
                lock_path.unlink(missing_ok=True)
                continue
            except PermissionError:
                pass
            raise RuntimeError(
                f"Another extractor already owns {lock_path}. {details}"
            ) from exc
    if descriptor is None:
        raise RuntimeError(f"Could not acquire extraction lock: {lock_path}")
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8") as handle:
            handle.write(
                json.dumps(
                    {
                        "pid": os.getpid(),
                        "started_at": datetime.now(timezone.utc).isoformat(),
                    }
                )
            )
        yield
    finally:
        lock_path.unlink(missing_ok=True)


def records_to_frames(
    reports: pd.DataFrame, records: dict[str, dict[str, Any]]
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    report_columns = ["report_key", "id", "url", "date", "coroner", "area", "receiver"]
    report_frame = reports[report_columns].rename(
        columns={"id": "report_id", "url": "report_url", "date": "report_date"}
    )
    occurrence_rows: list[dict[str, Any]] = []
    failure_rows: list[dict[str, Any]] = []
    report_lookup = reports.set_index("report_key", drop=False)
    for report_key, record in records.items():
        if report_key not in report_lookup.index:
            continue
        report = report_lookup.loc[report_key]
        if record.get("status") != "success":
            failure_rows.append(
                {
                    "report_key": report_key,
                    "report_id": report["id"],
                    "report_url": report["url"],
                    "error": clean_text(record.get("error")),
                }
            )
            continue
        for issue in record.get("issues") or []:
            issue = dict(issue)
            issue_warnings = list(issue.get("_validation_warnings") or [])
            if not issue.get("evidence_valid"):
                recovered_span = recover_source_span(issue, report)
                if recovered_span:
                    issue["evidence_quote"] = recovered_span
                    valid, section, _ = locate_evidence_section(recovered_span, report)
                    issue["evidence_valid"] = valid
                    issue["evidence_section"] = section
                    issue["evidence_status"] = "recovered_exact_sentence"
                    issue_warnings = [
                        warning
                        for warning in issue_warnings
                        if warning != "evidence_quote_not_found"
                    ]
                    issue_warnings.append("evidence_quote_recovered")
            public_issue = {
                key: value for key, value in issue.items() if key != "_validation_warnings"
            }
            for field in ARRAY_ENUM_FIELDS:
                public_issue[field] = "|".join(public_issue.get(field) or [])
            occurrence_rows.append(
                {
                    "issue_id": (
                        f"iss_{stable_hash(report_key, issue.get('evidence_quote'), issue.get('canonical_issue'))}"
                    ),
                    "pipeline_version": PIPELINE_VERSION,
                    "schema_version": SCHEMA_VERSION,
                    "report_key": report_key,
                    **public_issue,
                    "extraction_warnings": " | ".join(dict.fromkeys(issue_warnings)),
                }
            )
    return report_frame, pd.DataFrame(occurrence_rows), pd.DataFrame(failure_rows)


def extraction_quality_metrics(
    reports: pd.DataFrame, occurrences: pd.DataFrame, *, max_issues: int
) -> dict[str, Any]:
    report_counts = (
        occurrences.groupby("report_key").size()
        if not occurrences.empty
        else pd.Series(dtype=int)
    )
    weak_values: dict[str, dict[str, Any]] = {}
    for field in ENUM_FIELDS:
        if field not in occurrences:
            continue
        if field in ARRAY_ENUM_FIELDS:
            weak_count = int(
                occurrences[field]
                .map(clean_text)
                .str.split("|")
                .map(lambda values: "other_review" in values)
                .sum()
            )
        else:
            weak_count = int((occurrences[field] == "other_review").sum())
        weak_values[field] = {
            "count": weak_count,
            "percent": round(100.0 * weak_count / len(occurrences), 2)
            if len(occurrences)
            else 0.0,
        }
    canonical = occurrences.get("canonical_issue", pd.Series(dtype=str)).map(clean_text)
    return {
        "reports_in_scope": int(len(reports)),
        "reports_with_issues": int(report_counts.size),
        "reports_with_no_issues": int(len(reports) - report_counts.size),
        "reports_at_issue_cap": int((report_counts >= max_issues).sum()),
        "issue_occurrences": int(len(occurrences)),
        "mean_issues_per_report": round(float(len(occurrences) / len(reports)), 3)
        if len(reports)
        else 0.0,
        "valid_evidence_quotes": int(
            occurrences.get("evidence_valid", pd.Series(dtype=bool)).sum()
        ),
        "invalid_evidence_quotes": int(
            (~occurrences.get("evidence_valid", pd.Series(dtype=bool))).sum()
        ),
        "canonical_not_failure_framed": int(
            canonical.str.match(
                r"^(?:need|recommendation|requirement|risk)\s+(?:for|of)\b",
                case=False,
            ).sum()
        ),
        "weak_controlled_values": weak_values,
    }


def run_extraction_stage(
    *, run_dir: Path, reports: pd.DataFrame, args: argparse.Namespace, schema: dict[str, Any]
) -> Path:
    with extraction_lock(run_dir):
        ensure_ollama(args.ollama_host, args.model)
        checkpoint_path = run_dir / "01_extraction_checkpoint.jsonl"
        records = read_checkpoint(checkpoint_path)
        enums = schema_enums(schema)
        pending: list[pd.Series] = []
        for _, report in reports.iterrows():
            previous = records.get(report["report_key"])
            if previous and previous.get("status") == "success":
                continue
            if previous and not args.retry_failures:
                continue
            pending.append(report)
        print(
            f"Extraction: {len(records)} checkpointed; {len(pending)} reports pending."
        )
        if args.extraction_workers <= 1:
            for report in tqdm(
                pending, desc="Extracting structured issues", unit="report"
            ):
                record = extract_report(report, args=args, schema=schema, enums=enums)
                append_checkpoint(checkpoint_path, record)
                records[report["report_key"]] = record
        else:
            with ThreadPoolExecutor(max_workers=args.extraction_workers) as executor:
                futures = {
                    executor.submit(
                        extract_report, report, args=args, schema=schema, enums=enums
                    ): report["report_key"]
                    for report in pending
                }
                for future in tqdm(
                    as_completed(futures),
                    total=len(futures),
                    desc="Extracting structured issues",
                    unit="report",
                ):
                    record = future.result()
                    append_checkpoint(checkpoint_path, record)
                    records[futures[future]] = record
    report_frame, occurrence_frame, failures = records_to_frames(reports, records)
    report_frame.to_csv(run_dir / "01_reports.csv", index=False)
    occurrence_path = run_dir / "01_issue_occurrences.csv"
    occurrence_frame.to_csv(occurrence_path, index=False)
    failures.to_csv(run_dir / "01_extraction_failures.csv", index=False)
    scoped_records = [
        records[report_key]
        for report_key in reports["report_key"]
        if report_key in records
    ]
    validation = {
        **extraction_quality_metrics(reports, occurrence_frame, max_issues=args.max_issues),
        "reports_succeeded": int(
            sum(row.get("status") == "success" for row in scoped_records)
        ),
        "reports_failed": int(
            sum(row.get("status") != "success" for row in scoped_records)
        ),
    }
    (run_dir / "01_extraction_metrics.json").write_text(
        json.dumps(validation, indent=2), encoding="utf-8"
    )
    return occurrence_path


def embedding_text(row: pd.Series, canonical_field: str = "canonical_issue") -> str:
    parts = [clean_text(row.get(canonical_field))]
    facets = (
        ("Failure", row.get("failure_state")),
        ("Stage", row.get("process_stage")),
        ("Themes", row.get("issue_themes")),
        ("Communication", row.get("communication_direction")),
    )
    parts.extend(
        f"{label}: {clean_text(value).replace('_', ' ').replace('|', ', ')}"
        for label, value in facets
        if clean_text(value)
    )
    return ". ".join(parts)


def encode_embeddings(
    texts: list[str],
    model_name: str,
    *,
    allow_model_download: bool,
    batch_size: int = 48,
) -> np.ndarray:
    try:
        model = SentenceTransformer(
            model_name,
            local_files_only=not allow_model_download,
        )
    except Exception as exc:  # noqa: BLE001
        mode = "download enabled" if allow_model_download else "offline cache only"
        raise RuntimeError(
            f"Could not load embedding model '{model_name}' ({mode}). "
            "Cache the model first or pass --allow-model-download."
        ) from exc
    vectors = model.encode(
        texts,
        batch_size=batch_size,
        normalize_embeddings=True,
        convert_to_numpy=True,
        show_progress_bar=True,
    )
    return np.asarray(vectors, dtype=np.float32)


def blend_embedding_views(
    original: np.ndarray, normalized: np.ndarray, original_weight: float
) -> np.ndarray:
    if original.shape != normalized.shape:
        raise ValueError("Original and normalized embedding arrays must have equal shapes")
    blended = original_weight * original + (1.0 - original_weight) * normalized
    norms = np.linalg.norm(blended, axis=1, keepdims=True)
    return np.asarray(blended / np.maximum(norms, 1e-12), dtype=np.float32)


def mutual_neighbor_components(
    embeddings: np.ndarray,
    report_identities: list[str],
    *,
    top_k: int,
    threshold: float,
) -> tuple[list[list[int]], list[tuple[int, int, float]]]:
    count = len(embeddings)
    if count < 2:
        return [], []
    neighbours = min(count, max(2, top_k + 1))
    model = NearestNeighbors(n_neighbors=neighbours, metric="cosine", n_jobs=-1)
    model.fit(embeddings)
    distances, indices = model.kneighbors(embeddings)
    candidate_sets: list[dict[int, float]] = []
    for i in range(count):
        candidates: dict[int, float] = {}
        for distance, j in zip(distances[i], indices[i], strict=False):
            j = int(j)
            similarity = 1.0 - float(distance)
            if j == i or report_identities[j] == report_identities[i] or similarity < threshold:
                continue
            candidates[j] = similarity
        candidate_sets.append(candidates)
    adjacency: list[set[int]] = [set() for _ in range(count)]
    edges: list[tuple[int, int, float]] = []
    for i, candidates in enumerate(candidate_sets):
        for j, similarity in candidates.items():
            if i >= j or i not in candidate_sets[j]:
                continue
            adjacency[i].add(j)
            adjacency[j].add(i)
            edges.append((i, j, min(similarity, candidate_sets[j][i])))
    visited: set[int] = set()
    components: list[list[int]] = []
    for start in range(count):
        if start in visited or not adjacency[start]:
            continue
        stack = [start]
        members: list[int] = []
        visited.add(start)
        while stack:
            node = stack.pop()
            members.append(node)
            for neighbour in adjacency[node]:
                if neighbour not in visited:
                    visited.add(neighbour)
                    stack.append(neighbour)
        components.append(sorted(members))
    return components, edges


def split_component(
    members: list[int], embeddings: np.ndarray, split_similarity: float
) -> list[list[int]]:
    if len(members) <= 2:
        return [members]
    local = embeddings[members]
    clusterer = AgglomerativeClustering(
        n_clusters=None,
        metric="cosine",
        linkage="average",
        distance_threshold=max(0.0, 1.0 - split_similarity),
    )
    labels = clusterer.fit_predict(local)
    output: list[list[int]] = []
    for label in sorted(set(int(value) for value in labels)):
        output.append([members[i] for i in np.where(labels == label)[0].tolist()])
    return output


def apply_centroid_filter(
    members: list[int], embeddings: np.ndarray, minimum: float
) -> tuple[list[int], dict[int, float]]:
    retained = list(members)
    scores: dict[int, float] = {}
    # Recalculate after rejection so a weak member cannot continue to pull the
    # centroid away from the coherent core.
    for _ in range(3):
        if not retained:
            return [], scores
        local = embeddings[retained]
        centroid = local.mean(axis=0)
        norm = float(np.linalg.norm(centroid))
        if norm == 0:
            return [], scores
        centroid /= norm
        similarities = local @ centroid
        scores.update(
            {
                member: float(score)
                for member, score in zip(retained, similarities, strict=False)
            }
        )
        updated = [member for member in retained if scores[member] >= minimum]
        if updated == retained:
            break
        retained = updated
    return retained, scores


def dominant_value(frame: pd.DataFrame, column: str, fallback: str = "unclear") -> str:
    values = [clean_text(value) for value in frame.get(column, pd.Series(dtype=str)) if clean_text(value)]
    return Counter(values).most_common(1)[0][0] if values else fallback


def dominant_array_value(
    frame: pd.DataFrame, column: str, fallback: str = "other_review"
) -> str:
    values = [
        token
        for value in frame.get(column, pd.Series(dtype=str))
        for token in clean_text(value).split("|")
        if token
    ]
    return Counter(values).most_common(1)[0][0] if values else fallback


def safe_dates(values: pd.Series) -> tuple[str, str]:
    parsed = pd.to_datetime(values, errors="coerce")
    if not parsed.notna().any():
        return "", ""
    return parsed.min().strftime("%Y-%m-%d"), parsed.max().strftime("%Y-%m-%d")


def label_subissue(
    frame: pd.DataFrame,
    embeddings: np.ndarray,
    member_indices: list[int],
    *,
    args: argparse.Namespace,
    allow_llm: bool = True,
) -> tuple[str, str, int]:
    local = embeddings[member_indices]
    centroid = local.mean(axis=0)
    centroid /= max(float(np.linalg.norm(centroid)), 1e-12)
    similarities = local @ centroid
    order = np.argsort(-similarities)
    representative_index = member_indices[int(order[0])]
    representative = clean_text(frame.iloc[representative_index]["canonical_issue"])
    if not args.label_subissues or not allow_llm:
        return representative.rstrip("."), "", representative_index
    examples = [
        clean_text(frame.iloc[member_indices[int(local_index)]]["canonical_issue"])
        for local_index in order[: min(10, len(order))]
    ]
    try:
        payload, _ = ollama_json(
            host=args.ollama_host,
            model=args.model,
            system_prompt=LABEL_SYSTEM_PROMPT,
            user_prompt=LABEL_USER_PROMPT.format(examples=json.dumps(examples, indent=2)),
            schema=None,
            timeout=args.request_timeout,
            num_predict=180,
            num_ctx=args.ollama_num_ctx,
        )
        label = trim_words(payload.get("label"), 8).strip(" .") or representative.rstrip(".")
        description = trim_words(payload.get("description"), 32)
        return label, description, representative_index
    except Exception as exc:  # noqa: BLE001
        return representative.rstrip("."), f"Automatic label failed: {exc}", representative_index


def recurrence_status(report_count: int, minimum: int) -> str:
    if report_count >= minimum:
        return "recurring"
    if report_count == 2:
        return "emerging"
    return "isolated"


def build_index(
    occurrences: pd.DataFrame,
    *,
    run_dir: Path,
    args: argparse.Namespace,
) -> dict[str, Any]:
    required = {"issue_id", "report_key", "canonical_issue"}
    missing = required - set(occurrences.columns)
    if missing:
        raise ValueError(f"Occurrence CSV is missing columns: {sorted(missing)}")
    occurrences = occurrences.copy().fillna("").reset_index(drop=True)
    reports_path = run_dir / "01_reports.csv"
    if not reports_path.exists():
        raise FileNotFoundError(f"Report catalogue not found: {reports_path}")
    reports = pd.read_csv(reports_path).fillna("")
    report_columns = ["report_key", "report_id", "report_url", "report_date"]
    occurrences = occurrences.merge(
        reports[report_columns],
        on="report_key",
        how="left",
        validate="many_to_one",
    )
    requested_mode = getattr(args, "embedding_mode", "auto")
    original_weight = float(getattr(args, "original_view_weight", 0.44))
    has_original_view = (
        "canonical_issue_original" in occurrences.columns
        and occurrences["canonical_issue_original"].map(clean_text).str.len().gt(0).all()
    )
    if requested_mode == "dual" and not has_original_view:
        raise ValueError(
            "--embedding-mode dual requires a non-empty canonical_issue_original column"
        )
    effective_mode = (
        "dual" if requested_mode == "dual" or (requested_mode == "auto" and has_original_view) else "single"
    )
    occurrences["embedding_text"] = occurrences.apply(embedding_text, axis=1)
    if effective_mode == "dual":
        occurrences["embedding_text_original"] = occurrences.apply(
            embedding_text, axis=1, canonical_field="canonical_issue_original"
        )
    # Downstream tuning and repair must use the exact rows and representation
    # aligned to the cached vectors, not the pre-normalization extraction CSV.
    occurrences.to_csv(run_dir / "02_embedding_occurrences.csv", index=False)
    embeddings_path = run_dir / "02_issue_embeddings.npy"
    normalized_embeddings_path = run_dir / "02_issue_embeddings_normalized.npy"
    original_embeddings_path = run_dir / "02_issue_embeddings_original.npy"
    embeddings_meta_path = run_dir / "02_embeddings_meta.json"
    embedding_fingerprint = hashlib.sha256(
        "\n".join(
            f"{issue_id}\t{text}"
            for issue_id, text in zip(
                occurrences["issue_id"], occurrences["embedding_text"], strict=False
            )
        ).encode("utf-8")
    ).hexdigest()
    if effective_mode == "single":
        # Retain the historical metadata shape so existing single-view caches remain reusable.
        expected_embedding_meta = {
            "model": args.embedding_model,
            "row_count": int(len(occurrences)),
            "input_sha256": embedding_fingerprint,
        }
    else:
        original_fingerprint = hashlib.sha256(
            "\n".join(
                f"{issue_id}\t{text}"
                for issue_id, text in zip(
                    occurrences["issue_id"],
                    occurrences["embedding_text_original"],
                    strict=False,
                )
            ).encode("utf-8")
        ).hexdigest()
        expected_embedding_meta = {
            "model": args.embedding_model,
            "row_count": int(len(occurrences)),
            "embedding_mode": "dual",
            "original_view_weight": original_weight,
            "normalized_input_sha256": embedding_fingerprint,
            "original_input_sha256": original_fingerprint,
        }
    cached_embedding_meta: dict[str, Any] = {}
    if embeddings_meta_path.exists():
        cached_embedding_meta = json.loads(embeddings_meta_path.read_text(encoding="utf-8"))
    if embeddings_path.exists() and cached_embedding_meta == expected_embedding_meta:
        embeddings = np.load(embeddings_path)
        if len(embeddings) != len(occurrences):
            raise ValueError("Cached embeddings do not match the occurrence row count.")
        print(f"Reusing {len(embeddings):,} cached embeddings.")
    else:
        if effective_mode == "single":
            embeddings = encode_embeddings(
                occurrences["embedding_text"].tolist(),
                args.embedding_model,
                allow_model_download=args.allow_model_download,
            )
        else:
            cache_identity_matches = (
                cached_embedding_meta.get("model") == args.embedding_model
                and cached_embedding_meta.get("row_count") == len(occurrences)
                and cached_embedding_meta.get("embedding_mode") == "dual"
            )
            reuse_original = (
                cache_identity_matches
                and cached_embedding_meta.get("original_input_sha256")
                == original_fingerprint
                and original_embeddings_path.exists()
            )
            reuse_normalized = (
                cache_identity_matches
                and cached_embedding_meta.get("normalized_input_sha256")
                == embedding_fingerprint
                and normalized_embeddings_path.exists()
            )
            if reuse_original:
                original_embeddings = np.load(original_embeddings_path)
                if len(original_embeddings) != len(occurrences):
                    raise ValueError(
                        "Cached original-view embeddings do not match the occurrence row count."
                    )
                print(f"Reusing {len(original_embeddings):,} original-view embeddings.")
            else:
                original_embeddings = encode_embeddings(
                    occurrences["embedding_text_original"].tolist(),
                    args.embedding_model,
                    allow_model_download=args.allow_model_download,
                )
                np.save(original_embeddings_path, original_embeddings)
            if reuse_normalized:
                normalized_embeddings = np.load(normalized_embeddings_path)
                if len(normalized_embeddings) != len(occurrences):
                    raise ValueError(
                        "Cached normalized-view embeddings do not match the occurrence row count."
                    )
                print(f"Reusing {len(normalized_embeddings):,} normalized-view embeddings.")
            else:
                normalized_embeddings = encode_embeddings(
                    occurrences["embedding_text"].tolist(),
                    args.embedding_model,
                    allow_model_download=args.allow_model_download,
                )
                np.save(normalized_embeddings_path, normalized_embeddings)
            embeddings = blend_embedding_views(
                original_embeddings, normalized_embeddings, original_weight
            )
        np.save(embeddings_path, embeddings)
        embeddings_meta_path.write_text(
            json.dumps(expected_embedding_meta, indent=2), encoding="utf-8"
        )
    components, edges = mutual_neighbor_components(
        embeddings,
        occurrences["report_key"].map(clean_text).tolist(),
        top_k=args.top_k,
        threshold=args.edge_similarity,
    )
    edge_frame = pd.DataFrame(edges, columns=["left_index", "right_index", "cosine_similarity"])
    if not edge_frame.empty:
        edge_frame["left_issue_id"] = edge_frame["left_index"].map(occurrences["issue_id"])
        edge_frame["right_issue_id"] = edge_frame["right_index"].map(occurrences["issue_id"])
    edge_frame.to_csv(run_dir / "02_mutual_neighbour_edges.csv", index=False)

    candidates: list[list[int]] = []
    for component in tqdm(components, desc="Splitting chained components", unit="component"):
        candidates.extend(split_component(component, embeddings, args.split_similarity))

    final_candidates: list[tuple[list[int], dict[int, float]]] = []
    for members in candidates:
        filtered, scores = apply_centroid_filter(members, embeddings, args.min_centroid_similarity)
        if filtered:
            final_candidates.append((filtered, scores))
    final_candidates.sort(
        key=lambda item: (
            -occurrences.iloc[item[0]]["report_key"].nunique(),
            -len(item[0]),
            min(occurrences.iloc[item[0]]["issue_id"]),
        )
    )

    if args.label_subissues and final_candidates:
        ensure_ollama(args.ollama_host, args.model)
    assignment_rows: list[dict[str, Any]] = []
    subissue_rows: list[dict[str, Any]] = []
    parent_domains: set[str] = set()
    label_checkpoint_path = run_dir / "03_label_checkpoint.jsonl"
    label_records = read_keyed_jsonl(label_checkpoint_path, "subissue_id")
    for members, scores in tqdm(final_candidates, desc="Building sub-issues", unit="sub-issue"):
        subset = occurrences.iloc[members]
        report_count = int(subset["report_key"].nunique())
        status = recurrence_status(report_count, args.min_recurring_reports)
        member_ids = sorted(subset["issue_id"].map(clean_text).tolist())
        subissue_id = f"sub_{stable_hash(*member_ids)}"
        label, description, representative_index = label_subissue(
            occurrences,
            embeddings,
            members,
            args=args,
            allow_llm=False,
        )
        if args.label_subissues and status == "recurring":
            cached_label = label_records.get(subissue_id)
            if cached_label:
                label = clean_text(cached_label.get("label")) or label
                description = clean_text(cached_label.get("description"))
            else:
                label, description, representative_index = label_subissue(
                    occurrences,
                    embeddings,
                    members,
                    args=args,
                    allow_llm=True,
                )
                label_record = {
                    "subissue_id": subissue_id,
                    "label": label,
                    "description": description,
                    "completed_at": datetime.now(timezone.utc).isoformat(),
                }
                append_checkpoint(label_checkpoint_path, label_record)
                label_records[subissue_id] = label_record
        domain = dominant_array_value(subset, "issue_themes")
        parent_domains.add(domain)
        first_date, last_date = safe_dates(subset["report_date"])
        subissue_rows.append(
            {
                "subissue_id": subissue_id,
                "parent_group_id": f"grp_{domain}",
                "label": label,
                "description": description,
                "recurrence_status": status,
                "report_count": report_count,
                "issue_count": int(len(subset)),
                "first_date": first_date,
                "last_date": last_date,
                "representative_issue_id": occurrences.iloc[representative_index]["issue_id"],
                "median_centroid_similarity": float(np.median([scores[index] for index in members])),
                "failure_mode": dominant_value(subset, "failure_state"),
                "subject_domain": domain,
                "process_stage": dominant_value(subset, "process_stage"),
                "system_mechanism": "",
                "setting": dominant_array_value(subset, "service_contexts"),
                "sample_issues": " | ".join(subset["canonical_issue"].drop_duplicates().head(3)),
            }
        )
        for index in members:
            assignment_rows.append(
                {
                    "issue_id": occurrences.iloc[index]["issue_id"],
                    "subissue_id": subissue_id,
                    "assignment_similarity": scores[index],
                    "recurrence_status": status,
                }
            )

    assignments = pd.DataFrame(
        assignment_rows,
        columns=["issue_id", "subissue_id", "assignment_similarity", "recurrence_status"],
    )
    subissues = pd.DataFrame(subissue_rows)
    assignments.to_csv(run_dir / "03_issue_assignments.csv", index=False)
    subissues.to_csv(run_dir / "03_subissues.csv", index=False)
    parents = pd.DataFrame(
        [
            {
                "parent_group_id": f"grp_{domain}",
                "label": THEME_LABELS.get(domain, domain.replace("_", " ").title()),
                "subject_domain": domain,
            }
            for domain in sorted(parent_domains)
        ]
    )
    parents.to_csv(run_dir / "03_parent_groups.csv", index=False)

    joined = occurrences.merge(assignments, on="issue_id", how="left")
    joined["recurrence_status"] = joined["recurrence_status"].fillna("ungrouped")
    joined.to_csv(run_dir / "03_occurrences_indexed.csv", index=False)
    assigned = joined[joined["subissue_id"].fillna("").str.len() > 0].copy()
    if assigned.empty:
        presence = pd.DataFrame(
            columns=[
                "subissue_id", "report_key", "report_id", "report_url", "report_date",
                "representative_occurrence_id", "occurrence_count", "max_assignment_similarity",
                "recurrence_status",
            ]
        )
    else:
        assigned = assigned.sort_values("assignment_similarity", ascending=False)
        presence = (
            assigned.groupby(["subissue_id", "report_key"], as_index=False)
            .agg(
                report_id=("report_id", "first"),
                report_url=("report_url", "first"),
                report_date=("report_date", "first"),
                representative_occurrence_id=("issue_id", "first"),
                occurrence_count=("issue_id", "size"),
                max_assignment_similarity=("assignment_similarity", "max"),
                recurrence_status=("recurrence_status", "first"),
            )
        )
    presence.to_csv(run_dir / "03_issue_report_presence.csv", index=False)
    if presence.empty:
        timeseries = pd.DataFrame(columns=["subissue_id", "period", "report_count"])
    else:
        dated = presence.copy()
        dated["period"] = pd.to_datetime(dated["report_date"], errors="coerce").dt.to_period("M").astype(str)
        dated = dated[dated["period"] != "NaT"]
        timeseries = (
            dated.groupby(["subissue_id", "period"], as_index=False)["report_key"]
            .nunique()
            .rename(columns={"report_key": "report_count"})
        )
    timeseries.to_csv(run_dir / "03_subissue_timeseries_month.csv", index=False)

    recurring_ids = set(
        subissues.loc[subissues.get("recurrence_status", pd.Series(dtype=str)) == "recurring", "subissue_id"]
    ) if not subissues.empty else set()
    recurring_occurrences = assignments[assignments["subissue_id"].isin(recurring_ids)] if recurring_ids else assignments.iloc[0:0]
    metrics = {
        "embedding_mode": effective_mode,
        "original_view_weight": original_weight if effective_mode == "dual" else None,
        "total_issue_occurrences": int(len(occurrences)),
        "mutual_neighbour_edges": int(len(edges)),
        "initial_components": int(len(components)),
        "candidate_subissues": int(len(subissues)),
        "recurring_subissues": int(len(recurring_ids)),
        "emerging_subissues": int((subissues.get("recurrence_status", pd.Series(dtype=str)) == "emerging").sum()),
        "isolated_subissues": int((subissues.get("recurrence_status", pd.Series(dtype=str)) == "isolated").sum()),
        "recurring_issue_occurrences": int(len(recurring_occurrences)),
        "reports_with_recurring_issue": int(
            presence[presence["subissue_id"].isin(recurring_ids)]["report_key"].nunique()
        ) if recurring_ids else 0,
        "ungrouped_issue_occurrences": int((joined["recurrence_status"] == "ungrouped").sum()),
    }
    (run_dir / "04_index_metrics.json").write_text(json.dumps(metrics, indent=2), encoding="utf-8")
    return metrics


def resolve_run_dir(args: argparse.Namespace) -> Path:
    if args.run_dir:
        path = Path(args.run_dir).expanduser().resolve()
    else:
        timestamp = datetime.now(timezone.utc).strftime("%Y%m%d_%H%M%S")
        path = (Path(args.output_dir) / f"run_{timestamp}").resolve()
    path.mkdir(parents=True, exist_ok=True)
    return path


def write_manifest(
    run_dir: Path,
    *,
    args: argparse.Namespace,
    schema: dict[str, Any],
    metrics: dict[str, Any] | None = None,
) -> None:
    schema_bytes = json.dumps(schema, sort_keys=True).encode("utf-8")
    config = Config(
        pipeline_version=PIPELINE_VERSION,
        schema_version=clean_text(schema.get("$id")),
        input_csv=str(Path(args.input_csv).resolve()),
        run_dir=str(run_dir),
        stage=args.stage,
        subset_size=args.subset_size,
        random_seed=args.random_seed,
        ollama_host=args.ollama_host,
        model=args.model,
        embedding_model=args.embedding_model,
        embedding_mode=args.embedding_mode,
        original_view_weight=args.original_view_weight,
        allow_model_download=args.allow_model_download,
        max_issues=args.max_issues,
        continuation_issues=args.continuation_issues,
        continue_at_cap=args.continue_at_cap,
        canonical_max_words=args.canonical_max_words,
        max_source_chars=args.max_source_chars,
        retries=args.retries,
        extract_num_predict=args.extract_num_predict,
        extraction_workers=args.extraction_workers,
        ollama_num_ctx=args.ollama_num_ctx,
        top_k=args.top_k,
        edge_similarity=args.edge_similarity,
        split_similarity=args.split_similarity,
        min_centroid_similarity=args.min_centroid_similarity,
        min_recurring_reports=args.min_recurring_reports,
        label_subissues=args.label_subissues,
    )
    manifest = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "config": asdict(config),
        "schema_sha256": hashlib.sha256(schema_bytes).hexdigest(),
        "prompt_sha256": hashlib.sha256((SYSTEM_PROMPT + USER_PROMPT).encode("utf-8")).hexdigest(),
        "metrics": metrics or {},
    }
    (run_dir / "manifest.json").write_text(json.dumps(manifest, indent=2), encoding="utf-8")


def main() -> None:
    args = parse_args()
    if args.extraction_workers < 1:
        raise ValueError("--extraction-workers must be at least 1")
    if args.ollama_num_ctx < 8192:
        raise ValueError("--ollama-num-ctx must be at least 8192")
    if not 0.0 <= args.edge_similarity <= 1.0:
        raise ValueError("--edge-similarity must be between 0 and 1")
    if not 0.0 <= args.split_similarity <= 1.0:
        raise ValueError("--split-similarity must be between 0 and 1")
    if not 0.0 <= args.original_view_weight <= 1.0:
        raise ValueError("--original-view-weight must be between 0 and 1")
    if args.min_recurring_reports < 2:
        raise ValueError("--min-recurring-reports must be at least 2")
    schema = load_schema()
    run_dir = resolve_run_dir(args)
    print(f"Run directory: {run_dir}", flush=True)
    write_manifest(run_dir, args=args, schema=schema)
    occurrence_path: Path
    if args.stage in {"extract", "all"}:
        reports = load_reports(Path(args.input_csv), args.subset_size, args.random_seed)
        occurrence_path = run_extraction_stage(
            run_dir=run_dir, reports=reports, args=args, schema=schema
        )
        if args.stage == "extract":
            print(f"Extraction complete: {occurrence_path}")
            return
    else:
        occurrence_path = Path(args.occurrences_csv or run_dir / "01_issue_occurrences.csv")
    if not occurrence_path.exists():
        raise FileNotFoundError(f"Occurrence CSV not found: {occurrence_path}")
    occurrences = pd.read_csv(occurrence_path)
    if occurrences.empty:
        raise RuntimeError("No issue occurrences are available for indexing.")
    metrics = build_index(occurrences, run_dir=run_dir, args=args)
    write_manifest(run_dir, args=args, schema=schema, metrics=metrics)
    print("\nIssue index complete.")
    print(f"Run directory: {run_dir}")
    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print("Interrupted. Rerun with --run-dir to resume from the checkpoint.", file=sys.stderr)
        raise SystemExit(130)
