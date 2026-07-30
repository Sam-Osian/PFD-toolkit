#!/usr/bin/env python3
"""Validate manual thread annotations and write the feasibility evidence pack."""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from collections import Counter
from pathlib import Path
from typing import Any

import pandas as pd


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def comparable_text(value: Any) -> str:
    text = unicodedata.normalize("NFKC", clean_text(value)).casefold()
    text = text.translate(
        str.maketrans(
            {
                "“": '"',
                "”": '"',
                "‘": "'",
                "’": "'",
                "–": "-",
                "—": "-",
            }
        )
    )
    text = re.sub(r"[^a-z0-9]+", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def markdown_table(frame: pd.DataFrame) -> str:
    if frame.empty:
        return "_None._"
    columns = [str(column) for column in frame.columns]
    rows = [
        "| " + " | ".join(columns) + " |",
        "|" + "|".join("---" for _ in columns) + "|",
    ]
    for record in frame.fillna("").astype(str).to_dict(orient="records"):
        values = [
            record[column].replace("|", "\\|").replace("\n", " ")
            for column in columns
        ]
        rows.append("| " + " | ".join(values) + " |")
    return "\n".join(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("artifacts/prison_pfd_feasibility"),
    )
    parser.add_argument("--reports-csv", type=Path, default=Path("all_reports.csv"))
    parser.add_argument(
        "--annotations",
        type=Path,
        default=None,
        help="Defaults to <run-dir>/manual_thread_annotations.json.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    annotations_path = (
        args.annotations
        if args.annotations is not None
        else args.run_dir / "manual_thread_annotations.json"
    )
    annotations_payload = json.loads(annotations_path.read_text(encoding="utf-8"))
    reviews = pd.DataFrame(annotations_payload["reviews"])
    template = pd.read_csv(args.run_dir / "02_thread_review.csv")
    source_ledger = pd.read_csv(args.run_dir / "00_source_ledger.csv")
    document_ledger = pd.read_csv(args.run_dir / "01_document_ledger.csv")
    reports = pd.read_csv(args.reports_csv, low_memory=False)

    key_columns = ["case_id", "thread_id"]
    expected_keys = set(map(tuple, template[key_columns].itertuples(index=False, name=None)))
    actual_keys = set(map(tuple, reviews[key_columns].itertuples(index=False, name=None)))
    if expected_keys != actual_keys:
        missing = sorted(expected_keys - actual_keys)
        extra = sorted(actual_keys - expected_keys)
        raise ValueError(f"Annotation coverage mismatch; missing={missing}, extra={extra}")
    if reviews.duplicated(key_columns).any():
        raise ValueError("Manual annotations contain duplicate case/thread rows.")

    report_concerns = (
        source_ledger[["case_id", "report_url"]]
        .merge(
            reports[["url", "concerns"]].drop_duplicates("url"),
            left_on="report_url",
            right_on="url",
            how="left",
        )
        .set_index("case_id")["concerns"]
        .to_dict()
    )
    evidence_failures: list[str] = []
    for row in reviews[reviews["supported"]].to_dict(orient="records"):
        quote = comparable_text(row["concern_evidence_quote"])
        source = comparable_text(report_concerns.get(row["case_id"], ""))
        if not quote or quote not in source:
            evidence_failures.append(
                f"{row['case_id']}:{row['thread_id']} quote not found"
            )
    if evidence_failures:
        raise ValueError("Evidence validation failed: " + "; ".join(evidence_failures))

    response_ids = (
        document_ledger[document_ledger["document_type"].eq("response")]
        .groupby("case_id")["document_id"]
        .apply(lambda values: ";".join(values.astype(str)))
        .to_dict()
    )
    reviews["response_document_ids"] = reviews["case_id"].map(response_ids).fillna("")
    reviews["review_status"] = "single_reviewer_validated"
    reviews = reviews.merge(
        source_ledger[
            [
                "case_id",
                "report_id",
                "report_date",
                "report_url",
                "receiver",
                "official_response_count",
            ]
        ],
        on="case_id",
        how="left",
        validate="many_to_one",
    )
    reviews.to_csv(args.run_dir / "03_validated_thread_findings.csv", index=False)

    supported = reviews[reviews["supported"]].copy()
    supported["action_types"] = supported["action_type"].str.split(";")
    action_counts = Counter(
        action
        for values in supported["action_types"]
        for action in values
        if clean_text(action)
    )
    thread_summary = (
        reviews.groupby("thread_id", sort=False)
        .agg(
            candidate_case_thread_rows=("case_id", "size"),
            supported_case_thread_rows=("supported", "sum"),
            distinct_supported_cases=(
                "case_id",
                lambda values: values[reviews.loc[values.index, "supported"]].nunique(),
            ),
        )
        .reset_index()
    )
    direct_links = int(supported["concern_action_match"].eq("direct").sum())
    partial_links = int(
        supported["concern_action_match"].isin(["partial", "partial_to_direct"]).sum()
    )
    cases_with_responses = int(source_ledger["official_response_count"].gt(0).sum())
    source_documents = int(len(document_ledger))
    extracted_documents = int(
        document_ledger["extraction_status"].eq("text_extracted").sum()
    )

    summary = {
        "sample_cases": int(source_ledger["case_id"].nunique()),
        "candidate_case_thread_rows": int(len(reviews)),
        "supported_case_thread_rows": int(len(supported)),
        "rejected_candidate_case_thread_rows": int((~reviews["supported"]).sum()),
        "official_documents": source_documents,
        "official_response_documents": int(
            document_ledger["document_type"].eq("response").sum()
        ),
        "cases_with_official_response": cases_with_responses,
        "documents_with_extracted_text": extracted_documents,
        "direct_concern_action_links": direct_links,
        "partial_or_mixed_links": partial_links,
        "outcome_effectiveness_evidence_rows": 0,
        "evidence_quote_validation_failures": 0,
        "review_status": "provisional_single_reviewer",
        "action_type_counts": dict(action_counts.most_common()),
    }
    (args.run_dir / "04_feasibility_summary.json").write_text(
        json.dumps(summary, indent=2) + "\n", encoding="utf-8"
    )

    source_table = pd.DataFrame(
        [
            ["Selected reports", 20],
            ["Official report PDFs recovered", 20],
            ["Official response PDFs recovered", summary["official_response_documents"]],
            ["Cases with an official response", f"{cases_with_responses}/20"],
            ["PDFs yielding extracted text", f"{extracted_documents}/{source_documents}"],
            ["Candidate case-thread judgements", len(reviews)],
            ["Supported case-thread signals", len(supported)],
            ["Rejected candidate assignments", int((~reviews["supported"]).sum())],
        ],
        columns=["Measure", "Result"],
    )
    link_table = pd.DataFrame(
        [
            ["Direct concern-action match", direct_links],
            ["Partial or mixed match", partial_links],
            ["Supported rows with outcome-effectiveness evidence", 0],
        ],
        columns=["Linkage result", "Rows"],
    )
    action_table = pd.DataFrame(
        action_counts.most_common(12), columns=["Action mechanism", "Supported rows"]
    )

    report = f"""# Prison PFD concern–response feasibility exercise

## Status

**Feasible, with an important limit:** the documents support reproducible analysis of
concerns, organisational responses, action commitments and later recurrence signals.
They do not, by themselves, establish that an action was effective or causally
ineffective.

This is a purposive methodological exercise, not a prevalence study. It was completed
as a single-reviewer pass and must receive independent domain review before its
substantive findings are quoted externally.

## 1. Exercise design

Twenty reports from 2022–2024 were selected across five deliberately overlapping
candidate threads. Every candidate assignment was tested against the official
coroner's concerns rather than inferred from the death circumstances. This matters:
{int((~reviews["supported"]).sum())} of {len(reviews)} candidate case-thread assignments
were rejected because the thread appeared only in the circumstances, had already
been described as addressed, or was not the actual PFD concern.

{markdown_table(source_table)}

All {len(supported)} retained signals have a verbatim concern quote that was
mechanically validated against the local full-text concern field.

## 2. Supported signals by thread

{markdown_table(thread_summary.rename(columns={
    "thread_id": "Thread",
    "candidate_case_thread_rows": "Candidates",
    "supported_case_thread_rows": "Supported",
    "distinct_supported_cases": "Cases",
}))}

These are case-thread signals, not mutually exclusive reports and not final issue
types. The broad drug/deterioration/emergency thread contains several distinct
safeguards—clinical prescribing, medication diversion, intoxication monitoring,
NEWS2, CPR and cell entry—which must remain separate in the production taxonomy.

## 3. Concern-to-action linkage

{markdown_table(link_table)}

Direct linkage was usually possible because responses repeated the coroner's numbered
concerns. Partial links arose when respondents disputed the premise, relied on an
existing policy, allocated responsibility elsewhere, or offered a local action for a
potentially national concern.

The most common action mechanisms in the supported rows were:

{markdown_table(action_table)}

Training, guidance, notices and assurance processes are common. Implementation was
usually evidenced only by a statement that a policy, meeting, form, course or audit
existed. Two response chains supplied useful process measures—NEWS2 training or
compliance—but none supplied evidence of improved mortality or a comparable safety
outcome. “Claimed completed” must therefore remain distinct from “implemented” and
“outcome evaluated”.

## 4. Five golden threads

### ACCT and suicide-risk information

Five selected cases supported this thread. The recurring family includes missing
first-night safeguards, suicide-risk information omitted at transfer, weak ACCT
training and audit, cursory observations, fragmented records and failure to confirm
mental-health referrals. Responses repeatedly used local training, assurance,
briefings and new recording processes. The cases are related but not interchangeable:
handover omission, ACCT review quality and acute mental-health crisis require separate
issue types.

The defensible finding is a recurring **risk-information and ACCT assurance family**.
The sample does not show that one national ACCT intervention failed, because most
actions were local and later cases arose in different prisons.

### Observation, ligature safety and emergency cell entry

Six cases supported this thread. The clearest persistence signal is the Binfield
report: the coroner explicitly recorded that repeated notices after an earlier death
had not changed embedded staff practice around obscured observation panels. Sleaford
identified the same panel and cell-entry safeguard at another prison, while its
response again relied heavily on notices, briefings and acknowledgement.

Other cases concern different controls—safer-cell ligature points, razors, welfare
check quality and CPR after ligature. They should be reported as related subtypes,
not merged into one frequency.

### Mental-health recognition, information and access

Three selected cases contained unresolved PFD concerns in this thread. They show:
fragmented prison/health information for decision-making; absence of psychiatric MDT
input and rolling risk records; and lack of out-of-hours acute mental-health provision.
The Singh responses are especially informative methodologically because responsibility
was split between provider, commissioner and software supplier, producing a local
pilot but no clear national owner.

This thread supports analysis of ownership and scope mismatch more strongly than a
single frequency count.

### Drugs, physical deterioration and emergency response

Ten cases supported this broad candidate thread, but manual review shows it must be
split. Distinct signals include medication diversion, psychoactive-substance warnings,
night-state intoxication monitoring, methadone prescribing, NEWS2 deterioration
recognition, top-bunk resuscitation, CPR competence and welfare-check quality.

Two longitudinal patterns deserve further review:

1. The Braund response reported NEWS2 training and audit after a coroner stated that
   the issue was already recurrent. The later Smith PFD, involving the same healthcare
   provider, again reported failure to use NEWS2 and itself referred to repeated prior
   concerns. This is a strong provider-assurance signal. The deaths predated the
   responses, so it is not proof of post-intervention failure.
2. Johnson's 2022 response described entry-level first aid, an updated video and a
   policy review. Forrester and Sleaford later raised specific emergency-response
   competence gaps. This supports asking for training coverage and competence evidence,
   but differences in role, prison and task prevent a causal conclusion.

### Candour, record integrity and organisational learning

Five cases supported this thread. They identify four related but distinct safeguards:
electronic record auditability, post-death document handling, timely disclosure and
candour, and feedback of investigation findings to staff. Responses introduced
oversight meetings, investigators, templates, training and document-handling changes,
but supplied no measured evidence of disclosure timeliness, record-integrity compliance
or learning reaching practice.

This is a coherent Panel workstream, but not yet one homogeneous concern type.

## 5. What “persistence” can mean

The feasibility exercise supports three graded findings:

1. **Explicit persistence:** a coroner states that previous notices, recommendations
   or assurances did not resolve the problem. Binfield meets this standard.
2. **Post-action assurance gap:** a later PFD raises a comparable safeguard after an
   earlier relevant national or provider-level action should have operated. CPR/first
   aid and provider NEWS2 are candidates requiring domain verification.
3. **Recurring issue family:** later reports concern related processes but no action
   can safely be treated as applicable because scope, actor or safeguard differs.
   Most ACCT and mental-health cases currently sit here.

The production analysis must also distinguish the date of the death from the report,
response, promised completion and actual implementation dates. A later report about an
earlier death can demonstrate continued coronial concern or weak assurance, but not a
post-intervention adverse event.

## 6. Feasibility decisions

| Question | Decision | Reason |
|---|---|---|
| Can Andrew's cases be linked to full reports? | Yes | All 20 sample cases were reconciled and recovered. |
| Can official responses be recovered? | Yes | 19/20 cases had official response attachments; 32 response PDFs were recovered. |
| Can concern types be extracted with evidence? | Yes | All retained thread signals have source-validated quotes. |
| Can responses be decomposed into actions? | Yes | Actions, scope, stance and status were identifiable in the sample. |
| Can concerns be linked to actions? | Yes, with review | {direct_links} direct and {partial_links} partial/mixed thread-level links were identified. |
| Can action effectiveness be measured from PFD documents alone? | No | No response supplied comparable outcome-effectiveness evidence. |
| Can persistence and assurance gaps be analysed? | Yes | Explicit recurrence and cautious post-action comparisons are supportable. |

## 7. Required production data model

The exercise confirms that the production model needs separate entities for:

- report and death dates;
- exact concern occurrence and broader issue family;
- respondent and response document;
- individual action commitment;
- action scope, owner, status, deadline and claimed completion date;
- implementation evidence and outcome evidence;
- concern-action match;
- later comparable concern; and
- persistence grade with a recorded rationale.

The current general issue schema remains suitable for concern extraction. A parallel
response-action schema and human-review workflow are required.

## 8. Quality and limitations

- The sample is purposive and cannot estimate corpus prevalence.
- One reviewer made the current judgements.
- Andrew's summaries informed sample selection but were not treated as evidence.
- A response not located on the Judiciary page is “not located”, not proof that no
  organisation responded.
- Similar wording is insufficient for persistence: actor, failed object, process,
  scope and timing must all be checked.
- Self-reported completion is not independent implementation evidence.

The next defensible step is independent review of all {len(reviews)} case-thread
judgements, followed by a complete date-bounded corpus and response scrape. The five
golden threads can then seed, but must not constrain, the full concern taxonomy.
"""
    (args.run_dir / "FEASIBILITY_REPORT.md").write_text(
        report.strip() + "\n", encoding="utf-8"
    )
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
