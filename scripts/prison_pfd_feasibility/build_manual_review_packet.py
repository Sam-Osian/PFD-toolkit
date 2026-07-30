#!/usr/bin/env python3
"""Build a human-review packet and thread-level annotation template.

This stage makes no model calls. It combines the local structured report text,
the source-recovery ledger, and the extracted official responses so that a
reviewer can verify one evidence chain at a time.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any

import pandas as pd


ANNOTATION_COLUMNS = [
    "case_id",
    "report_id",
    "report_date",
    "thread_id",
    "thread_label",
    "concern_present",
    "canonical_concern",
    "concern_evidence_quote",
    "responsible_actor",
    "response_document_ids",
    "response_position",
    "action_summary",
    "action_type",
    "action_scope",
    "action_status",
    "action_deadline",
    "implementation_evidence",
    "evaluation_evidence",
    "concern_action_match",
    "later_recurrence_candidate",
    "reviewer_notes",
    "review_status",
]


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def read_text(path: str) -> str:
    source = Path(clean_text(path))
    if not source.exists():
        return ""
    return source.read_text(encoding="utf-8", errors="replace").strip()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--run-dir",
        type=Path,
        default=Path("artifacts/prison_pfd_feasibility"),
    )
    parser.add_argument("--reports-csv", type=Path, default=Path("all_reports.csv"))
    parser.add_argument(
        "--sample",
        type=Path,
        default=Path(__file__).with_name("feasibility_sample.json"),
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    source_ledger = pd.read_csv(args.run_dir / "00_source_ledger.csv")
    document_ledger = pd.read_csv(args.run_dir / "01_document_ledger.csv")
    reports = pd.read_csv(args.reports_csv, low_memory=False)
    sample = json.loads(args.sample.read_text(encoding="utf-8"))
    thread_labels = sample["threads"]

    report_fields = reports[
        ["url", "investigation", "circumstances", "concerns"]
    ].drop_duplicates("url")
    cases = source_ledger.merge(
        report_fields, left_on="report_url", right_on="url", how="left"
    )

    annotation_rows: list[dict[str, Any]] = []
    packet_parts = [
        "# Prison PFD feasibility: manual evidence review packet",
        "",
        "This packet contains source text for a purposive feasibility sample. "
        "Andrew's summaries are comparison material, not ground truth. "
        "All analytical fields must be supported by the official report or response.",
        "",
    ]
    for case in cases.sort_values("case_id").to_dict(orient="records"):
        case_id = case["case_id"]
        case_documents = document_ledger[
            document_ledger["case_id"].eq(case_id)
        ].sort_values("document_id")
        response_documents = case_documents[
            case_documents["document_type"].eq("response")
        ]
        threads = clean_text(case["threads"]).split(";")
        for thread_id in threads:
            annotation_rows.append(
                {
                    "case_id": case_id,
                    "report_id": case["report_id"],
                    "report_date": case["report_date"],
                    "thread_id": thread_id,
                    "thread_label": thread_labels[thread_id],
                    "concern_present": "",
                    "canonical_concern": "",
                    "concern_evidence_quote": "",
                    "responsible_actor": "",
                    "response_document_ids": ";".join(
                        response_documents["document_id"].astype(str)
                    ),
                    "response_position": "",
                    "action_summary": "",
                    "action_type": "",
                    "action_scope": "",
                    "action_status": "",
                    "action_deadline": "",
                    "implementation_evidence": "",
                    "evaluation_evidence": "",
                    "concern_action_match": "",
                    "later_recurrence_candidate": "",
                    "reviewer_notes": "",
                    "review_status": "",
                }
            )

        packet_parts.extend(
            [
                f"## {case_id}: {case['report_id']} ({case['report_date']})",
                "",
                f"Official page: {case['report_url']}",
                "",
                f"Review threads: {', '.join(threads)}",
                "",
                "### Andrew's supplied summary",
                "",
                f"**Circumstances:** {clean_text(case['andrew_circumstances_summary'])}",
                "",
                f"**Concerns:** {clean_text(case['andrew_concern_summary'])}",
                "",
                f"**Response:** {clean_text(case['andrew_response_summary'])}",
                "",
                f"**Notes:** {clean_text(case['andrew_issue_action_notes'])}",
                "",
                "### Local full-text fields",
                "",
                "**Investigation**",
                "",
                clean_text(case.get("investigation")),
                "",
                "**Circumstances**",
                "",
                clean_text(case.get("circumstances")),
                "",
                "**Coroner's concerns**",
                "",
                clean_text(case.get("concerns")),
                "",
                "### Official response documents",
                "",
            ]
        )
        if response_documents.empty:
            packet_parts.extend(["No official response attachment was located.", ""])
        for document in response_documents.to_dict(orient="records"):
            packet_parts.extend(
                [
                    f"#### {document['document_id']}: {document['link_text']}",
                    "",
                    f"Source: {document['source_url']}",
                    "",
                    read_text(document["local_text_path"]),
                    "",
                ]
            )

    template = pd.DataFrame(annotation_rows, columns=ANNOTATION_COLUMNS)
    template.to_csv(args.run_dir / "02_thread_review.csv", index=False)
    (args.run_dir / "manual_review_packet.md").write_text(
        "\n".join(packet_parts).strip() + "\n", encoding="utf-8"
    )
    print(
        json.dumps(
            {
                "cases": int(len(cases)),
                "thread_review_rows": int(len(template)),
                "response_documents": int(
                    document_ledger["document_type"].eq("response").sum()
                ),
                "packet": str(args.run_dir / "manual_review_packet.md"),
                "template": str(args.run_dir / "02_thread_review.csv"),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
