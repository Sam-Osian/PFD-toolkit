#!/usr/bin/env python3
"""Build the source ledger for the prison-PFD manual feasibility exercise.

The script deliberately keeps generated material under ``artifacts/`` (ignored
by git). It reconciles the selected rows from Andrew Harris's DOCX to the local
PFD dataset, inventories official Judiciary attachments, downloads them when
requested, and extracts auditable PDF text without making any analytical
judgement about the contents.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import re
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass
from pathlib import Path
from typing import Any
from urllib.parse import urljoin, urlparse
from xml.etree import ElementTree as ET
from zipfile import ZipFile

import fitz
import pandas as pd
import requests
from bs4 import BeautifulSoup


W = "{http://schemas.openxmlformats.org/wordprocessingml/2006/main}"
R = "{http://schemas.openxmlformats.org/officeDocument/2006/relationships}"
P = "{http://schemas.openxmlformats.org/package/2006/relationships}"
URL_RE = re.compile(r"https?://[^\s<>]+")


@dataclass(frozen=True)
class AndrewRow:
    identifier: str
    circumstances_summary: str
    concern_summary: str
    organisation_summary: str
    response_summary: str
    issue_action_notes: str
    embedded_links: tuple[str, ...]


def clean_text(value: Any) -> str:
    return " ".join(str(value or "").split())


def safe_slug(value: str, *, max_length: int = 90) -> str:
    slug = re.sub(r"[^a-z0-9]+", "-", value.casefold()).strip("-")
    return slug[:max_length].rstrip("-") or "document"


def sha256_bytes(content: bytes) -> str:
    return hashlib.sha256(content).hexdigest()


def parse_andrew_docx(path: Path) -> dict[str, AndrewRow]:
    with ZipFile(path) as archive:
        document = ET.fromstring(archive.read("word/document.xml"))
        relationships = ET.fromstring(
            archive.read("word/_rels/document.xml.rels")
        )

    targets = {
        rel.attrib["Id"]: rel.attrib.get("Target", "")
        for rel in relationships.findall(P + "Relationship")
    }
    output: dict[str, AndrewRow] = {}
    for table in document.findall(".//" + W + "tbl"):
        for row in table.findall("./" + W + "tr"):
            cells: list[str] = []
            links: list[str] = []
            for cell in row.findall("./" + W + "tc"):
                cell_text = clean_text(
                    "".join(node.text or "" for node in cell.findall(".//" + W + "t"))
                )
                cells.append(cell_text)
                links.extend(URL_RE.findall(cell_text))
                for hyperlink in cell.findall(".//" + W + "hyperlink"):
                    relationship_id = hyperlink.attrib.get(R + "id")
                    if relationship_id and targets.get(relationship_id):
                        links.append(targets[relationship_id])
            if len(cells) != 6 or not re.match(r"^\s*\d+", cells[0]):
                continue
            unique_links = tuple(dict.fromkeys(link.rstrip(".,);") for link in links))
            output[cells[0]] = AndrewRow(
                identifier=cells[0],
                circumstances_summary=cells[1],
                concern_summary=cells[2],
                organisation_summary=cells[3],
                response_summary=cells[4],
                issue_action_notes=cells[5],
                embedded_links=unique_links,
            )
    return output


def load_sample(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    cases = payload.get("cases")
    if not isinstance(cases, list) or len(cases) != 20:
        raise ValueError("The feasibility sample must contain exactly 20 cases.")
    identifiers = [clean_text(case.get("andrew_identifier")) for case in cases]
    if len(set(identifiers)) != len(identifiers):
        raise ValueError("The feasibility sample contains duplicate identifiers.")
    return payload


def build_source_ledger(
    andrew_rows: dict[str, AndrewRow],
    reports: pd.DataFrame,
    sample: dict[str, Any],
) -> pd.DataFrame:
    if "url" not in reports.columns:
        raise ValueError("The local report dataset does not contain a url column.")
    indexed_reports = reports.drop_duplicates("url").set_index("url", drop=False)
    records: list[dict[str, Any]] = []
    for case_number, case in enumerate(sample["cases"], start=1):
        identifier = clean_text(case["andrew_identifier"])
        if identifier not in andrew_rows:
            raise ValueError(f"Selected identifier is missing from the DOCX: {identifier}")
        report_url = clean_text(case["report_url"])
        if report_url not in indexed_reports.index:
            raise ValueError(f"Selected report URL is missing from all_reports.csv: {report_url}")
        row = andrew_rows[identifier]
        report = indexed_reports.loc[report_url]
        records.append(
            {
                "case_id": f"F{case_number:02d}",
                "andrew_identifier": identifier,
                "threads": ";".join(case["threads"]),
                "report_url": report_url,
                "report_id": clean_text(report.get("id")),
                "report_date": clean_text(report.get("date")),
                "coroner": clean_text(report.get("coroner")),
                "area": clean_text(report.get("area")),
                "receiver": clean_text(report.get("receiver")),
                "local_investigation_chars": len(clean_text(report.get("investigation"))),
                "local_circumstances_chars": len(clean_text(report.get("circumstances"))),
                "local_concerns_chars": len(clean_text(report.get("concerns"))),
                "andrew_circumstances_summary": row.circumstances_summary,
                "andrew_concern_summary": row.concern_summary,
                "andrew_organisation_summary": row.organisation_summary,
                "andrew_response_summary": row.response_summary,
                "andrew_issue_action_notes": row.issue_action_notes,
                "andrew_embedded_links": ";".join(row.embedded_links),
                "source_match_status": "manually_confirmed",
            }
        )
    return pd.DataFrame(records)


def classify_attachment(text: str, href: str) -> str:
    combined = f"{text} {href}".casefold()
    if "response" in combined:
        return "response"
    if ".pdf" in combined:
        return "report_or_related_pdf"
    return "other"


def extract_pdf_text(content: bytes) -> tuple[str, int]:
    document = fitz.open(stream=content, filetype="pdf")
    pages = [page.get_text("text", sort=True) for page in document]
    return "\\n\\n".join(pages).strip(), document.page_count


def fetch_case(
    case: dict[str, Any],
    output_dir: Path,
    *,
    timeout: int,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    case_id = case["case_id"]
    report_url = case["report_url"]
    case_dir = output_dir / "source_documents" / case_id
    text_dir = output_dir / "extracted_text" / case_id
    case_dir.mkdir(parents=True, exist_ok=True)
    text_dir.mkdir(parents=True, exist_ok=True)

    page_response = requests.get(report_url, timeout=timeout)
    page_response.raise_for_status()
    page_content = page_response.content
    page_path = case_dir / "report_page.html"
    page_path.write_bytes(page_content)

    soup = BeautifulSoup(page_content, "html.parser")
    attachments: list[tuple[str, str, str]] = []
    for anchor in soup.find_all("a", href=True):
        title = clean_text(anchor.get_text(" ", strip=True))
        href = urljoin(report_url, anchor["href"])
        if ".pdf" not in href.casefold() and "response" not in title.casefold():
            continue
        attachments.append((title, href, classify_attachment(title, href)))
    attachments = list(dict.fromkeys(attachments))

    document_records: list[dict[str, Any]] = []
    for document_number, (title, url, document_type) in enumerate(attachments, start=1):
        response = requests.get(url, timeout=timeout)
        response.raise_for_status()
        content = response.content
        url_name = Path(urlparse(response.url).path).name
        suffix = Path(url_name).suffix.casefold() or ".bin"
        stem = safe_slug(Path(url_name).stem or title)
        file_name = f"{document_number:02d}_{document_type}_{stem}{suffix}"
        document_path = case_dir / file_name
        document_path.write_bytes(content)

        extraction_status = "not_pdf"
        extracted_chars = 0
        page_count = 0
        text_path = ""
        extraction_error = ""
        if suffix == ".pdf" or content.startswith(b"%PDF"):
            try:
                text, page_count = extract_pdf_text(content)
                extracted_chars = len(text)
                text_output = text_dir / f"{Path(file_name).stem}.txt"
                text_output.write_text(text, encoding="utf-8")
                text_path = str(text_output)
                extraction_status = "text_extracted" if text else "empty_text"
            except Exception as exc:  # retained in the audit ledger
                extraction_status = "extraction_failed"
                extraction_error = f"{type(exc).__name__}: {exc}"

        document_records.append(
            {
                "case_id": case_id,
                "document_id": f"{case_id}-D{document_number:02d}",
                "document_type": document_type,
                "link_text": title,
                "source_url": url,
                "resolved_url": response.url,
                "http_status": response.status_code,
                "content_type": response.headers.get("content-type", ""),
                "byte_count": len(content),
                "sha256": sha256_bytes(content),
                "page_count": page_count,
                "extracted_chars": extracted_chars,
                "extraction_status": extraction_status,
                "extraction_error": extraction_error,
                "local_document_path": str(document_path),
                "local_text_path": text_path,
            }
        )

    case_result = {
        "case_id": case_id,
        "report_page_http_status": page_response.status_code,
        "report_page_sha256": sha256_bytes(page_content),
        "report_page_path": str(page_path),
        "official_attachment_count": len(document_records),
        "official_response_count": sum(
            record["document_type"] == "response" for record in document_records
        ),
        "source_recovery_status": (
            "report_and_response_recovered"
            if any(record["document_type"] == "response" for record in document_records)
            else "report_recovered_no_official_response_attachment"
        ),
    }
    return case_result, document_records


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--docx", type=Path, required=True)
    parser.add_argument("--reports-csv", type=Path, default=Path("all_reports.csv"))
    parser.add_argument(
        "--sample",
        type=Path,
        default=Path(__file__).with_name("feasibility_sample.json"),
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/prison_pfd_feasibility"),
    )
    parser.add_argument(
        "--fetch",
        action="store_true",
        help="Fetch official pages and attachments. Without this flag only the source ledger is built.",
    )
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--timeout", type=int, default=60)
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)
    andrew_rows = parse_andrew_docx(args.docx)
    reports = pd.read_csv(args.reports_csv, low_memory=False)
    sample = load_sample(args.sample)
    ledger = build_source_ledger(andrew_rows, reports, sample)
    ledger_path = args.output_dir / "00_source_ledger.csv"
    ledger.to_csv(ledger_path, index=False)

    manifest = {
        "sample_version": sample["sample_version"],
        "sample_size": len(ledger),
        "andrew_docx": str(args.docx),
        "andrew_docx_sha256": sha256_bytes(args.docx.read_bytes()),
        "reports_csv": str(args.reports_csv),
        "reports_csv_sha256": sha256_bytes(args.reports_csv.read_bytes()),
        "fetch_performed": bool(args.fetch),
    }

    if args.fetch:
        case_results: list[dict[str, Any]] = []
        document_records: list[dict[str, Any]] = []
        cases = ledger.to_dict(orient="records")
        with ThreadPoolExecutor(max_workers=args.workers) as executor:
            futures = {
                executor.submit(
                    fetch_case, case, args.output_dir, timeout=args.timeout
                ): case["case_id"]
                for case in cases
            }
            for future in as_completed(futures):
                case_result, documents = future.result()
                case_results.append(case_result)
                document_records.extend(documents)

        recovery = pd.DataFrame(case_results)
        ledger = ledger.merge(recovery, on="case_id", how="left")
        ledger.sort_values("case_id").to_csv(ledger_path, index=False)
        documents = pd.DataFrame(document_records).sort_values(
            ["case_id", "document_id"]
        )
        documents.to_csv(args.output_dir / "01_document_ledger.csv", index=False)
        manifest.update(
            {
                "official_documents_recovered": int(len(documents)),
                "official_responses_recovered": int(
                    documents["document_type"].eq("response").sum()
                ),
                "cases_with_official_response": int(
                    ledger["official_response_count"].fillna(0).gt(0).sum()
                ),
                "documents_with_extracted_text": int(
                    documents["extraction_status"].eq("text_extracted").sum()
                ),
            }
        )

    (args.output_dir / "source_manifest.json").write_text(
        json.dumps(manifest, indent=2) + "\\n", encoding="utf-8"
    )
    print(json.dumps(manifest, indent=2))


if __name__ == "__main__":
    main()
