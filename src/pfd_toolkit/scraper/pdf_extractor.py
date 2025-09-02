from __future__ import annotations

import logging
import os
from io import BytesIO
from typing import Dict
import pandas as pd
from urllib.parse import urlparse, unquote

from bs4 import BeautifulSoup
import requests
import pymupdf

from dataclasses import dataclass
from typing import List, Optional

from ..text_utils import clean_text

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class PdfSectionConfig:
    """Configuration for extracting a single section from PDF text."""

    key: str
    start_keys: List[str]
    end_keys: List[str]
    rem_strs: List[str]
    min_len: Optional[int]
    max_len: Optional[int]

# Default keyword patterns for PDF extraction
# ...We take word 'sandwiches' through specifying the start and end strings/keys.
#    From here, we remove extracted string snippets, & set min_len & max_len 
#    values for validation.

DEFAULT_PDF_SECTIONS: List[PdfSectionConfig] = [
    PdfSectionConfig(
        "coroner",
        start_keys=["I am", "CORONER"],
        end_keys=["CORONER'S LEGAL POWERS", "paragraph 7"], 
        rem_strs=["I am", "CORONER", "CORONER'S LEGAL POWERS", "paragraph 7"],
        min_len=5,
        max_len=20,
    ),
    PdfSectionConfig(
        "area",
        start_keys=["area of"],
        end_keys=["LEGAL POWERS", "LEGAL POWER", "paragraph 7"],
        rem_strs=["area of", "CORONER'S", "CORONER", "CORONERS", "paragraph 7"],
        min_len=4,
        max_len=40,
    ),
    PdfSectionConfig(
        "receiver",
        start_keys=[" SENT ", "SENT TO:"],
        end_keys=["CORONER", "CIRCUMSTANCES"],
        rem_strs=["TO:"],
        min_len=5,
        max_len=None,
    ),
    PdfSectionConfig(
        "investigation",
        start_keys=["INVESTIGATION and INQUEST", "3 INQUEST"],
        end_keys=["CIRCUMSTANCES"],
        rem_strs=[],
        min_len=30,
        max_len=None,
    ),
    PdfSectionConfig(
        "circumstances",
        start_keys=["CIRCUMSTANCES OF DEATH", "CIRCUMSTANCES OF THE DEATH"],
        end_keys=["CORONER'S CONCERNS", "as follows"],
        rem_strs=[],
        min_len=30,
        max_len=None,
    ),
    PdfSectionConfig(
        "concerns",
        start_keys=["CORONER'S CONCERNS", "as follows"],
        end_keys=["ACTION SHOULD BE TAKEN"],
        rem_strs=[],
        min_len=30,
        max_len=None,
    ),
]

class PdfExtractor:
    """Utility class encapsulating PDF scraping helpers."""

    def __init__(
        self,
        cfg,
        timeout: int,
        not_found_text: str,
        verbose: bool = False,
    ) -> None:
        # Cache configuration and downloaded files
        self.cfg = cfg
        self.timeout = timeout
        self.not_found_text = not_found_text
        self.verbose = verbose
        self._last_pdf_bytes: bytes | None = None
        self._pdf_cache: Dict[str, bytes] = {}

    # ------------------------------------------------------------------
    # Basic helpers
    # ------------------------------------------------------------------
    def get_pdf_link(self, soup: BeautifulSoup) -> str | None:
        """Grab the first download link on the page."""
        # Newer reports use a ``wp-block-file`` container with a download button
        pdf_link = None
        block = soup.find("div", class_="wp-block-file")
        if block:
            anchor = block.find(
                "a",
                href=True,
                class_="wp-block-file__button",
            )
            if not anchor:
                anchor = block.find("a", href=True)
            if anchor:
                pdf_link = anchor.get("href")

        # Fallback to the older GOV.UK button format
        if not pdf_link:
            pdf_link = next(
                (
                    a["href"]
                    for a in soup.find_all("a", class_="govuk-button")
                    if a.get("href")
                ),
                None,
            )
        return pdf_link

    def fetch_pdf_bytes(self, report_url: str) -> bytes | None:
        """Download the report page and locate the PDF link"""
        try:
            soup = BeautifulSoup(
                self.cfg.session.get(report_url, timeout=self.timeout).content,
                "html.parser",
            )
            pdf_link = self.get_pdf_link(soup)
        except Exception:
            return None

        if not pdf_link:
            return None

        if pdf_link in self._pdf_cache:
            # Use cached bytes if available
            return self._pdf_cache[pdf_link]

        try:
            pdf_response = self.cfg.session.get(pdf_link, timeout=self.timeout)
            pdf_response.raise_for_status()
            content = pdf_response.content
            self._pdf_cache[pdf_link] = content
            return content
        except Exception as e:
            logger.error("Failed to fetch PDF for report at %s: %s", report_url, e)
            return None

    def extract_text_from_pdf(self, pdf_url: str) -> str:
        parsed_url = urlparse(pdf_url)
        path = unquote(parsed_url.path)
        ext = os.path.splitext(path)[1].lower()
        if self.verbose:
            logger.debug(f"Processing .pdf {pdf_url}.")
        file_bytes: bytes | None = None
        
        if pdf_url in self._pdf_cache:
            file_bytes = self._pdf_cache[pdf_url]
        else:
            try:
                response = self.cfg.session.get(pdf_url, timeout=self.timeout)
                response.raise_for_status()
                file_bytes = response.content
                self._pdf_cache[pdf_url] = file_bytes
            except requests.RequestException as e:
                logger.error("Failed to fetch file: %s; Error: %s", pdf_url, e)
                return self.not_found_text

        pdf_bytes_to_process: bytes | None = None
        if ext != ".pdf":
            logger.info(
                "File %s is not a .pdf (extension %s). Skipping this file...",
                pdf_url,
                ext,
            )
            return self.not_found_text
        else:
            # Use extracted bytes if file is PDF
            pdf_bytes_to_process = file_bytes
            self._pdf_cache[pdf_url] = pdf_bytes_to_process

        if pdf_bytes_to_process is None:
            return "N/A: Source file not PDF"
        # Keep the last processed bytes for debugging
        self._last_pdf_bytes = pdf_bytes_to_process

        try:
            pdf_buffer = BytesIO(pdf_bytes_to_process)
            pdf_document = pymupdf.open(stream=pdf_buffer, filetype="pdf")
            text = "".join(page.get_text() for page in pdf_document)
            pdf_document.close()
        except Exception as e:
            logger.error("Error processing .pdf %s: %s", pdf_url, e)
            return "N/A: Source file not PDF"
        return clean_text(text)

    def extract_pdf_section(self, text: str, start_keywords: list[str], end_keywords: list[str]) -> str:
        """Locate the text between the start and end keywords"""
        lower_text = text.lower()
        for start_kw in start_keywords:
            start_kw_lower = start_kw.lower()
            start_index = lower_text.find(start_kw_lower)
            if start_index != -1:
                section_start_offset = start_index + len(start_kw_lower)
                end_indices_found: list[int] = []
                for end_kw in end_keywords:
                    end_kw_lower = end_kw.lower()
                    end_index = lower_text.find(end_kw_lower, section_start_offset)
                    if end_index != -1:
                        end_indices_found.append(end_index)
                if end_indices_found:
                    section_end_offset = min(end_indices_found)
                    extracted_section_text = text[section_start_offset:section_end_offset]
                else:
                    extracted_section_text = text[section_start_offset:]
                return extracted_section_text
        return self.not_found_text

    def apply_pdf_fallback(self, pdf_text: str, fields: Dict[str, str], include_flags: Dict[str, bool]) -> None:
        # Only attempt sections that are missing
        missing = {k for k, v in fields.items() if pd.isna(v)}
        for cfg in self.cfg.pdf_sections:
            if cfg.key not in missing or not include_flags.get(cfg.key, False):
                continue
            raw = self.extract_pdf_section(
                pdf_text,
                start_keywords=cfg.start_keys,
                end_keywords=cfg.end_keys,
            )
            if pd.isna(raw):
                continue
            cleaned = clean_text(raw)
            for rem in cfg.rem_strs:
                cleaned = cleaned.replace(rem, "")
            cleaned = cleaned.strip()
            if (cfg.min_len is not None and len(cleaned) < cfg.min_len) or (
                cfg.max_len is not None and len(cleaned) > cfg.max_len
            ):
                continue
            # Update fields dict with extracted section
            fields[cfg.key] = cleaned if cleaned else self.not_found_text