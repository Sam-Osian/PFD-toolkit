from __future__ import annotations

import logging
from typing import Dict
import pandas as pd

from bs4 import BeautifulSoup
import requests

from dataclasses import dataclass
from typing import List, Optional

from ..text_utils import clean_text, process_extracted_field

logger = logging.getLogger(__name__)


@dataclass(frozen=True)
class HtmlFieldConfig:
    """Configuration for extracting a single field from HTML."""

    key: str
    para_keys: Optional[List[str]]
    sec_keys: Optional[List[str]]
    rem_strs: List[str]
    min_len: Optional[int]
    max_len: Optional[int]
    is_date: bool

# Default patterns used when scraping from HTML
# ...the script looks at either paragraph (para) or section (sec) tags
#    based on provided strings that act as 'keys'. min_length and 
#    max_length provide data validation.
DEFAULT_HTML_FIELDS: List[HtmlFieldConfig] = [
    HtmlFieldConfig(
        "id",
        para_keys=["Ref:"],
        sec_keys=None,
        rem_strs=["Ref:"],
        min_len=None,
        max_len=None,
        is_date=False,
    ),
    HtmlFieldConfig(
        "date",
        para_keys=["Date of report:"],
        sec_keys=None,
        rem_strs=["Date of report:"],
        min_len=None,
        max_len=None,
        is_date=True,
    ),
    HtmlFieldConfig(
        "receiver",
        para_keys=["This report is being sent to:", "Sent to:"],
        sec_keys=None,
        rem_strs=["This report is being sent to:", "Sent to:", "TO:"],
        min_len=5,
        max_len=20,
        is_date=False,
    ),
    HtmlFieldConfig(
        "coroner",
        para_keys=["Coroners name:", "Coroner name:", "Coroner's name:"],
        sec_keys=None,
        rem_strs=["Coroners name:", "Coroner name:", "Coroner's name:"],
        min_len=5,
        max_len=20,
        is_date=False,
    ),
    HtmlFieldConfig(
        "area",
        para_keys=["Coroners Area:", "Coroner Area:", "Coroner's Area:"],
        sec_keys=None,
        rem_strs=["Coroners Area:", "Coroner Area:", "Coroner's Area:"],
        min_len=4,
        max_len=40,
        is_date=False,
    ),
    HtmlFieldConfig(
        "investigation",
        para_keys=None,
        sec_keys=[
            "INVESTIGATION and INQUEST",
            "INVESTIGATION & INQUEST",
            "3 INQUEST",
        ],
        rem_strs=[
            "INVESTIGATION and INQUEST",
            "INVESTIGATION & INQUEST",
            "3 INQUEST",
        ],
        min_len=30,
        max_len=None,
        is_date=False,
    ),
    HtmlFieldConfig(
        "circumstances",
        para_keys=None,
        sec_keys=[
            "CIRCUMSTANCES OF THE DEATH",
            "CIRCUMSTANCES OF DEATH",
            "CIRCUMSTANCES OF",
        ],
        rem_strs=[
            "CIRCUMSTANCES OF THE DEATH",
            "CIRCUMSTANCES OF DEATH",
            "CIRCUMSTANCES OF",
        ],
        min_len=30,
        max_len=None,
        is_date=False,
    ),
    HtmlFieldConfig(
        "concerns",
        para_keys=None,
        sec_keys=[
            "CORONER'S CONCERNS",
            "CORONERS CONCERNS",
            "CORONER CONCERNS",
        ],
        rem_strs=[
            "CORONER'S CONCERNS",
            "CORONERS CONCERNS",
            "CORONER CONCERNS",
        ],
        min_len=30,
        max_len=None,
        is_date=False,
    ),
]



class HtmlExtractor:
    """Utility class encapsulating HTML scraping helpers."""

    def __init__(
        self,
        cfg,
        timeout: int,
        id_pattern,
        not_found_text: str,
        verbose: bool = False,
    ) -> None:
        self.cfg = cfg
        self.timeout = timeout
        self.id_pattern = id_pattern
        self.not_found_text = not_found_text
        self.verbose = verbose

    def fetch_report_page(self, url: str) -> BeautifulSoup | None:
        """Fetch and parse a report HTML page and convert to BeautifulSoup"""
        try:
            response = self.cfg.session.get(url, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error("Failed to fetch %s; Error: %s", url, e)
            return None
        return BeautifulSoup(response.content, "html.parser")

    def extract_html_paragraph_text(self, soup: BeautifulSoup, keywords: list[str]) -> str:
        """Scan <p> tags for provided keywords"""
        for keyword in keywords:
            element = soup.find(lambda tag: tag.name == "p" and keyword in tag.get_text(), recursive=True)
            if element:
                return clean_text(element.get_text())
        return self.not_found_text

    def extract_html_section_text(self, soup: BeautifulSoup, header_keywords: list[str]) -> str:
        """Look for section headers in <strong> tags"""
        for strong_tag in soup.find_all("strong"):
            header_text = strong_tag.get_text(strip=True)
            if any(keyword.lower() in header_text.lower() for keyword in header_keywords):
                content_parts: list[str] = []
                for sibling in strong_tag.next_siblings:
                    if isinstance(sibling, str):
                        text = sibling.strip()
                        if text:
                            content_parts.append(text)
                    else:
                        text = sibling.get_text(separator=" ", strip=True)
                        if text:
                            content_parts.append(text)
                if content_parts:
                    return clean_text(" ".join(content_parts))
        return self.not_found_text

    def extract_fields_from_html(
        self,
        soup: BeautifulSoup,
        fields: Dict[str, str],
        include_flags: Dict[str, bool],
    ) -> None:
        """Populate *fields* dict with data extracted from HTML."""
        # Iterate through configured fields
        for cfg in self.cfg.html_fields:
            if not include_flags.get(cfg.key, False):
                continue
            
            if cfg.sec_keys:
                raw = self.extract_html_section_text(soup, cfg.sec_keys)
            else:
                raw = self.extract_html_paragraph_text(soup, cfg.para_keys or [])
                
            # Special handling for the report reference number
            # ...which requires unique format validation
            if cfg.key == "id" and pd.notna(raw):
                m = self.id_pattern.search(raw)
                fields["id"] = m.group(1) if m else self.not_found_text
            else:
                fields[cfg.key] = process_extracted_field(
                    raw,
                    cfg.rem_strs,
                    self.not_found_text,
                    min_len=cfg.min_len,
                    max_len=cfg.max_len,
                    is_date=cfg.is_date,
                    verbose=self.verbose,
                )