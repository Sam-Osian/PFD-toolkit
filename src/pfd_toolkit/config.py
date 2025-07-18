"""
Central configuration for pfd_toolkit. This module contains two classes:

* `GeneralConfig` – constants that are useful package-wide
* `ScraperConfig` – network, retry, throttling and LLM-prompt settings
  that the `PFDScraper` class will import and use internally

'Why does this exist?'
The intention is to keep the other modules as lean as possible to improve readability
and maintainability.

"""

from __future__ import annotations

import logging
import random
import threading
import time
from dataclasses import dataclass, field
from typing import Dict, List, Tuple
import re

import pandas as pd
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from .scraper.html_extractor import HtmlFieldConfig, DEFAULT_HTML_FIELDS
from .scraper.pdf_extractor import PdfSectionConfig, DEFAULT_PDF_SECTIONS

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Package-wide constants
# --------------------------------------------------------------------------- #
class GeneralConfig:
    """Constants that may be reused by other modules in the toolkit."""

    # Flag for missing data
    NOT_FOUND_TEXT: object = pd.NA

    # Column names used across every DataFrame we emit
    COL_URL = "url"
    COL_ID = "id"
    COL_DATE = "date"
    COL_CORONER_NAME = "coroner"
    COL_AREA = "area"
    COL_RECEIVER = "receiver"
    COL_INVESTIGATION = "investigation"
    COL_CIRCUMSTANCES = "circumstances"
    COL_CONCERNS = "concerns"
    COL_DATE_SCRAPED = "date_scraped"

    ID_PATTERN = re.compile(r"(\d{4}-\d{4})")


# --------------------------------------------------------------------------- #
# HTML & PDF extraction logic row types
# Configuration classes live in pfd_toolkit.scraper.html_extractor and
# pfd_toolkit.scraper.pdf_extractor
# --------------------------------------------------------------------------- #

# --------------------------------------------------------------------------- #
# Scraper-specific configuration & helpers
# --------------------------------------------------------------------------- #
@dataclass
class ScraperConfig:
    """
    All knobs that influence network I/O *and* the static strings
    PFDScraper relies on (category templates, LLM keys & prompts).
    """

    max_workers: int = 10
    max_requests: int = 5
    delay_range: Tuple[float, float] = (1.0, 2.0)
    timeout: int = 60
    retries_total: int = 30
    retries_connect: int = 10
    backoff_factor: float = 1.0
    status_forcelist: Tuple[int, ...] = (429, 502, 503, 504)

    # Static scraper strings
    # URL templates for every judiciary.uk category
    CATEGORY_TEMPLATES = {
        "all": "https://www.judiciary.uk/page/{page}/?s&pfd_report_type&post_type=pfd&order=relevance"
        "&after-day={after_day}&after-month={after_month}&after-year={after_year}"
        "&before-day={before_day}&before-month={before_month}&before-year={before_year}",
        "suicide": "https://www.judiciary.uk/page/{page}/?s&pfd_report_type=suicide-from-2015&post_type=pfd&order=relevance"
        "&after-day={after_day}&after-month={after_month}&after-year={after_year}"
        "&before-day={before_day}&before-month={before_month}&before-year={before_year}",
        "accident_work_safety": "https://www.judiciary.uk/page/{page}/?s&pfd_report_type=accident-at-work-and-health-and-safety-related-deaths&post_type=pfd&order=relevance"
        "&after-day={after_day}&after-month={after_month}&after-year={after_year}"
        "&before-day={before_day}&before-month={before_month}&before-year={before_year}",
        "alcohol_drug_medication": "https://www.judiciary.uk/page/{page}/?s&pfd_report_type=alcohol-drug-and-medication-related-deaths&post_type=pfd&order=relevance"
        "&after-day={after_day}&after-month={after_month}&after-year={after_year}"
        "&before-day={before_day}&before-month={before_month}&before-year={before_year}",
        "care_home": "https://www.judiciary.uk/page/{page}/?s&pfd_report_type=care-home-health-related-deaths&post_type=pfd&order=relevance"
        "&after-day={after_day}&after-month={after_month}&after-year={after_year}"
        "&before-day={before_day}&before-month={before_month}&before-year={before_year}",
        "child_death": "https://www.judiciary.uk/page/{page}/?s&pfd_report_type=child-death-from-2015&post_type=pfd&order=relevance"
        "&after-day={after_day}&after-month={after_month}&after-year={after_year}"
        "&before-day={before_day}&before-month={before_month}&before-year={before_year}",
        "community_health_emergency": "https://www.judiciary.uk/page/{page}/?s&pfd_report_type=community-health-care-and-emergency-services-related-deaths&post_type=pfd&order=relevance"
        "&after-day={after_day}&after-month={after_month}&after-year={after_year}"
        "&before-day={before_day}&before-month={before_month}&before-year={before_year}",
        "emergency_services": "https://www.judiciary.uk/page/{page}/?s&pfd_report_type=emergency-services-related-deaths-2019-onwards&post_type=pfd&order=relevance"
        "&after-day={after_day}&after-month={after_month}&after-year={after_year}"
        "&before-day={before_day}&before-month={before_month}&before-year={before_year}",
        "hospital_deaths": "https://www.judiciary.uk/page/{page}/?s&pfd_report_type=hospital-death-clinical-procedures-and-medical-management-related-deaths&post_type=pfd&order=relevance"
        "&after-day={after_day}&after-month={after_month}&after-year={after_year}"
        "&before-day={before_day}&before-month={before_month}&before-year={before_year}",
        "mental_health": "https://www.judiciary.uk/page/{page}/?s&pfd_report_type=mental-health-related-deaths&post_type=pfd&order=relevance"
        "&after-day={after_day}&after-month={after_month}&after-year={after_year}"
        "&before-day={before_day}&before-month={before_month}&before-year={before_year}",
        "police": "https://www.judiciary.uk/page/{page}/?s&pfd_report_type=police-related-deaths&post_type=pfd&order=relevance"
        "&after-day={after_day}&after-month={after_month}&after-year={after_year}"
        "&before-day={before_day}&before-month={before_month}&before-year={before_year}",
        "product": "https://www.judiciary.uk/page/{page}/?s&pfd_report_type=product-related-deaths&post_type=pfd&order=relevance"
        "&after-day={after_day}&after-month={after_month}&after-year={after_year}"
        "&before-day={before_day}&before-month={before_month}&before-year={before_year}",
        "railway": "https://www.judiciary.uk/page/{page}/?s&pfd_report_type=railway-related-deaths&post_type=pfd&order=relevance"
        "&after-day={after_day}&after-month={after_month}&after-year={after_year}"
        "&before-day={before_day}&before-month={before_month}&before-year={before_year}",
        "road": "https://www.judiciary.uk/page/{page}/?s&pfd_report_type=road-highways-safety-related-deaths&post_type=pfd&order=relevance"
        "&after-day={after_day}&after-month={after_month}&after-year={after_year}"
        "&before-day={before_day}&before-month={before_month}&before-year={before_year}",
        "service_personnel": "https://www.judiciary.uk/page/{page}/?s&pfd_report_type=service-personnel-related-deaths&post_type=pfd&order=relevance"
        "&after-day={after_day}&after-month={after_month}&after-year={after_year}"
        "&before-day={before_day}&before-month={before_month}&before-year={before_year}",
        "custody": "https://www.judiciary.uk/page/{page}/?s&pfd_report_type=state-custody-related-deaths&post_type=pfd&order=relevance"
        "&after-day={after_day}&after-month={after_month}&after-year={after_year}"
        "&before-day={before_day}&before-month={before_month}&before-year={before_year}",
        "wales": "https://www.judiciary.uk/page/{page}/?s&pfd_report_type=wales-prevention-of-future-deaths-reports-2019-onwards&post_type=pfd&order=relevance"
        "&after-day={after_day}&after-month={after_month}&after-year={after_year}"
        "&before-day={before_day}&before-month={before_month}&before-year={before_year}",
        "other": "https://www.judiciary.uk/page/{page}/?s&pfd_report_type=other-related-deaths&post_type=pfd&order=relevance"
        "&after-day={after_day}&after-month={after_month}&after-year={after_year}"
        "&before-day={before_day}&before-month={before_month}&before-year={before_year}",
    }

    # HTML extraction logic config
    html_fields: List[HtmlFieldConfig] = field(default_factory=lambda: DEFAULT_HTML_FIELDS.copy())

    # PDF extraction logic config
    pdf_sections: List[PdfSectionConfig] = field(default_factory=lambda: DEFAULT_PDF_SECTIONS.copy())

    # Keys sent to / returned from the LLM
    LLM_KEY_DATE: str = "date of report"
    LLM_KEY_CORONER: str = "coroner's name"
    LLM_KEY_AREA: str = "area"
    LLM_KEY_RECEIVER: str = "receiver"
    LLM_KEY_INVESTIGATION: str = "investigation and inquest"
    LLM_KEY_CIRCUMSTANCES: str = "circumstances of death"
    LLM_KEY_CONCERNS: str = "coroner's concerns"

    # Default field-specific guidance prompts for LLM text extraction
    LLM_PROMPTS: Dict[str, str] = field(
        default_factory=lambda: {
            "date of report": "[Date of the report, not the death]",
            "coroner's name": "[Name of the coroner. Provide the name only.]",
            "area": "[Area/location of the Coroner. Provide the location itself only.]",
            "receiver": "[Name or names of the recipient(s) as provided in the report.]",
            "investigation and inquest": "[The text from the Investigation/Inquest section.]",
            "circumstances of death": "[The text from the Circumstances of Death section.]",
            "coroner's concerns": "[The text from the Coroner's Concerns section.]",
        }
    )

    # Runtime attributes
    session: requests.Session = field(init=False, repr=False)
    domain_semaphore: threading.Semaphore = field(init=False, repr=False)

    # Build the requests session and semaphore
    def __post_init__(self) -> None:
        self.session = self._build_session()
        self.domain_semaphore = threading.Semaphore(self.max_requests)

    # ----------------------------------------------------------------------- #
    # Helpers
    # ----------------------------------------------------------------------- #
    def url_template(self, category: str) -> str:
        """Return the formatted URL template for a category (case-insensitive)."""
        try:
            return self.CATEGORY_TEMPLATES[category.lower()]
        except KeyError as exc:
            valid = ", ".join(sorted(self.CATEGORY_TEMPLATES))
            raise ValueError(
                f"Unknown category '{category}'. Valid options are: {valid}"
            ) from exc

    def apply_random_delay(self) -> None:
        """Sleep for a random interval, honouring the configured delay range."""
        low, high = self.delay_range or (0.0, 0.0)
        if high:  # (0, 0) disables waiting entirely
            time.sleep(random.uniform(low, high))

    def _build_session(self) -> requests.Session:
        """Create and return a configured `requests.Session` object."""
        session = requests.Session()
        retries = Retry(
            total=self.retries_total,
            connect=self.retries_connect,
            backoff_factor=self.backoff_factor,
            status_forcelist=self.status_forcelist,
        )
        adapter = HTTPAdapter(
            max_retries=retries,
            pool_connections=100,
            pool_maxsize=100,
        )
        session.mount("http://", adapter)
        session.mount("https://", adapter)
        return session
