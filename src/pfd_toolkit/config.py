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
from typing import Any, Dict, List, Tuple, Optional
import re

import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry

logger = logging.getLogger(__name__)


# --------------------------------------------------------------------------- #
# Package-wide constants
# --------------------------------------------------------------------------- #
class GeneralConfig:
    """Constants that may be reused by other modules in the toolkit."""

    # Flag for missing data
    NOT_FOUND_TEXT: str = "N/A: Not found"

    # Column names used across every DataFrame we emit
    COL_URL = "URL"
    COL_ID = "ID"
    COL_DATE = "Date"
    COL_CORONER_NAME = "CoronerName"
    COL_AREA = "Area"
    COL_RECEIVER = "Receiver"
    COL_INVESTIGATION = "InvestigationAndInquest"
    COL_CIRCUMSTANCES = "CircumstancesOfDeath"
    COL_CONCERNS = "MattersOfConcern"
    COL_DATE_SCRAPED = "DateScraped"
    
    ID_PATTERN = re.compile(r'(\d{4}-\d{4})')


# --------------------------------------------------------------------------- #
# HTML & PDF extraction logic row types
# The _FIELD_HTML_CONFIG and _FIELD_PDF_CONFIG attributes in ScraperConfig
# refer to these
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class HtmlFieldConfig:
    """
    Defines how to extract a single field from HTML:
      key       - the internal name for this field (matches dict key and DataFrame column)
      para_keys - list of keywords to look for in <p> tags (if present)
      sec_keys  - list of keywords in <strong> or header tags to define a section (if para_keys is None)
      rem_strs  - substrings to strip from the raw text before cleaning
      min_len   - minimum acceptable length after cleaning (None = no minimum)
      max_len   - maximum acceptable length after cleaning (None = no maximum)
      is_date   - whether to run the cleaned text through the date normaliser
    """
    key:       str
    para_keys: Optional[List[str]]
    sec_keys:  Optional[List[str]]
    rem_strs:  List[str]
    min_len:   Optional[int]
    max_len:   Optional[int]
    is_date:   bool


@dataclass(frozen=True)
class PdfSectionConfig:
    """
    Defines how to extract a single field from PDF text fallback:
      key        – the internal name for this field (matches dict key and DataFrame column)
      start_keys – list of phrases that mark the start of the desired section
      end_keys   – list of phrases that mark the end of the desired section
      rem_strs   – substrings to strip from the extracted raw section before cleaning
      min_len    – minimum acceptable length after cleaning (None = no minimum)
      max_len    – maximum acceptable length after cleaning (None = no maximum)
    """
    key:        str
    start_keys: List[str]
    end_keys:   List[str]
    rem_strs:   List[str]
    min_len:    Optional[int]
    max_len:    Optional[int]



# --------------------------------------------------------------------------- #
# Scraper-specific configuration & helpers
# --------------------------------------------------------------------------- #
@dataclass
class ScraperConfig:
    """
    All knobs that influence network I/O *and* the static strings
    PFDScraper relies on (category templates, LLM keys & prompts).
    """

    max_workers:    int                    = 10
    max_requests:   int                    = 5
    delay_range:    Tuple[float, float]    = (1.0, 2.0)
    timeout:        int                    = 60
    retries_total:  int                    = 30
    retries_connect:int                    = 10
    backoff_factor: float                  = 1.0
    status_forcelist: Tuple[int, ...]      = (429, 502, 503, 504)

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
                 "&before-day={before_day}&before-month={before_month}&before-year={before_year}"
    }
    
    
    # HTML extraction logic config
    html_fields: List[HtmlFieldConfig] = field(default_factory=lambda: [
        HtmlFieldConfig("id",
                        para_keys=["Ref:"], sec_keys=None,
                        rem_strs=["Ref:"], min_len=None, max_len=None,
                        is_date=False),
        HtmlFieldConfig("date",
                        para_keys=["Date of report:"], sec_keys=None,
                        rem_strs=["Date of report:"], min_len=None, max_len=None,
                        is_date=True),
        HtmlFieldConfig("receiver",
                        para_keys=["This report is being sent to:", "Sent to:"], sec_keys=None,
                        rem_strs=["This report is being sent to:", "Sent to:", "TO:"], min_len=5, max_len=20,
                        is_date=False),
        HtmlFieldConfig("coroner",
                        para_keys=["Coroners name:", "Coroner name:", "Coroner's name:"], sec_keys=None,
                        rem_strs=["Coroners name:", "Coroner name:", "Coroner's name:"], min_len=5, max_len=20,
                        is_date=False),
        HtmlFieldConfig("area",
                        para_keys=["Coroners Area:", "Coroner Area:", "Coroner's Area:"], sec_keys=None,
                        rem_strs=["Coroners Area:", "Coroner Area:", "Coroner's Area:"], min_len=4, max_len=40,
                        is_date=False),
        HtmlFieldConfig("investigation",
                        para_keys=None,
                        sec_keys=["INVESTIGATION and INQUEST", "INVESTIGATION & INQUEST", "3 INQUEST"],
                        rem_strs=["INVESTIGATION and INQUEST", "INVESTIGATION & INQUEST", "3 INQUEST"],
                        min_len=30, max_len=None, is_date=False),
        HtmlFieldConfig("circumstances",
                        para_keys=None,
                        sec_keys=["CIRCUMSTANCES OF THE DEATH", "CIRCUMSTANCES OF DEATH", "CIRCUMSTANCES OF"],
                        rem_strs=["CIRCUMSTANCES OF THE DEATH", "CIRCUMSTANCES OF DEATH", "CIRCUMSTANCES OF"],
                        min_len=30, max_len=None, is_date=False),
        HtmlFieldConfig("concerns",
                        para_keys=None,
                        sec_keys=["CORONER'S CONCERNS", "CORONERS CONCERNS", "CORONER CONCERNS"],
                        rem_strs=["CORONER'S CONCERNS", "CORONERS CONCERNS", "CORONER CONCERNS"],
                        min_len=30, max_len=None, is_date=False),
    ])

    # PDF extraction logic config
    pdf_sections: List[PdfSectionConfig] = field(default_factory=lambda: [
        PdfSectionConfig("coroner",
                         start_keys=["I am", "CORONER"],
                         end_keys=["CORONER'S LEGAL POWERS", "paragraph 7"],
                         rem_strs=["I am", "CORONER", "CORONER'S LEGAL POWERS", "paragraph 7"],
                         min_len=5, max_len=20),
        PdfSectionConfig("area",
                         start_keys=["area of"],
                         end_keys=["LEGAL POWERS", "LEGAL POWER", "paragraph 7"],
                         rem_strs=["area of", "CORONER'S", "CORONER", "CORONERS", "paragraph 7"],
                         min_len=4, max_len=40),
        PdfSectionConfig("receiver",
                         start_keys=[" SENT ", "SENT TO:"],
                         end_keys=["CORONER", "CIRCUMSTANCES"],
                         rem_strs=["TO:"],
                         min_len=5, max_len=None),
        PdfSectionConfig("investigation",
                         start_keys=["INVESTIGATION and INQUEST", "3 INQUEST"],
                         end_keys=["CIRCUMSTANCES"],
                         rem_strs=[], min_len=30, max_len=None),
        PdfSectionConfig("circumstances",
                         start_keys=["CIRCUMSTANCES OF DEATH", "CIRCUMSTANCES OF THE DEATH"],
                         end_keys=["CORONER'S CONCERNS", "as follows"],
                         rem_strs=[], min_len=30, max_len=None),
        PdfSectionConfig("concerns",
                         start_keys=["CORONER'S CONCERNS", "as follows"],
                         end_keys=["ACTION SHOULD BE TAKEN"],
                         rem_strs=[], min_len=30, max_len=None),
    ])


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
            raise ValueError(f"Unknown category '{category}'. Valid options are: {valid}") from exc

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