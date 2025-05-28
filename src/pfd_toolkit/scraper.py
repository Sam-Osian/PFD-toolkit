from __future__ import annotations

from typing import Dict, List, Tuple, Any
import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import pymupdf
import pandas as pd
import re
from dateutil import parser as date_parser
from concurrent.futures import ThreadPoolExecutor, as_completed
from urllib.parse import urlparse, unquote
from io import BytesIO
import os
import time
import random
import threading
from datetime import datetime
from tqdm.auto import tqdm 

# -----------------------------------------------------------------------------
# Logging Configuration:
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__) 

class PFDScraper:
    """Scrape UK “Prevention of Future Death” (PFD) reports into a
    :class:`pandas.DataFrame`.

    The extractor runs in three cascading layers
    (`html → pdf → llm`), each independently switchable.

    1. **HTML scrape** – parse metadata and rich sections directly from
       the web page.
    2. **PDF fallback** – download the attached PDF and extract text with
       *PyMuPDF* for any missing fields.
    3. **LLM fallback** – delegate unresolved gaps to a Large Language
       Model supplied via *llm*.

    All three layers are optional and independently switchable.

    Parameters
    ----------
    llm : LLM | None
        Client implementing ``_call_llm_fallback()``; required only when
        *llm_fallback* is *True*.
    category : str
        Judiciary category slug (e.g. ``"suicide"``, ``"hospital_deaths"``)
        or ``"all"``.
    start_date : str
        Inclusive lower bound for the **report date** in the ``YYYY-MM-DD``
        format.
    end_date : str
        Inclusive upper bound for the **report date** in the ``YYYY-MM-DD``
        format.
    max_workers : int
        Thread-pool size for concurrent scraping.
    max_requests : int
        Maximum simultaneous requests per host (enforced with a semaphore).
    delay_range : tuple[float, float] | None
        Random delay *(seconds)* before every request.  
        Use ``None`` to disable (not recommended).
    timeout : int
        Per-request timeout in seconds.
    html_scraping, pdf_fallback, llm_fallback : bool
        Toggles for the three extraction layers.
    include_* : bool
        Flags that control which columns appear in the output DataFrame.
    verbose : bool
        Emit debug-level logs when *True*.

    Attributes
    ----------
    reports : pandas.DataFrame | None
        Cached result of the last call to :py:meth:`scrape_reports`
        or :py:meth:`top_up`.
    report_links : list[str]
        URLs discovered by :py:meth:`get_report_links`.
    NOT_FOUND_TEXT : str
        Placeholder value set when a field cannot be extracted.

    Examples
    --------
    >>> from pfd_toolkit import PFDScraper
    >>> scraper = PFDScraper(category="suicide",
    ...                      start_date="2020-01-01",
    ...                      end_date="2022-12-31",
    ...                      llm_fallback=True,
    ...                      llm=my_llm_client)         # Configured in LLM class
    >>> df = scraper.scrape_reports()          # full scrape
    >>> newer_df = scraper.top_up(df)             # later "top-up"
    >>> added_llm_df = scraper.run_llm_fallback(df)   # apply LLM retro-actively
    """

    # Constants for reused strings and keys to ensure consistency and avoid typos
    NOT_FOUND_TEXT = "N/A: Not found"  # Standard text for missing data fields

    # DataFrame column names
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

    # Keys used for LLM interaction when requesting missing fields
    LLM_KEY_DATE = "date of report"
    LLM_KEY_CORONER = "coroner's name"
    LLM_KEY_AREA = "area"
    LLM_KEY_RECEIVER = "receiver"
    LLM_KEY_INVESTIGATION = "investigation and inquest"
    LLM_KEY_CIRCUMSTANCES = "circumstances of death"
    LLM_KEY_CONCERNS = "coroner's concerns"

    # URL templates for different PFD categories on the judiciary.uk website
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

    def __init__(
        self,
        llm: "LLM" = None,
        # Web page and search criteria
        category: str = 'all',
        start_date: str = "2000-01-01",
        end_date: str = "2050-01-01",
        # Threading and HTTP request configuration
        max_workers: int = 10,
        max_requests: int = 5,
        delay_range: tuple[int | float, int | float] | None = (1, 2),
        timeout: int = 60,
        # Scraping strategy configuration
        html_scraping: bool = True,
        pdf_fallback: bool = True,
        llm_fallback: bool = False,
        # Output DataFrame column inclusion flags
        include_url: bool = True,
        include_id: bool = True,
        include_date: bool = True,
        include_coroner: bool = True,
        include_area: bool = True,
        include_receiver: bool = True,
        include_investigation: bool = True,
        include_circumstances: bool = True,
        include_concerns: bool = True,
        include_time_stamp: bool = False,
        verbose: bool = False
    ) -> None:
        
        self.category = category.lower()
        
        # Parse date strings into datetime objects
        self.start_date = date_parser.parse(start_date)
        self.end_date = date_parser.parse(end_date)
        
        # Store date components for formatting into search URLs
        self.date_params = {
            "after_day": self.start_date.day,
            "after_month": self.start_date.month,
            "after_year": self.start_date.year,
            "before_day": self.end_date.day,
            "before_month": self.end_date.month,
            "before_year": self.end_date.year,
        }
        
        # Hardcode in always starting from page 1
        self.start_page = 1
        
        # Store threading and request parameters
        self.max_workers = max_workers
        self.max_requests = max_requests
        self.delay_range = delay_range
        self.timeout = timeout
        
        # Store scraping strategy flags
        self.html_scraping = html_scraping
        self.pdf_fallback = pdf_fallback
        self.llm_fallback = llm_fallback
        self.llm = llm
        
        # Store output column inclusion flags
        self.include_url = include_url
        self.include_id = include_id
        self.include_date = include_date
        self.include_coroner = include_coroner
        self.include_area = include_area
        self.include_receiver = include_receiver
        self.include_investigation = include_investigation
        self.include_circumstances = include_circumstances
        self.include_concerns = include_concerns
        self.include_time_stamp = include_time_stamp
        
        self.verbose = verbose
        
        # Semaphore to limit concurrent requests per domain
        self.domain_semaphore = threading.Semaphore(self.max_requests)
        
        # Initialise storage for results and links
        self.reports: pd.DataFrame | None = None
        self.report_links: list[str] = []
        self._last_pdf_bytes: bytes | None = None
        
        # Store LLM model name if LLM client is provided
        self.llm_model = self.llm.model if self.llm else "None"
        
        # Configure url template
        self.page_template: str = ""
        self._configure_url_template()
        
        # Normalise delay_range if set to 0 or None
        if self.delay_range is None or self.delay_range == 0:
            self.delay_range = (0, 0)
        
        # Validate param
        self._validate_init_params()
        self._warn_if_suboptimal_config()
        
        # Set up requests Session with retry logic
        self._configure_session()
        
        # Pre-compile regex for extracting report IDs (e.g. "2025-0296")
        self._id_pattern = re.compile(r'(\d{4}-\d{4})')
        
        # Configuration for dynamically building the list of required columns in top_up()
        self._COLUMN_CONFIG: List[Tuple[bool, str]] = [
            (self.include_url, self.COL_URL),
            (self.include_id, self.COL_ID),
            (self.include_date, self.COL_DATE),
            (self.include_coroner, self.COL_CORONER_NAME),
            (self.include_area, self.COL_AREA),
            (self.include_receiver, self.COL_RECEIVER),
            (self.include_investigation, self.COL_INVESTIGATION),
            (self.include_circumstances, self.COL_CIRCUMSTANCES),
            (self.include_concerns, self.COL_CONCERNS),
            (self.include_time_stamp, self.COL_DATE_SCRAPED),
        ]
        
        # Configuration for identifying missing fields for LLM fallback
        self._LLM_FIELD_CONFIG: List[Tuple[bool, str, str, str]] = [
            (self.include_date, self.COL_DATE, self.LLM_KEY_DATE, "[Date of the report, not the death]"),
            (self.include_coroner, self.COL_CORONER_NAME, self.LLM_KEY_CORONER, "[Name of the coroner. Provide the name only.]"),
            (self.include_area, self.COL_AREA, self.LLM_KEY_AREA, "[Area/location of the Coroner. Provide the location itself only.]"),
            (self.include_receiver, self.COL_RECEIVER, self.LLM_KEY_RECEIVER, "[Name or names of the recipient(s) as provided in the report.]"),
            (self.include_investigation, self.COL_INVESTIGATION, self.LLM_KEY_INVESTIGATION, "[The text from the Investigation/Inquest section.]"),
            (self.include_circumstances, self.COL_CIRCUMSTANCES, self.LLM_KEY_CIRCUMSTANCES, "[The text from the Circumstances of Death section.]"),
            (self.include_concerns, self.COL_CONCERNS, self.LLM_KEY_CONCERNS, "[The text from the Coroner's Concerns section.]"),
        ]
        
        # Mapping from LLM response keys back to DataFrame column names
        self._LLM_TO_DF_MAPPING: Dict[str, str] = {
            self.LLM_KEY_DATE: self.COL_DATE,
            self.LLM_KEY_CORONER: self.COL_CORONER_NAME,
            self.LLM_KEY_AREA: self.COL_AREA,
            self.LLM_KEY_RECEIVER: self.COL_RECEIVER,
            self.LLM_KEY_INVESTIGATION: self.COL_INVESTIGATION,
            self.LLM_KEY_CIRCUMSTANCES: self.COL_CIRCUMSTANCES,
            self.LLM_KEY_CONCERNS: self.COL_CONCERNS,
        }

    # -----------------------------------------------------------------------------
    # Initialisation and configuration helper methods
    # -----------------------------------------------------------------------------
    def _configure_url_template(self) -> None:
        """Configure the search page URL template based on the selected category."""
        if self.category in self.CATEGORY_TEMPLATES:
            self.page_template = self.CATEGORY_TEMPLATES[self.category]
        else:
            valid_options = ", ".join(sorted(self.CATEGORY_TEMPLATES.keys()))
            raise ValueError(f"Unknown category '{self.category}'. Valid options are: {valid_options}")

    def _validate_init_params(self) -> None:
        """Validate initialisation parameters and raise errors for invalid configs."""
        if self.start_date > self.end_date:
            raise ValueError("start_date must be before end_date.")
        if self.llm_fallback and not self.llm:
            raise ValueError("LLM Client must be provided if LLM fallback is enabled. \nPlease create an instance of the LLM class and pass this in the llm parameter. \nGet an API key from https://platform.openai.com/.")
        if self.max_workers <= 0:
            raise ValueError("max_workers must be a positive integer.")
        if self.max_requests <= 0:
            raise ValueError("max_requests must be a positive integer.")
        if not isinstance(self.delay_range, tuple) or len(self.delay_range) != 2 or not all(isinstance(i, (int, float)) for i in self.delay_range):
            raise ValueError("delay_range must be a tuple of two numbers (int or float) - e.g. (1, 2) or (1.5, 2.5). If you are attempting to disable delays, set to (0,0).")
        if self.delay_range[1] < self.delay_range[0]:
            raise ValueError("Upper bound of delay_range must be greater than or equal to lower bound.")
        if not (self.html_scraping or self.pdf_fallback or self.llm_fallback):
            raise ValueError("At least one of 'html_scraping', 'pdf_fallback', or 'llm_fallback' must be enabled.")
        if not any([self.include_id, self.include_date, self.include_coroner, self.include_area, self.include_receiver, self.include_investigation, self.include_circumstances, self.include_concerns]):
            raise ValueError("At least one field must be included in the output. Please set one or more of the following to True:\n 'include_id', 'include_date', 'include_coroner', 'include_area', 'include_receiver', 'include_investigation', 'include_circumstances', 'include_concerns'.\n")

    def _warn_if_suboptimal_config(self) -> None:
        """Log warnings for configurations that might lead to suboptimal scraping."""
        if self.html_scraping and not self.pdf_fallback and not self.llm_fallback:
            logger.warning("Only HTML scraping is enabled. \nConsider enabling .pdf or LLM fallback for more complete data extraction.\n")
        if not self.html_scraping and self.pdf_fallback and not self.llm_fallback:
            logger.warning("Only .pdf fallback is enabled. \nConsider enabling HTML scraping or LLM fallback for more complete data extraction.\n")
        if not self.html_scraping and not self.pdf_fallback and self.llm_fallback:
            logger.warning("Only LLM fallback is enabled. \nWhile this is a high-performance option, large API costs may be incurred, especially for large requests. \nConsider enabling HTML scraping or .pdf fallback for more cost-effective data extraction.\n")
        if self.max_workers > 50:
            logger.warning("max_workers is set to a high value (>50). \nDepending on your system, this may cause performance issues. It could also trigger anti-scraping measures by the host, leading to temporary or permanent IP bans. \nWe recommend setting to between 10 and 50.\n")
        if self.max_workers < 10:
            logger.warning("max_workers is set to a low value (<10). \nThis may result in slower scraping speeds. Consider increasing the value for faster performance. \nWe recommend setting to between 10 and 50.\n")
        if self.max_requests > 10:
            logger.warning("max_requests is set to a high value (>10). \nThis may trigger anti-scraping measures by the host, leading to temporary or permanent IP bans. \nWe recommend setting to between 3 and 10.\n")
        if self.max_requests < 3:
            logger.warning("max_requests is set to a low value (<3). \nThis may result in slower scraping speeds. Consider increasing the value for faster performance. \nWe recommend setting to between 3 and 10.\n")
        if self.delay_range == (0, 0):
            logger.warning("delay_range has been disabled. \nThis will disable delays between requests. This may trigger anti-scraping measures by the host, leading to temporary or permanent IP bans. \nWe recommend setting to (1,2).\n")
        elif self.delay_range[0] < 0.5 and self.delay_range[1] != 0:
            logger.warning("delay_range is set to a low value (<0.5 seconds). \nThis may trigger anti-scraping measures by the host, leading to temporary or permanent IP bans. We recommend setting to between (1, 2).\n")
        if self.delay_range[1] > 5:
            logger.warning("delay_range is set to a high value (>5 seconds). \nThis may result in slower scraping speeds. Consider decreasing the value for faster performance. We recommend setting to between (1, 2).\n")

    def _configure_session(self) -> None:
        """Initialise the requests Session with retry logic for robust HTTP requests."""
        self.session = requests.Session()
        retries = Retry(total=30, connect=10, backoff_factor=1, status_forcelist=[429, 502, 503, 504])
        adapter = HTTPAdapter(max_retries=retries, pool_connections=100, pool_maxsize=100)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)

    # -----------------------------------------------------------------------------
    # Link fetching logic: Methods for discovering report URLs from search pages
    # -----------------------------------------------------------------------------
    def _get_report_href_values(self, url: str) -> list[str]:
        """
        Parses through a **single page** of PFD search results and extracts individual report URLs via href values.
        
        Applies a random delay and uses a semaphore to limit concurrent requests.

        :param url: The URL of the search results page to scrape.
        :return: A list of href strings, each being a URL to a PFD report page.
                 Returns an empty list if the page fetch fails or no links are found.
        """
        with self.domain_semaphore:
            time.sleep(random.uniform(*self.delay_range))
            try:
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status()
                if self.verbose:
                    logger.debug(f"Fetched URL: {url} (Status: {response.status_code})")
                    
            except requests.RequestException as e:
                if self.verbose:
                    logger.error("Failed to fetch page: %s; Error: %s", url, e)
                return []
            
        soup = BeautifulSoup(response.text, 'html.parser')
        links = soup.find_all('a', class_='card__link')
        return [link.get('href') for link in links if link.get('href')]

    def get_report_links(self) -> list[str] | None:
        """Discover individual report URLs for the current query, across all pages.
        
        Iterates through _get_report_href_values (which collects URLs for a single page).

        Pagination continues until a page yields zero new links.

        Returns
        -------
        list[str] | None
            All discovered URLs, or *None* if **no** links were found for
            the given category/date window.
        """
        self.report_links = []
        page = self.start_page
        pbar = tqdm(desc="Fetching pages", unit=" page", leave=False, initial=page, position=0)
        
        while True:
            page_url = self.page_template.format(page=page, **self.date_params)
            href_values = self._get_report_href_values(page_url)
            # Break the loop when no new URLs are found 
            if not href_values:
                break
            self.report_links.extend(href_values)
            page += 1
            pbar.update(1)
            if self.verbose:
                logger.info("Scraped %d links from %s", len(href_values), page_url)
        pbar.close()
        if not self.report_links:
            return None
        
        logger.info("Total collected report links: %d", len(self.report_links))
        return self.report_links

    # -----------------------------------------------------------------------------------------
    # PUBLIC METHODS: The user interface for the scraper
    # -----------------------------------------------------------------------------------------
    def scrape_reports(self) -> pd.DataFrame:
        """Execute a full scrape with the Class configuration.

        Workflow
        --------
        1. Call :py:meth:`get_report_links`.  
        2. Extract each report in parallel via
           :py:meth:`_extract_report_info`.  
        3. Optionally invoke :py:meth:`run_llm_fallback`.  
        4. Cache the final DataFrame to :pyattr:`self.reports`.

        Returns
        -------
        pandas.DataFrame
            One row per report.  Column presence matches the *include_* flags.
            The DataFrame is empty if nothing was scraped.

        Examples
        --------
        >>> df = scraper.scrape_reports()
        >>> df.columns
        Index(['URL', 'ID', 'Date', ...], dtype='object')
        """
        # Check to see if get_report_links() has already been run; if not, run it.
        if not self.report_links:
            fetched_links = self.get_report_links()
            if fetched_links is None:
                self.reports = pd.DataFrame()
                return self.reports
            
        report_data = self._scrape_report_details(self.report_links)
        reports_df = pd.DataFrame(report_data)
        
        # Run the LLM fallback if enabled
        if self.llm_fallback and self.llm:
            reports_df = self.run_llm_fallback(reports_df if not reports_df.empty else None)
        
        # Output the timestamp of scraping completion for each report, if enabled
        if self.include_date:
            reports_df = reports_df.sort_values(by=[self.COL_DATE], ascending=False)
        self.reports = reports_df.copy()
        
        return reports_df

    def _scrape_report_details(self, urls: list[str]) -> list[dict[str, Any]]:
        """Handles the mechanics of scraping PFD reports for all given URLs using multithreading, 
        returning a list of result dicts."""
        results: list[dict[str, Any] | None] = []
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._extract_report_info, url) for url in urls]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Scraping reports", position=0, leave=False):
                results.append(future.result())
        report_dicts = [r for r in results if r is not None]
        
        return report_dicts

    def top_up(self, old_reports: pd.DataFrame | None = None, start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame | None:
        """Checks to see if there are any unscraped PFD reports within Class instance parameters.
        
        If so, it reruns the scraper and appends new reports to  
        :pyattr:`self.reports` under Class instance parameters.

        Any URL (or ID) already present in *old_reports* is skipped.
        
        Optionally, you can override the *start_date* and *end_date*
        parameters from `self`.

        Parameters
        ----------
        old_reports : pandas.DataFrame | None
            Existing DataFrame.  Defaults to :pyattr:`self.reports`.
        start_date, end_date : str | None
            Optionally override the scraper’s date window *for this call only*.

        Returns
        -------
        pandas.DataFrame | None
            Updated DataFrame if new reports were added; *None* if no new
            records were found **and** *old_reports* was *None*.

        Raises
        ------
        ValueError
            If *old_reports* lacks columns required for duplicate checks.

        Examples
        --------
        >>> updated = scraper.top_up(df, end_date="2023-01-01")
        >>> len(updated) - len(df)     # number of new reports
        3
        """
        logger.info("Attempting to 'top up' the existing reports with new data.")
        
        # Update date range for this top_up if new dates provided
        if start_date is not None or end_date is not None:
            new_start_date = date_parser.parse(start_date) if start_date is not None else self.start_date
            new_end_date = date_parser.parse(end_date) if end_date is not None else self.end_date
            if new_start_date > new_end_date:
                raise ValueError("start_date must be before end_date.")
            self.start_date = new_start_date
            self.end_date = new_end_date
            self.date_params.update({
                "after_day": self.start_date.day,
                "after_month": self.start_date.month,
                "after_year": self.start_date.year,
                "before_day": self.end_date.day,
                "before_month": self.end_date.month,
                "before_year": self.end_date.year,
            })
            
        # If provided, update provided DataFrame. Else, update the internal attribute
        base_df = old_reports if old_reports is not None else self.reports
        # Ensure base_df has required columns for duplicate checking
        if base_df is not None:
            required_columns = [col_name for include_flag, col_name in self._COLUMN_CONFIG if include_flag]
            missing_cols = [col for col in required_columns if col not in base_df.columns]
            if missing_cols:
                raise ValueError(f"Required columns missing from the provided DataFrame: {missing_cols}")
            
        # Determine unique key for identifying existing/duplicate reports: URL or ID
        if self.include_url:
            unique_key = self.COL_URL
        elif self.include_id:
            unique_key = self.COL_ID
        else:
            logger.error("No unique identifier available for duplicate checking.\nEnsure include_url or include_id was set to True in instance initialisation.")
            return None
        existing_identifiers = set(base_df[unique_key].tolist()) if base_df is not None and unique_key in base_df.columns else set()
        
        # Fetch updated list of report links within current date range
        updated_links = self.get_report_links()
        if updated_links is None:
            updated_links = []
        new_links = [link for link in updated_links if link not in existing_identifiers]
        logger.info("Top-up: %d new report(s) found; %d duplicate(s) which won't be added", len(new_links), len(updated_links) - len(new_links))
        if not new_links:
            return None if base_df is None and old_reports is None else base_df
        
        # Scrape details for new links
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            new_results = list(tqdm(executor.map(self._extract_report_info, new_links), total=len(new_links), desc="Topping up reports", position=0, leave=True))
        new_records = [record for record in new_results if record is not None]
        if new_records:
            new_df = pd.DataFrame(new_records)
            updated_reports_df = pd.concat([base_df, new_df], ignore_index=True) if base_df is not None else new_df
        else:
            updated_reports_df = base_df if base_df is not None else pd.DataFrame()
        if self.include_date:
            updated_reports_df = updated_reports_df.sort_values(by=[self.COL_DATE], ascending=False)
        self.reports = updated_reports_df.copy()
        return updated_reports_df

    def run_llm_fallback(self, reports_df: pd.DataFrame | None = None) -> pd.DataFrame:
        """Ask the LLM to fill cells still set to :pyattr:`self.NOT_FOUND_TEXT`.

        Only the missing fields requested via *include_* flags are sent to
        the model, along with the report’s PDF bytes (when available).

        Parameters
        ----------
        reports_df : pandas.DataFrame | None
            DataFrame to process.  Defaults to :pyattr:`self.reports`.

        Returns
        -------
        pandas.DataFrame
            Same shape as *reports_df*, updated in place and re-cached to
            :pyattr:`self.reports`.

        Raises
        ------
        ValueError
            If no LLM client was supplied at construction time.

        Examples
        --------
        >>> updated_df = scraper.run_llm_fallback()
        """
        # Make sure llm param is set
        if not self.llm:
            raise ValueError("LLM client (self.llm) not provided. Cannot run LLM fallback.")
        
        current_reports_df: pd.DataFrame
        if reports_df is None:
            if self.reports is None:
                raise ValueError("No scraped reports found (reports_df is None and self.reports is None). Please run scrape_reports() first or provide a DataFrame.")
            current_reports_df = self.reports.copy()
        else:
            current_reports_df = reports_df.copy()
        if current_reports_df.empty:
            logger.info("Report DataFrame is empty. Skipping LLM fallback.")
            return current_reports_df

        # --- Helper function to process a single row for LLM fallback ---
        def _process_row(idx: int, row_data: pd.Series) -> tuple[int, dict[str, str]]:
            """Identifies missing fields for a given row and calls LLM for them."""
            missing_fields: dict[str, str] = {}
            
            # Build dictionary of fields needing LLM extraction based on _LLM_FIELD_CONFIG
            for include_flag, df_col_name, llm_key, llm_prompt in self._LLM_FIELD_CONFIG:
                if include_flag and row_data.get(df_col_name, "") == self.NOT_FOUND_TEXT:
                    missing_fields[llm_key] = llm_prompt
            if not missing_fields:
                return idx, {}
            pdf_bytes: bytes | None = None
            report_url = row_data.get(self.COL_URL)
            if report_url:
                pdf_bytes = self._fetch_pdf_bytes(report_url)
            if not pdf_bytes and self.verbose:
                logger.warning(f"Could not obtain PDF bytes for URL {report_url} (row {idx}). LLM fallback for this row might be impaired.")
                
            # Call the LLM client's fallback method
            updates = self.llm._call_llm_fallback(
                pdf_bytes=pdf_bytes,
                missing_fields=missing_fields,
                report_url=str(report_url) if report_url else "N/A",
                verbose=self.verbose,
                tqdm_extra_kwargs={"disable": True}
            )
            return idx, updates if updates else {}

        # --- Process rows for LLM fallback  ---
        results_map: Dict[int, Dict[str, str]] = {}
        use_parallel = self.llm and hasattr(self.llm, 'max_workers') and self.llm.max_workers > 1
        if use_parallel:
            with ThreadPoolExecutor(max_workers=self.llm.max_workers) as executor:
                future_to_idx = {
                    executor.submit(_process_row, idx, row_series): idx
                    for idx, row_series in current_reports_df.iterrows()
                }
                for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc="LLM fallback (parallel processing)", position=0, leave=True):
                    idx = future_to_idx[future]
                    try:
                        _, updates = future.result()
                        results_map[idx] = updates
                    except Exception as e:
                        logger.error(f"LLM fallback failed for row index {idx}: {e}")
        else:
            for idx, row_series in tqdm(current_reports_df.iterrows(), total=len(current_reports_df), desc="LLM fallback (sequential processing)", position=0, leave=True):
                try:
                    _, updates = _process_row(idx, row_series)
                    results_map[idx] = updates
                except Exception as e:
                    logger.error(f"LLM fallback failed for row index {idx}: {e}")
                    
        # Apply updates from LLM to the DataFrame
        for idx, updates_dict in results_map.items():
            if not updates_dict:
                continue
            for llm_key, value_from_llm in updates_dict.items():
                df_col_name = self._LLM_TO_DF_MAPPING.get(llm_key)
                if df_col_name:
                    if llm_key == self.LLM_KEY_DATE:
                        if value_from_llm != self.NOT_FOUND_TEXT:
                            current_reports_df.at[idx, df_col_name] = self._normalise_date(value_from_llm)
                    else:
                        current_reports_df.at[idx, df_col_name] = value_from_llm
        self.reports = current_reports_df.copy()
        return current_reports_df

    # -----------------------------------------------------------------------------
    # Text processing helper functions
    # -----------------------------------------------------------------------------
    @staticmethod
    def _normalise_apostrophes(text: str) -> str:
        """
        Replaces typographic (curly) apostrophes with standard (straight) apostrophes.
        This helps in standardising text for consistent processing.

        :param text: The input string.
        :return: The string with normalised apostrophes.
        """
        return text.replace("’", "'").replace("‘", "'")

    def _clean_text(self, text: str) -> str:
        """
        Cleans text by normalising apostrophes and collapsing multiple whitespace characters
        (including newlines, tabs) into single spaces, and stripping leading/trailing whitespace.

        :param text: The input string.
        :return: The cleaned string.
        """
        normalised_apostrophes = self._normalise_apostrophes(text)
        return " ".join(normalised_apostrophes.split())

    def _normalise_date(self, raw_date_str: str) -> str:
        """
        Converts a human-readable date string to ISO-8601 format (YYYY-MM-DD),
        attempting to clean known non-date suffixes like "Ref:" before parsing.

        :param raw_date_str: The raw date string to parse.
        :return: The date string in "YYYY-MM-DD" format, or the original string if parsing fails.
        """
        text_being_processed = self._clean_text(raw_date_str).strip()
        final_text_to_parse = text_being_processed
        try:
            match = re.match(r"(.+?)(Ref[:\s]|$)", text_being_processed, re.IGNORECASE)
            if match:
                potential_date_part = match.group(1).strip()
                if potential_date_part:
                    final_text_to_parse = potential_date_part
            if not final_text_to_parse:
                if self.verbose:
                    logger.warning(f"Date string empty after trying to remove 'Ref...' from '{text_being_processed}'. Raw: '{raw_date_str}'. Keeping raw.")
                return text_being_processed
            dt = date_parser.parse(final_text_to_parse, fuzzy=True, dayfirst=True)
            return dt.strftime("%Y-%m-%d")
        except Exception as e:
            if self.verbose:
                logger.warning(
                    f"Date parse failed for raw '{raw_date_str}' (processed to '{text_being_processed}', attempted '{final_text_to_parse}') "
                    f"– keeping raw. Error: {e}"
                )
            return text_being_processed

    def _process_extracted_field(self, text: str, strings_to_remove: list[str],
                                 min_len: int | None = None, max_len: int | None = None,
                                 is_date: bool = False) -> str:
        """
        Helper function to process a raw extracted text field. It performs several steps:
        1. Returns `NOT_FOUND_TEXT` if the input is already `NOT_FOUND_TEXT`.
        2. Removes specified leading substrings (e.g., "Date of report:").
        3. Strips leading/trailing whitespace.
        4. Applies general text cleaning (apostrophes, multiple spaces via `_clean_text`).
        5. If `is_date` is True, normalises the text to "YYYY-MM-DD" format.
        6. Validates the length of the processed text against `min_len` and `max_len`.
           If length checks fail, or if text becomes empty and `min_len` > 0, returns `NOT_FOUND_TEXT`.

        :param text: The raw extracted text.
        :param strings_to_remove: A list of substrings to remove from the beginning of the text.
        :param min_len: Optional minimum length for the final text.
        :param max_len: Optional maximum length for the final text.
        :param is_date: Boolean flag indicating if the field is a date.
        :return: The processed and validated text, or `NOT_FOUND_TEXT`.
        """
        if text == self.NOT_FOUND_TEXT:
            return self.NOT_FOUND_TEXT
        processed_text = text
        for s_to_remove in strings_to_remove:
            processed_text = processed_text.replace(s_to_remove, "")
        processed_text = processed_text.strip()
        processed_text = self._clean_text(processed_text)
        if is_date:
            return self._normalise_date(processed_text)
        if not processed_text and min_len is not None and min_len > 0:
            return self.NOT_FOUND_TEXT
        if min_len is not None and len(processed_text) < min_len:
            return self.NOT_FOUND_TEXT
        if max_len is not None and len(processed_text) > max_len:
            return self.NOT_FOUND_TEXT
        return processed_text

    # -----------------------------------------------------------------------------
    # PDF and HTML content extraction logic
    # -----------------------------------------------------------------------------
    def _extract_text_from_pdf(self, pdf_url: str) -> str:
        """
        Downloads a PDF from a URL and extracts all text content from it.
        Caches the downloaded PDF bytes if LLM fallback is enabled.

        :param pdf_url: The URL of the .pdf file.
        :return: The cleaned, concatenated text from all pages of the PDF.
                 Returns `NOT_FOUND_TEXT` or a specific error message ("N/A: Source file not PDF") on failure.
        """
        parsed_url = urlparse(pdf_url)
        path = unquote(parsed_url.path)
        ext = os.path.splitext(path)[1].lower()
        if self.verbose:
            logger.debug(f"Processing .pdf {pdf_url}.")
        try:
            response = self.session.get(pdf_url, timeout=self.timeout)
            response.raise_for_status()
            file_bytes = response.content
        except requests.RequestException as e:
            logger.error("Failed to fetch file: %s; Error: %s", pdf_url, e)
            return self.NOT_FOUND_TEXT
        pdf_bytes_to_process: bytes | None = None
        if ext != ".pdf":
            logger.info("File %s is not a .pdf (extension %s). Skipping this file...", pdf_url, ext)
            return self.NOT_FOUND_TEXT
        else:
            pdf_bytes_to_process = file_bytes
        if pdf_bytes_to_process is None:
            return "N/A: Source file not PDF"
        self._last_pdf_bytes = pdf_bytes_to_process
        try:
            pdf_buffer = BytesIO(pdf_bytes_to_process)
            pdf_document = pymupdf.open(stream=pdf_buffer, filetype="pdf")
            text = "".join(page.get_text() for page in pdf_document)
            pdf_document.close()
        except Exception as e:
            logger.error("Error processing .pdf %s: %s", pdf_url, e)
            return "N/A: Source file not PDF"
        return self._clean_text(text)

    def _extract_html_paragraph_text(self, soup: BeautifulSoup, keywords: list[str]) -> str:
        """
        Extracts text from the first `<p>` (paragraph) element in HTML that contains any of the provided keywords.
        This is typically used for extracting metadata fields.

        :param soup: A BeautifulSoup object representing the parsed HTML page.
        :param keywords: A list of keywords to search for within paragraph tags.
        :return: The cleaned text content of the found paragraph, or `NOT_FOUND_TEXT`.
        """
        for keyword in keywords:
            # Find a <p> tag whose text contains the keyword.
            element = soup.find(lambda tag: tag.name == 'p' and keyword in tag.get_text(), recursive=True)
            if element:
                return self._clean_text(element.get_text())
        return self.NOT_FOUND_TEXT

    def _extract_html_section_text(self, soup: BeautifulSoup, header_keywords: list[str]) -> str:
        """
        Extracts a block of text from HTML following a header.
        It looks for a `<strong>` tag (assumed to be a section header) containing one of the `header_keywords`.
        If found, it collects text from all subsequent sibling elements until the next header or end of parent.

        :param soup: A BeautifulSoup object representing the parsed HTML page.
        :param header_keywords: A list of keyword variations to identify the section header.
        :return: The cleaned, concatenated text of the section, or `NOT_FOUND_TEXT`.
        """
        for strong_tag in soup.find_all('strong'):
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
                    return self._clean_text(" ".join(content_parts))
        return self.NOT_FOUND_TEXT

    def _extract_pdf_section(self, text: str, start_keywords: list[str], end_keywords: list[str]) -> str:
        """
        Extracts a section of text from a larger body of PDF text based on start and end keywords.
        The search for keywords is case-insensitive.

        :param text: The full text from which to extract the section.
        :param start_keywords: A list of possible keywords that mark the beginning of the section.
        :param end_keywords: A list of possible keywords that mark the end of the section.
        :return: The cleaned text of the extracted section. If no start keyword is found, or if a start
                 is found but no end keyword, it may return part of the text or `NOT_FOUND_TEXT`.
        """
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
        return self.NOT_FOUND_TEXT

    def _fetch_report_page(self, url: str) -> BeautifulSoup | None:
        """Fetch the HTML content of a report page and return a BeautifulSoup object, or None on failure."""
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error("Failed to fetch %s; Error: %s", url, e)
            return None
        return BeautifulSoup(response.content, 'html.parser')

    def _get_pdf_link(self, soup: BeautifulSoup) -> str | None:
        """Extract the first PDF link from a report's HTML page, or None if not found."""
        pdf_links = [a['href'] for a in soup.find_all('a', class_='govuk-button') if a.get('href')]
        return pdf_links[0] if pdf_links else None

    def _extract_fields_from_html(self, soup: BeautifulSoup, fields: dict[str, str]) -> None:
        """Extract configured fields from the HTML content and update the fields dict in-place."""
        if self.verbose:
            logger.debug(f"Extracting data from HTML for URL.")
        if self.include_id:
            ref_text = self._extract_html_paragraph_text(soup, ["Ref:"])
            if ref_text != self.NOT_FOUND_TEXT:
                match = self._id_pattern.search(ref_text)
                fields["id"] = match.group(1) if match else self.NOT_FOUND_TEXT
        if self.include_date:
            date_raw = self._extract_html_paragraph_text(soup, ["Date of report:"])
            fields["date"] = self._process_extracted_field(date_raw, ["Date of report:"], is_date=True)
        if self.include_receiver:
            receiver_raw = self._extract_html_paragraph_text(soup, ["This report is being sent to:", "Sent to:"])
            fields["receiver"] = self._process_extracted_field(receiver_raw,
                                                               ["This report is being sent to:", "Sent to:", "TO:"],
                                                               min_len=5, max_len=20)
        if self.include_coroner:
            coroner_raw = self._extract_html_paragraph_text(soup, ["Coroners name:", "Coroner name:", "Coroner's name:"])
            fields["coroner"] = self._process_extracted_field(coroner_raw,
                                                              ["Coroners name:", "Coroner name:", "Coroner's name:"],
                                                              min_len=5, max_len=20)
        if self.include_area:
            area_raw = self._extract_html_paragraph_text(soup, ["Coroners Area:", "Coroner Area:", "Coroner's Area:"])
            fields["area"] = self._process_extracted_field(area_raw,
                                                           ["Coroners Area:", "Coroner Area:", "Coroner's Area:"],
                                                           min_len=4, max_len=40)
        if self.include_investigation:
            investigation_raw = self._extract_html_section_text(soup, ["INVESTIGATION and INQUEST", "INVESTIGATION & INQUEST", "3 INQUEST"])
            fields["investigation"] = self._process_extracted_field(investigation_raw,
                                                                    ["INVESTIGATION and INQUEST", "INVESTIGATION & INQUEST", "3 INQUEST"],
                                                                    min_len=30)
        if self.include_circumstances:
            circumstances_raw = self._extract_html_section_text(soup, ["CIRCUMSTANCES OF THE DEATH", "CIRCUMSTANCES OF DEATH", "CIRCUMSTANCES OF"])
            fields["circumstances"] = self._process_extracted_field(circumstances_raw,
                                                                     ["CIRCUMSTANCES OF THE DEATH", "CIRCUMSTANCES OF DEATH", "CIRCUMSTANCES OF"],
                                                                     min_len=30)
        if self.include_concerns:
            concerns_raw = self._extract_html_section_text(soup, ["CORONER'S CONCERNS", "CORONERS CONCERNS", "CORONER CONCERNS"])
            fields["concerns"] = self._process_extracted_field(concerns_raw,
                                                               ["CORONER'S CONCERNS", "CORONERS CONCERNS", "CORONER CONCERNS"],
                                                               min_len=30)

    def _apply_pdf_fallback(self, pdf_text: str, fields: dict[str, str]) -> None:
        """Use PDF text to fill any missing fields in the fields dict (in-place)."""
        # Skip if no PDF content available
        if pdf_text == self.NOT_FOUND_TEXT or pdf_text == "N/A: Source file not PDF":
            return
        # Determine if any field that PDF can supply is missing
        fields_to_fill = ["coroner", "area", "receiver", "investigation", "circumstances", "concerns"]
        if not any(fields.get(key) == self.NOT_FOUND_TEXT for key in fields_to_fill):
            return
        if self.verbose:
            logger.debug("Initiating .pdf fallback because one or more fields are missing.")
        if "coroner" in fields and fields["coroner"] == self.NOT_FOUND_TEXT:
            raw = self._extract_pdf_section(pdf_text, start_keywords=["I am", "CORONER"], end_keywords=["CORONER'S LEGAL POWERS", "paragraph 7"])
            if raw != self.NOT_FOUND_TEXT:
                coroner_clean = raw.replace("I am", "").replace("CORONER'S LEGAL POWERS", "").replace("CORONER", "").replace("paragraph 7", "").strip()
                coroner_clean = self._clean_text(coroner_clean)
                fields["coroner"] = coroner_clean if coroner_clean else self.NOT_FOUND_TEXT
        if "area" in fields and fields["area"] == self.NOT_FOUND_TEXT:
            raw = self._extract_pdf_section(pdf_text, start_keywords=["area of"], end_keywords=["LEGAL POWERS", "LEGAL POWER", "paragraph 7"])
            if raw != self.NOT_FOUND_TEXT:
                area_clean = raw.replace("area of", "").replace("CORONER'S", "").replace("CORONER", "").replace("CORONERS", "").replace("paragraph 7", "").strip()
                area_clean = self._clean_text(area_clean)
                fields["area"] = area_clean if area_clean else self.NOT_FOUND_TEXT
        if "receiver" in fields and fields["receiver"] == self.NOT_FOUND_TEXT:
            raw = self._extract_pdf_section(pdf_text, start_keywords=[" SENT ", "SENT TO:"], end_keywords=["CORONER", "CIRCUMSTANCES OF THE DEATH", "CIRCUMSTANCES OF"])
            if raw != self.NOT_FOUND_TEXT:
                temp_receiver = self._clean_text(raw).replace("TO:", "").strip()
                fields["receiver"] = temp_receiver if len(temp_receiver) >= 5 else self.NOT_FOUND_TEXT
        if "investigation" in fields and fields["investigation"] == self.NOT_FOUND_TEXT:
            raw = self._extract_pdf_section(pdf_text, start_keywords=["INVESTIGATION and INQUEST", "3 INQUEST"], end_keywords=["CIRCUMSTANCES OF DEATH", "CIRCUMSTANCES OF THE DEATH", "CIRCUMSTANCES OF"])
            if raw != self.NOT_FOUND_TEXT:
                temp_invest = self._clean_text(raw)
                fields["investigation"] = temp_invest if len(temp_invest) >= 30 else self.NOT_FOUND_TEXT
        if "circumstances" in fields and fields["circumstances"] == self.NOT_FOUND_TEXT:
            raw = self._extract_pdf_section(pdf_text, start_keywords=["CIRCUMSTANCES OF DEATH", "CIRCUMSTANCES OF THE DEATH", "CIRCUMSTANCES OF"], end_keywords=["CORONER'S CONCERNS", "CORONER CONCERNS", "CORONERS CONCERNS", "as follows"])
            if raw != self.NOT_FOUND_TEXT:
                temp_circ = self._clean_text(raw)
                fields["circumstances"] = temp_circ if len(temp_circ) >= 30 else self.NOT_FOUND_TEXT
        if "concerns" in fields and fields["concerns"] == self.NOT_FOUND_TEXT:
            raw = self._extract_pdf_section(pdf_text, start_keywords=["CORONER'S CONCERNS", "as follows"], end_keywords=["ACTION SHOULD BE TAKEN"])
            if raw != self.NOT_FOUND_TEXT:
                temp_concerns = self._clean_text(raw)
                fields["concerns"] = temp_concerns if len(temp_concerns) >= 30 else self.NOT_FOUND_TEXT

    def _assemble_report(self, url: str, fields: dict[str, str]) -> dict[str, Any]:
        """Assemble a single report's data into a dictionary based on included fields."""
        report: dict[str, Any] = {}
        if self.include_url:
            report[self.COL_URL] = url
        if self.include_id:
            report[self.COL_ID] = fields.get("id", self.NOT_FOUND_TEXT)
        if self.include_date:
            report[self.COL_DATE] = fields.get("date", self.NOT_FOUND_TEXT)
        if self.include_coroner:
            report[self.COL_CORONER_NAME] = fields.get("coroner", self.NOT_FOUND_TEXT)
        if self.include_area:
            report[self.COL_AREA] = fields.get("area", self.NOT_FOUND_TEXT)
        if self.include_receiver:
            report[self.COL_RECEIVER] = fields.get("receiver", self.NOT_FOUND_TEXT)
        if self.include_investigation:
            report[self.COL_INVESTIGATION] = fields.get("investigation", self.NOT_FOUND_TEXT)
        if self.include_circumstances:
            report[self.COL_CIRCUMSTANCES] = fields.get("circumstances", self.NOT_FOUND_TEXT)
        if self.include_concerns:
            report[self.COL_CONCERNS] = fields.get("concerns", self.NOT_FOUND_TEXT)
        if self.include_time_stamp:
            report[self.COL_DATE_SCRAPED] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return report

    def _extract_report_info(self, url: str) -> dict[str, Any] | None:
        """
        Extracts metadata and section text from a single PFD report page (given by URL).

        The process involves:
          1. Fetching and parsing the HTML of the report page.
          2. Identifying and downloading the associated PDF report.
          3. Extracting all text from the PDF.
          4. If `html_scraping` is enabled, attempting to extract all configured fields from the HTML.
          5. If `pdf_fallback` is enabled and any fields are still missing, attempting to extract them
             from the PDF text using keyword-based section extraction.
          6. (LLM fallback is handled by `run_llm_fallback` method if enabled globally).

        :param url: The URL of the PFD report's HTML page.
        :return: A dictionary containing the extracted report information.
                 Returns None if the page fetch fails or essential components (like PDF link) are missing.
        """
        # Initialise all fields with default "not found" text
        fields: dict[str, str] = {
            "id": self.NOT_FOUND_TEXT,
            "date": self.NOT_FOUND_TEXT,
            "receiver": self.NOT_FOUND_TEXT,
            "coroner": self.NOT_FOUND_TEXT,
            "area": self.NOT_FOUND_TEXT,
            "investigation": self.NOT_FOUND_TEXT,
            "circumstances": self.NOT_FOUND_TEXT,
            "concerns": self.NOT_FOUND_TEXT
        }
        # Fetch HTML page
        soup = self._fetch_report_page(url)
        if soup is None:
            return None
        # Find PDF download link
        pdf_link = self._get_pdf_link(soup)
        if not pdf_link:
            logger.error("No .pdf links found on %s", url)
            return None
        # Download and extract PDF text
        pdf_text = self._extract_text_from_pdf(pdf_link)
        # Extract fields from HTML if enabled
        if self.html_scraping:
            self._extract_fields_from_html(soup, fields)
        # Use PDF fallback if enabled and PDF text is available
        if self.pdf_fallback and pdf_text not in (self.NOT_FOUND_TEXT, "N/A: Source file not PDF"):
            if any(fields[key] == self.NOT_FOUND_TEXT for key in ["coroner", "area", "receiver", "investigation", "circumstances", "concerns"]):
                if self.verbose:
                    logger.debug(f"Initiating .pdf fallback for URL: {url} because one or more fields are missing.")
                self._apply_pdf_fallback(pdf_text, fields)
        # Assemble result dictionary
        report = self._assemble_report(url, fields)
        return report

    def _fetch_pdf_bytes(self, report_url: str) -> bytes | None:
        """
        Fetches the PDF file content as bytes from a report's HTML page.
        It first finds the PDF download link on the HTML page, then downloads the PDF.

        :param report_url: The URL of the HTML page containing the link to the PDF report.
        :return: The PDF content as bytes, or None if fetching fails or no PDF link is found.
        """
        try:
            page_response = self.session.get(report_url, timeout=self.timeout)
            page_response.raise_for_status()
            soup = BeautifulSoup(page_response.content, 'html.parser')
            pdf_links = [a['href'] for a in soup.find_all('a', class_='govuk-button') if a.get('href')]
            if pdf_links:
                pdf_link = pdf_links[0]
                pdf_response = self.session.get(pdf_link, timeout=self.timeout)
                pdf_response.raise_for_status()
                return pdf_response.content
            else:
                logger.error("No PDF link found for report at %s", report_url)
                return None
        except Exception as e:
            logger.error("Failed to fetch PDF for report at %s: %s", report_url, e)
            return None