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
logging.basicConfig(level=logging.INFO, force=True) 

# Reduce verbosity from the 'httpx' library by setting its log level to WARNING
logging.getLogger("httpx").setLevel(logging.WARNING)

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
    >>> newer_df = scraper.top_up(df)             # later “top-up”
    >>> added_llm_df = scraper.run_llm_fallback(df)   # apply LLM retro-actively

    """

    # Constants for reused strings and keys to ensure consistency and avoid typos
    NOT_FOUND_TEXT = "N/A: Not found" # Standard text for missing data fields

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

    def __init__(
        self,
        llm: "LLM" = None,
        
        # Web page and search criteria
        category: str = 'all', # Category of PFD reports to scrape
        start_date: str = "2000-01-01", # Start date for filtering reports (YYYY-MM-DD)
        end_date: str = "2050-01-01", # End date for filtering reports (YYYY-MM-DD)
        
        # Threading and HTTP request configuration
        max_workers: int = 10, # Maximum number of concurrent threads for scraping
        max_requests: int = 5, # Maximum concurrent requests to the same domain
        delay_range: tuple[int | float, int | float] | None = (1, 2), # Min/max delay (seconds) between requests
        timeout: int = 60, # Timeout for response requests
        
        # Scraping strategy configuration
        html_scraping: bool = True, # Whether to attempt HTML-based scraping
        pdf_fallback: bool = True, # Whether to use PDF scraping as a fallback
        llm_fallback: bool = False, # Whether to use LLM as a final fallback
        
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
        include_time_stamp: bool = False, # Whether to include a 'DateScraped' timestamp
        
        verbose: bool = False # Whether to print verbose logging output.
    ) -> None:

        self.category = category.lower() # Normalise category to lowercase if user specifies otherwise
        
        # Parse date strings into datetime objects (for internal use)
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
        
        self.start_page = 1 # Pagination always starts from page 1
        
        # Store threading and request parameters
        self.max_workers = max_workers
        self.max_requests = max_requests
        self.delay_range = delay_range
        self.timeout = timeout
        
        # Store scraping strategy flags
        self.html_scraping = html_scraping
        self.pdf_fallback = pdf_fallback
        self.llm_fallback = llm_fallback
        self.llm = llm # LLM client instance

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
        
        # Semaphore to limit concurrent requests to the website per domain
        self.domain_semaphore = threading.Semaphore(self.max_requests)
        
        self.reports: pd.DataFrame | None = None # DataFrame to store scraped reports
        self.report_links: list[str] = [] # List to store URLs of individual PFD reports
        self._last_pdf_bytes: bytes | None = None # Cache for the last downloaded PDF bytes (for LLM fallback)

        self.llm_model = self.llm.model if self.llm else "None" # Store LLM model name if LLM client is provided
        
        # URL templates for different PFD categories on the judiciary.uk website
        category_templates = {
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
        self.page_template: str = "" # URL template for the selected category

        # Normalise delay_range if set to 0 or None, to (0,0) meaning no delay
        if self.delay_range is None or self.delay_range == 0: 
             self.delay_range = (0, 0)

        # -----------------------------------------------------------------------------
        # Error and Warning Handling for Initialisation Parameters
        # -----------------------------------------------------------------------------
        
        # Validate category and set the page template
        if self.category in category_templates:
            self.page_template = category_templates[self.category]
        else:
            valid_options = ", ".join(sorted(category_templates.keys()))
            raise ValueError(f"Unknown category '{self.category}'. Valid options are: {valid_options}")
        
        # Validate date range
        if self.start_date > self.end_date:
            raise ValueError("start_date must be before end_date.")
        
        # Validate LLM configuration
        if self.llm_fallback and not self.llm:
            raise ValueError("LLM Client must be provided if LLM fallback is enabled. \nPlease create an instance of the LLM class and pass this in the llm parameter. \nGet an API key from https://platform.openai.com/.")
        
        # Validate worker and request limits
        if self.max_workers <= 0:
            raise ValueError("max_workers must be a positive integer.")
        if self.max_requests <= 0:
            raise ValueError("max_requests must be a positive integer.")
        
        # Validate delay_range format and values
        if not isinstance(self.delay_range, tuple) or len(self.delay_range) != 2 or not all(isinstance(i, (int, float)) for i in self.delay_range):
            raise ValueError("delay_range must be a tuple of two numbers (int or float) - e.g. (1, 2) or (1.5, 2.5). If you are attempting to disable delays, set to (0,0).")
        if self.delay_range[1] < self.delay_range[0]:
            raise ValueError("Upper bound of delay_range must be greater than or equal to lower bound.")
        
        # Validate scraping strategy (at least one method must be enabled)
        if not self.html_scraping and not self.pdf_fallback and not self.llm_fallback:
            raise ValueError("At least one of 'html_scraping', 'pdf_fallback', or 'llm_fallback' must be enabled.")

        # Validate output configuration (at least one field must be included)
        if not any([self.include_id, self.include_date, self.include_coroner, self.include_area, self.include_receiver, self.include_investigation, self.include_circumstances, self.include_concerns]):
            raise ValueError("At least one field must be included in the output. Please set one or more of the following to True:\n 'include_id', 'include_date', 'include_coroner', 'include_area', 'include_receiver', 'include_investigation', 'include_circumstances', 'include_concerns'.\n")
        
        # Warnings for potentially suboptimal configurations (code will still run)
        if self.html_scraping and not self.pdf_fallback and not self.llm_fallback: # Only HTML scraping enabled
            logger.warning("Only HTML scraping is enabled. \nConsider enabling .pdf or LLM fallback for more complete data extraction.\n")
        if not self.html_scraping and self.pdf_fallback and not self.llm_fallback: # Only .pdf scraping enabled
            logger.warning("Only .pdf fallback is enabled. \nConsider enabling HTML scraping or LLM fallback for more complete data extraction.\n")
        if not self.html_scraping and not self.pdf_fallback and self.llm_fallback: # Only LLM scraping enabled
            logger.warning("Only LLM fallback is enabled. \nWhile this is a high-performance option, large API costs may be incurred, especially for large requests. \nConsider enabling HTML scraping or .pdf fallback for more cost-effective data extraction.\n")
        if self.max_workers > 50: # Too many max workers
            logger.warning("max_workers is set to a high value (>50). \nDepending on your system, this may cause performance issues. It could also trigger anti-scraping measures by the host, leading to temporary or permanent IP bans. \nWe recommend setting to between 10 and 50.\n")
        if self.max_workers < 10: # Too few max workers
            logger.warning("max_workers is set to a low value (<10). \nThis may result in slower scraping speeds. Consider increasing the value for faster performance. \nWe recommend setting to between 10 and 50.\n")
        if self.max_requests > 10: # Too many max requests
            logger.warning("max_requests is set to a high value (>10). \nThis may trigger anti-scraping measures by the host, leading to temporary or permanent IP bans. \nWe recommend setting to between 3 and 10.\n")
        if self.max_requests < 3: # Too few max requests
            logger.warning("max_requests is set to a low value (<3). \nThis may result in slower scraping speeds. Consider increasing the value for faster performance. \nWe recommend setting to between 3 and 10.\n")
        if self.delay_range == (0, 0): # Delay range too high or low, or disabled
            logger.warning("delay_range has been disabled. \nThis will disable delays between requests. This may trigger anti-scraping measures by the host, leading to temporary or permanent IP bans. \nWe recommend setting to (1,2).\n")
        elif self.delay_range[0] < 0.5 and self.delay_range[1] != 0: # type: ignore[operator] # delay_range elements are numbers
            logger.warning("delay_range is set to a low value (<0.5 seconds). \nThis may trigger anti-scraping measures by the host, leading to temporary or permanent IP bans. We recommend setting to between (1, 2).\n")
        if self.delay_range[1] > 5: # type: ignore[operator] # delay_range elements are numbers
            logger.warning("delay_range is set to a high value (>5 seconds). \nThis may result in slower scraping speeds. Consider decreasing the value for faster performance. We recommend setting to between (1, 2).\n")


        # Set up a requests session with retry logic to handle temporary network and rate limit issues
        self.session = requests.Session()
        retries = Retry(total=50, backoff_factor=1, status_forcelist=[429, 502, 503, 504]) # 50 retry attempts before failing
        adapter = HTTPAdapter(max_retries=retries, pool_connections=100, pool_maxsize=100)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Pre-compile regular expression for extracting report IDs (e.g. "2025-0296")
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

        # Mapping from LLM response keys back to DataFrame column names for updating
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
    # Link fetching logic: Methods for discovering report URLs from search pages
    # -----------------------------------------------------------------------------
        
    def _get_report_href_values(self, url: str) -> list[str]:
        """
        Extracts individual PFD report URLs (href values) from a single search results page.
        Applies a random delay and uses a semaphore to limit concurrent requests.

        :param url: The URL of the search results page to scrape.
        :return: A list of href strings, each being a URL to a PFD report page.
                 Returns an empty list if the page fetch fails or no links are found.
        """
        with self.domain_semaphore: 
            time.sleep(random.uniform(*self.delay_range))  # Introduce a random delay between requests to be polite to the server
            try:
                response = self.session.get(url, timeout=self.timeout)
                response.raise_for_status() # Raise HTTPError for bad responses (4XX or 5XX)
                if self.verbose:
                    logger.debug(f"Fetched URL: {url} (Status: {response.status_code})")
            except requests.RequestException as e:
                if self.verbose:
                    logger.error("Failed to fetch page: %s; Error: %s", url, e)
                return [] # Return empty list on failure
        
        # Parse the page content and find links
        soup = BeautifulSoup(response.text, 'html.parser')
        links = soup.find_all('a', class_='card__link') # Find <a> tags with class 'card__link'
        return [link.get('href') for link in links if link.get('href')] # Extract href values

    def get_report_links(self) -> list[str] | None:
        """Discover individual report URLs for the current query.

        Pagination continues until a page yields zero new links.

        Returns
        -------
        list[str] | None
            All discovered URLs, or *None* if **no** links were found for
            the given category/date window.

        Examples
        --------
        >>> links = scraper.get_report_links()
        >>> len(links)
        42
        """
        self.report_links = [] # Reset internal list of report links
        page = self.start_page 
        
        # Initialise progress bar for fetching pages
        pbar = tqdm(desc="Fetching pages", unit=" page", leave=False, initial=page, position=0)
        
        while True:
            # Format the search URL with the current page number and date parameters
            page_url = self.page_template.format(page=page, **self.date_params)
            href_values = self._get_report_href_values(page_url) # Get links from the current page
            pbar.update(1) # Increment progress bar
            
            if self.verbose:
                logger.info("Scraped %d links from %s", len(href_values), page_url)
            
            if not href_values:
                # If no more links are returned on a successive page, assume we've reached the end of search results & break link fetching
                break
            self.report_links.extend(href_values) # Add found links to the main list
            page += 1  # Move to the next page
        
        pbar.close() # Close the progress bar

        if not self.report_links: # If no links were collected across *all* pages...
            logger.error("\nNo report links found. Please check your date range.")
            return None 
        
        logger.info("Total collected report links: %d", len(self.report_links))
        return self.report_links

    # -----------------------------------------------------------------------------
    # Text processing helper functions -- all for internal use
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
        return " ".join(normalised_apostrophes.split()) # Splits by whitespace and rejoins with single spaces

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
            # Attempt to isolate date part if "Ref" (case-insensitive) acts as a suffix
            match = re.match(r"(.+?)(Ref[:\s]|$)", text_being_processed, re.IGNORECASE)
            if match:
                potential_date_part = match.group(1).strip()
                if potential_date_part:
                    final_text_to_parse = potential_date_part
            
            if not final_text_to_parse:
                if self.verbose:
                    logger.warning(f"Date string empty after trying to remove 'Ref...' from '{text_being_processed}'. Raw: '{raw_date_str}'. Keeping raw.")
                return text_being_processed # Return original

            dt = date_parser.parse(final_text_to_parse, fuzzy=True, dayfirst=True)
            return dt.strftime("%Y-%m-%d") # Return parsed date
            
        except Exception as e:
            if self.verbose:
                logger.warning(
                    f"Date parse failed for raw '{raw_date_str}' (processed to '{text_being_processed}', attempted '{final_text_to_parse}') "
                    f"– keeping raw. Error: {e}"
                )
            return text_being_processed # Return original

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

        # Length validation: If text is empty after cleaning and a minimum length is expected...
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
        path = unquote(parsed_url.path) # Decode URL-encoded characters in path
        ext = os.path.splitext(path)[1].lower() # Get file extension
        
        if self.verbose:
            logger.debug(f"Processing .pdf {pdf_url}.")
        
        # Download the file content
        try:
            response = self.session.get(pdf_url, timeout=self.timeout)
            response.raise_for_status()
            file_bytes = response.content
        except requests.RequestException as e:
            logger.error("Failed to fetch file: %s; Error: %s", pdf_url, e)
            return self.NOT_FOUND_TEXT 

        pdf_bytes_to_process: bytes | None = None
        if ext != ".pdf": # Check that file is a .pdf
            logger.info("File %s is not a .pdf (extension %s). Skipping this file...", pdf_url, ext)
            return self.NOT_FOUND_TEXT 
        else:
            pdf_bytes_to_process = file_bytes
        
        if pdf_bytes_to_process is None:
            return "N/A: Source file not PDF"

        # Cache the downloaded PDF bytes for potential later use
        self._last_pdf_bytes = pdf_bytes_to_process

        # Use PyMuPDF to open the PDF from bytes and extract text
        try:
            pdf_buffer = BytesIO(pdf_bytes_to_process) # Create in-memory binary stream
            pdf_document = pymupdf.open(stream=pdf_buffer, filetype="pdf")
            text = "".join(page.get_text() for page in pdf_document) # Concatenate text from all pages
            pdf_document.close()
        except Exception as e:
            logger.error("Error processing .pdf %s: %s", pdf_url, e)
            return "N/A: Source file not PDF"
        
        return self._clean_text(text) # Clean the extracted text
              
    def _extract_paragraph_text_by_keywords(self, soup: BeautifulSoup, keywords: list[str]) -> str:
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
                return self._clean_text(element.get_text()) # Return cleaned text of the first match
        return self.NOT_FOUND_TEXT # Return if no matching paragraph is found
              
    def _extract_section_text_by_keywords(self, soup: BeautifulSoup, header_keywords: list[str]) -> str:
        """
        Extracts a block of text from HTML following a header.
        It looks for a `<strong>` tag (assumed to be a section header) containing one of the `header_keywords`.
        If found, it collects text from all subsequent sibling elements until the next header or end of parent.

        :param soup: A BeautifulSoup object representing the parsed HTML page.
        :param header_keywords: A list of keyword variations to identify the section header.
        :return: The cleaned, concatenated text of the section, or `NOT_FOUND_TEXT`.
        """
        for strong_tag in soup.find_all('strong'): # Iterate over all <strong> tags
            header_text = strong_tag.get_text(strip=True)
            # Check if any header keyword is present in the <strong> tag's text (case-insensitive)
            if any(keyword.lower() in header_text.lower() for keyword in header_keywords):
                content_parts = []
                # Iterate over sibling elements that appear *after* the found <strong> header
                for sibling in strong_tag.next_siblings:
                    if isinstance(sibling, str): # If the sibling is a NavigableString...
                        text = sibling.strip()
                        if text:
                            content_parts.append(text)
                    else: # If the sibling is a Tag...
                        # Get text content, using space as separator for elements like <br>
                        text = sibling.get_text(separator=" ", strip=True)
                        if text:
                            content_parts.append(text)
                if content_parts:
                    return self._clean_text(" ".join(content_parts)) # Join and clean
        return self.NOT_FOUND_TEXT # Return if no matching section header is found

    def _extract_section_from_pdf_text(self, text: str, start_keywords: list[str], end_keywords: list[str]) -> str:
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
            start_index = lower_text.find(start_kw_lower) # Find the first occurrence of a start keyword
            
            if start_index != -1: # If a start keyword is found...
                section_start_offset = start_index + len(start_kw_lower) # Position after the start keyword
                
                end_indices_found = []
                for end_kw in end_keywords:
                    end_kw_lower = end_kw.lower()
                    # Find the end/exit keyword, starting search from after the start keyword
                    end_index = lower_text.find(end_kw_lower, section_start_offset)
                    if end_index != -1:
                        end_indices_found.append(end_index)
                
                extracted_section_text: str
                if end_indices_found:
                    # If one or more end keywords are found, use the earliest one
                    section_end_offset = min(end_indices_found)
                    extracted_section_text = text[section_start_offset:section_end_offset]
                else:
                    # If no end keyword is found, take text from start keyword to the end
                    extracted_section_text = text[section_start_offset:]
                return extracted_section_text # Return raw extracted section

        return self.NOT_FOUND_TEXT # Return if no start keyword is found
              
    # -----------------------------------------------------------------------------
    # LLM Report Extraction Logic
    # -----------------------------------------------------------------------------      

    def _fetch_pdf_bytes(self, report_url: str) -> bytes | None:
        """
        Fetches the PDF file content as bytes from a report's HTML page.
        It first finds the PDF download link on the HTML page, then downloads the PDF.

        :param report_url: The URL of the HTML page containing the link to the PDF report.
        :return: The PDF content as bytes, or None if fetching fails or no PDF link is found.
        """
        try:  # Get the HTML page of the report
            page_response = self.session.get(report_url, timeout=self.timeout)
            page_response.raise_for_status()
            soup = BeautifulSoup(page_response.content, 'html.parser')
            
            # Find the PDF download link (we're looking for a 'govuk-button' as its named on the site)
            pdf_links = [a['href'] for a in soup.find_all('a', class_='govuk-button') if a.get('href')]
            if pdf_links:
                pdf_link = pdf_links[0] # There are often multiple buttons; fortunately the report is always the first one (0 index)
                # Download the PDF file
                pdf_response = self.session.get(pdf_link, timeout=self.timeout)
                pdf_response.raise_for_status()
                return pdf_response.content # Return PDF content as bytes
            else:
                logger.error("No PDF link found for report at %s", report_url)
                return None
        except Exception as e:
            logger.error("Failed to fetch PDF for report at %s: %s", report_url, e)
            return None
              
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
        # Initialise all potential fields with the default "not found" text
        report_id = self.NOT_FOUND_TEXT
        date = self.NOT_FOUND_TEXT
        receiver = self.NOT_FOUND_TEXT
        coroner = self.NOT_FOUND_TEXT
        area = self.NOT_FOUND_TEXT
        investigation = self.NOT_FOUND_TEXT
        circumstances = self.NOT_FOUND_TEXT
        concerns = self.NOT_FOUND_TEXT
        
        # Fetch the main HTML report page
        try:
            response = self.session.get(url, timeout=self.timeout)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error("Failed to fetch %s; Error: %s", url, e)
            return None # Return None if page fetch fails
        
        soup = BeautifulSoup(response.content, 'html.parser')
        
        # Find the link to the PDF document on the HTML page
        pdf_links = [a['href'] for a in soup.find_all('a', class_='govuk-button') if a.get('href')]
        if not pdf_links:
            logger.error("No .pdf links found on %s", url)
            return None # Return None if no PDF link is found
        
        report_link = pdf_links[0] # Assume the first link is the primary PDF
        # Extract text from the PDF. This also caches PDF bytes via self._last_pdf_bytes
        pdf_text = self._extract_text_from_pdf(report_link)
        
        # --- HTML Data Extraction ---
        if self.html_scraping:
            if self.verbose:
                logger.debug(f"Extracting data from HTML for URL: {url}")
            
            # Report ID (Ref:) from HTML
            if self.include_id:
                ref_element_text = self._extract_paragraph_text_by_keywords(soup, ["Ref:"])
                if ref_element_text != self.NOT_FOUND_TEXT:
                    match = self._id_pattern.search(ref_element_text)
                    report_id = match.group(1) if match else self.NOT_FOUND_TEXT
            
            # Date of report from HTML
            if self.include_date:
                date_raw = self._extract_paragraph_text_by_keywords(soup, ["Date of report:"])
                date = self._process_extracted_field(date_raw, ["Date of report:"], is_date=True)

            # Receiver (Sent to:) from HTML
            if self.include_receiver:
                receiver_raw = self._extract_paragraph_text_by_keywords(soup, ["This report is being sent to:", "Sent to:"])
                receiver = self._process_extracted_field(receiver_raw, 
                                                         ["This report is being sent to:", "Sent to:", "TO:"],
                                                         min_len=5, max_len=20)
            # Coroner's name from HTML
            if self.include_coroner:
                coroner_raw = self._extract_paragraph_text_by_keywords(soup, ["Coroners name:", "Coroner name:", "Coroner's name:"])
                coroner = self._process_extracted_field(coroner_raw,
                                                        ["Coroners name:", "Coroner name:", "Coroner's name:"],
                                                        min_len=5, max_len=20)
            # Coroner's area from HTML
            if self.include_area:
                area_raw = self._extract_paragraph_text_by_keywords(soup, ["Coroners Area:", "Coroner Area:", "Coroner's Area:"])
                area = self._process_extracted_field(area_raw,
                                                     ["Coroners Area:", "Coroner Area:", "Coroner's Area:"],
                                                     min_len=4, max_len=40)
            
            # Investigation and Inquest section from HTML
            if self.include_investigation:
                investigation_raw = self._extract_section_text_by_keywords(soup, ["INVESTIGATION and INQUEST", "INVESTIGATION & INQUEST", "3 INQUEST"])
                investigation = self._process_extracted_field(investigation_raw,
                                                             ["INVESTIGATION and INQUEST", "INVESTIGATION & INQUEST", "3 INQUEST"],
                                                             min_len=30)
            # Circumstances of Death section from HTML
            if self.include_circumstances:
                circumstances_raw = self._extract_section_text_by_keywords(soup, ["CIRCUMSTANCES OF THE DEATH", "CIRCUMSTANCES OF DEATH", "CIRCUMSTANCES OF"])
                circumstances = self._process_extracted_field(circumstances_raw,
                                                              ["CIRCUMSTANCES OF THE DEATH", "CIRCUMSTANCES OF DEATH", "CIRCUMSTANCES OF"],
                                                              min_len=30)
            # Coroner's Concerns section from HTML
            if self.include_concerns:
                concerns_raw = self._extract_section_text_by_keywords(soup, ["CORONER'S CONCERNS", "CORONERS CONCERNS", "CORONER CONCERNS"])
                concerns = self._process_extracted_field(concerns_raw,
                                                         ["CORONER'S CONCERNS", "CORONERS CONCERNS", "CORONER CONCERNS"],
                                                         min_len=30)

        # --- .pdf Data Extraction Fallback ---
        # This runs if pdf_fallback is enabled AND if any of the main text fields are still "N/A: Not found"
        if self.pdf_fallback and pdf_text != self.NOT_FOUND_TEXT and pdf_text != "N/A: Source file not PDF":
            # Check which fields (among those included) are still missing...
            fields_to_check_for_pdf_fallback = []
            if self.include_coroner: fields_to_check_for_pdf_fallback.append(coroner)
            if self.include_area: fields_to_check_for_pdf_fallback.append(area)
            if self.include_receiver: fields_to_check_for_pdf_fallback.append(receiver)
            if self.include_investigation: fields_to_check_for_pdf_fallback.append(investigation)
            if self.include_circumstances: fields_to_check_for_pdf_fallback.append(circumstances)
            if self.include_concerns: fields_to_check_for_pdf_fallback.append(concerns)
            
            if self.NOT_FOUND_TEXT in fields_to_check_for_pdf_fallback:
                if self.verbose:
                    logger.debug(f"Initiating .pdf fallback for URL: {url} because one or more fields are missing.")

                # Coroner name from PDF
                if self.include_coroner and coroner == self.NOT_FOUND_TEXT:
                    coroner_raw_pdf = self._extract_section_from_pdf_text(pdf_text, start_keywords=["I am", "CORONER"], end_keywords=["CORONER'S LEGAL POWERS", "paragraph 7"])
                    if coroner_raw_pdf != self.NOT_FOUND_TEXT:
                        coroner = coroner_raw_pdf.replace("I am", "").replace("CORONER'S LEGAL POWERS", "").replace("CORONER", "").replace("paragraph 7", "").strip()
                        coroner = self._clean_text(coroner) 
                        if not coroner: coroner = self.NOT_FOUND_TEXT


                # Area from PDF
                if self.include_area and area == self.NOT_FOUND_TEXT:
                    area_raw_pdf = self._extract_section_from_pdf_text(pdf_text, start_keywords=["area of"], end_keywords=["LEGAL POWERS", "LEGAL POWER", "paragraph 7"])
                    if area_raw_pdf != self.NOT_FOUND_TEXT:
                        area = area_raw_pdf.replace("area of", "").replace("CORONER'S", "").replace("CORONER", "").replace("CORONERS", "").replace("paragraph 7", "").strip()
                        area = self._clean_text(area) 
                        if not area: area = self.NOT_FOUND_TEXT


                # Receiver from PDF
                if self.include_receiver and receiver == self.NOT_FOUND_TEXT:
                    receiver_raw_pdf = self._extract_section_from_pdf_text(pdf_text, start_keywords=[" SENT ", "SENT TO:"], end_keywords=["CORONER", "CIRCUMSTANCES OF THE DEATH", "CIRCUMSTANCES OF"])
                    if receiver_raw_pdf != self.NOT_FOUND_TEXT: 
                        temp_receiver = self._clean_text(receiver_raw_pdf).replace("TO:", "").strip()
                        if len(temp_receiver) >= 5 : 
                            receiver = temp_receiver
                        else:
                            receiver = self.NOT_FOUND_TEXT
                
                # Investigation from PDF
                if self.include_investigation and investigation == self.NOT_FOUND_TEXT:
                    investigation_raw_pdf = self._extract_section_from_pdf_text(pdf_text, start_keywords=["INVESTIGATION and INQUEST", "3 INQUEST"], end_keywords=["CIRCUMSTANCES OF DEATH", "CIRCUMSTANCES OF THE DEATH", "CIRCUMSTANCES OF"])
                    if investigation_raw_pdf != self.NOT_FOUND_TEXT:
                        temp_investigation = self._clean_text(investigation_raw_pdf) 
                        if len(temp_investigation) >= 30: 
                            investigation = temp_investigation
                        else:
                            investigation = self.NOT_FOUND_TEXT
                
                # Circumstances from PDF
                if self.include_circumstances and circumstances == self.NOT_FOUND_TEXT:
                    circumstances_raw_pdf = self._extract_section_from_pdf_text(pdf_text, start_keywords=["CIRCUMSTANCES OF DEATH", "CIRCUMSTANCES OF THE DEATH", "CIRCUMSTANCES OF"], end_keywords=["CORONER'S CONCERNS", "CORONER CONCERNS", "CORONERS CONCERNS", "as follows"])
                    if circumstances_raw_pdf != self.NOT_FOUND_TEXT:
                        temp_circumstances = self._clean_text(circumstances_raw_pdf)
                        if len(temp_circumstances) >= 30:
                            circumstances = temp_circumstances
                        else:
                            circumstances = self.NOT_FOUND_TEXT

                # Concerns from PDF
                if self.include_concerns and concerns == self.NOT_FOUND_TEXT:
                    concerns_raw_pdf = self._extract_section_from_pdf_text(pdf_text, start_keywords=["CORONER'S CONCERNS", "as follows"], end_keywords=["ACTION SHOULD BE TAKEN"])
                    if concerns_raw_pdf != self.NOT_FOUND_TEXT:
                        temp_concerns = self._clean_text(concerns_raw_pdf) 
                        if len(temp_concerns) >= 30:
                            concerns = temp_concerns
                        else:
                            concerns = self.NOT_FOUND_TEXT
        
        # Assemble the report dictionary with extracted (or default) values
        report: Dict[str, Any] = {}
        if self.include_url: report[self.COL_URL] = url
        if self.include_id: report[self.COL_ID] = report_id
        if self.include_date: report[self.COL_DATE] = date
        if self.include_coroner: report[self.COL_CORONER_NAME] = coroner
        if self.include_area: report[self.COL_AREA] = area
        if self.include_receiver: report[self.COL_RECEIVER] = receiver
        if self.include_investigation: report[self.COL_INVESTIGATION] = investigation
        if self.include_circumstances: report[self.COL_CIRCUMSTANCES] = circumstances
        if self.include_concerns: report[self.COL_CONCERNS] = concerns
        if self.include_time_stamp: report[self.COL_DATE_SCRAPED] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        return report

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
        
        if not self.report_links: # If links haven't been fetched yet...
            fetched_links = self.get_report_links()
            if fetched_links is None: # No links found by get_report_links...
                 self.reports = pd.DataFrame() # Ensure self.reports is an empty DF
                 return self.reports # Return empty DataFrame

        # Use a thread pool to concurrently scrape information from each report URL.
        results = []
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            # Submit all report extraction tasks to the executor & collect with progress bar
            futures = [executor.submit(self._extract_report_info, url) for url in self.report_links]
            for future in tqdm(as_completed(futures), total=len(futures), desc="Scraping reports", position=0, leave=False):
                results.append(future.result())
        
        # Filter out any None results (from failed extractions)
        reports_data = [report for report in results if report is not None]
        reports_df = pd.DataFrame(reports_data) # Convert list of dicts to DataFrame

        # If LLM fallback is enabled and an LLM client is configured, apply it
        if self.llm_fallback and self.llm:
            reports_df = self.run_llm_fallback(reports_df if not reports_df.empty else None)
            
        # Sort dataframe by date column, if this is present
        if self.include_date == True:
            reports_df = reports_df.sort_values(by=[self.COL_DATE], ascending=False) # Latest reports at the top
            
        self.reports = reports_df.copy() # Store the final DataFrame internally
        return reports_df


    def top_up(self, old_reports: pd.DataFrame | None = None, start_date: str | None = None, end_date: str | None = None) -> pd.DataFrame | None:
        """Append **new** reports to an existing DataFrame.

        Any URL (or ID) already present in *old_reports* is skipped.

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
        
        # Update date range if new dates are provided for the top-up
        if start_date is not None or end_date is not None:
            new_start_date = date_parser.parse(start_date) if start_date is not None else self.start_date
            new_end_date = date_parser.parse(end_date) if end_date is not None else self.end_date
            if new_start_date > new_end_date:
                raise ValueError("start_date must be before end_date.")
            # Update scraper's internal date range and parameters
            self.start_date = new_start_date
            self.end_date = new_end_date
            self.date_params.update({
                "after_day": self.start_date.day, "after_month": self.start_date.month, "after_year": self.start_date.year,
                "before_day": self.end_date.day, "before_month": self.end_date.month, "before_year": self.end_date.year,
            })

        # Determine the base DataFrame to top up
        base_df = old_reports if old_reports is not None else self.reports

        # Check if the base DataFrame has the required columns for merging/comparison
        required_columns = [col_name for include_flag, col_name in self._COLUMN_CONFIG if include_flag]
        if base_df is not None:
            missing_cols = [col for col in required_columns if col not in base_df.columns]
            if missing_cols:
                raise ValueError(f"Required columns missing from the provided DataFrame: {missing_cols}")

        # Fetch the latest list of report links based on current settings
        updated_links = self.get_report_links()
        if updated_links is None: # No links found for the current date range
            logger.info("No links found for the specified date range during top-up.")
            return base_df if base_df is not None else None


        # Determine the unique key for checking existing reports (URL or ID)
        unique_key = ""
        if self.include_url: unique_key = self.COL_URL
        elif self.include_id: unique_key = self.COL_ID
        else:
            logger.error("No unique identifier available for duplicate checking.\nEnsure include_url or include_id was set to True in instance initialisation.")
            return base_df if base_df is not None else pd.DataFrame() # Return existing or empty DF

        # Get identifiers (URLs or IDs) from the existing reports
        existing_identifiers = set(base_df[unique_key].tolist()) if base_df is not None and unique_key in base_df.columns else set()
        
        # Filter out links that are already in existing_identifiers
        new_links = [link for link in updated_links if link not in existing_identifiers]
        logger.info("Top-up: %d new report(s) found; %d duplicate(s) which won't be added", len(new_links), len(updated_links) - len(new_links))

        if not new_links:
            logger.info("No new reports to scrape during top-up.")
            return None if base_df is None and not old_reports else base_df


        # Scrape the new reports
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            new_results = list(tqdm(executor.map(self._extract_report_info, new_links), total=len(new_links), desc="Topping up reports", position=0, leave=True))
        
        new_records = [record for record in new_results if record is not None]
        new_df = pd.DataFrame(new_records) # DataFrame of newly scraped reports

        updated_reports_df: pd.DataFrame
        if not new_df.empty: # If new reports were successfully scraped...
            updated_reports_df = pd.concat([base_df, new_df], ignore_index=True) if base_df is not None else new_df
        else: # No new valid records were scraped from the new_links...
            updated_reports_df = base_df if base_df is not None else pd.DataFrame() # Return original or empty
        
        # Sort updates reports by date
        if self.include_date == True:
            updated_reports_df = updated_reports_df.sort_values(by=[self.COL_DATE], ascending=False)
        
        self.reports = updated_reports_df.copy() # Update internal reports DataFrame
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
        if not self.llm: # Check if LLM client is configured
             raise ValueError("LLM client (self.llm) not provided. Cannot run LLM fallback.")

        current_reports_df: pd.DataFrame
        if reports_df is None: # If no DataFrame is passed, use the internal one
            if self.reports is None:
                raise ValueError("No scraped reports found (reports_df is None and self.reports is None). Please run scrape_reports() first or provide a DataFrame.")
            current_reports_df = self.reports.copy() 
        else:
            current_reports_df = reports_df.copy()

        if current_reports_df.empty: # If DataFrame is empty, nothing to process
            logger.info("Report DataFrame is empty. Skipping LLM fallback.")
            return current_reports_df

        # --- Helper function to process a single row for LLM fallback ---
        def process_row(idx: int, row_data: pd.Series) -> tuple[int, dict[str, str]]:
            """Identifies missing fields in a row and calls LLM for them."""
            missing_fields: dict[str, str] = {}
            # Build dictionary of fields needing LLM extraction based on _LLM_FIELD_CONFIG.
            for include_flag, df_col_name, llm_key, llm_prompt in self._LLM_FIELD_CONFIG:
                if include_flag and row_data.get(df_col_name, "") == self.NOT_FOUND_TEXT:
                    missing_fields[llm_key] = llm_prompt # { "llm_key": "prompt_for_llm" }
            
            if not missing_fields: # If no fields are missing for this row...
                return idx, {}

            pdf_bytes: bytes | None = None
            report_url = row_data.get(self.COL_URL) # Get URL to fetch/find PDF bytes
            
            if report_url:
                # Attempt to use cached PDF bytes if available and matches the current context
                pdf_bytes = self._fetch_pdf_bytes(report_url)

            if not pdf_bytes and self.verbose:
                 logger.warning(f"Could not obtain PDF bytes for URL {report_url} (row {idx}). LLM fallback for this row might be impaired.")

            # Call the LLM client's fallback method
            # Assumes self.llm is not None due to the check at the start of run_llm_fallback -- maybe this needs to change?
            updates = self.llm._call_llm_fallback( 
                pdf_bytes=pdf_bytes,
                missing_fields=missing_fields,
                report_url=str(report_url) if report_url else "N/A", 
                verbose=self.verbose,
                tqdm_extra_kwargs={"disable": True}
            )
            return idx, updates if updates else {} # Ensure a dict is returned

        # --- Process rows for LLM fallback (parallel or sequential) ---
        results_map: Dict[int, Dict[str, str]] = {} # To store {row_index: {field: value}} from LLM.
        
        # Determine if parallel processing should be used based on LLM client's max_workers
        use_parallel = self.llm and hasattr(self.llm, 'max_workers') and self.llm.max_workers > 1 

        if use_parallel:
            with ThreadPoolExecutor(max_workers=self.llm.max_workers) as executor: 
                future_to_idx = { # Map futures to row indices for result correlation
                    executor.submit(process_row, idx, row_series): idx 
                    for idx, row_series in current_reports_df.iterrows()
                }
                for future in tqdm(as_completed(future_to_idx), total=len(future_to_idx), desc="LLM fallback (parallel processing)", position=0, leave=True):
                    idx = future_to_idx[future]
                    try:
                        _, updates = future.result() # Get (original_idx, updates_dict).
                        results_map[idx] = updates
                    except Exception as e:
                        logger.error(f"LLM fallback failed for row index {idx}: {e}")
        else: # Sequential processing...
            for idx, row_series in tqdm(current_reports_df.iterrows(), total=len(current_reports_df), desc="LLM fallback (sequential processing)", position=0, leave=True):
                try:
                    _, updates = process_row(idx, row_series) 
                    results_map[idx] = updates
                except Exception as e:
                     logger.error(f"LLM fallback failed for row index {idx}: {e}")

        # Apply the updates received from LLM back to the DataFrame
        for idx, updates_dict in results_map.items():
            if not updates_dict: # Skip if LLM returned no updates for this row
                continue
            for llm_key, value_from_llm in updates_dict.items():
                df_col_name = self._LLM_TO_DF_MAPPING.get(llm_key) # Get corresponding DataFrame column name
                if df_col_name:
                    if llm_key == self.LLM_KEY_DATE:
                        # For dates, normalise only if LLM found something other than "N/A: Not found"
                        if value_from_llm != self.NOT_FOUND_TEXT:
                            current_reports_df.at[idx, df_col_name] = self._normalise_date(value_from_llm)
                    else:
                        # For other fields, update with LLM's value (even if it's "N/A: Not found")
                        current_reports_df.at[idx, df_col_name] = value_from_llm
        
        self.reports = current_reports_df.copy()
        return current_reports_df