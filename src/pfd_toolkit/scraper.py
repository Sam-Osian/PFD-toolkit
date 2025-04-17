from __future__ import annotations

import logging
import requests
from requests.adapters import HTTPAdapter
from urllib3.util.retry import Retry
from bs4 import BeautifulSoup
import pymupdf
import pandas as pd
import re
from dateutil import parser
import tempfile
from concurrent.futures import ThreadPoolExecutor
from pydantic import create_model, BaseModel
from io import BytesIO
from urllib.parse import urlparse, unquote
import base64
from pdf2image import convert_from_bytes
from dotenv import load_dotenv
import os
import subprocess
import time
import random
import threading
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import as_completed

from typing import TYPE_CHECKING, Optional, Type

# Only import LLM if needed
if TYPE_CHECKING:                           
    from pfd_toolkit.llm import LLM         

#from pfd_toolkit.llm import LLM

# -----------------------------------------------------------------------------
# Logging Configuration:
# - Sets up logging for the module. The logger is used to record events,
#   debugging messages, and error messages.
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, force=True)

# Set the log level for the 'httpx' library to WARNING to reduce verbosity
logging.getLogger("httpx").setLevel(logging.WARNING)

# Define PFDScraper class
class PFDScraper:
    """Web scraper for extracting Prevention of Future Death (PFD) reports from the UK Judiciary website.
    
    This class handles:
      - Fetching PFD URLs.
      - Parsing HTML to extract report data.
      - Fallback to .pdf scraping if HTML fails for any given field.
      - Fallback to OpenAI LLM for image-based .pdf extraction if scraping fails for any given field.
      
    """
    
    def __init__(
        self,
        llm: Optional["LLM"] = None, # Quoted LLM so that runtime import not needed
        # Web page logic
        category: str = 'all',
        date_from: str = "2000-01-01",
        date_to: str = "2030-01-01",
        
        # Threading and request logic
        max_workers: int = 10,
        max_requests: int = 5, 
        delay_range = (1, 2),
        
        # Straping strategy
        html_scraping: bool = True,
        pdf_fallback: bool = True,
        llm_fallback: bool = False,
        # Document conversion
        docx_conversion: str = "None",
        
        # Output configuration
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
        """
        Initialises the scraper.
        :param llm: The LLM client to use when llm_fallback is True.
        :param category: Category of reports as categorised on the judiciary.uk website. Options are 'all' (default), 'suicide', 'accident_work_safety', 'alcohol_drug_medication', 'care_home', 'child_death', 'community_health_emergency', 'emergency_services', 'hospital_deaths', 'mental_health', 'police', 'product', 'railway', 'road', 'service_personnel', 'custody', 'wales', 'other'.
        :param date_from: In "YYYY-MM-DD" format. Only reports published on or after this date will be scraped.
        :param date_to: In "YYYY-MM-DD" format. Only reports published on or before this date will be scraped.
        :param max_workers: The total number of concurrent threads the scraper can use for fetching data across all pages.
        :param max_requests: Maximum number of requests per domain to avoid IP address block.
        :param delay_range: None, or a tuple of two integers representing the range of seconds to delay between requests. Default is (1, 2) for a random delay between 1 and 2 seconds.
        :param html_scraping: Whether to attempt HTML-based scraping.
        :param pdf_fallback: Whether to fallback to .pdf scraping if missing values remain following HTML scraping (if set).
        :param llm_fallback: Whether to fallback to LLM scraping if missing values remain following previous method(s), if set. OpenAI API key must provided.
        :param docx_conversion: Conversion method for .docx files; "MicrosoftWord", "LibreOffice", or "None" (default).
        :param include_url: Whether to add a URL column to the output file.
        :param include_id: Whether to add a report ID column to the output file.
        :param include_date: Whether to add a date column to the output file.
        :param include_coroner: Whether to add a coroner column to the output file.
        :param include_area: Whether to add an area column to the output file.
        :param include_receiver: Whether to add a receiver column to the output file.
        :param include_investigation: Whether to add an investigation column to the output file.
        :param include_circumstances: Whether to add a circumstances column to the output file.
        :param include_concerns: Whether to add a concerns column to the output file.
        :param include_time_stamp: Whether to add a timestamp column to the output file.
        :param verbose: Whether to print verbose output.
        """
        self.category = category.lower()
        
        # Parsing dates into datetime objects
        self.date_from = parser.parse(date_from)
        self.date_to = parser.parse(date_to)
        
        # Storing the parsed date parts for the URL formatting that comes later
        self.date_params = {
            "after_day": self.date_from.day,
            "after_month": self.date_from.month,
            "after_year": self.date_from.year,
            "before_day": self.date_to.day,
            "before_month": self.date_to.month,
            "before_year": self.date_to.year,
        }
        
        self.start_page = 1
        
        self.max_workers = max_workers
        self.max_requests = max_requests
        self.delay_range = delay_range
        
        self.html_scraping = html_scraping
        self.pdf_fallback = pdf_fallback
        self.llm_fallback = llm_fallback
        self.llm = llm

        self.docx_conversion = docx_conversion
        
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
        
        self.domain_semaphore = threading.Semaphore(self.max_requests) # Semaphore to limit requests per domain
        
        self.reports = None # ...So that the user can access them later as an internal attribute
        self.report_links = [] 
        
        self.llm_model = self.llm.model if self.llm else "None"
        
        # Define URL templates for different PFD categories.
        # ...Some categories (like 'all' and 'suicide') have unique URL formats, which is why we're specifying them individually
        
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

        
        if self.category in category_templates:
            self.page_template = category_templates[self.category]
        else:
            valid_options = ", ".join(sorted(category_templates.keys()))
            raise ValueError(f"Unknown category '{self.category}'. Valid options are: {valid_options}")
        
        # Set pagination parameter
        self.start_page = 1
        
        # Normalise delay_range if set to 0 or None
        if self.delay_range is None or self.delay_range == 0:
            self.delay_range = (0, 0)

        
        # -----------------------------------------------------------------------------
        # Error and Warning Handling for Initialisation Parameters
        # -----------------------------------------------------------------------------
        
        ### Errors
        
        # If category is not one of the allowed values
        if self.category in category_templates:
            self.page_template = category_templates[self.category]
        else:
            valid_options = ", ".join(sorted(category_templates.keys()))
            raise ValueError(f"Unknown category '{self.category}'. Valid options are: {valid_options}")
        
        # If date_from is after date_to
        if self.date_from > self.date_to:
            raise ValueError("date_from must be before date_to.")
        
        # If LLM fallback is enabled but no LLM client is provided
        if self.llm_fallback and not self.llm:
            raise ValueError("LLM Client must be provided if LLM fallback is enabled. \nPlease create an instance of the LLM class and pass this in the llm parameter. \nGet an API key from https://platform.openai.com/.")
        
        # If max_workers is set to 0 or a negative number
        if self.max_workers <= 0:
            raise ValueError("max_workers must be a positive integer.")
        
        # If max_requests is set to 0 or a negative number
        if self.max_requests <= 0:
            raise ValueError("max_requests must be a positive integer.")
        
        # If delay_range is not a tuple of two numbers (int or float)
        if not isinstance(self.delay_range, tuple) or len(self.delay_range) != 2 or not all(isinstance(i, (int, float)) for i in self.delay_range):
            raise ValueError("delay_range must be a tuple of two numbers (int or float) - e.g. (1, 2) or (1.5, 2.5). If you are attempting to disable delays, set to (0,0).")

        # If upper bound of delay_range is less than lower bound
        if self.delay_range[1] < self.delay_range[0]:
            raise ValueError("Upper bound of delay_range must be greater than or equal to lower bound.")
        
        # If docx_conversion is not one of the allowed values
        if self.docx_conversion not in ["MicrosoftWord", "LibreOffice", "None"]:
            raise ValueError("docx_conversion must be one of 'MicrosoftWord', 'LibreOffice', or 'None'.")
        
        # If OpenAI API key or client is not provided when LLM fallback is enabled
        if self.llm_fallback and not self.llm:
            raise ValueError("LLM Client must be provided if LLM fallback is enabled. \nPlease create an instance of the LLM class and pass this in the llm parameter. \nGet an API key from https://platform.openai.com/.")
        
        # If no scrape method is enabled
        if not self.html_scraping and not self.pdf_fallback and not self.llm_fallback:
            raise ValueError("At least one of 'html_scraping', 'pdf_fallback', or 'llm_fallback' must be enabled.")

        # If no fields are included
        if not any([self.include_id, self.include_date, self.include_coroner, self.include_area, self.include_receiver, self.include_investigation, self.include_circumstances, self.include_concerns]):
            raise ValueError("At least one field must be included in the output. Please set one or more of the following to True:\n 'include_id', 'include_date', 'include_coroner', 'include_area', 'include_receiver', 'include_investigation', 'include_circumstances', 'include_concerns'.\n")
        
        ### Warnings (code will still run)
        
        # If only html_scraping is enabled
        if self.html_scraping and not self.pdf_fallback and not self.llm_fallback:
            logger.warning("Only HTML scraping is enabled. \nConsider enabling .pdf or LLM fallback for more complete data extraction.\n")
        
        # If only pdf_fallback is enabled
        if not self.html_scraping and self.pdf_fallback and not self.llm_fallback:
            logger.warning("Only .pdf fallback is enabled. \nConsider enabling HTML scraping or LLM fallback for more complete data extraction.\n")
            
        # If only llm_fallback is enabled
        if not self.html_scraping and not self.pdf_fallback and self.llm_fallback:
            logger.warning("Only LLM fallback is enabled. \nWhile this is a high-performance option, large API costs may be incurred, especially for large requests. \nConsider enabling HTML scraping or .pdf fallback for more cost-effective data extraction.\n")
        
        # If max_workers is set above 50
        if self.max_workers > 50:
            logger.warning("max_workers is set to a high value (>50). \nDepending on your system, this may cause performance issues. It could also trigger anti-scraping measures by the host, leading to temporary or permanent IP bans. \nWe recommend setting to between 10 and 50.\n")
        
        # If max_workers is set below 10
        if self.max_workers < 10:
            logger.warning("max_workers is set to a low value (<10). \nThis may result in slower scraping speeds. Consider increasing the value for faster performance. \nWe recommend setting to between 10 and 50.\n")
        
        # If max_requests is set above 10
        if self.max_requests > 10:
            logger.warning("max_requests is set to a high value (>10). \nThis may trigger anti-scraping measures by the host, leading to temporary or permanent IP bans. \nWe recommend setting to between 3 and 10.\n")
            
        # If max_requests is set below 3
        if self.max_requests < 3:
            logger.warning("max_requests is set to a low value (<3). \nThis may result in slower scraping speeds. Consider increasing the value for faster performance. \nWe recommend setting to between 3 and 10.\n")

        # If delay range is set to (0,0)
        if self.delay_range == (0, 0):
            logger.warning("delay_range has been disabled. \nThis will disable delays between requests. This may trigger anti-scraping measures by the host, leading to temporary or permanent IP bans. \nWe recommend setting to (1,2).\n")
        elif self.delay_range[0] < 0.5 and self.delay_range[1] != 0:
            logger.warning("delay_range is set to a low value (<0.5 seconds). \nThis may trigger anti-scraping measures by the host, leading to temporary or permanent IP bans. We recommend setting to between (1, 2).\n")
        
        # If delay_range upper bound is set above 5 seconds
        if self.delay_range[1] > 5:
            logger.warning("delay_range is set to a high value (>5 seconds). \nThis may result in slower scraping speeds. Consider decreasing the value for faster performance. We recommend setting to between (1, 2).\n")

        # -----------------------------------------------------------------------------
        # Log the initialisation parameters for debug if verbose is enabled
        # -----------------------------------------------------------------------------
        
        if verbose:
            logger.info(
                "\nPFDScraper initialised with parameters:\n "
                f"Category: {self.category}\n "
                f"Date Range: {self.date_from} to {self.date_to}\n "
                f"Max Workers: {self.max_workers}\n "
                f"Max Requests: {self.max_requests}\n "
                f"Delay Range: {self.delay_range}\n "
                f"HTML Scraping: {self.html_scraping}\n "
                f"PDF Fallback: {self.pdf_fallback}\n "
                f"LLM Fallback: {self.llm_fallback}\n "
                f"LLM Model: {self.llm_model}\n "
                f"Docx Conversion: {self.docx_conversion}\n "
                f"Include URL: {'Yes' if self.include_url else 'No'}\n "
                f"Include ID: {'Yes' if self.include_id else 'No'}\n "
                f"Include Date: {'Yes' if self.include_date else 'No'}\n "
                f"Include Coroner: {'Yes' if self.include_coroner else 'No'}\n "
                f"Include Area: {'Yes' if self.include_area else 'No'}\n "
                f"Include Receiver: {'Yes' if self.include_receiver else 'No'}\n "
                f"Include Investigation: {'Yes' if self.include_investigation else 'No'}\n "
                f"Include Circumstances: {'Yes' if self.include_circumstances else 'No'}\n "
                f"Include Concerns: {'Yes' if self.include_concerns else 'No'}\n "
                f"Include Time Stamp: {'Yes' if self.include_time_stamp else 'No'}\n "
                f"Verbose: {'Yes' if self.verbose else 'No'}\n "
    )
        
        # -----------------------------------------------------------------------------
        # Setting up a requests session with retry logic:
        # - Configures a session that automatically retries failed requests.
        # - Should handle temporary network issues.
        # -----------------------------------------------------------------------------
        
        self.session = requests.Session()
        retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504])
        adapter = HTTPAdapter(max_retries=retries, pool_connections=100, pool_maxsize=100)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # -----------------------------------------------------------------------------
        # Pre-compile regular expression for extracting report IDs.
        # Example report ID format: "2025-0296".
        # -----------------------------------------------------------------------------
        self._id_pattern = re.compile(r'(\d{4}-\d{4})')
        
        
    # -----------------------------------------------------------------------------
    # Link fetching logic
    # -----------------------------------------------------------------------------
        
    def _get_report_href_values(self, url: str) -> list:
        """
        Extracts URLs from <a> tags on a page, applying a random delay and limiting concurrent
        requests to the host using a semaphore.
        """
        with self.domain_semaphore:
            # Introduce a random delay between delay_range[0] and delay_range[1] secs. Default is between 1 and 2 secs.
            time.sleep(random.uniform(*self.delay_range))
            try:
                response = self.session.get(url)
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
    
    def get_report_links(self) -> list:
        """
        Dynamically collects all PFD report links from consecutive pages until a page returns no new links.
        
        :return: A list of report URLs.
        """
        self.report_links = []
        page = self.start_page  # Always start at page 1
        
        # Create a progress bar with an unknown total
        pbar = tqdm(desc="Fetching pages", unit=" page(s)", leave=False, initial=page)
        
        while True:
            # Format the URL with both the page number and the date parameters
            page_url = self.page_template.format(page=page, **self.date_params)
            href_values = self._get_report_href_values(page_url)
            pbar.update(1)
            
            if self.verbose:
                logger.info("Scraped %d links from %s", len(href_values), page_url)
            if not href_values:
                # If no links are returned, assume we've reached the end and stop the loop
                break
            self.report_links.extend(href_values)
            page += 1  # Move to the next page
        
        # Throw error if no report links are found
        if len(self.report_links) == 0:
            logger.error("\nNo report links found. Please check your date range.")
            return 
        
        logger.info("Total collected report links: %d", len(self.report_links))
        return self.report_links

    # -----------------------------------------------------------------------------
    # Report extraction logic
    # -----------------------------------------------------------------------------
    
    @staticmethod
    def _normalise_apostrophes(text: str) -> str:
        """Helper function to replace ‘fancy’ (typographic) apostrophes with the standard apostrophe.
        
        Some reports use fancy apostrophes (‘ and ’) over typical 'keyboard' apostrophes (') which can cause issues with text processing.
        
        """
        return text.replace("’", "'").replace("‘", "'")

    def _clean_text(self, text: str) -> str:
        """Helper function to clean text by removing excessive whitespace & replacing typographic apostrophes."""
        normalised = self._normalise_apostrophes(text)
        return " ".join(normalised.split())

    def _normalise_date(self, raw: str) -> str:
        """
        Convert any human‑readable date to ISO‑8601 (YYYY‑MM‑DD).
        Returns the original string on failure.
        """
        try:
            dt = parser.parse(raw, fuzzy=True, dayfirst=True)
            return dt.strftime("%Y-%m-%d")
        except Exception as e:
            logger.warning("Date parse failed for '%s' – keeping raw (%s)", raw, e)
            return raw

    def _extract_text_from_pdf(self, pdf_url: str) -> str:
        """
        Internal function to download and extract text from a .pdf report. If the file is not in .pdf format (.docx or .doc),
        converts it to .pdf using the method specified by self.docx_conversion.
        
        :param pdf_url: URL of the file to extract text from.
        :return: Cleaned text extracted from the .pdf, or "N/A" on failure.
        """
        # Find the file extension from URL, in case it's not .pdf as expected
        parsed_url = urlparse(pdf_url)
        path = unquote(parsed_url.path)
        ext = os.path.splitext(path)[1].lower()
        
        if self.verbose:
            logger.debug(f"Processing .pdf {pdf_url}.")
        
        # Download the file content as bytes
        try:
            response = self.session.get(pdf_url)
            response.raise_for_status()
            file_bytes = response.content
        except requests.RequestException as e:
            logger.error("Failed to fetch file: %s; Error: %s", pdf_url, e)
            return "N/A"
        
        pdf_bytes = None
        if ext != ".pdf":
            logger.info("File %s is not a .pdf (extension %s)", pdf_url, ext)
            if self.docx_conversion == "MicrosoftWord":
                logger.info("Attempting conversion using Microsoft Word...")
                try:
                    from docx2pdf import convert
                except ImportError:
                    logger.error("docx2pdf is not installed. Please install it with 'pip install docx2pdf'.")
                    return "N/A"
                try:
                    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_in:
                        tmp_in.write(file_bytes)
                        tmp_in.flush()
                        input_path = tmp_in.name
                    output_path = input_path.rsplit(ext, 1)[0] + ".pdf"
                    convert(input_path, output_path)
                    with open(output_path, "rb") as f:
                        pdf_bytes = f.read()
                    os.remove(input_path)
                    os.remove(output_path)
                    logger.info("Conversion successful using Microsoft Word! Proceeding with .pdf extraction...")
                except Exception as e:
                    logger.error("Conversion using Microsoft Word failed: %s", e)
                    return "N/A"
            elif self.docx_conversion == "LibreOffice":
                logger.info("Attempting conversion using LibreOffice...")
                try:
                    with tempfile.NamedTemporaryFile(suffix=ext, delete=False) as tmp_in:
                        tmp_in.write(file_bytes)
                        tmp_in.flush()
                        input_path = tmp_in.name
                    output_path = input_path.rsplit(ext, 1)[0] + ".pdf"
                    subprocess.run(
                        ["soffice", "--headless", "--convert-to", "pdf", input_path, "--outdir", os.path.dirname(input_path)],
                        check=True
                    )
                    with open(output_path, "rb") as f:
                        pdf_bytes = f.read()
                    os.remove(input_path)
                    os.remove(output_path)
                    logger.info("Conversion successful using LibreOffice! Proceeding with .pdf extraction...")
                except Exception as e:
                    logger.error("Conversion using LibreOffice failed: %s", e)
                    return "N/A"
            else:
                logger.info("docx_conversion is set to 'None'; skipping conversion!")
                return "N/A"
        else:
            pdf_bytes = file_bytes
        
        # If llm_fallback is enabled, cache the downloaded .pdf bytes for later reuse
        if self.llm_fallback:
            self._last_pdf_bytes = pdf_bytes
        
        # Use pymupdf to read and extract text from the .pdf
        try:
            pdf_buffer = BytesIO(pdf_bytes)
            pdf_document = pymupdf.open(stream=pdf_buffer, filetype="pdf")
            text = "".join(page.get_text() for page in pdf_document)
            pdf_document.close()
        except Exception as e:
            logger.error("Error processing .pdf %s: %s", pdf_url, e)
            return "N/A"
        
        return self._clean_text(text)
    
    # Method 1 of HTML extraction that looks for a <p> tag containing keywords
    # This is intended for the report metadata, such as date, receiver, id - which are usually in a <p> tag.
    def _extract_paragraph_text_by_keywords(self, soup: BeautifulSoup, keywords: list) -> str:
        """
        Internal function to search for a <p> element in the HTML that contains any of the provided keywords.
        
        :param soup: BeautifulSoup object of the page.
        :param keywords: List of keywords to search for.
        :return: Extracted text or 'N/A: Not found'.
        """
        for keyword in keywords:
            element = soup.find(lambda tag: tag.name == 'p' and keyword in tag.get_text(), recursive=True)
            if element:
                return self._clean_text(element.get_text())
        return 'N/A: Not found'
    
    # Method 2 of HTML extraction that looks for a <strong> tag containing keywords
    # This is intended for the main report sections, such as Investigation, Circumstances of Death, etc.
    def _extract_section_text_by_keywords(
        self, soup: BeautifulSoup, header_keywords: list
    ) -> str:
        """
        Extracts a block of text from HTML by locating a header (within <strong> tags) that matches
        one of the provided header keywords, then collecting all sibling elements that follow.
        

        :param soup: BeautifulSoup object of the page.
        :param header_keywords: List of header keyword variations to search for.
        :return: Extracted section text or 'N/A: Not found'.
        """
        # The below is more extensively commented because it has a low success rate and could possibly be improved.
        
        # Look for all <strong> tags which (hopefully!) contain section headers
        for strong in soup.find_all('strong'):
            header_text = strong.get_text(strip=True)
            # Check if any of the header keywords match
            if any(keyword.lower() in header_text.lower() for keyword in header_keywords):
                content_parts = []
                # Iterate over siblings that follow the header within the same parent element
                for sibling in strong.next_siblings:
                    # If the sibling is a string, strip it
                    if isinstance(sibling, str):
                        text = sibling.strip()
                        if text:
                            content_parts.append(text)
                    else:
                        # For tags, get their text content (thru handling inner <br> tags by using a space as separator
                        text = sibling.get_text(separator=" ", strip=True)
                        if text:
                            content_parts.append(text)
                # Return the concatenated content if found
                if content_parts:
                    return self._clean_text(" ".join(content_parts))
        return "N/A: Not found"

    # Our single method for extracting a section from the .pdf version of the report
    def _extract_section_from_pdf_text(self, text: str, start_keywords: list, end_keywords: list) -> str:
        """
        Internal function to extract a section from text using multiple start and end keywords.
        Uses case-insensitive search to locate the markers.
        
        :param text: The full text to search in.
        :param start_keywords: List of possible starting keywords.
        :param end_keywords: List of possible ending keywords.
        :return: Extracted section text or "N/A: Not found" if markers are missing.
        """
        lower_text = text.lower()
        for start in start_keywords:
            start_lower = start.lower()
            start_index = lower_text.find(start_lower)
            if start_index != -1:
                section_start = start_index + len(start_lower)
                end_indices = []
                for end in end_keywords:
                    end_lower = end.lower()
                    end_index = lower_text.find(end_lower, section_start)
                    if end_index != -1:
                        end_indices.append(end_index)
                if end_indices:
                    section_end = min(end_indices)
                    return text[section_start:section_end]
                else:
                    return text[section_start:]
        return "N/A: Not found"
    
    # -----------------------------------------------------------------------------
    # LLM Report Extraction Logic
    # ----------------------------------------------------------------------------- 

    def _fetch_pdf_bytes(self, report_url: str) -> bytes:
        """
        Helper to fetch the .pdf bytes given a report URL.
        """
        try:
            page_response = self.session.get(report_url)
            page_response.raise_for_status()
            soup = BeautifulSoup(page_response.content, 'html.parser')
            pdf_links = [a['href'] for a in soup.find_all('a', class_='govuk-button') if a.get('href')]
            if pdf_links:
                pdf_link = pdf_links[0]
                pdf_response = self.session.get(pdf_link)
                pdf_response.raise_for_status()
                return pdf_response.content
            else:
                logger.error("No PDF link found for report at %s", report_url)
                return None
        except Exception as e:
            logger.error("Failed to fetch PDF for report at %s: %s", report_url, e)
            return None

    def _construct_missing_fields_model(self, required_fields: list[str]) -> Type[BaseModel]:
        fields = {field: (str, ...) for field in required_fields}
        return create_model(
            "MissingFields",
            **fields,
            __doc__="Missing fields populated with the required information from the PFD report.",
        )

    def _call_llm_fallback(self, pdf_bytes: bytes, missing_fields: dict, report_url: str | None = None) -> dict:
        """
        Helper that converts pdf_bytes to images, builds a prompt based on missing_fields,
        calls the LLM API, and parses the response.
        
        Returns a dictionary of fallback updates.
        """
        base64_images = [] # ...OpenAI requires images to be base64 encoded
        if pdf_bytes:
            try:
                images = convert_from_bytes(pdf_bytes)
                for img in images:
                    buffered = BytesIO()
                    img.save(buffered, format="JPEG")
                    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    base64_images.append(img_str)
            except Exception as e:
                logger.error("Error converting PDF to images: %s", e)
        
        prompt = (
            "Your goal is to transcribe the **exact** text from this report, presented as images.\n\n"
            "Please extract the following section(s):\n"
        )
        response_fields = []
        for field, instruction in missing_fields.items():
            response_fields.append(field)
            prompt += f"\n{field}: {instruction}\n"
        prompt += (
            "\nRespond with nothing else whatsoever. You must not respond in your own 'voice' or even acknowledge the task.\n"
            'If you are unable to identify the text from the image for any given section, simply respond: "N/A: Not found" for that section.\n'
            "Sometimes text may be redacted with a black box; transcribe it as '[REDACTED]'.\n"
            "Make sure you transcribe the *full* text for each section, not just a snippet.\n"
            "Do *not* change the section title(s) from the above format.\n"
            "Respond in the specified response format\n"
        )
        #if self.verbose:
        #    logger.info("LLM prompt:\n\n%s", prompt)
        # Construct dynamic response_format model
        missing_fields_model = self._construct_missing_fields_model(
            required_fields=response_fields
        )
        try:
            output = self.llm.generate(
                prompt=prompt,
                images=base64_images,
                response_format=missing_fields_model,
            )

            if self.verbose:
                # Display the returned populated model
                logger.info(
                    "LLM fallback response:\n%s\n\n", output.model_dump_json(indent=2)
                )
        except Exception as e:
            logger.error(f"LLM fallback failed: {e}")
            return {}
        
        fallback_updates = {}
        output_json = output.model_dump()
        
        if self.verbose:
            logger.info("LLM fallback for report: %s", report_url)
            logger.info("Output JSON: %s", output_json)
            logger.info("Missing fields were: %s", response_fields)
            
        for field in response_fields:
            try:
                fallback_updates[field] = output_json[field]
            except Exception as e:
                print(e)
                fallback_updates[field] = "LLM Fallback failed"
    
        return fallback_updates
    
    def _extract_report_info(self, url: str) -> dict:
        """
        Extract metadata and text from a PFD report webpage.
        
        Process:
          1. Download the webpage and parse it using BeautifulSoup.
          2. Identify the .pdf download link.
          3. Extract text from the .pdf file.
          4. Scrape various metadata (report ID, date, receiver, coroner name, area,
             investigation details, circumstances, and concerns) from the HTML.
          5. If one or more fields are missing and PDF fallback is enabled, attempt to extract missing
             data from the PDF text.
          6. Optionally, if llm_fallback is enabled, use OpenAI GPT to extract missing data from images
             generated from the PDF.
        
        :param url: URL of the report page.
        :return: Dictionary containing extracted report information.
        """
        
        date_scraped = datetime.now().strftime("%Y-%m-%d %H:%M:%S") if self.include_time_stamp else None
        
        # Initialise all fields with default missing values.
        #   We do this because if `html_scraping` is disabled, we need to ensure all fields are still set,
        #   because our .pdf and LLM fallbacks only kick in if a field is missing.
        
        report_id = "N/A: Not found"
        date = "N/A: Not found"
        receiver = "N/A: Not found"
        coroner = "N/A: Not found"
        area = "N/A: Not found"
        investigation = "N/A: Not found"
        circumstances = "N/A: Not found"
        concerns = "N/A: Not found"
        
        try:
            response = self.session.get(url)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error("Failed to fetch %s; Error: %s", url, e)
            return None
        
        soup = BeautifulSoup(response.content, 'html.parser')
        pdf_links = [a['href'] for a in soup.find_all('a', class_='govuk-button') if a.get('href')]
        if not pdf_links:
            logger.error("No .pdf links found on %s", url)
            return None
        
        report_link = pdf_links[0]
        pdf_text = self._extract_text_from_pdf(report_link)
        
        
        # -----------------------------------------------------------------------
        #                          HTML Data Extraction                          
        # -----------------------------------------------------------------------
        
        # The below is the primary method of data extraction, using the HTML content. 
        # `html_scraping` must be set to True (default).
        
        if self.html_scraping:
            if self.verbose:
                logger.debug(f"Extracting data from HTML for URL: {url}")
        
            # Report ID extraction using compiled regex (e.g. "2025-0296")
            if self.include_id:
                ref_element = soup.find(lambda tag: tag.name == 'p' and 'Ref:' in tag.get_text(), recursive=True)
                if ref_element:
                    match = self._id_pattern.search(ref_element.get_text())
                    report_id = match.group(1) if match else 'N/A: Not found'
                else:
                    report_id = "N/A: Not found"
            
            
            # Date of report extraction
            if self.include_date:
                date_element = self._extract_paragraph_text_by_keywords(soup, ["Date of report:"])
                if date_element != "N/A: Not found":
                    date_element = date_element.replace("Date of report:", "").strip()
                    date = self._normalise_date(date_element)
                
            
            # Receiver extraction (who the report is sent to)
            if self.include_receiver:
                receiver_element = self._extract_paragraph_text_by_keywords(
                    soup, ["This report is being sent to:", "Sent to:"]
                )
                receiver = receiver_element.replace("This report is being sent to:", "") \
                                            .replace("Sent to:", "") \
                                            .strip()
                                            
                if len(receiver) < 5 or len(receiver) > 20:
                    receiver = 'N/A: Not found'
                            
            
            # Name of coroner extraction
            if self.include_coroner:
                coroner_element = self._extract_paragraph_text_by_keywords(
                    soup, ["Coroners name:", "Coroner name:", "Coroner's name:"]
                )
                coroner = coroner_element.replace("Coroners name:", "") \
                                            .replace("Coroner name:", "") \
                                            .replace("Coroner's name:", "") \
                                            .strip()
                if len(coroner) < 5 or len(coroner) > 20:
                    coroner = 'N/A: Not found'
            
            
            # Area extraction
            if self.include_area:
                area_element = self._extract_paragraph_text_by_keywords(
                    soup, ["Coroners Area:", "Coroner Area:", "Coroner's Area:"]
                )
                area = area_element.replace("Coroners Area:", "") \
                                    .replace("Coroner Area:", "") \
                                    .replace("Coroner's Area:", "") \
                                    .strip()
                                    
                if len(area) < 4 or len(area) > 40:
                    area = 'N/A: Not found'
            
            
            # Investigation and Inquest extraction
            if self.include_investigation:
                investigation_section = self._extract_section_text_by_keywords(
                    soup, ["INVESTIGATION and INQUEST", "INVESTIGATION & INQUEST", "3 INQUEST"]
                )
                investigation = investigation_section.replace("INVESTIGATION and INQUEST", "") \
                                                    .replace("INVESTIGATION & INQUEST", "") \
                                                    .replace("3 INQUEST", "") \
                                                    .strip()

                if len(investigation) < 30:
                    investigation = 'N/A: Not found'
            
            
            # Circumstances of the Death extraction
            if self.include_circumstances:
                circumstances_section = self._extract_section_text_by_keywords(
                    soup, 
                    ["CIRCUMSTANCES OF THE DEATH", "CIRCUMSTANCES OF DEATH", "CIRCUMSTANCES OF"]
                )
                circumstances = circumstances_section.replace("CIRCUMSTANCES OF THE DEATH", "") \
                                                    .replace("CIRCUMSTANCES OF DEATH", "") \
                                                    .replace("CIRCUMSTANCES OF", "") \
                                                    .strip()
                if len(circumstances) < 30:
                    circumstances = 'N/A: Not found'


            # Coroner's Concerns extraction
            if self.include_concerns:
                concerns_text = self._extract_section_text_by_keywords(
                    soup, 
                    ["CORONER'S CONCERNS", "CORONERS CONCERNS", "CORONER CONCERNS"]
                )
                concerns = concerns_text.replace("CORONER'S CONCERNS", "") \
                                                    .replace("CORONERS CONCERNS", "") \
                                                    .replace("CORONER CONCERNS", "") \
                                                    .strip()
                if len(concerns) < 30:
                    concerns = 'N/A: Not found'

        
        # -----------------------------------------------------------------------------
        #                          .pdf Data Extraction Fallback                           
        # -----------------------------------------------------------------------------
        
        # The below will only run if (1) pdf_fallback is enabled and (2) one or more fields are missing
        #    following previous extraction method (if any). This will always run if `html_scraping` is 
        #    disabled.
        
        if self.pdf_fallback and (
            "N/A: Not found" in [coroner, area, receiver, investigation, circumstances, concerns]
        ):
            if self.verbose:
                logger.debug(f"Initiating .pdf fallback for URL: {url} because one or more fields are missing.")

            if self.include_coroner and coroner == "N/A: Not found":
                coroner_element = self._extract_section_from_pdf_text(
                    pdf_text,
                    start_keywords=["I am", "CORONER"],
                    end_keywords=["CORONER'S LEGAL POWERS", "paragraph 7"]
                )
                coroner = coroner_element.replace("I am", "") \
                                        .replace("CORONER'S LEGAL POWERS", "") \
                                        .replace("CORONER", "") \
                                        .replace("paragraph 7", "") \
                                        .strip()
            # Area extraction if missing
            if self.include_area and area == "N/A: Not found":
                area_element = self._extract_section_from_pdf_text(
                    pdf_text,
                    start_keywords=["area of"],
                    end_keywords=["LEGAL POWERS", "LEGAL POWER", "paragraph 7"]
                )
                area = area_element.replace("area of", "") \
                                .replace("CORONER'S", "") \
                                .replace("CORONER", "") \
                                .replace("CORONERS", "") \
                                .replace("paragraph 7", "") \
                                .strip()

            # Receiver extraction if missing
            if self.include_receiver and receiver == "N/A: Not found":
                receiver_element = self._extract_section_from_pdf_text(
                    pdf_text,
                    start_keywords=[" SENT ", "SENT TO:"],
                    end_keywords=["CORONER", "CIRCUMSTANCES OF THE DEATH", "CIRCUMSTANCES OF"]
                )
                receiver = self._clean_text(receiver_element).replace("TO:", "").strip()
                if len(receiver) < 5:
                    receiver = 'N/A: Not found'
            # Investigation & Inquest extraction if missing
            if self.include_investigation and investigation == "N/A: Not found":
                investigation_element = self._extract_section_from_pdf_text(
                    pdf_text,
                    start_keywords=["INVESTIGATION and INQUEST", "3 INQUEST"],
                    end_keywords=["CIRCUMSTANCES OF DEATH", "CIRCUMSTANCES OF THE DEATH", "CIRCUMSTANCES OF"]
                )
                investigation = self._clean_text(investigation_element)
                if len(investigation) < 30:
                    investigation = 'N/A: Not found'
            # Circumstances of Death extraction if missing
            if self.include_circumstances and circumstances == "N/A: Not found":
                circumstances_section = self._extract_section_from_pdf_text(
                    pdf_text,
                    start_keywords=["CIRCUMSTANCES OF DEATH", "CIRCUMSTANCES OF THE DEATH", "CIRCUMSTANCES OF"],
                    end_keywords=["CORONER'S CONCERNS", "CORONER CONCERNS", "CORONERS CONCERNS", "as follows"]
                )
                circumstances = self._clean_text(circumstances_section)
                if len(circumstances) < 30:
                    circumstances = 'N/A: Not found'
            # Matters of Concern extraction if missing
            if self.include_concerns and concerns == "N/A: Not found":
                concerns_section = self._extract_section_from_pdf_text(
                    pdf_text,
                    start_keywords=["CORONER'S CONCERNS", "as follows"],
                    end_keywords=["ACTION SHOULD BE TAKEN"]
                )
                concerns = self._clean_text(concerns_section)
                if len(concerns) < 30:
                    concerns = 'N/A: Not found'

        # -----------------------------------------------------------------------------
        #                         LLM Data Extraction Fallback                          
        # -----------------------------------------------------------------------------
        
        # The below will only run if (1) llm_fallback is enabled and (2) one or more elements are missing
        #   following previous extraction method. This will always run if both `html_scraping` and` 
        #   `pdf_fallback` are disabled.
        
        if self.llm_fallback:
            missing_fields = {}
            if self.include_date and date == "N/A: Not found":
                missing_fields["date of report"] = "[Date of the report, not the death]"
            if self.include_coroner and coroner == "N/A: Not found":
                missing_fields["coroner's name"] = "[Name of the coroner. Provide the name only.]"
            if self.include_area and area == "N/A: Not found":
                missing_fields["area"] = "[Area/location of the Coroner. Provide the location itself only.]"
            if self.include_receiver and receiver == "N/A: Not found":
                missing_fields["receiver"] = "[Recipients of the report. Always extract the role & organisation of the recipients, if provided, and *not* individual names -- if no organisational details are provided the name is fine. ]"
            if self.include_investigation and investigation == "N/A: Not found":
                missing_fields["investigation and inquest"] = "[The text from the Investigation/Inquest section.]"
            if self.include_circumstances and circumstances == "N/A: Not found":
                missing_fields["circumstances of death"] = "[The text from the Circumstances of Death section.]"
            if self.include_concerns and concerns == "N/A: Not found":
                missing_fields["coroner's concerns"] = "[The text from the Coroner's Concerns section. This is sometimes under 'Matters of Concern'.]"
            if missing_fields:
                # Attempt to use cached PDF bytes or re-fetch if needed
                pdf_bytes = getattr(self, '_last_pdf_bytes', None)
                if pdf_bytes is None:
                    pdf_bytes = self._fetch_pdf_bytes(report_link)

                fallback_updates = self._call_llm_fallback(pdf_bytes, missing_fields, report_url=report_url)
                if fallback_updates:
                    if ("date of report" in fallback_updates
                            and fallback_updates["date of report"] != "N/A: Not found"):
                        fallback_updates["date of report"] = self._normalise_date(
                            fallback_updates["date of report"]
                        )
                        date = fallback_updates["date of report"]
                    if self.include_coroner and coroner == "N/A: Not found" and "coroner's name" in fallback_updates:
                        coroner = fallback_updates["coroner's name"]
                    if self.include_area and area == "N/A: Not found" and "area" in fallback_updates:
                        area = fallback_updates["area"]
                    if self.include_receiver and receiver == "N/A: Not found" and "receiver" in fallback_updates:
                        receiver = fallback_updates["receiver"]
                    if self.include_investigation and investigation == "N/A: Not found" and "investigation and inquest" in fallback_updates:
                        investigation = fallback_updates["investigation and inquest"]
                    if self.include_circumstances and circumstances == "N/A: Not found" and "circumstances of death" in fallback_updates:
                        circumstances = fallback_updates["circumstances of death"]
                    if self.include_concerns and concerns == "N/A: Not found" and "coroner's concerns" in fallback_updates:
                        concerns = fallback_updates["coroner's concerns"]



        # Return the extracted report information
        report = {}
        if self.include_url:
            report["URL"] = url
        if self.include_id:
            report["ID"] = report_id
        if self.include_date:
            report["Date"] = date
        if self.include_coroner:
            report["CoronerName"] = coroner
        if self.include_area:
            report["Area"] = area
        if self.include_receiver:
            report["Receiver"] = receiver
        if self.include_investigation:
            report["InvestigationAndInquest"] = investigation
        if self.include_circumstances:
            report["CircumstancesOfDeath"] = circumstances
        if self.include_concerns:
            report["MattersOfConcern"] = concerns
        if self.include_time_stamp:
            report["DateScraped"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return report

    # -----------------------------------------------------------------------------------------
    # PUBLIC METHODS - These are the two main methods that the user will interact with.
    # -----------------------------------------------------------------------------------------
    # 1) scrape_reports(): Scrapes reports from the collected report links based on the user configuration.
    # 2) top_up(): Adds new reports to the existing scraped reports based on the user configuration.
    # 3) run_llm_fallback(): Runs the LLM fallback. This is a separate method to allow the user to run this
    #                        separately if needed.
    # 4) estimate_api_costs(): Estimates the API costs for the current configuration. The user can supply
    #                          a dataframe with their existing scrape
    
    def scrape_reports(self) -> pd.DataFrame:
        """
        Scrapes reports from the collected report links based on the user configuration of the PFDScraper class instance (self).
        
        :return: A pandas DataFrame containing one row per scraped report.
        """
        if not self.report_links:
            self.get_report_links()
        
        # Use a thread pool to concurrently scrape the new report links
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            futures = [executor.submit(self._extract_report_info, url) for url in self.report_links]
            results = []
            for future in tqdm(as_completed(futures), total=len(futures), desc="Scraping reports"):
                results.append(future.result())
            
            
        # Filter out any failed report extractions 
        reports = [report for report in results if report is not None]
        
        # Create timestamp if parameter is set to True
        reports_df = pd.DataFrame(reports)
        
        # Save the reports internally in case the user forgets to assign
        self.reports = pd.DataFrame(reports_df)
        
        return reports_df


    def top_up(self, old_reports: pd.DataFrame = None, date_from: str = None, date_to: str = None) -> pd.DataFrame:
        """
        Adds new reports to the existing scraped reports based on the user configuration of the PFDScraper class instance (self).
        Duplicate checking is based on the URL as a unique identifier – by default the URL (if include_url is True)
        or the report ID otherwise.
        
        Parameters:
            old_reports (pd.DataFrame): Optional DataFrame containing previously scraped reports. If not supplied,
                                         the internal self.reports will be used.
            date_from (str): Optional new start date in YYYY-MM-DD format.
            date_to (str): Optional new end date in YYYY-MM-DD format.
        
        Returns:
            pd.DataFrame: Updated DataFrame containing both the old and new scraped reports.
        """
        
        logger.info("Attempting to 'top up' the existing reports with new data.")
        
        # Update the date range if new parameters are provided
        if date_from is not None or date_to is not None:
            new_date_from = parser.parse(date_from) if date_from is not None else self.date_from
            new_date_to = parser.parse(date_to) if date_to is not None else self.date_to
            
            if new_date_from > new_date_to:
                raise ValueError("date_from must be before date_to.")
            
            self.date_from = new_date_from
            self.date_to = new_date_to
            self.date_params = {
                "after_day": self.date_from.day,
                "after_month": self.date_from.month,
                "after_year": self.date_from.year,
                "before_day": self.date_to.day,
                "before_month": self.date_to.month,
                "before_year": self.date_to.year,
            }
        
        # Use the provided DataFrame if supplied, or fall back to the internal self.reports
        base_df = old_reports if old_reports is not None else self.reports

        # Check that base_df contains the required columns; if not, throw an error
        # This is an interim measure! We need to implement a mapping feature so that if the user
        #    renames columns, the top up scraper can still work.
        required_columns = []
        if self.include_url:
            required_columns.append("URL")
        if self.include_id:
            required_columns.append("ID")
        if self.include_date:
            required_columns.append("Date")
        if self.include_coroner:
            required_columns.append("CoronerName")
        if self.include_area:
            required_columns.append("Area")
        if self.include_receiver:
            required_columns.append("Receiver")
        if self.include_investigation:
            required_columns.append("InvestigationAndInquest")
        if self.include_circumstances:
            required_columns.append("CircumstancesOfDeath")
        if self.include_concerns:
            required_columns.append("MattersOfConcern")
        if self.include_time_stamp:
            required_columns.append("DateScraped")

        if base_df is not None:
            missing_cols = [col for col in required_columns if col not in base_df.columns]
            if missing_cols:
                raise ValueError(f"Required columns missing from the provided DataFrame: {missing_cols}")

        # Retrieve the latest report links from the website.
        updated_links = self.get_report_links()
        
        # Determine which unique key to use for duplicate checking
        if self.include_url:
            unique_key = "URL"
        elif self.include_id:
            unique_key = "ID"
        else:
            logger.error("No unique identifier available for duplicate checking.\n"
                         "Ensure include_url or include_id was set to True in instance initialisation.")
            return base_df if base_df is not None else pd.DataFrame()

        # Gather identifiers from the base DataFrame
        if base_df is not None and unique_key in base_df.columns:
            existing_identifiers = set(base_df[unique_key].tolist())
        else:
            existing_identifiers = set()

        # Identify new report links by filtering out duplicates
        new_links = [link for link in updated_links if link not in existing_identifiers]
        duplicates_count = len(updated_links) - len(new_links)
        new_count = len(new_links)
        
        logger.info("Top-up: %d new report(s) found; %d duplicate(s) which won't be added", new_count, duplicates_count)

        if not new_links:
            logger.info("No new reports to scrape during top-up.")
            return None  # Don't return anything if there are no new reports to scrape

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            new_results = list(tqdm(executor.map(self._extract_report_info, new_links), total=len(new_links), desc="Topping up reports"))

        new_records = [record for record in new_results if record is not None]

        # Convert new records to DataFrame
        new_df = pd.DataFrame(new_records)

        # Add the new records to the old dataframe
        if base_df is not None:
            updated_reports = pd.concat([base_df, new_df], ignore_index=True)
        else:
            updated_reports = new_df

        # Update the internal self.reports if new reports were successfully added
        self.reports = updated_reports
        return updated_reports


    def run_llm_fallback(self, reports_df: pd.DataFrame = None):
        """ 
        Runs the LLM fallback on already scraped reports that have at least one missing field.
        For each report (row) where any enabled field has the value "N/A: Not found", it will
        re-fetch the .pdf and update those missing fields.
        
        Parameters:
            reports_df (pd.DataFrame): Optional DataFrame of scraped reports. If not provided,
                                       self.reports will be used.
                                       
        Returns:
            pd.DataFrame: The updated DataFrame with fallback values filled in.
        """
        
        if reports_df is None:
            if self.reports is None:
                raise ValueError("No scraped reports found. Please run scrape_reports() first.")
            reports_df = self.reports.copy()
        
        # Iterate over each report row in the DataFrame with tqdm progress bar
        
        for idx, row in tqdm(reports_df.iterrows(), total=len(reports_df), desc="Running LLM Fallback"):
            missing_fields = {}
            if self.include_date and row.get("Date", "") == "N/A: Not found":
                missing_fields["date of report"] = "[Date of the report, not the death]"
            if self.include_coroner and row.get("CoronerName", "") == "N/A: Not found":
                missing_fields["coroner's name"] = "[Name of the coroner. Provide the name only.]"
            if self.include_area and row.get("Area", "") == "N/A: Not found":
                missing_fields["area"] = "[Area/location of the Coroner. Provide the location itself only.]"
            if self.include_receiver and row.get("Receiver", "") == "N/A: Not found":
                missing_fields["receiver"] = "[Name or names of the recipient(s) as provided in the report.]"
            if self.include_investigation and row.get("InvestigationAndInquest", "") == "N/A: Not found":
                missing_fields["investigation and inquest"] = "[The text from the Investigation/Inquest section.]"
            if self.include_circumstances and row.get("CircumstancesOfDeath", "") == "N/A: Not found":
                missing_fields["circumstances of death"] = "[The text from the Circumstances of Death section.]"
            if self.include_concerns and row.get("MattersOfConcern", "") == "N/A: Not found":
                missing_fields["coroner's concerns"] = "[The text from the Coroner's Concerns section.]"
            
            if missing_fields:
                report_url = row.get("URL")
                if report_url:
                    pdf_bytes = self._fetch_pdf_bytes(report_url)
                else:
                    pdf_bytes = None

                fallback_updates = self._call_llm_fallback(pdf_bytes, missing_fields, report_url=report_url)
                # Update the dataframe row with any fallback values that were returned.
                if ("date of report" in fallback_updates
                        and fallback_updates["date of report"] != "N/A: Not found"):
                    fallback_updates["date of report"] = self._normalise_date(
                        fallback_updates["date of report"]
                    )
                    reports_df.at[idx, "Date"] = fallback_updates["date of report"]
                if "coroner's name" in fallback_updates:
                    reports_df.at[idx, "CoronerName"] = fallback_updates["coroner's name"]
                if "area" in fallback_updates:
                    reports_df.at[idx, "Area"] = fallback_updates["area"]
                if "receiver" in fallback_updates:
                    reports_df.at[idx, "Receiver"] = fallback_updates["receiver"]
                if "investigation and inquest" in fallback_updates:
                    reports_df.at[idx, "InvestigationAndInquest"] = fallback_updates["investigation and inquest"]
                if "circumstances of death" in fallback_updates:
                    reports_df.at[idx, "CircumstancesOfDeath"] = fallback_updates["circumstances of death"]
                if "coroner's concerns" in fallback_updates:
                    reports_df.at[idx, "MattersOfConcern"] = fallback_updates["coroner's concerns"]
        
        # Update the internal reports attribute and return the updated DataFrame.
        self.reports = reports_df.copy()
        return reports_df


    def estimate_api_costs(self, df: pd.DataFrame = None) -> float:
        """
        Estimates the API cost in USD for LLM fallback based on the number of missing fields
        in the scraped reports. This method uses the pricing structure for tokens and looks at
        each missing cell in the DataFrame.
        
        Parameters:
            df (pd.DataFrame): Optional DataFrame containing scraped reports. If not supplied,
                               self.reports is used.
                               
        Returns:
            float: The estimated API cost in USD.
        """
        
        # Check if the LLM client is set
        if self.llm is None:
            raise RuntimeError(
                "estimate_api_costs() needs an LLM client. "
                "Pass one to the constructor or skip this call."
            )
            
        # Use the provided DataFrame or default to self.reports
        if df is None:
            if self.reports is None:
                logger.error("No scraped reports available for cost estimation. Please run scrape_reports() first.")
                return 0.0
            df = self.reports

        # Pricing structure per million tokens (input and output)
        # ...unfortunately, OpenAI does not (I think!) provide a public API for querying the pricing structure
        #    so we have to hardcode it.
        MODEL_PRICING_PER_1M_TOKENS = {
            "gpt-4o-mini": {"input": 0.15, "output": 0.60, "cached_input": 0.075},
            "gpt-4o": {"input": 2.50, "output": 10.00, "cached_input": 1.25},
            "gpt-4.5-preview": {"input": 75.00, "output": 150.00, "cached_input": 37.50},
            "o1-mini": {"input": 1.10, "output": 4.40, "cached_input": 0.55},
            "o1": {"input": 15.00, "output": 60.00, "cached_input": 7.50},
            "o1-pro": {"input": 150.00, "output": 600.00, "cached_input": 600.00},
            "o3-mini": {"input": 1.10, "output": 4.40, "cached_input": 0.55},
            "gpt-4-turbo": {"input": 10.00, "output": 30.00, "cached_input": 10.00},
            "gpt-4": {"input": 30.00, "output": 60.00, "cached_input": 30.00},
            "gpt-3.5-turbo": {"input": 0.50, "output": 1.50, "cached_input": 0.50},
        }
        
        # Throw error if the model is not found in the pricing structure
        if self.llm.model not in MODEL_PRICING_PER_1M_TOKENS:
            raise ValueError(
                f"LLM model '{self.llm.model}' is not supported in the pricing estimation."
            )

        # Extract the pricing per million tokens for the selected model
        input_price = MODEL_PRICING_PER_1M_TOKENS[self.llm.model]["input"]
        output_price = MODEL_PRICING_PER_1M_TOKENS[self.llm.model]["output"]
        cached_input_price = MODEL_PRICING_PER_1M_TOKENS[self.llm.model]["cached_input"]

        # Determine which columns are relevant (only those that are enabled)
        relevant_columns = []
        if self.include_date:
            relevant_columns.append("Date")
        if self.include_coroner:
            relevant_columns.append("CoronerName")
        if self.include_area:
            relevant_columns.append("Area")
        if self.include_receiver:
            relevant_columns.append("Receiver")
        if self.include_investigation:
            relevant_columns.append("InvestigationAndInquest")
        if self.include_circumstances:
            relevant_columns.append("CircumstancesOfDeath")
        if self.include_concerns:
            relevant_columns.append("MattersOfConcern")

        # Count total missing fields (cells with "N/A: Not found")
        total_missing_fields = 0
        for col in relevant_columns:
            if col in df.columns:
                missing_count = df[col].eq("N/A: Not found").sum()
                total_missing_fields += missing_count

        # Assume average tokens per missing field...
        #   In testing, we ran an experiment where we disabled the LLM fallback and ran the scraper, resulting 
        #     in 394 missing fields.
        #   We then ran the LLM fallback on these fields and calculated the average number of tokens used via 
        #     the OpenAI API web interface.
        #
        #   From here, we observed:
        #       - Total input tokens: 21,201,984
        #       - Total output tokens: 68,989
        #       - Total cache input tokens: 0
        #
        #   Dividing this by the 394 missing fields gave us:
        AVERAGE_INPUT_TOKENS = 53812
        AVERAGE_OUTPUT_TOKENS = 175
        AVERAGE_CACHE_INPUT_TOKENS = 0

        # Calculate total tokens required based on missing fields
        total_input_tokens = total_missing_fields * AVERAGE_INPUT_TOKENS
        total_output_tokens = total_missing_fields * AVERAGE_OUTPUT_TOKENS
        total_cached_input_tokens = total_missing_fields * AVERAGE_CACHE_INPUT_TOKENS

        # Calculate cost in USD: convert tokens to millions
        total_cost = ((total_input_tokens / 1_000_000.0) * input_price +
                      (total_cached_input_tokens / 1_000_000.0) * cached_input_price +
                      (total_output_tokens / 1_000_000.0) * output_price)
        logger.info("Estimated API cost for LLM fallback (model: %s): $%.2f based on %d missing fields.",
                    self.llm.model, total_cost, total_missing_fields)



# -----------------------------------------------------------------------------------------
# TESTING

# # Load OpenAI API key
# load_dotenv("api.env")
# openai_api_key = os.getenv("OPENAI_API_KEY")
# llm = LLM(api_key=openai_api_key)

# # Run the scraper! :D
# scraper = PFDScraper(
#     llm=llm,
#     category="all",
#     date_from="2024-01-10",
#     date_to="2024-04-19",
#     html_scraping=True,
#     pdf_fallback=False,
#     llm_fallback=False,
#     # docx_conversion="LibreOffice", # Doesn't currently seem to work; need to debug.
#     include_time_stamp=False,
#     delay_range=None,
#     verbose=False,
# )
# scraper.scrape_reports()
# scraper.estimate_api_costs()

# scraper.run_llm_fallback()
# scraper.top_up(date_to="2025-03-19")
# scraper.reports
# scraper.reports.to_csv("../../data/testreports.csv")

# scraper.get_report_links()
