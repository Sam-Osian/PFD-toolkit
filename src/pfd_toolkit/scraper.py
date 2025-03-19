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
from io import BytesIO 
from urllib.parse import urlparse, unquote
import base64
from pdf2image import convert_from_bytes
from openai import OpenAI
from dotenv import load_dotenv
import os
import subprocess
import time
import random
import threading
from itertools import chain
from datetime import datetime
from tqdm import tqdm
from concurrent.futures import as_completed

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
        
        # OpenAI API configuration
        openai_api_key: str = None,
        openai_client: OpenAI = None,
        llm_model: str = "gpt-4o-mini",
        
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
        
        verbose: bool = True  
    ) -> None:
        
        """
        Initialises the scraper.
        
        :param category: Category of reports as categorised on the judiciary.uk website. Options are 'all' (default), 'suicide', 'accident_work_safety', 'alcohol_drug_medication', 'care_home', 'child_death', 'community_health_emergency', 'emergency_services', 'hospital_deaths', 'mental_health', 'police', 'product', 'railway', 'road', 'service_personnel', 'custody', 'wales', 'other'.
        :param date_from: Only reports published on or after this date will be scraped.
        :param date_to: Only reports published on or before this date will be scraped.
        :param max_workers: The total number of concurrent threads the scraper can use for fetching data across all pages.
        :param max_requests: Maximum number of requests per domain to avoid IP address block.
        :param delay_range: None, or a tuple of two integers representing the range of seconds to delay between requests. Default is (1, 2) for a random delay between 1 and 2 seconds.
        :param html_scraping: Whether to attempt HTML-based scraping.
        :param pdf_fallback: Whether to fallback to .pdf scraping if missing values remain following HTML scraping (if set).
        :param llm_fallback: Whether to fallback to LLM scraping if missing values remain following previous method(s), if set. OpenAI API key must provided.
        :param openai_api_key: OpenAI API Key
        :param llm_model: The specific OpenAI LLM model to use, if llm_fallback is set to True. Default is "gpt_4o_mini".
        :param docx_conversion: Conversion method for .docx files; "MicrosoftWord", "LibreOffice", or "None" (default).
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
        
        self.openai_api_key = openai_api_key
        self.llm_model = llm_model
        
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
        
        # Use injected LLM client if provided; otherwise, create one from the API key.
        # This allows the user to pass in their own OpenAI client instance as an alternative to supplying an API key.
        if self.llm_fallback:
            if openai_client is not None:
                self.openai_client = openai_client
            elif openai_api_key is not None:
                self.openai_client = OpenAI(api_key=self.openai_api_key)
            else:
                raise ValueError("LLM fallback enabled, but neither an API key nor OpenAI client was provided.")
        else:
            self.openai_client = openai_client 
        
        
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
        if self.llm_fallback and not self.openai_api_key and not self.openai_client:
            raise ValueError("OpenAI API Key or Client key must be provided if LLM fallback is enabled. \nPlease set either 'openai_api_key' or 'openai_client' parameters. \nGet your API key from https://platform.openai.com/.")
        
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
                f"OpenAI API Key Provided: {'Yes' if self.openai_api_key else 'No'}\n " # Hide the API key 
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
    
    def _get_report_links(self) -> list:
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

    @staticmethod
    def _clean_text(text: str) -> str:
        """Helper function to clean text by removing excessive whitespace & replacing typographic apostrophes."""
        normalised = PFDScraper._normalise_apostrophes(text)
        return ' '.join(normalised.split())
    
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
    def _extract_section_text_by_keywords(self, soup: BeautifulSoup, header_keywords: list) -> str:
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
    # Core report extraction logic
    # ----------------------------------------------------------------------------- 
    # This pieces together the above functions to extract all report information from a given URL.
    # It serves as the internal core of the scraper.
    
    def _extract_report_info(self, url: str) -> dict:
        """
        Internal function to extract metadata and text from a PFD report webpage.
        
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
            # We use fuzzy date parsing to handle variations in date formats (e.g. "1st January 2025" and "01/01/2025")
            if self.include_date:
                date_element = self._extract_paragraph_text_by_keywords(soup, ["Date of report:"])
                if date_element != "N/A: Not found":
                    date_element = date_element.replace("Date of report:", "").strip()
                    try:
                        parsed_date = parser.parse(date_element, fuzzy=True)
                        date = parsed_date.strftime("%Y-%m-%d")
                    except Exception as e:
                        logger.error("Error parsing date '%s': %s", date_element, e)
                        date = date_element
                else:
                    date = 'N/A: Not found'
                
            
            # Receiver extraction (who the report is sent to)
            if self.include_receiver:
                receiver_element = self._extract_paragraph_text_by_keywords(
                    soup, ["This report is being sent to:", "Sent to:"]
                )
                receiver = receiver_element.replace("This report is being sent to:", "") \
                                            .replace("Sent to:", "") \
                                            .strip()
                                            
                if len(receiver) < 5 or len(receiver) > 30:
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
                if len(coroner) < 5 or len(coroner) > 30:
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
                missing_fields["Date of Report"] = "[Date of the report, not the death]"
            if self.include_coroner and coroner == "N/A: Not found":
                missing_fields["Coroner's Name"] = "[Name of the coroner. Provide the name only.]"
            if self.include_area and area == "N/A: Not found":
                missing_fields["Area"] = "[Area/location of the Coroner. Provide the location itself only.]"
            if self.include_receiver and receiver == "N/A: Not found":
                missing_fields["Receiver"] = "[Name or names of the recipient(s) as provided in the report.]"
            if self.include_investigation and investigation == "N/A: Not found":
                missing_fields["Investigation and Inquest"] = "[The text from the Investigation/Inquest section.]"
            if self.include_circumstances and circumstances == "N/A: Not found":
                missing_fields["Circumstances of Death"] = "[The text from the Circumstances of Death section.]"
            if self.include_concerns and concerns == "N/A: Not found":
                missing_fields["Coroner's Concerns"] = "[The text from the Coroner's Concerns section.]"

            if missing_fields:
                if self.verbose:
                    logger.debug(
                        f"Initiating LLM fallback for URL: {url}. Missing fields: {missing_fields}"
                    )
                logger.info("Attempting LLM fallback for %s", url)

                # Reuse previously downloaded .pdf bytes if available
                pdf_bytes = getattr(self, '_last_pdf_bytes', None)
                if pdf_bytes is None:
                    try:
                        pdf_response = self.session.get(report_link)
                        pdf_response.raise_for_status()
                        pdf_bytes = pdf_response.content
                        self._last_pdf_bytes = pdf_bytes
                    except Exception as e:
                        logger.error("Failed to fetch .pdf for image conversion: %s", e)
                        pdf_bytes = None

                base64_images = [] # ...OpenAI requires base64 encoded images
                # Convert the .pdf to images using pdf2image
                # Each page of the .pdf is converted to a separate image
                if pdf_bytes:
                    try:
                        images = convert_from_bytes(pdf_bytes)
                    except Exception as e:
                        logger.error("Error converting .pdf to images: %s", e)
                        images = []
                    for img in images:
                        buffered = BytesIO()
                        img.save(buffered, format="JPEG")
                        img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                        base64_images.append(img_str)

                prompt = (
                    "Your goal is to transcribe the **exact** text from this report, presented as images.\n\n"
                    "Please extract the following section(s):\n"
                )
                # Add missing fields to the prompt
                for field, instruction in missing_fields.items():
                    prompt += f"\n{field}: {instruction}\n"
                # Additional instructions for the LLM
                prompt += (
                    "\nRespond with nothing else whatsoever. You must not respond in your own 'voice' or even acknowledge the task.\n"
                    "If you are unable to identify the text from the image for any given section, simply respond: \"N/A: Not found\" for that section.\n"
                    "Sometimes text may be redacted with a black box; transcribe it as '[REDACTED]'.\n"
                    "Make sure you transcribe the *full* text for each section, not just a snippet.\n"
                    "Do *not* change the section title(s) from the above format.\n"
                )
                # If verbose, print out the prompt for debugging
                if self.verbose:
                    logger.info("LLM prompt:\n\n%s", prompt)

                # Construct the messages to send to the LLM (first the prompt text then each image)
                messages = [{"type": "text", "text": prompt}]
                for b64_img in base64_images:
                    messages.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
                    })

                response = self.openai_client.chat.completions.create(
                    model=self.llm_model,
                    messages=[{"role": "user", "content": messages}],
                )
                # Extract the LLM response
                llm_text = response.choices[0].message.content
                if self.verbose:
                    logger.info("LLM fallback response:\n%s\n\n", llm_text)

                # Parse the LLM response to update only missing fields
                fallback_date = 'N/A: Not found'
                fallback_coroner = 'N/A: Not found'
                fallback_area = 'N/A: Not found'
                fallback_receiver = 'N/A: Not found'
                fallback_investigation = 'N/A: Not found'
                fallback_circumstances = 'N/A: Not found'
                fallback_concerns = 'N/A: Not found'

                # NEED TO CHANGE
                # The below looks for lines that start with the field name and then extracts the text after the colon.
                # This is problematic, because the LLM does not always output exactly as instructued
                # It also requires the LLM to output everything on a single line, which is doesn't currently do.
                # Structured outputs would (hopefully!) be better, but this requires a different approach.
                for line in llm_text.splitlines():
                    line_strip = line.strip()
                    line_lower = line_strip.lower()
                    if line_lower.startswith("date of report:"):
                        fallback_date = line_strip.split(":", 1)[1].strip()
                    elif line_lower.startswith("coroner's name:"):
                        fallback_coroner = line_strip.split(":", 1)[1].strip()
                    elif line_lower.startswith("area:"):
                        fallback_area = line_strip.split(":", 1)[1].strip()
                    elif line_lower.startswith("receiver:"):
                        fallback_receiver = line_strip.split(":", 1)[1].strip()
                    elif line_lower.startswith("investigation and inquest:"):
                        fallback_investigation = line_strip.split(":", 1)[1].strip()
                    elif line_lower.startswith("circumstances of death:"):
                        fallback_circumstances = line_strip.split(":", 1)[1].strip()
                    elif line_lower.startswith("coroner's concerns:"):
                        fallback_concerns = line_strip.split(":", 1)[1].strip()

                # Parse the date into YYYY-MM_DD format
                if fallback_date != 'N/A: Not found':
                    try:
                        parsed_date = parser.parse(fallback_date, fuzzy=True)
                        fallback_date = parsed_date.strftime("%Y-%m-%d")
                    except Exception as e:
                        logger.error("LLM fallback: could not parse date '%s': %s", fallback_date, e)

                # Update each field **only** if the HTML/.pdf extraction failed to find the information
                if self.include_date and date == 'N/A: Not found':
                    date = fallback_date
                if self.include_coroner and coroner == 'N/A: Not found':
                    coroner = fallback_coroner
                if self.include_area and area == 'N/A: Not found':
                    area = fallback_area
                if self.include_receiver and receiver == 'N/A: Not found':
                    receiver = fallback_receiver
                if self.include_investigation and investigation == 'N/A: Not found':
                    investigation = fallback_investigation
                if self.include_circumstances and circumstances == 'N/A: Not found':
                    circumstances = fallback_circumstances
                if self.include_concerns and concerns == 'N/A: Not found':
                    concerns = fallback_concerns


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
    
    def scrape_reports(self) -> pd.DataFrame:
        """
        Scrapes reports from the collected report links based on the user configuration of the PFDScraper class instance (self).
        
        :return: A pandas DataFrame containing one row per scraped report.
        """
        if not self.report_links:
            self._get_report_links()
        
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


    def top_up(self, old_reports: pd.DataFrame = None) -> pd.DataFrame:
        """
        Adds new reports to the existing scraped reports based on the user configuration of the PFDScraper class instance (self).
        Duplicate checking is based on the URL as a unique identifier – by default the URL (if include_url is True)
        or the report ID otherwise.
        
        Parameters:
            old_reports (pd.DataFrame): Optional DataFrame containing previously scraped reports. If not supplied,
                                         the internal self.reports will be used.
        
        Returns:
            pd.DataFrame: Updated DataFrame containing both the old and new scraped reports.
        """
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
        updated_links = self._get_report_links()

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
        
        logger.info("Top-up: %d new report(s) found; %d duplicate(s) not added", new_count, duplicates_count)

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



# -----------------------------------------------------------------------------------------
# TESTING

# Load OpenAI API key
load_dotenv('api.env')
openai_api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=openai_api_key)

# Run the scraper! :D
scraper = PFDScraper(
    category='accident_work_safety', 
    date_from="2023-01-01",
    date_to="2025-02-07",
    html_scraping=True,
    pdf_fallback=True,
    llm_fallback=False,
    openai_client = client,
    llm_model="gpt-4o-mini",
    #docx_conversion="LibreOffice", # Doesn't currently seem to work; need to debug.
    include_time_stamp=False,
    delay_range = None,
    verbose=False
)
scraper.scrape_reports()
scraper.top_up()
scraper.reports


#reports.to_csv('../../data/testreports.csv')