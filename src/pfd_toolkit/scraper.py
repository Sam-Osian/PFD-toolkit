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

# -----------------------------------------------------------------------------
# Logging Configuration:
# - Sets up logging for the module. The logger is used to record events,
#   debugging messages, and error messages.
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, force=True)

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
        category: str = 'all',
        start_page: int = 1,
        end_page: int = 559,
        max_workers: int = 10,
        max_requests: int = 5, 
        delay_range = (1, 2),
        html_scraping: bool = True,
        pdf_fallback: bool = True,
        llm_fallback: bool = False,
        api_key: str = None,
        llm_model: str = "gpt-4o-mini",
        docx_conversion: str = "None",
        time_stamp: str = False,
        verbose: bool = True  
    ) -> None:
        
        """
        Initialises the scraper.
        
        :param category: Category of reports as categorised on the judiciary.uk website. Options are 'all' (default), 'suicide', 'accident_work_safety', 'alcohol_drug_medication', 'care_home', 'child_death', 'community_health_emergency', 'emergency_services', 'hospital_deaths', 'mental_health', 'police', 'product', 'railway', 'road', 'service_personnel', 'custody', 'wales', 'other'.
        :param start_page: The first page to scrape.
        :param end_page: The last page to scrape.
        :param max_workers: The total number of concurrent threads the scraper can use for fetching data across all pages.
        :param max_requests: Maximum number of requests per domain to avoid IP address block.
        :param delay_range: None, or a tuple of two integers representing the range of seconds to delay between requests. Default is (1, 2) for a random delay between 1 and 2 seconds.
        :param html_scraping: Whether to attempt HTML-based scraping.
        :param pdf_fallback: Whether to fallback to .pdf scraping if missing values remain following HTML scraping (if set).
        :param llm_fallback: Whether to fallback to LLM scraping if missing values remain following previous method(s), if set. OpenAI API key must provided.
        :param api_key: OpenAI API Key
        :param llm_model: The specific OpenAI LLM model to use, if llm_fallback is set to True. Default is "gpt_4o_mini".
        :param docx_conversion: Conversion method for .docx files; "MicrosoftWord", "LibreOffice", or "None" (default).
        :param time_stamp: Whether to add a timestamp column to the output file.
        :param verbose: Whether to print verbose output.
        """
        self.category = category.lower()
        self.start_page = start_page
        self.end_page = end_page
        self.max_workers = max_workers
        self.max_requests = max_requests
        self.delay_range = delay_range
        self.html_scraping = html_scraping
        self.pdf_fallback = pdf_fallback
        self.llm_fallback = llm_fallback
        self.api_key = api_key
        self.llm_model = llm_model
        self.docx_conversion = docx_conversion
        self.time_stamp = time_stamp
        self.verbose = verbose
        
        self.domain_semaphore = threading.Semaphore(self.max_requests) # Semaphore to limit requests per domain
        self.report_links = [] # List to store all report URLs
        
        # Define URL templates for different PFD categories.
        # ...Some categories (like 'all' and 'suicide') have unique URL formats.
        category_templates = {
            "all": "https://www.judiciary.uk/prevention-of-future-death-reports/page/{page}/",
            "suicide": "https://www.judiciary.uk/prevention-of-future-death-reports/page/{page}/?s&pfd_report_type=suicide-from-2015",
            "accident_work_safety": "https://www.judiciary.uk/page/{page}/?s&pfd_report_type=accident-at-work-and-health-and-safety-related-deaths",
            "alcohol_drug_medication": "https://www.judiciary.uk/page/{page}/?s&pfd_report_type=alcohol-drug-and-medication-related-deaths",
            "care_home": "https://www.judiciary.uk/page/{page}/?s&pfd_report_type=care-home-health-related-deaths",
            "child_death": "https://www.judiciary.uk/page/{page}/?s&pfd_report_type=child-death-from-2015",
            "community_health_emergency": "https://www.judiciary.uk/page/{page}/?s&pfd_report_type=community-health-care-and-emergency-services-related-deaths",
            "emergency_services": "https://www.judiciary.uk/page/{page}/?s&pfd_report_type=emergency-services-related-deaths-2019-onwards",
            "hospital_deaths": "https://www.judiciary.uk/page/{page}/?s&pfd_report_type=hospital-death-clinical-procedures-and-medical-management-related-deaths",
            "mental_health": "https://www.judiciary.uk/page/{page}/?s&pfd_report_type=mental-health-related-deaths",
            "police": "https://www.judiciary.uk/page/{page}/?s&pfd_report_type=police-related-deaths",
            "product": "https://www.judiciary.uk/page/{page}/?s&pfd_report_type=product-related-deaths",
            "railway": "https://www.judiciary.uk/page/{page}/?s&pfd_report_type=railway-related-deaths",
            "road": "https://www.judiciary.uk/page/{page}/?s&pfd_report_type=road-highways-safety-related-deaths",
            "service_personnel": "https://www.judiciary.uk/page/{page}/?s&pfd_report_type=service-personnel-related-deaths",
            "custody": "https://www.judiciary.uk/page/{page}/?s&pfd_report_type=state-custody-related-deaths",
            "wales": "https://www.judiciary.uk/page/{page}/?s&pfd_report_type=wales-prevention-of-future-deaths-reports-2019-onwards",
            "other": "https://www.judiciary.uk/page/{page}/?s&pfd_report_type=other-related-deaths",
        }
        
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
        
        # If max_workers is set to 0 or a negative number
        if self.max_workers <= 0:
            raise ValueError("max_workers must be a positive integer.")
        
        # If max_requests is set to 0 or a negative number
        if self.max_requests <= 0:
            raise ValueError("max_requests must be a positive integer.")
        
        # If start_page is set to 0 or a negative number
        if self.start_page <= 0:
            raise ValueError("start_page must be a positive integer.")
        
        # If end_page is set to 0 or a negative number
        if self.end_page <= 0:
            raise ValueError("end_page must be a positive integer.")
        
        # If end_page is less than start_page
        if self.end_page < self.start_page:
            raise ValueError("end_page must be greater than or equal to start_page.")
        
        # If delay_range is not a tuple of two numbers (int or float)
        if not isinstance(self.delay_range, tuple) or len(self.delay_range) != 2 or not all(isinstance(i, (int, float)) for i in self.delay_range):
            raise ValueError("delay_range must be a tuple of two numbers (int or float) - e.g. (1, 2) or (1.5, 2.5). If you are attempting to disable delays, set to (0,0).")

        # If upper bound of delay_range is less than lower bound
        if self.delay_range[1] < self.delay_range[0]:
            raise ValueError("Upper bound of delay_range must be greater than or equal to lower bound.")
        
        # If docx_conversion is not one of the allowed values
        if self.docx_conversion not in ["MicrosoftWord", "LibreOffice", "None"]:
            raise ValueError("docx_conversion must be one of 'MicrosoftWord', 'LibreOffice', or 'None'.")
        
        # If OpenAI API key is not provided when LLM fallback is enabled
        if self.llm_fallback and not self.api_key:
            raise ValueError("OpenAI API key must be provided if LLM fallback is enabled. Please set 'api_key' parameter. \nGet your API key from https://platform.openai.com/.")
        
        # If no scrape method is enabled
        if not self.html_scraping and not self.pdf_fallback and not self.llm_fallback:
            raise ValueError("At least one of 'html_scraping', 'pdf_fallback', or 'llm_fallback' must be enabled.")

        
        ### Warnings (code will still run)
        
        # If only html_scraping is enabled
        if self.html_scraping and not self.pdf_fallback and not self.llm_fallback:
            logger.warning("Only HTML scraping is enabled. Consider enabling .pdf or LLM fallback for more complete data extraction.")
        
        # If only pdf_fallback is enabled
        if not self.html_scraping and self.pdf_fallback and not self.llm_fallback:
            logger.warning("Only .pdf fallback is enabled. Consider enabling HTML scraping or LLM fallback for more complete data extraction.")
            
        # If only llm_fallback is enabled
        if not self.html_scraping and not self.pdf_fallback and self.llm_fallback:
            logger.warning("Only LLM fallback is enabled. While this is a high-performance option, large API costs may be incurred, especially for large requests. Consider enabling HTML scraping or .pdf fallback for more cost-effective data extraction.")
        
        # If max_workers is set above 50
        if self.max_workers > 50:
            logger.warning("max_workers is set to a high value (>50). Depending on your system, this may cause performance issues. It could also trigger anti-scraping measures by the host, leading to temporary or permanent IP bans. We recommend setting to between 10 and 50.")
        
        # If max_workers is set below 10
        if self.max_workers < 10:
            logger.warning("max_workers is set to a low value (<10). This may result in slower scraping speeds. Consider increasing the value for faster performance. We recommend setting to between 10 and 50.")
        
        # If max_requests is set above 10
        if self.max_requests > 10:
            logger.warning("max_requests is set to a high value (>10). This may trigger anti-scraping measures by the host, leading to temporary or permanent IP bans. We recommend setting to between 3 and 10.")
            
        # If max_requests is set below 3
        if self.max_requests < 3:
            logger.warning("max_requests is set to a low value (<3). This may result in slower scraping speeds. Consider increasing the value for faster performance. We recommend setting to between 3 and 10.")

        # If delay range is set to (0,0)
        if self.delay_range == (0, 0):
            logger.warning("delay_range has been disabled. This will disable delays between requests. This may trigger anti-scraping measures by the host, leading to temporary or permanent IP bans. We recommend setting to (1,2).")
        elif self.delay_range[0] < 0.5 and self.delay_range[1] != 0:
            logger.warning("delay_range is set to a low value (<0.5 seconds). This may trigger anti-scraping measures by the host, leading to temporary or permanent IP bans. We recommend setting to between (1, 2).")
        
        # If delay_range upper bound is set above 5 seconds
        if self.delay_range[1] > 5:
            logger.warning("delay_range is set to a high value (>5 seconds). This may result in slower scraping speeds. Consider decreasing the value for faster performance. We recommend setting to between (1, 2).")

        # -----------------------------------------------------------------------------
        # Log the initialisation parameters for debug if verbose is enabled
        # -----------------------------------------------------------------------------
        # Log all initialisation parameters
        if verbose:
            logger.info(
                "\nPFDScraper initialised with parameters:\n "
                f"category='{self.category}',\n "
                f"start_page={self.start_page},\n "
                f"end_page={self.end_page},\n "
                f"max_workers={self.max_workers},\n "
                f"max_requests={self.max_requests},\n "
                f"delay_range={self.delay_range},\n "
                f"html_scraping={self.html_scraping},\n "
                f"pdf_fallback={self.pdf_fallback},\n "
                f"llm_fallback={self.llm_fallback},\n "
                f"api_key={'provided' if self.api_key else 'not provided'},\n " # ...Preventing the API key from being printed
                f"llm_model='{self.llm_model}',\n "
                f"docx_conversion='{self.docx_conversion}',\n "
                f"verbose={self.verbose}"
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
                logger.error("Failed to fetch page: %s; Error: %s", url, e)
                return []
        
        soup = BeautifulSoup(response.text, 'html.parser')
        links = soup.find_all('a', class_='card__link')
        return [link.get('href') for link in links if link.get('href')]
    
    def _get_report_links(self) -> list:
        """
        Internal function to collect all PFD report links from the paginated pages.
        
        :return: A list of report URLs.
        """
        self.report_links = []
        pages = list(range(self.start_page, self.end_page + 1))
        
        def _fetch_page_links(page_number: int) -> list:
            page_url = self.page_template.format(page=page_number)
            href_values = self._get_report_href_values(page_url)
            logger.info("Scraped %d links from %s", len(href_values), page_url)
            return href_values
        
        # Use a thread pool to concurrently fetch multiple pages
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(_fetch_page_links, pages))
        
        # Flatten the list of lists into a single list of report URLs -- replaced for loop with chain from itertools
        self.report_links = list(chain.from_iterable(results))
        logger.info("Total collected report links: %d", len(self.report_links))
        return self.report_links

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
        
        date_scraped = datetime.now().strftime("%Y-%m-%d %H:%M:%S") if self.time_stamp else None
        
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
            ref_element = soup.find(lambda tag: tag.name == 'p' and 'Ref:' in tag.get_text(), recursive=True)
            if ref_element:
                match = self._id_pattern.search(ref_element.get_text())
                report_id = match.group(1) if match else 'N/A: Not found'
            else:
                report_id = "N/A: Not found"
            
            
            # Date of report extraction
            # We use fuzzy date parsing to handle variations in date formats (e.g. "1st January 2025" and "01/01/2025")
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
            receiver_element = self._extract_paragraph_text_by_keywords(
                soup, ["This report is being sent to:", "Sent to:"]
            )
            receiver = receiver_element.replace("This report is being sent to:", "") \
                                        .replace("Sent to:", "") \
                                        .strip()
                                        
            if len(receiver) < 5 or len(receiver) > 30:
                receiver = 'N/A: Not found'
                        
            
            # Name of coroner extraction
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
            'N/A: Not found' in [
                #date, # Tricky to implement due to date placement in .pdfs. However, HTML extraction is usually successful.
                coroner,
                area,
                receiver,
                investigation,
                circumstances,
                concerns
            ]
        ):
            if self.verbose:
                logger.debug(f"Initiating .pdf fallback for URL: {url} because one or more fields are missing.")
            
            # Coroner name extraction if missing
            if coroner == "N/A: Not found":
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
            if area == "N/A: Not found":
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
            if receiver == "N/A: Not found":
                receiver_element = self._extract_section_from_pdf_text(
                    pdf_text,
                    start_keywords=[" SENT ", "SENT TO:"],
                    end_keywords=["CORONER", "CIRCUMSTANCES OF THE DEATH", "CIRCUMSTANCES OF"]
                )
                receiver = self._clean_text(receiver_element).replace("TO:", "").strip()
                
                if len(receiver) < 5:
                    receiver = 'N/A: Not found'
                
            
            # Investigation & Inquest extraction if missing
            if investigation == "N/A: Not found":
                investigation_element = self._extract_section_from_pdf_text(
                    pdf_text,
                    start_keywords=["INVESTIGATION and INQUEST", "3 INQUEST"],
                    end_keywords=["CIRCUMSTANCES OF DEATH", "CIRCUMSTANCES OF THE DEATH", "CIRCUMSTANCES OF"]
                )
                investigation = self._clean_text(investigation_element)
                
                if len(investigation) < 30:
                    investigation = 'N/A: Not found'
            
            
            # Circumstances of Death extraction if missing
            if circumstances == "N/A: Not found":
                circumstances_section = self._extract_section_from_pdf_text(
                    pdf_text,
                    start_keywords=["CIRCUMSTANCES OF DEATH", "CIRCUMSTANCES OF THE DEATH", "CIRCUMSTANCES OF"],
                    end_keywords=["CORONER'S CONCERNS", "CORONER CONCERNS", "CORONERS CONCERNS", "as follows"]
                )
                circumstances = self._clean_text(circumstances_section)
                
                if len(circumstances) < 30:
                    circumstances = 'N/A: Not found'
            
            
            # Matters of Concern extraction if missing
            if concerns == "N/A: Not found":
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
        
        if self.llm_fallback and (
            'N/A: Not found' in [
                date,
                coroner,
                area,
                receiver,
                investigation,
                circumstances,
                concerns
            ]
        ):
            if self.verbose:
                logger.debug(
                    f"Initiating LLM fallback for URL: {url}. Missing fields: " +
                    f"date='{date}', coroner='{coroner}', " +
                    f"area='{area}', receiver='{receiver}', investigation='{investigation}', " +
                    f"circumstances='{circumstances}', concerns='{concerns}'"
                )
            logger.info("Attempting LLM fallback for %s", url)
            
            # Reuse previously downloaded .pdf bytes if available; otherwise, download again
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
            

            base64_images = []  # ...OpenAI requires base64 encoded images
            
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
            
            # Dictionary of missing fields and their extraction instructions.
            # This is formatted in this way because we use dynamic prompting for the LLM, only
            #   asking it to fill in the fields that remain missing.
            
            missing_fields = {}
            if date == "N/A: Not found":
                missing_fields["Date of Report"] = "[Date of the report, not the death]"
            if coroner == "N/A: Not found":
                missing_fields["Coroner's Name"] = "[Name of the coroner. Provide the name only.]"
            if area == "N/A: Not found":
                missing_fields["Area"] = "[Area/location of the Coroner. Provide the location itself only.]"
            if receiver == "N/A: Not found":
                missing_fields["Receiver"] = "[Name or names of the person/people or organisation(s) the report is sent to, including job roles if provided.]"
            if investigation == "N/A: Not found":
                missing_fields["Investigation and Inquest"] = "[The text from the Investigation/Inquest section.]"
            if circumstances == "N/A: Not found":
                missing_fields["Circumstances of Death"] = "[The text from the Circumstances of Death section.]"
            if concerns == "N/A: Not found":
                missing_fields["Coroner's Concerns"] = "[The text from the Coroner's Concerns or Matters of Concern section. Sometimes this will follow boilerplate text (e.g. 'The matters of concern are as follows...')]"
            
            # Main prompt set-up
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
                "Do *not* change the section title(s) from the above format. For example, you must **not** change Circumstances of Death to Circumstances of the Death.\n"
            )
            
            # If verbose, print out the prompt for debugging
            if self.verbose:
                logger.info("Report: %s", url)
                logger.info("LLM prompt:\n\n%s", prompt)
            
            # Construct the messages to send to the LLM (first the prompt text then each image)
            messages = [
                {"type": "text", "text": prompt}
            ]
            for b64_img in base64_images:
                messages.append({
                    "type": "image_url",
                    "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}
                })
            
            response = client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "user", "content": messages}],
            )
            
            # Extract the LLM response
            llm_text = response.choices[0].message.content
            
            if self.verbose:
                logger.info("LLM fallback response:\n%s\n\n", llm_text)
            else:
                logger.info("LLM fallback response received.")
            
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
                line_lower = line_strip.lower() # ...convert to lowercase, in case the LLM outputs in a different case
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
            
            # Parse the date into YYYY-MM-DD format
            if fallback_date != 'N/A: Not found':
                try:
                    parsed_date = parser.parse(fallback_date, fuzzy=True)
                    fallback_date = parsed_date.strftime("%Y-%m-%d")
                except Exception as e:
                    logger.error("LLM fallback: could not parse date '%s': %s", fallback_date, e)
            
            # Update each field **only** if the HTML/.pdf extraction failed to find the information
            if date == 'N/A: Not found':
                date = fallback_date
            if coroner == 'N/A: Not found':
                coroner = fallback_coroner
            if area == 'N/A: Not found':
                area = fallback_area
            if receiver == 'N/A: Not found':
                receiver = fallback_receiver
            if investigation == 'N/A: Not found':
                investigation = fallback_investigation
            if circumstances == 'N/A: Not found':
                circumstances = fallback_circumstances
            if concerns == 'N/A: Not found':
                concerns = fallback_concerns

        # Return the extracted report information
        report = {
            "URL": url,
            "ID": report_id,
            "Date": date,
            "CoronerName": coroner,
            "Area": area,
            "Receiver": receiver,
            "InvestigationAndInquest": investigation,
            "CircumstancesOfDeath": circumstances,
            "MattersOfConcern": concerns
        }
        if self.time_stamp:
            report["DateScraped"] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return report

    
    # The below function serves as the user entry point for the scraper.
    # It is currently the only function that is not internal (i.e. doesn't start with a _)
    def scrape_all_reports(self) -> pd.DataFrame:
        """
        Scrapes all reports from the collected report links.
        
        :return: A pandas DataFrame containing one row per scraped report.
        """
        if not self.report_links:
            self._get_report_links()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(self._extract_report_info, self.report_links))
        # Filter out any failed report extractions (None)
        records = [record for record in results if record is not None]
        # Create timestamp if parameter is set to True
        records = pd.DataFrame(records)
        return records



# -----------------------------------------------------------------------------------------
# TESTING

# Load OpenAI API key
load_dotenv('api.env')
openai_api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=openai_api_key)

# Run the scraper! :D
scraper = PFDScraper(
    category='alcohol_drug_medication', 
    start_page=1, 
    end_page=3, 
    max_workers=15,
    html_scraping=True,
    pdf_fallback=True,
    llm_fallback=False,
    api_key=openai_api_key,
    llm_model="gpt-4o-mini",
    docx_conversion="LibreOffice", # Doesn't currently seem to work; need to debug.
    time_stamp=False,
    delay_range = None,
    verbose=True
)
reports = scraper.scrape_all_reports()
reports

#reports.to_csv('../../data/testreports.csv')