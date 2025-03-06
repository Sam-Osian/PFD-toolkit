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

# Configure error logging for the module
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class PFDScraper:
    """Web scraper for extracting Prevention of Future Death (PFD) reports from the UK Judiciary website."""
    
    def __init__(
        self,
        category: str = 'all',
        start_page: int = 1,
        end_page: int = 559,
        max_workers: int = 10,
        pdf_fallback: bool = True,
        llm_fallback: bool = False,
        api_key: str = None,
        llm_model: str = "gpt-4o-mini",
        docx_conversion: str = "None",
        verbose: bool = True  
    ) -> None:
        """
        Initialises the scraper.
        
        :param category: Category of reports as categorised on the judiciary.uk website. Options are 'all' (default), 'suicide', 'accident_work_safety', 'alcohol_drug_medication', 'care_home', 'child_death', 'community_health_emergency', 'emergency_services', 'hospital_deaths', 'mental_health', 'police', 'product', 'railway', 'road', 'service_personnel', 'custody', 'wales', 'other'.
        :param start_page: The first page to scrape.
        :param end_page: The last page to scrape.
        :param max_workers: Maximum number of workers for concurrent fetching.
        :param pdf_fallback: If HTML scraping fails for any given report section, whether to fallback to .pdf scraping. 
        :param llm_fallback: If previous scraping method fails for any given report section, whether to fallback to OpenAI LLM to process reports as images. OpenAI API key must be set.
        :param api_key: OpenAI API Key
        :param llm_model: The specific OpenAI LLM model to use, if llm_fallback is set to True.
        :param docx_conversion: Conversion method for .docx files; "MicrosoftWord", "LibreOffice", or "None" (default).
        :param verbose: Whether to print verbose output.
        """
        self.category = category.lower()
        self.start_page = start_page
        self.end_page = end_page
        self.max_workers = max_workers
        self.pdf_fallback = pdf_fallback
        self.llm_fallback = llm_fallback
        self.api_key = api_key
        self.llm_model = llm_model
        self.docx_conversion = docx_conversion
        self.verbose = verbose
        self.report_links = []
        
        if self.llm_fallback and not self.api_key:
            raise ValueError("OpenAI API key must be provided if LLM fallback is enabled.")
        
        if self.verbose:
            logger.debug(
                f"Initialised PFDScraper with category='{self.category}', "
                f"start_page={self.start_page}, end_page={self.end_page}, "
                f"max_workers={self.max_workers}, pdf_fallback={self.pdf_fallback}, "
                f"llm_fallback={self.llm_fallback}, docx_conversion='{self.docx_conversion}'"
            )
        
        # Setting up a requests session with retry logic
        self.session = requests.Session()
        retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504])
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Compile regex for report ID extraction (e.g. "2025-0296")
        self._id_pattern = re.compile(r'(\d{4}-\d{4})')
        
        # Define URL templates for different PFD categories. 'All' and 'suicide' have different formats, so we can't use a universal URL format.
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
        if self.category in category_templates:
            self.page_template = category_templates[self.category]
        else:
            valid_options = ", ".join(sorted(category_templates.keys()))
            raise ValueError(f"Unknown category '{self.category}'. Valid options are: {valid_options}")
    
    def _get_href_values(self, url: str) -> list:
        """
        Internal function to extract href values from <a> elements with class 'card__link'.
        
        :param url: URL of the page to scrape.
        :return: List of href values.
        """
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
            href_values = self._get_href_values(page_url)
            logger.info("Scraped %d links from %s", len(href_values), page_url)
            return href_values
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(_fetch_page_links, pages))
        
        for href_values in results:
            self.report_links.extend(href_values)
        logger.info("Total collected report links: %d", len(self.report_links))
        return self.report_links

    @staticmethod
    def _normalise_apostrophes(text: str) -> str:
        """Helper function to replace ‘fancy’ (typographic) apostrophes with the standard apostrophe."""
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
        
        # Download the file bytes
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
        
        try:
            pdf_buffer = BytesIO(pdf_bytes)
            pdf_document = pymupdf.open(stream=pdf_buffer, filetype="pdf")
            text = "".join(page.get_text() for page in pdf_document)
            pdf_document.close()
        except Exception as e:
            logger.error("Error processing .pdf %s: %s", pdf_url, e)
            return "N/A"
        
        return self._clean_text(text)
    
    def _extract_paragraph_text_by_keywords(self, soup: BeautifulSoup, keywords: list) -> str:
        """
        Internal function to extract text from a <p> element containing any of the given keywords.
        
        :param soup: BeautifulSoup object of the page.
        :param keywords: List of keywords to search for.
        :return: Extracted text or 'N/A: Not found'.
        """
        for keyword in keywords:
            element = soup.find(lambda tag: tag.name == 'p' and keyword in tag.get_text(), recursive=True)
            if element:
                return self._clean_text(element.get_text())
        return 'N/A: Not found'
    
    def _extract_section_text_by_keywords(self, soup: BeautifulSoup, header_keywords: list) -> str:
        """
        Internal function to extract text from a section of HTML that spans multiple elements,
        based on a header that matches any of the provided keywords. The search is case-insensitive,
        and the function accepts multiple keyword variations for a single section.

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
    
    
    def _extract_report_info(self, url: str) -> dict:
        """
        Internal function to extract metadata and text from a PFD report webpage.
        
        :param url: URL of the report page.
        :return: Dictionary containing extracted report information.
        """
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
        
        
        # -----------------------------------------------------------------------#
        #                          HTML Data Extraction                          #
        #                                                                        #
        #   The below attempts to scrape data from each report's HTML webpage.   #
        # -----------------------------------------------------------------------#
        
        
        # Report ID extraction using compiled regex
        ref_element = soup.find(lambda tag: tag.name == 'p' and 'Ref:' in tag.get_text(), recursive=True)
        if ref_element:
            match = self._id_pattern.search(ref_element.get_text())
            report_id = match.group(1) if match else 'N/A: Not found'
        else:
            report_id = "N/A: Not found"
        
        
        # Date of report extraction with fuzzy date parsing
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
        
        
        # Receiver extraction
        receiver_element = self._extract_paragraph_text_by_keywords(
            soup, ["This report is being sent to:", "Sent to:"]
        )
        receiver = receiver_element.replace("This report is being sent to:", "") \
                                       .replace("Sent to:", "") \
                                       .strip()
                                       
        if len(receiver) < 5 or len(receiver) > 30:
            receiver = 'N/A: Not found'
        
        
        # Name of deceased extraction
        deceased_element = self._extract_paragraph_text_by_keywords(
            soup, ["Deceased name:", "Deceased's name:", "Deceaseds name:"]
        )
        deceased = deceased_element.replace("Deceased name:", "") \
                                       .replace("Deceased's name:", "") \
                                       .replace("Deceaseds name:", "") \
                                       .strip()
                                       
        if len(deceased) < 5 or len(deceased) > 30:
            deceased = 'N/A: Not found'
        
        
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

        
        # -------------------------------------------------------------------------------------------- #
        #                               .pdf Data Extraction Fallback                                  #
        #                                                                                              #
        # If pdf.fallback is set to True (default), any PFD reports where our HTML scraper was unable  #
        # to scrape any given section will switch to .pdf scraping. Any successfully scraped sections  #
        # will be unchanged.                                                                           #
        # ---------------------------------------------------------------------------------------------#
        
        
        if self.pdf_fallback and (
            'N/A: Not found' in [
                #date, # Tricky to implement due to date placement in .pdfs; 
                #deceased, # Unable to read from .pdfs; information not _structurally_ recorded but is sometimes mentioned in anothe section. LLM fallback better suited to this.
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
            else:
                logger.info("Attempting .pdf fallback for %s", url)
            
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
        
        # -------------------------------------------------------------------------------------------- #
        #                                LLM Data Extraction Fallback                                  #
        #                                                                                              #
        # If the previous method of scraping (whether HTML or .pdf) was unsuccessful for any given     #
        # section, the below code will (re-)download .pdfs, convertws to images, and fed it to a GPT   #
        # Vision model to extract text.                                                                #
        # ---------------------------------------------------------------------------------------------#
        
        if self.llm_fallback and (
            'N/A: Not found' in [
                date,
                deceased,
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
                    f"date='{date}', deceased='{deceased}', coroner='{coroner}', " +
                    f"area='{area}', receiver='{receiver}', investigation='{investigation}', " +
                    f"circumstances='{circumstances}', concerns='{concerns}'"
                )
            logger.info("Attempting LLM fallback for %s", url)
            
            # (Re-)downloading the .pdf for image conversion
            try:
                pdf_response = self.session.get(report_link)
                pdf_response.raise_for_status()
                pdf_bytes = pdf_response.content
            except Exception as e:
                logger.error("Failed to fetch .pdf for image conversion: %s", e)
                pdf_bytes = None
            
            base64_images = []
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
            missing_fields = {}
            if date == "N/A: Not found":
                missing_fields["Date of Report"] = "[Date of the report, not the death]"
            if deceased == "N/A: Not found":
                missing_fields["Deceased Name(s)"] = "[Name or names of the deceased.]"
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
            
            # Dynamic prompt only including the missing fields
            prompt = (
                "Your goal is to transcribe the **exact** text from this report, presented as images.\n\n"
                "Please extract the following section(s):\n"
            )
            for field, instruction in missing_fields.items():
                prompt += f"\n{field}: {instruction}\n"
            prompt += (
                "\nRespond with nothing else whatsoever. You must not respond in your own 'voice' or even acknowledge the task.\n"
                "If you are unable to identify the text from the image for any given section, simply respond: \"N/A: Not found\" for that section.\n"
                "Sometimes text may be redacted with a black box; transcribe it as '[REDACTED]'.\n"
                "Make sure you transcribe the *full* text for each section, not just a snippet.\n"
                "Do *not* change the section title(s) from the above format. For example, you must not change Circumstances of Death to Circumstances of _the_ Death.\n"
            )
            
            # If verbose, print out the prompt for debugging
            if self.verbose:
                logger.info("Report: %s", url)
                logger.info("LLM prompt:\n\n%s", prompt)
            
            # Construct messages for GPT vision call
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
            fallback_deceased = 'N/A: Not found'
            fallback_coroner = 'N/A: Not found'
            fallback_area = 'N/A: Not found'
            fallback_receiver = 'N/A: Not found'
            fallback_investigation = 'N/A: Not found'
            fallback_circumstances = 'N/A: Not found'
            fallback_concerns = 'N/A: Not found'
            
            for line in llm_text.splitlines():
                line_strip = line.strip()
                line_lower = line_strip.lower()
                if line_lower.startswith("date of report:"):
                    fallback_date = line_strip.split(":", 1)[1].strip()
                elif line_lower.startswith("deceased name(s):"):
                    fallback_deceased = line_strip.split(":", 1)[1].strip()
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
            
            # Update each field only if it's still missing
            if date == 'N/A: Not found':
                date = fallback_date
            if deceased == 'N/A: Not found':
                deceased = fallback_deceased
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
        return {
            "URL": url,
            "ID": report_id,
            "Date": date,
            "DeceasedName": deceased,
            "CoronerName": coroner,
            "Area": area,
            "Receiver": receiver,
            "InvestigationAndInquest": investigation,
            "CircumstancesOfDeath": circumstances,
            "MattersOfConcern": concerns
        }
    
    # The below function serves as the entry point for the scraper.
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
        return pd.DataFrame(records)



# -----------------------------------------------------------------------------------------
# TESTING

# Load OpenAI API key
load_dotenv('api.env')
openai_api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=openai_api_key)

# Run the scraper :D
scraper = PFDScraper(
    category='alcohol_drug_medication', 
    start_page=3, 
    end_page=3, 
    max_workers=10,
    pdf_fallback=True,
    llm_fallback=True,
    api_key=openai_api_key,
    llm_model="gpt-4o-mini",
    docx_conversion="LibreOffice",
    verbose=True
)
reports = scraper.scrape_all_reports()
reports

# reports.to_csv('../data/testreports.csv')