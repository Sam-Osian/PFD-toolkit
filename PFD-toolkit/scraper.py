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
from io import BytesIO  # For in-memory buffer
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
        llm_fallback: bool = False,
        llm_model: str = "gpt-4o-mini",
        docx_conversion: str = "None"  
    ) -> None:
        """
        Initialises the scraper.
        
        :param category: Category of reports as categorised on the judiciary.uk website. Options are 'all', 'suicide', 'accident_work_safety', 'alcohol_drug_medication', 'care_home', 'child_death', 'community_health_emergency', 'emergency_services', 'hospital_deaths', 'mental_health', 'police', 'product', 'railway', 'road', 'service_personnel', 'custody', 'wales', 'other'.
        :param start_page: The first page to scrape.
        :param end_page: The last page to scrape.
        :param max_workers: Maximum number of workers for concurrent fetching. 
        :param llm_fallback: If PDF scraping fails, whether to fallback to OpenAI LLM to process reports as images. OpenAI API key must be set.
        :param llm_model: The specific OpenAI LLM model to use, if llm_fallback is set to True.
        :param docx_conversion: Conversion method for .docx files; "MicrosoftWord", "LibreOffice", or "None" (default).
        """
        self.category = category.lower()
        self.start_page = start_page
        self.end_page = end_page
        self.max_workers = max_workers
        self.llm_fallback = llm_fallback
        self.llm_model = llm_model
        self.docx_conversion = docx_conversion
        self.report_links = []
        
        # Setup a requests session with retry logic.
        self.session = requests.Session()
        retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504])
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Compile regex for report ID extraction (e.g. "2025-0296").
        self._id_pattern = re.compile(r'(\d{4}-\d{4})')
        
        # Define URL templates for different PFD categories using a dictionary mapping.
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
        
    def get_href_values(self, url: str) -> list:
        """
        Extracts href values from <a> elements with class 'card__link'.
        
        :param url: URL of the page to scrape.
        :return: List of href values.
        """
        try:
            response = self.session.get(url)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error("Failed to fetch page: %s; Error: %s", url, e)
            return []
        
        soup = BeautifulSoup(response.text, 'html.parser')
        links = soup.find_all('a', class_='card__link')
        return [link.get('href') for link in links if link.get('href')]
    
    def get_report_links(self) -> list:
        """
        Collects all PFD report links from the paginated pages.
        
        :return: A list of report URLs.
        """
        self.report_links = []
        pages = list(range(self.start_page, self.end_page + 1))
        
        def fetch_page_links(page_number: int) -> list:
            page_url = self.page_template.format(page=page_number)
            href_values = self.get_href_values(page_url)
            logger.info("Scraped %d links from %s", len(href_values), page_url)
            return href_values
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(fetch_page_links, pages))
        
        for href_values in results:
            self.report_links.extend(href_values)
        logger.info("Total collected report links: %d", len(self.report_links))
        return self.report_links
    
    @staticmethod
    def clean_text(text: str) -> str:
        """Cleans text by removing excessive whitespace."""
        return ' '.join(text.split())
    
    def extract_text_from_pdf(self, pdf_url: str) -> str:
        """
        Downloads and extracts text from a PDF report. If the file is not in PDF format (.docx or .doc),
        converts it to PDF using the method specified by self.docx_conversion.
        
        :param pdf_url: URL of the file to extract text from.
        :return: Cleaned text extracted from the PDF, or "N/A" on failure.
        """
        # Determine file extension from URL
        parsed_url = urlparse(pdf_url)
        path = unquote(parsed_url.path)
        ext = os.path.splitext(path)[1].lower()
        
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
            logger.info("File %s is not a PDF (extension %s)", pdf_url, ext)
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
                    logger.info("Conversion successful using Microsoft Word!")
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
                    logger.info("Conversion successful using LibreOffice!")
                except Exception as e:
                    logger.error("Conversion using LibreOffice failed: %s", e)
                    return "N/A"
            else:
                logger.info("docx_conversion is set to 'None'; skipping conversion.")
                return "N/A"
        else:
            pdf_bytes = file_bytes
        
        # Process pdf_bytes as PDF
        try:
            pdf_buffer = BytesIO(pdf_bytes)
            pdf_document = pymupdf.open(stream=pdf_buffer, filetype="pdf")
            text = "".join(page.get_text() for page in pdf_document)
            pdf_document.close()
        except Exception as e:
            logger.error("Error processing PDF %s: %s", pdf_url, e)
            return "N/A"
        
        return self.clean_text(text)
    
    def _extract_text_by_keywords(self, soup: BeautifulSoup, keywords: list) -> str:
        """
        Helper function to extract text from a <p> element containing any of the given keywords.
        
        :param soup: BeautifulSoup object of the page.
        :param keywords: List of keywords to search for.
        :return: Extracted text or 'N/A: Not found'.
        """
        for keyword in keywords:
            element = soup.find(lambda tag: tag.name == 'p' and keyword in tag.get_text(), recursive=True)
            if element:
                return element.get_text().strip()
        return 'N/A: Not found'
    
    def _extract_section_from_text(self, text: str, start_keywords: list, end_keywords: list) -> str:
        """
        Helper function to extract a section from text using multiple start and end keywords.
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
    
    def extract_report_info(self, url: str) -> dict:
        """
        Extracts metadata and text from a PFD report webpage.
        
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
            logger.error("No PDF links found on %s", url)
            return None
        
        report_link = pdf_links[0]
        pdf_text = self.extract_text_from_pdf(report_link)
        
        # ------------------
        # HTML Data Extraction
        # ------------------
        # Report ID extraction using compiled regex
        ref_element = soup.find(lambda tag: tag.name == 'p' and 'Ref:' in tag.get_text(), recursive=True)
        if ref_element:
            match = self._id_pattern.search(ref_element.get_text())
            report_id = match.group(1) if match else 'N/A: Not found'
        else:
            report_id = 'N/A: Not found'
        
        # Date of report extraction with fuzzy date parsing
        date_text = self._extract_text_by_keywords(soup, ["Date of report:"])
        if date_text != 'N/A: Not found':
            date_text = date_text.replace("Date of report:", "").strip()
            try:
                parsed_date = parser.parse(date_text, fuzzy=True)
                report_date = parsed_date.strftime("%Y-%m-%d")
            except Exception as e:
                logger.error("Error parsing date '%s': %s", date_text, e)
                report_date = date_text
        else:
            report_date = 'N/A: Not found'
        
        # Name of deceased extraction
        deceased_text = self._extract_text_by_keywords(
            soup, ["Deceased name:", "Deceased's name:", "Deceaseds name:"]
        )
        report_deceased = deceased_text.replace("Deceased name:", "") \
                                       .replace("Deceased's name:", "") \
                                       .replace("Deceaseds name:", "") \
                                       .strip()
        if len(report_deceased) < 5 or len(report_deceased) > 30:
            report_deceased = 'N/A: Not found'
        
        # Name of coroner extraction
        coroner_text = self._extract_text_by_keywords(
            soup, ["Coroners name:", "Coroner name:", "Coroner's name:"]
        )
        report_coroner = coroner_text.replace("Coroners name:", "") \
                                     .replace("Coroner name:", "") \
                                     .replace("Coroner's name:", "") \
                                     .strip()
        if len(report_coroner) < 5 or len(report_coroner) > 30:
            report_coroner = 'N/A: Not found'
        
        # Area extraction
        area_text = self._extract_text_by_keywords(
            soup, ["Coroners Area:", "Coroner Area:", "Coroner's Area:"]
        )
        report_area = area_text.replace("Coroners Area:", "") \
                               .replace("Coroner Area:", "") \
                               .replace("Coroner's Area:", "") \
                               .strip()
        if len(report_area) < 5 or len(report_area) > 30:
            report_area = 'N/A: Not found'
        
        # ------------------
        # PDF Data Extraction
        # ------------------
        
        # Receiver extraction
        try:
            receiver_section = pdf_text.split(" SENT ")[1].split("CORONER")[0]
        except IndexError:
            receiver_section = "N/A: Not found"
        receiver = self.clean_text(receiver_section).replace("TO:", "").strip()
        if len(receiver) < 5:
            receiver = 'N/A: Not found'
        
        # Investigation & inquest extraction
        investigation_section = self._extract_section_from_text(
            pdf_text,
            start_keywords=["INVESTIGATION and INQUEST", "3 INQUEST"],
            end_keywords=["CIRCUMSTANCES OF DEATH", "CIRCUMSTANCES OF THE DEATH", "CIRCUMSTANCES OF"]
        )
        investigation = self.clean_text(investigation_section)
        if len(investigation) < 30:
            investigation = 'N/A: Not found'
        
        # Circumstances of Death extraction
        circumstances_section = self._extract_section_from_text(
            pdf_text,
            start_keywords=["CIRCUMSTANCES OF DEATH", "CIRCUMSTANCES OF THE DEATH", "CIRCUMSTANCES OF"],
            end_keywords=["CORONER'S CONCERNS", "CORONER CONCERNS", "CORONERS CONCERNS", "as follows"]
        )
        circumstances = self.clean_text(circumstances_section)
        if len(circumstances) < 30:
            circumstances = 'N/A: Not found'
        
        # Matters of Concern extraction
        concerns_section = self._extract_section_from_text(
            pdf_text,
            start_keywords=["CORONER’S CONCERNS", "as follows"],
            end_keywords=["ACTION SHOULD BE TAKEN"]
        )
        concerns = self.clean_text(concerns_section)
        if len(concerns) < 30:
            concerns = 'N/A: Not found'
        
        # ------------------
        # LLM Fallback (Only if any field is "N/A: Not found")
        # ------------------
        if self.llm_fallback and (
            'N/A: Not found' in [
                report_date,
                report_deceased,
                report_coroner,
                report_area,
                receiver,
                investigation,
                circumstances,
                concerns
            ]
        ):
            logger.info("Attempting LLM fallback for %s", url)
            
            # Load OpenAI API key
            load_dotenv('api.env')
            openai_api_key = os.getenv('OPENAI_API_KEY')
            client = OpenAI(api_key=openai_api_key)
            
            # Re-download the PDF for image conversion
            try:
                pdf_response = self.session.get(report_link)
                pdf_response.raise_for_status()
                pdf_bytes = pdf_response.content
            except Exception as e:
                logger.error("Failed to fetch PDF for image conversion: %s", e)
                pdf_bytes = None
            
            base64_images = []
            if pdf_bytes:
                try:
                    images = convert_from_bytes(pdf_bytes)
                except Exception as e:
                    logger.error("Error converting PDF to images: %s", e)
                    images = []
                for img in images:
                    buffered = BytesIO()
                    img.save(buffered, format="JPEG")
                    img_str = base64.b64encode(buffered.getvalue()).decode("utf-8")
                    base64_images.append(img_str)
            
            # Construct messages for GPT vision call
            messages = [
                {
                    "type": "text",
                    "text": (
                        "Your goal is to transcribe the **exact** text from this report, presented as images.\n\n"
                        "You must return the text in the **exact** format given below:\n\n"
                        "Date of Report: [Date in YYYY-MM-DD format if it is not already.]\n"
                        "Deceased's Name: [Name of the deceased. Provide the name only with no additional information.]\n"
                        "Coroner's Name: [Name of the coroner writing the report. Provide the name only, with no additional information.]\n"
                        "Area: [Area/location of the Coroner. Provide the location only, with no additional information.]\n"
                        "Receiver: [Name of the person or people the report is sent to. Provide the name only with no additional information.]\n"
                        "Investigation and Inquest: [The text from the Investigation and Inquest section. Do not extract from any other section.]\n"
                        "Circumstances of Death: [The text from the Circumstances of Death section. Do not extract from any other section.]\n"
                        "Coroner's Concerns: [The text from the Coroner's Concerns or Matters of Concern section. Do not extract from any other section.]\n\n"
                        "*\n\n"
                        "Respond with nothing else whatsoever. You must not respond in your own 'voice' or even acknowledge the task.\n\n"
                        "If you are unable to identify the text from the image for any given section, simply respond: \"N/A: Not found\" for each section.\n\n"
                        "Sometimes text may be redacted with a black box; transcribe it as '[REDACTED]'.\n\n"
                        "Make sure you transcribe the *full* text for each section, not just a snippet.\n"
                    )
                }
            ]
            
            # Append each image as an image message
            for b64_img in base64_images:
                messages.append({
                    "type": "image_url",
                    "image_url": {
                        "url": f"data:image/jpeg;base64,{b64_img}"
                    }
                })
            
            response = client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "user", "content": messages}
                ],
            )
            
            # Correct extraction of the response text from the ChatCompletion object.
            llm_text = response.choices[0].message.content
            logger.info("LLM fallback response: %s", llm_text)
            
            # Parse the LLM response to update the fields only if they are still "N/A: Not found".
            
            fallback_date = 'N/A: Not found'
            fallback_deceased = 'N/A: Not found'
            fallback_coroner = 'N/A: Not found'
            fallback_area = 'N/A: Not found'
            fallback_receiver = 'N/A: Not found'
            fallback_investigation = 'N/A: Not found'
            fallback_circumstances = 'N/A: Not found'
            fallback_concerns = 'N/A: Not found'
            
            for line in llm_text.splitlines():
                line = line.strip()
                if line.startswith("Date of Report:"):
                    fallback_date = line.split("Date of Report:", 1)[1].strip()
                elif line.startswith("Deceased's Name:"):
                    fallback_deceased = line.split("Deceased's Name:", 1)[1].strip()
                elif line.startswith("Coroner's Name:"):
                    fallback_coroner = line.split("Coroner's Name:", 1)[1].strip()
                elif line.startswith("Area:"):
                    fallback_area = line.split("Area:", 1)[1].strip()
                elif line.startswith("Receiver:"):
                    fallback_receiver = line.split("Receiver:", 1)[1].strip()
                elif line.startswith("Investigation and Inquest:"):
                    fallback_investigation = line.split("Investigation and Inquest:", 1)[1].strip()
                elif line.startswith("Circumstances of Death:"):
                    fallback_circumstances = line.split("Circumstances of Death:", 1)[1].strip()
                elif line.startswith("Coroner's Concerns:"):
                    fallback_concerns = line.split("Coroner's Concerns:", 1)[1].strip()
            
            # Update each field only if it is "N/A: Not found"
            if fallback_date != 'N/A: Not found':
                try:
                    parsed_date = parser.parse(fallback_date, fuzzy=True)
                    fallback_date = parsed_date.strftime("%Y-%m-%d")
                except Exception as e:
                    logger.error("LLM fallback: could not parse date '%s': %s", fallback_date, e)
            
            if report_date == 'N/A: Not found':
                report_date = fallback_date
            if report_deceased == 'N/A: Not found':
                report_deceased = fallback_deceased
            if report_coroner == 'N/A: Not found':
                report_coroner = fallback_coroner
            if report_area == 'N/A: Not found':
                report_area = fallback_area
            if receiver == 'N/A: Not found':
                receiver = fallback_receiver
            if investigation == 'N/A: Not found':
                investigation = fallback_investigation
            if circumstances == 'N/A: Not found':
                circumstances = fallback_circumstances
            if concerns == 'N/A: Not found':
                concerns = fallback_concerns
        
        return {
            "URL": url,
            "ID": report_id,
            "Date": report_date,
            "DeceasedName": report_deceased,
            "CoronerName": report_coroner,
            "Area": report_area,
            "Receiver": receiver,
            "InvestigationAndInquest": investigation,
            "CircumstancesOfDeath": circumstances,
            "MattersOfConcern": concerns
        }
    
    def scrape_all_reports(self) -> pd.DataFrame:
        """
        Scrapes all reports from the collected report links.
        
        :return: A pandas DataFrame containing one row per scraped report.
        """
        if not self.report_links:
            self.get_report_links()
        
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            results = list(executor.map(self.extract_report_info, self.report_links))
        # Filter out any failed report extractions (None)
        records = [record for record in results if record is not None]
        return pd.DataFrame(records)


# Example usage:
scraper = PFDScraper(
    category='accident_work_safety', 
    start_page=2, 
    end_page=2, 
    max_workers=10,
    llm_fallback=False,
    llm_model="gpt-4o-mini",
    docx_conversion="LibreOffice"  # Options: "MicrosoftWord", "LibreOffice", "None"
)
reports = scraper.scrape_all_reports()
reports

# reports.to_csv('../data/testreports.csv')
