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

# Configure logging for the module
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO)

class PFDScraper:
    """Web scraper for extracting Prevention of Future Death (PFD) reports from the UK Judiciary website."""
    
    def __init__(self, category: str = 'all', start_page: int = 1, end_page: int = 559, max_workers: int = 10) -> None:
        """
        Initialises the scraper.
        
        :param category: Category of reports as categorised on the judiciary.uk website.
        :param start_page: The first page to scrape.
        :param end_page: The last page to scrape.
        :param max_workers: Maximum number of workers for concurrent fetching.
        """
        self.category = category.lower()
        self.start_page = start_page
        self.end_page = end_page
        self.max_workers = max_workers
        self.report_links = []
        
        # Setup a requests session with retry logic.
        self.session = requests.Session()
        retries = Retry(total=5, backoff_factor=1, status_forcelist=[502, 503, 504])
        adapter = HTTPAdapter(max_retries=retries)
        self.session.mount("http://", adapter)
        self.session.mount("https://", adapter)
        
        # Compile regex for report ID extraction (e.g. "2025-0296").
        self._id_pattern = re.compile(r'(\d{4}-\d{4})')
        
        # Define URL templates for different PFD categories.
        if self.category == "all":
            self.page_template = "https://www.judiciary.uk/prevention-of-future-death-reports/page/{page}/"
        elif self.category == "suicide":
            self.page_template = "https://www.judiciary.uk/prevention-of-future-death-reports/page/{page}/?s&pfd_report_type=suicide-from-2015"
        elif self.category == "accident_work_safety":
            self.page_template = "https://www.judiciary.uk/page/{page}/?s&pfd_report_type=accident-at-work-and-health-and-safety-related-deaths"
        elif self.category == "alcohol_drug_medication":
            self.page_template = "https://www.judiciary.uk/page/{page}/?s&pfd_report_type=alcohol-drug-and-medication-related-deaths"
        elif self.category == "care_home":
            self.page_template = "https://www.judiciary.uk/page/{page}/?s&pfd_report_type=care-home-health-related-deaths"
        elif self.category == "child_death":
            self.page_template = "https://www.judiciary.uk/page/{page}/?s&pfd_report_type=child-death-from-2015"
        elif self.category == "community_health_emergency":
            self.page_template = "https://www.judiciary.uk/page/{page}/?s&pfd_report_type=community-health-care-and-emergency-services-related-deaths"
        elif self.category == "emergency_services":
            self.page_template = "https://www.judiciary.uk/page/{page}/?s&pfd_report_type=emergency-services-related-deaths-2019-onwards"
        elif self.category == "hospital_deaths":
            self.page_template = "https://www.judiciary.uk/page/{page}/?s&pfd_report_type=hospital-death-clinical-procedures-and-medical-management-related-deaths"
        elif self.category == "mental_health":
            self.page_template = "https://www.judiciary.uk/page/{page}/?s&pfd_report_type=mental-health-related-deaths"
        elif self.category == "police":
            self.page_template = "https://www.judiciary.uk/page/{page}/?s&pfd_report_type=police-related-deaths"
        elif self.category == "product":
            self.page_template = "https://www.judiciary.uk/page/{page}/?s&pfd_report_type=product-related-deaths"
        elif self.category == "railway":
            self.page_template = "https://www.judiciary.uk/page/{page}/?s&pfd_report_type=railway-related-deaths"
        elif self.category == "road":
            self.page_template = "https://www.judiciary.uk/page/{page}/?s&pfd_report_type=road-highways-safety-related-deaths"
        elif self.category == "service_personnel":
            self.page_template = "https://www.judiciary.uk/page/{page}/?s&pfd_report_type=service-personnel-related-deaths"
        elif self.category == "custody":
            self.page_template = "https://www.judiciary.uk/page/{page}/?s&pfd_report_type=state-custody-related-deaths"
        elif self.category == "wales":
            self.page_template = "https://www.judiciary.uk/page/{page}/?s&pfd_report_type=wales-prevention-of-future-deaths-reports-2019-onwards"
        elif self.category == "other":
            self.page_template = "https://www.judiciary.uk/page/{page}/?s&pfd_report_type=other-related-deaths"
        else:
            valid_options = (
                "'all', 'suicide', 'accident_work_safety', 'alcohol_drug_medication', 'care_home', "
                "'child_death', 'community_health_emergency', 'emergency_services', 'hospital_deaths', "
                "'mental_health', 'police', 'product', 'railway', 'road', 'service_personnel', 'custody', "
                "'wales', 'other'"
            )
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
        Downloads and extracts text from a PDF report.

        :param pdf_url: URL of the PDF to extract text from.
        :return: Cleaned text extracted from the PDF.
        """
        try:
            response = self.session.get(pdf_url)
            response.raise_for_status()
        except requests.RequestException as e:
            logger.error("Failed to fetch PDF: %s; Error: %s", pdf_url, e)
            return "N/A"
        
        # Use tempfile for automatic clean-up of temporary files
        try:
            with tempfile.NamedTemporaryFile(suffix=".pdf", delete=True) as temp_file:
                temp_file.write(response.content)
                temp_file.flush()
                pdf_document = pymupdf.open(temp_file.name)
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
        :return: Extracted text or 'N/A - Not found'.
        """
        for keyword in keywords:
            element = soup.find(lambda tag: tag.name == 'p' and keyword in tag.get_text(), recursive=True)
            if element:
                return element.get_text().strip()
        return 'N/A - Not found'
    
    def _extract_section_from_text(self, text: str, start_keywords: list, end_keywords: list) -> str:
        """
        Helper function to extract a section from text using multiple start and end keywords.
        Uses case-insensitive search to locate the markers.

        :param text: The full text to search in.
        :param start_keywords: List of possible starting keywords.
        :param end_keywords: List of possible ending keywords.
        :return: Extracted section text or "N/A - Not found" if markers are missing.
        """
        lower_text = text.lower()
        for start in start_keywords:
            start_lower = start.lower()
            start_index = lower_text.find(start_lower)
            if start_index != -1:
                section_start = start_index + len(start_lower)
                # Find all occurrences of end keywords after section_start
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
        return "N/A - Not found"

    
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
        
        ## Extract HTML report data
        
        # Report ID extraction using compiled regex
        ref_element = soup.find(lambda tag: tag.name == 'p' and 'Ref:' in tag.get_text(), recursive=True)
        if ref_element:
            match = self._id_pattern.search(ref_element.get_text())
            report_id = match.group(1) if match else 'N/A - Not found'
        else:
            report_id = 'N/A - Not found'
        
        # Date of report extraction
        date_text = self._extract_text_by_keywords(soup, ["Date of report:"])
        if date_text != 'N/A - Not found':
            date_text = date_text.replace("Date of report:", "").strip()
            try:
                parsed_date = parser.parse(date_text, fuzzy=True)
                report_date = parsed_date.strftime("%Y-%m-%d")
            except Exception as e:
                logger.error("Error parsing date '%s': %s", date_text, e)
                report_date = date_text
        else:
            report_date = 'N/A - Not found'
        
        # Name of coroner extraction
        coroner_text = self._extract_text_by_keywords(soup, ["Coroners name:", "Coroner name:", "Coroner's name:"])
        report_coroner = coroner_text.replace("Coroners name:", "").replace("Coroner name:", "").strip()
        
        # Area extraction.
        area_text = self._extract_text_by_keywords(soup, ["Coroners Area:", "Coroner Area:", "Coroner's Area:"])
        report_area = area_text.replace("Coroners Area:", "").replace("Coroner Area:", "").replace("Coroner's Area:", "").strip()
        
        
        ## Extract PDF report data
        # Receiver extraction
        try:
            receiver_section = pdf_text.split(" SENT ")[1].split("CORONER")[0]
        except IndexError:
            receiver_section = "N/A - Not found"
        receiver = self.clean_text(receiver_section).replace("TO:", "").strip()
        
        # Investigation & Inquest extraction
        investigation_section = self._extract_section_from_text(
            pdf_text,
            start_keywords=["INVESTIGATION and INQUEST", "3 INQUEST"],
            end_keywords=["CIRCUMSTANCES OF DEATH", "CIRCUMSTANCES OF THE DEATH", "CIRCUMSTANCES OF"]
        )
        investigation = self.clean_text(investigation_section)
        
       # try:
        #    investigation_section = pdf_text.split("INVESTIGATION")[1].split("CIRCUMSTANCES OF THE DEATH")[0]
        #except IndexError:
         #   investigation_section = "N/A - Not found"
        #investigation = self.clean_text(investigation_section).replace("and INQUEST", "").strip()
        
        # Circumstances of Death extraction
        circumstances_section = self._extract_section_from_text(
            pdf_text,
            start_keywords=["CIRCUMSTANCES OF DEATH", "CIRCUMSTANCES OF THE DEATH", "CIRCUMSTANCES OF"],
            end_keywords=["CORONER'S CONCERNS", "CORONER CONCERNS", "CORONERS CONCERNS", "as follows"]
        )
        circumstances = self.clean_text(circumstances_section)
        
        # Matters of Concern extraction
        concerns_section = self._extract_section_from_text(
            pdf_text,
            start_keywords=["CORONER’S CONCERNS", "as follows"],
            end_keywords=["ACTION SHOULD BE TAKEN"]
        )
        concerns = self.clean_text(concerns_section)
        
        return {
            "URL": url,
            "ID": report_id,
            "Date": report_date,
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
        records = []
        for url in self.report_links:
            report_data = self.extract_report_info(url)
            if report_data:
                records.append(report_data)
        return pd.DataFrame(records)



scraper = PFDScraper(category='road', start_page=2, end_page=2, max_workers=10)
reports = scraper.scrape_all_reports()
reports

#reports.to_csv('../data/testreports.csv')