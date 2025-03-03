import requests
from bs4 import BeautifulSoup
import pymupdf
import os
import pandas as pd
import re
from dateutil import parser

class PFDScraper:
    """Web scraper for extracting Prevention of Future Death (PFD) reports from the UK Judiciary website."""
    
    def __init__(self, category='all', start_page=1, end_page=559):
        """
        Initialises the scraper.
        
        :param category: Category of reports as categorised on the judiciary.uk website.
        :param base_url: Base URL of the PFD reports page.
        :param start_page: The first page to scrape.
        :param end_page: The last page to scrape.
        """
        self.category = category.lower()
        self.start_page = start_page
        self.end_page = end_page
        self.report_links = []
        
       # Define URL templates for different PFD categories from judiciary.uk website
        if self.category == "all":
            self.page_template = "https://www.judiciary.uk/prevention-of-future-death-reports/page/{page}/"
        elif self.category == "suicide":
            self.page_template = "https://www.judiciary.uk/prevention-of-future-death-reports/page/{page}/?s&pfd_report_type=suicide-from-2015"
        elif self.category == "accident":
            self.page_template = "https://www.judiciary.uk/page/{page}/?s&pfd_report_type=accident-at-work-and-health-and-safety-related-deaths"
        elif self.category == "alcohol drug":
            self.page_template = "https://www.judiciary.uk/page/{page}/?s&pfd_report_type=alcohol-drug-and-medication-related-deaths"
        elif self.category == "care home":
            self.page_template = "https://www.judiciary.uk/page/{page}/?s&pfd_report_type=care-home-health-related-deaths"
        else:
            raise ValueError(f"Unknown category '{self.category}'. Valid options are: 'all', 'accident', 'alcohol drug', 'care home'")
            
            
    def get_href_values(self, url):
        """Extracts href values from <a> elements with class 'card__link'."""
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to fetch page: {url}")
            return []
        soup = BeautifulSoup(response.text, 'html.parser')
        links = soup.find_all('a', class_='card__link')
        return [link.get('href') for link in links]
    
    def get_report_links(self):
        """
        Collects all PFD report links from the paginated pages.
        
        Returns:
            A list of report URLs.
        """
        self.report_links = []
        for page_number in range(self.start_page, self.end_page + 1):
            page_url = self.page_template.format(page=page_number)
            href_values = self.get_href_values(page_url)
            self.report_links.extend(href_values)
            print(f"Scraped {len(href_values)} links from {page_url}")
        print(f"Total collected report links: {len(self.report_links)}")
        return self.report_links
    
    def clean_text(self, text):
        """Cleans text by removing excessive whitespace."""
        return ' '.join(text.split())
    
    def extract_text_from_pdf(self, pdf_url):
        """Downloads and extracts text from a PDF report."""
        response = requests.get(pdf_url)
        if response.status_code != 200:
            print(f"Failed to fetch PDF: {pdf_url}")
            return "N/A"
        temp_file = "temp.pdf"
        with open(temp_file, "wb") as pdf_file:
            pdf_file.write(response.content)
        text = ""
        try:
            pdf_document = pymupdf.open(temp_file)
            for page in pdf_document:
                text += page.get_text()
            pdf_document.close()
        except Exception as e:
            print(f"Error processing PDF {pdf_url}: {e}")
            text = "N/A"
        os.remove(temp_file)
        return self.clean_text(text)
    
    def extract_report_info(self, url):
        """Extracts metadata and text from a PFD report webpage."""
        response = requests.get(url)
        if response.status_code != 200:
            print(f"Failed to fetch {url}")
            return None

        soup = BeautifulSoup(response.content, 'html.parser')
        pdf_links = [a['href'] for a in soup.find_all('a', class_='govuk-button')]
        if not pdf_links:
            print(f"No PDF links found on {url}")
            return None

        report_link = pdf_links[0]
        pdf_text = self.extract_text_from_pdf(report_link)

        ## Extract HTML report data
        # Report ID
        ref_element = soup.find(lambda tag: tag.name == 'p' and 'Ref:' in tag.get_text(), recursive=True)
        if ref_element:
            # Extract a pattern of four digits, a hyphen, and four digits (e.g. 2025-0296)
            match = re.search(r'(\d{4}-\d{4})', ref_element.get_text())
            report_id = match.group(1) if match else 'N/A - Not found'
        else:
            report_id = 'N/A - Not found'
        
        # Date of report
        # ...PFD reports use various formats for dates (e.g. '03/03/2025' and '3rd March 2025')
        # ...to solve this, use `dateutil` to parse through the text and standardise into YYYY-MM-DD
        date_element = soup.find(lambda tag: tag.name == 'p' and 'Date of report:' in tag.get_text(), recursive=True)
        if date_element:
            text = date_element.get_text().replace("Date of report:", "").strip()
            try:
                parsed_date = parser.parse(text, fuzzy=True)
                report_date = parsed_date.strftime("%Y-%m-%d")
            except Exception as e:
                report_date = text
        else:
            report_date = 'N/A - Not found'
                
        
        # Name of coroner
        coroner_element = soup.find(lambda tag: tag.name == 'p' and "Coroners name: " in tag.get_text(), recursive=True)
        report_coroner = coroner_element.get_text() if coroner_element else 'N/A - Not found'
        
        # ...Sometimes the name of the coroner is listed under 'Coroner name:' or 'Coroner's name'.
        if report_coroner == 'N/A - Not found':
            coroner_element = soup.find(lambda tag: tag.name == 'p' and "Coroner name: " in tag.get_text(), recursive=True)
            report_coroner = coroner_element.get_text() if coroner_element else 'N/A - Not found'
        
        if report_coroner == 'N/A - Not found':
            coroner_element = soup.find(lambda tag: tag.name == 'p' and "Coroner's name: " in tag.get_text(), recursive=True)
            report_coroner = coroner_element.get_text() if coroner_element else 'N/A - Not found'
        
        report_coroner = report_coroner.replace("Coroners name:", "").strip()
        report_coroner = report_coroner.replace("Coroner name:", "").strip()
        
        # Area
        area_element = soup.find(lambda tag: tag.name == 'p' and "Coroners Area:" in tag.get_text(), recursive=True)
        report_area = area_element.get_text() if area_element else 'N/A - Not found'
        
        # ...Sometimes the area of the coroner is listed under 'Coroner Area' or 'Coroner's area'
        
        if report_area == 'N/A - Not found':
            area_element = soup.find(lambda tag: tag.name == 'p' and "Coroner Area:" in tag.get_text(), recursive=True)
            report_area = area_element.get_text() if area_element else 'N/A - Not found'
        
        if report_area == 'N/A - Not found':
            area_element = soup.find(lambda tag: tag.name == 'p' and "Coroner's Area:" in tag.get_text(), recursive=True)
            report_area = area_element.get_text() if area_element else 'N/A - Not found'
        
        report_area = report_area.replace("Coroners Area:", "").strip()
        report_area = report_area.replace("Coroner Area:", "").strip()
        report_area = report_area.replace("Coroner's Area:", "").strip()
        
        
        # Extract PDF report data
        
        # ...Receiver
        try:
            receiver_section = pdf_text.split(" SENT ")[1].split("CORONER")[0]
        except IndexError:
            receiver_section = "N/A - Not found"

        receiver = self.clean_text(receiver_section)
        receiver = receiver.replace("TO:", "").strip()
        
        # ...Investigation & Inquest
        try:
            investigation_section = pdf_text.split("INVESTIGATION")[1].split("CIRCUMSTANCES OF THE DEATH")[0]
        except IndexError:
            investigation_section = "N/A - Not found"
        
        investigation = self.clean_text(investigation_section)
        investigation = investigation.replace("and INQUEST", "").strip()
        
        # ...Cirumstances of Death
        try:
            circumstances_section = pdf_text.split("CIRCUMSTANCES OF")[1].split("CORONER’S CONCERNS")[0]
        except IndexError:
            circumstances_section = "N/A - Not found"
            
        circumstances = circumstances_section.replace("THE DEATH", "").strip()
        circumstances = circumstances.replace("DEATH", "").strip()
        circumstances = self.clean_text(circumstances)
        
        # ...Matters of Concern
        try:
            concerns_section = pdf_text.split("as follows")[1].split("ACTION SHOULD BE TAKEN")[0]
        except IndexError:
            concerns_section = "N/A - Not found"

        concerns = self.clean_text(concerns_section)


        # Create structure of dataframe
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

    
    def scrape_all_reports(self):
        """
        Scrapes all reports from the collected report links.
        If report links haven't been gathered yet, it runs get_report_links first.
        
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



scraper = PFDScraper(category='care home', start_page=2, end_page=2)

reports = scraper.scrape_all_reports()
reports

#reports.to_csv('../data/reports')