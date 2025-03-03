import requests
from bs4 import BeautifulSoup
import pymupdf
import os
import pandas as pd

class PFDScraper:
    """Web scraper for extracting Prevention of Future Death (PFD) reports from the UK Judiciary website."""
    
    def __init__(self, base_url="https://www.judiciary.uk/prevention-of-future-death-reports/page/", start_page=1, end_page=559):
        """
        Initialises the scraper.
        
        :param base_url: Base URL of the PFD reports page.
        :param start_page: The first page to scrape.
        :param end_page: The last page to scrape.
        """
        self.base_url = base_url
        self.start_page = start_page
        self.end_page = end_page
        self.report_links = []
        
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
            page_url = f"{self.base_url}{page_number}/"
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

        # Extract HTML report data
        
        # ...Report ID
        ref_element = soup.find(lambda tag: tag.name == 'p' and 'Ref:' in tag.get_text(), recursive=True)
        report_id = ref_element.get_text() if ref_element else 'N/A - Not found'
        report_id = report_id.replace("Ref:", "").strip()
        
        # ...Date of report
        date_element = soup.find(lambda tag: tag.name == 'p' and 'Date of report:' in tag.get_text(), recursive=True)
        report_date = date_element.get_text() if date_element else 'N/A - Not found'
        report_date = report_date.replace("Date of report:", "").strip()
        
        
        # ...Name of coroner
        coroner_element = soup.find(lambda tag: tag.name == 'p' and 'Coroners name:' in tag.get_text(), recursive=True)
        report_coroner = coroner_element.get_text() if coroner_element else 'N/A - Not found'
        report_coroner = report_coroner.replace("Coroners name:", "").strip()
        
        # ...Area
        area_element = soup.find(lambda tag: tag.name == 'p' and 'Coroners Area:' in tag.get_text(), recursive=True)
        report_area = area_element.get_text() if area_element else 'N/A - Not found'
        report_area = report_area.replace("Coroners Area:", "").strip()

        
        
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
            investigation_section = pdf_text.split("INVESTIGATION and INQUEST")[1].split("CIRCUMSTANCES OF THE DEATH")[0]
        except IndexError:
            investigation_section = "N/A - Not found"
        
        investigation = self.clean_text(investigation_section)
        
        # ...Cirumstances of the Death
        try:
            circumstances_section = pdf_text.split("CIRCUMSTANCES OF THE DEATH")[1].split("CORONER’S CONCERNS")[0]
        except IndexError:
            circumstances_section = "N/A - Not found"
            
        circumstances = self.clean_text(circumstances_section)
        
        # ...Concerns
        try:
            concerns_section = pdf_text.split("CORONER’S CONCERNS")[1].split("ACTION SHOULD BE TAKEN ")[0]
        except IndexError:
            concerns_section = "N/A - Not found"

        concerns = self.clean_text(concerns_section)


        # Create structure of dataframe
        return {
            "url": url,
            "id": report_id,
            "date": report_date,
            "coroner": report_coroner,
            "area": report_area,
            "receiver": receiver,
            "investigation": investigation,
            "circumstances": circumstances,
            "concerns": concerns
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



scraper = PFDScraper(start_page=2, end_page=2)

reports = scraper.scrape_all_reports()
reports

#reports.to_csv('../data/reports')