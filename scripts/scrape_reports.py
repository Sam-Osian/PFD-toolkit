#!/usr/bin/env python3
"""
Helper script for scraping all PFD reports contained
within `../data/full_reports.csv.
"""

from pfd_toolkit import LLM, PFDScraper
from dotenv import load_dotenv
import os

# Load OpenAI API key
load_dotenv("../notebooks/api.env")
openai_api_key = os.getenv("OPENAI_API_KEY")
llm_client = LLM(api_key=openai_api_key, max_workers=30)

# Set up scraper
scraper = PFDScraper(
    llm=llm_client,
    html_scraping=True,
    pdf_fallback=True,
    llm_fallback=True,
    delay_range=None,
)

# Run scraper
scraper.scrape_reports()

reports = scraper.reports

# reports.to_csv('../src/pfd_toolkit/data/all_reports.csv')
