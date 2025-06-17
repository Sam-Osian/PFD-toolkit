#!/usr/bin/env python3
"""
Helper script for scraping all PFD reports contained
within `../data/full_reports.csv.
"""

from pfd_toolkit import LLM, Scraper, load_reports, Cleaner
from dotenv import load_dotenv
import os

# Load OpenAI API key
load_dotenv("../notebooks/api.env")
openai_api_key = os.getenv("OPENAI_API_KEY")
llm_client = LLM(api_key=openai_api_key, max_workers=25,
                 timeout=150, seed=123)

# Set up scraper
scraper = Scraper(
    llm=llm_client,
    scraping_strategy=[2, 3, 1],
    max_workers=30,
    max_requests=30
)

# Run scraper & save reports
scraper.scrape_reports()

reports = scraper.reports

# Clean reports
cleaner = Cleaner(llm=llm_client,
                  reports=reports)

cleaned_reports = cleaner.clean_reports()

cleaned_reports.to_csv('../src/pfd_toolkit/data/all_reports.csv', index=False)