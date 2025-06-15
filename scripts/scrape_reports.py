#!/usr/bin/env python3
"""
Helper script for scraping all PFD reports contained
within `../data/full_reports.csv.
"""

from pfd_toolkit import LLM, Scraper
from dotenv import load_dotenv
import os

# Load OpenAI API key
load_dotenv("../notebooks/api.env")
openai_api_key = os.getenv("OPENAI_API_KEY")
llm_client = LLM(api_key=openai_api_key, max_workers=25,
                 timeout=150)

# Set up scraper
scraper = Scraper(
    start_date="2025-05-01",
    llm=llm_client,
    scraping_strategy=[3, 2, 1],
)

# Run scraper & save reports
scraper.scrape_reports()

reports = scraper.reports

reports.to_csv('../src/pfd_toolkit/data/all_reports_test.csv')
