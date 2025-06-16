from pfd_toolkit import LLM, Scraper
from dotenv import load_dotenv
import os
import pandas as pd

reports = pd.read_csv('../src/pfd_toolkit/data/all_reports_test_2.csv')

# Load OpenAI API key
load_dotenv("../notebooks/api.env")
openai_api_key = os.getenv("OPENAI_API_KEY")
llm_client = LLM(api_key=openai_api_key, max_workers=15,
                 timeout=300, seed=123)

# Set up scraper
scraper = Scraper(
    llm=llm_client,
    scraping_strategy=[2, 3, 1],
    max_workers=30,
    max_requests=30
)

scraper.reports = reports

# Run scraper & save reports
scraper.run_llm_fallback()

reports = scraper.reports

reports.to_csv('../src/pfd_toolkit/data/all_reports_test_2.csv')
