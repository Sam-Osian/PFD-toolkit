from pfd_toolkit import PFDScraper
import pandas as pd


# -- INITALISE SCRAPER ENGINE --
scraper = PFDScraper()

# -- GET REPORTS --
# Commented out as we only need to run this once

#scraper.scrape_reports()

# Save reports
#reports = scraper.reports
#reports.to_csv('../src/pfd_toolkit/data/all_reports.csv')


# -- UPDATE REPORTS -- 