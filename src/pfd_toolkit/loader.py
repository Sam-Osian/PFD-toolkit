from pfd_toolkit import PFDScraper
import pandas as pd


# -- INITALISE SCRAPER ENGINE --
scraper = PFDScraper()

# -- GET REPORTS --
# Commented out as we only need to run this once

#scraper.scrape_reports()

# Save reports
#reports = scraper.reports
#reports.to_csv('data/all_reports.csv', index=False)


# -- UPDATE REPORTS -- 

# Read the existing set of reports
reports = pd.read_csv('data/all_reports.csv')

# Try to 'top up' with new ones
scraper.top_up(old_reports=reports, date_from="2025-05-09")