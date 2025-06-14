# Load PFD reports

PFD Toolkit offers two ways to bring reports into a pandas DataFrame.
Most users should call [`load_reports`](load_reports.md) to download the
weekly dataset that ships with the package.  Advanced users can take
full control of the scraping pipeline using the [`Scraper`](scraper.md)
class.

The pages below explain each approach in detail:

- [Loading report data](load_reports.md) – get a ready-made DataFrame with a single function call.
- [Scraping module](scraper.md) – build your own scraping workflow for maximum flexibility.
