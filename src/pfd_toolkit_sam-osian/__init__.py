# The below lets users run `from pfd_toolkit import PFDScraper` instead of `from pfd_toolkit.scraper import PFDScraper`
from .scraper import PFDScraper
__all__ = ['PFDScraper']