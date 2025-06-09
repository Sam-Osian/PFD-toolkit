import warnings

# Suppress PyMuPDF SWIG-related DeprecationWarnings from SwigPyPacked and
# swigvarlink objects only
warnings.filterwarnings(
    "ignore",
    message=r"builtin type (SwigPyPacked|swigvarlink|SwigPyObject) has no __module__ attribute",
    category=DeprecationWarning,
)


# The below lets users run `from pfd_toolkit import PFDScraper` instead of `from pfd_toolkit.scraper import PFDScraper`
# Same for the other modules
from pfd_toolkit.scraper import (
    Scraper,
    HtmlFieldConfig,
    PdfSectionConfig,
)
from pfd_toolkit.cleaner import Cleaner
from pfd_toolkit.llm import LLM
from pfd_toolkit.loader import load_reports
from pfd_toolkit.screener import Screener
from pfd_toolkit.extractor import Extractor
from pfd_toolkit.config import (
    GeneralConfig,
    ScraperConfig,
)

__all__ = [
    "Scraper",
    "Cleaner",
    "LLM",
    "load_reports",
    "Screener",
    "Extractor",
    "GeneralConfig",
    "ScraperConfig",
    "HtmlFieldConfig",
    "PdfSectionConfig",
]
