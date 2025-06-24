from __future__ import annotations

from typing import Dict, List, Tuple, Any
import logging
from bs4 import BeautifulSoup
import pandas as pd
from dateutil import parser as date_parser
from concurrent.futures import ThreadPoolExecutor
from tqdm import tqdm
import requests
from itertools import count
from datetime import datetime

from .html_extractor import HtmlExtractor
from .pdf_extractor import PdfExtractor
from ..cleaner import Cleaner

from .llm_extractor import run_llm_fallback as _run_llm_fallback
from ..llm import LLM
from ..config import GeneralConfig, ScraperConfig

# -----------------------------------------------------------------------------
# Logging Configuration:
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)


class Scraper:
    """Scrape UK “Prevention of Future Death” (PFD) reports into a
    pandas.DataFrame.

    The extractor runs in three cascading layers
    (`html → pdf → llm`), each independently switchable.

    1. **HTML scrape** – parse metadata and rich sections directly from
       the web page.
    2. **PDF fallback** – download the attached PDF and extract text with
       *PyMuPDF* for any missing fields.
    3. **LLM fallback** – delegate unresolved gaps to a Large Language
       Model supplied via *llm*.

    Each layer can be enabled or disabled via ``scraping_strategy``.

    Parameters
    ----------
    llm : LLM | None
        Client implementing ``_call_llm_fallback()``; required only when the
        LLM stage is enabled.
    category : str
        Judiciary category slug (e.g. ``"suicide"``, ``"hospital_deaths"``)
        or ``"all"``.
    start_date : str
        Inclusive lower bound for the **report date** in the ``YYYY-MM-DD``
        format.
    end_date : str
        Inclusive upper bound for the **report date** in the ``YYYY-MM-DD``
        format.
    max_workers : int
        Thread-pool size for concurrent scraping.
    max_requests : int
        Maximum simultaneous requests per host (enforced with a semaphore).
    delay_range : tuple[float, float] | None
        Random delay *(seconds)* before every request.
        Use ``None`` to disable (not recommended).
    timeout : int
        Per-request timeout in seconds.
    scraping_strategy : list[int] | tuple[int, int, int]
        Defines the order in which HTML, PDF and LLM scraping are attempted.
        The sequence indexes correspond to ``(HTML, PDF, LLM)``. Provide
        ``-1`` to disable a stage.  For example ``[1, 2, -1]`` runs HTML first,
        then PDF, and disables LLM scraping.
    include_url : bool, optional
        Include the ``url`` column. Defaults to ``True``.
    include_id : bool, optional
        Include the ``id`` column. Defaults to ``True``.
    include_date : bool, optional
        Include the ``date`` column. Defaults to ``True``.
    include_coroner : bool, optional
        Include the ``coroner`` column. Defaults to ``True``.
    include_area : bool, optional
        Include the ``area`` column. Defaults to ``True``.
    include_receiver : bool, optional
        Include the ``receiver`` column. Defaults to ``True``.
    include_investigation : bool, optional
        Include the ``investigation`` column. Defaults to ``True``.
    include_circumstances : bool, optional
        Include the ``circumstances`` column. Defaults to ``True``.
    include_concerns : bool, optional
        Include the ``concerns`` column. Defaults to ``True``.
    include_time_stamp : bool, optional
        Include a ``date_scraped`` column. Defaults to ``False``.
    verbose : bool
        Emit debug-level logs when *True*.

    Attributes
    ----------
    reports : pandas.DataFrame | None
        Cached result of the last call to ``scrape_reports`` or ``top_up``.
    report_links : list[str]
        URLs discovered by ``get_report_links``.
    NOT_FOUND_TEXT : str
        Placeholder value set when a field cannot be extracted.

    Examples
    --------

        from pfd_toolkit import Scraper
        scraper = Scraper(
            category="suicide",
            start_date="2020-01-01",
            end_date="2022-12-31",
            scraping_strategy=[1, 2, 3],
            llm=my_llm_client,
        )
        df = scraper.scrape_reports()          # full scrape
        newer_df = scraper.top_up(df)          # later "top-up"
        added_llm_df = scraper.run_llm_fallback(df)  # apply LLM retro-actively
    """

    # Constants for reused strings and keys to ensure consistency and avoid typos
    NOT_FOUND_TEXT = GeneralConfig.NOT_FOUND_TEXT

    # DataFrame column names
    COL_URL = GeneralConfig.COL_URL
    COL_ID = GeneralConfig.COL_ID
    COL_DATE = GeneralConfig.COL_DATE
    COL_CORONER_NAME = GeneralConfig.COL_CORONER_NAME
    COL_AREA = GeneralConfig.COL_AREA
    COL_RECEIVER = GeneralConfig.COL_RECEIVER
    COL_INVESTIGATION = GeneralConfig.COL_INVESTIGATION
    COL_CIRCUMSTANCES = GeneralConfig.COL_CIRCUMSTANCES
    COL_CONCERNS = GeneralConfig.COL_CONCERNS
    COL_DATE_SCRAPED = GeneralConfig.COL_DATE_SCRAPED

    # Keys used for LLM interaction when requesting missing fields
    LLM_KEY_DATE = ScraperConfig.LLM_KEY_DATE
    LLM_KEY_CORONER = ScraperConfig.LLM_KEY_CORONER
    LLM_KEY_AREA = ScraperConfig.LLM_KEY_AREA
    LLM_KEY_RECEIVER = ScraperConfig.LLM_KEY_RECEIVER
    LLM_KEY_INVESTIGATION = ScraperConfig.LLM_KEY_INVESTIGATION
    LLM_KEY_CIRCUMSTANCES = ScraperConfig.LLM_KEY_CIRCUMSTANCES
    LLM_KEY_CONCERNS = ScraperConfig.LLM_KEY_CONCERNS

    # URL templates for different PFD categories on the judiciary.uk website
    CATEGORY_TEMPLATES = ScraperConfig.CATEGORY_TEMPLATES

    def __init__(
        self,
        llm: LLM | None = None,
        # Web page and search criteria
        category: str = "all",
        start_date: str = "2000-01-01",
        end_date: str = "2050-01-01",
        # Threading and HTTP request configuration
        max_workers: int = 10,
        max_requests: int = 5,
        delay_range: tuple[int | float, int | float] | None = (1, 2),
        timeout: int = 60,
        # Scraping strategy configuration
        scraping_strategy: list[int] | tuple[int, int, int] = [1, 2, 3],
        # Output DataFrame column inclusion flags
        include_url: bool = True,
        include_id: bool = True,
        include_date: bool = True,
        include_coroner: bool = True,
        include_area: bool = True,
        include_receiver: bool = True,
        include_investigation: bool = True,
        include_circumstances: bool = True,
        include_concerns: bool = True,
        include_time_stamp: bool = False,
        verbose: bool = False,
    ) -> None:

        # Network configuration 
        self.cfg = ScraperConfig(
            max_workers=max_workers,
            max_requests=max_requests,
            delay_range=delay_range,
            timeout=timeout,
        )

        self.category = category

        # Parse date strings into datetime objects
        self.start_date = date_parser.parse(start_date)
        self.end_date = date_parser.parse(end_date)

        # Store date components for formatting into search URLs
        self.date_params = {
            "after_day": self.start_date.day,
            "after_month": self.start_date.month,
            "after_year": self.start_date.year,
            "before_day": self.end_date.day,
            "before_month": self.end_date.month,
            "before_year": self.end_date.year,
        }

        # Hardcode in always starting from page 1
        self.start_page = 1

        # Store threading and request parameters
        self.max_workers = self.cfg.max_workers
        self.max_requests = self.cfg.max_requests
        self.delay_range = self.cfg.delay_range
        self.timeout = self.cfg.timeout

        # Prepare scraping order container before parsing strategy
        self._scraping_order: list[str] | None = None

        # Parse scraping strategy
        self.scraping_strategy = scraping_strategy
        self.llm = llm
        self._parse_scraping_strategy(scraping_strategy)

        # Store output column inclusion flags
        self.include_url = include_url
        self.include_id = include_id
        self.include_date = include_date
        self.include_coroner = include_coroner
        self.include_area = include_area
        self.include_receiver = include_receiver
        self.include_investigation = include_investigation
        self.include_circumstances = include_circumstances
        self.include_concerns = include_concerns
        self.include_time_stamp = include_time_stamp

        self.verbose = verbose

        # Initialise storage for results and links
        self.reports: pd.DataFrame | None = None
        self.report_links: list[str] = []

        # Store LLM model name if LLM client is provided
        self.llm_model = self.llm.model if self.llm else "None"

        # Configure url template
        self.page_template = self.cfg.url_template(self.category)

        # Normalise delay_range if set to 0 or None
        if self.delay_range is None or self.delay_range == 0:
            self.delay_range = (0, 0)

        # Validate param
        self._validate_init_params()
        self._warn_if_suboptimal_config()

        # Pre-compile regex for extracting report IDs
        self._id_pattern = GeneralConfig.ID_PATTERN

        # Configuration for dynamically building the list of required columns in top_up()
        # ...flags controlling which columns appear in the output:
        self._COLUMN_CONFIG: List[Tuple[bool, str]] = [
            (self.include_url, self.COL_URL),
            (self.include_id, self.COL_ID),
            (self.include_date, self.COL_DATE),
            (self.include_coroner, self.COL_CORONER_NAME),
            (self.include_area, self.COL_AREA),
            (self.include_receiver, self.COL_RECEIVER),
            (self.include_investigation, self.COL_INVESTIGATION),
            (self.include_circumstances, self.COL_CIRCUMSTANCES),
            (self.include_concerns, self.COL_CONCERNS),
            (self.include_time_stamp, self.COL_DATE_SCRAPED),
        ]

        # Prompts for missing fields passed to the LLM
        self._LLM_FIELD_CONFIG: List[Tuple[bool, str, str, str]] = [
            (
                self.include_date,
                self.COL_DATE,
                self.LLM_KEY_DATE,
                f"[Date of the report, not the death. Return {GeneralConfig.NOT_FOUND_TEXT} if not found]",
            ),
            (
                self.include_coroner,
                self.COL_CORONER_NAME,
                self.LLM_KEY_CORONER,
                f"[Name of the coroner. Provide the name only. Return {GeneralConfig.NOT_FOUND_TEXT} if not found]",
            ),
            (
                self.include_area,
                self.COL_AREA,
                self.LLM_KEY_AREA,
                f"[Area/location of the Coroner. Provide the location itself only. Return {GeneralConfig.NOT_FOUND_TEXT} if not found]",
            ),
            (
                self.include_receiver,
                self.COL_RECEIVER,
                self.LLM_KEY_RECEIVER,
                f"[Name or names of the recipient(s) as provided in the report. Return {GeneralConfig.NOT_FOUND_TEXT} if not found]",
            ),
            (
                self.include_investigation,
                self.COL_INVESTIGATION,
                self.LLM_KEY_INVESTIGATION,
                f"[The text from the Investigation/Inquest section. Return {GeneralConfig.NOT_FOUND_TEXT} if not found]",
            ),
            (
                self.include_circumstances,
                self.COL_CIRCUMSTANCES,
                self.LLM_KEY_CIRCUMSTANCES,
                f"[The text from the Circumstances of Death section. Return {GeneralConfig.NOT_FOUND_TEXT} if not found]",
            ),
            (
                self.include_concerns,
                self.COL_CONCERNS,
                self.LLM_KEY_CONCERNS,
                f"[The text from the Coroner's Concerns section. Return {GeneralConfig.NOT_FOUND_TEXT} if not found]",
            ),
        ]

        # Mapping used when writing LLM results to the DataFrame
        self._LLM_TO_DF_MAPPING: Dict[str, str] = {
            self.LLM_KEY_DATE: self.COL_DATE,
            self.LLM_KEY_CORONER: self.COL_CORONER_NAME,
            self.LLM_KEY_AREA: self.COL_AREA,
            self.LLM_KEY_RECEIVER: self.COL_RECEIVER,
            self.LLM_KEY_INVESTIGATION: self.COL_INVESTIGATION,
            self.LLM_KEY_CIRCUMSTANCES: self.COL_CIRCUMSTANCES,
            self.LLM_KEY_CONCERNS: self.COL_CONCERNS,
        }

        # Helper extractors
        self._html_extractor = HtmlExtractor(
            self.cfg,
            timeout=self.timeout,
            id_pattern=self._id_pattern,
            not_found_text=self.NOT_FOUND_TEXT,
            verbose=self.verbose,
        )
        self._pdf_extractor = PdfExtractor(
            self.cfg,
            timeout=self.timeout,
            not_found_text=self.NOT_FOUND_TEXT,
            verbose=self.verbose,
        )

        self._include_flags: Dict[str, bool] = {
            "id": self.include_id,
            "date": self.include_date,
            "coroner": self.include_coroner,
            "area": self.include_area,
            "receiver": self.include_receiver,
            "investigation": self.include_investigation,
            "circumstances": self.include_circumstances,
            "concerns": self.include_concerns,
        }



    def _parse_scraping_strategy(self, strategy: list[int] | tuple[int, int, int]) -> None:
        """Parse ``scraping_strategy`` into flags and ordered stages."""
        if (
            not isinstance(strategy, (list, tuple))
            or len(strategy) != 3
            or not all(isinstance(i, int) for i in strategy)
        ):
            raise ValueError(
                "scraping_strategy must be a list or tuple of three integers"
            )

        stage_map = {0: "html", 1: "pdf", 2: "llm"}

        self.html_scraping = strategy[0] != -1
        self.pdf_fallback = strategy[1] != -1
        self.llm_fallback = strategy[2] != -1

        provided = [(val, stage_map[idx]) for idx, val in enumerate(strategy) if val != -1]
        # Sort by provided order
        ordered = sorted(provided, key=lambda x: x[0])
        self._scraping_order = [name for _, name in ordered]

        numbers = [num for num, _ in provided]
        valid_seq = sorted(numbers) == list(range(1, len(numbers) + 1)) and len(numbers) == len(set(numbers))
        if not valid_seq:
            enabled = [s.upper() for s in self._scraping_order]
            disabled = [stage_map[i].upper() for i in range(3) if strategy[i] == -1]
            logger.warning(
                "Unexpected scraping_strategy %s interpreted as: enabled=%s; disabled=%s; order=%s",
                strategy,
                ", ".join(enabled) if enabled else "None",
                ", ".join(disabled) if disabled else "None",
                " > ".join(enabled) if enabled else "None",
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Public API
    # ──────────────────────────────────────────────────────────────────────────

    def get_report_links(self) -> list[str] | None:
        """Discover individual report URLs for the current query, across all pages.

        Iterates through _get_report_href_values (which collects URLs for a single page).

        Pagination continues until a page yields zero new links.

        Returns
        -------
        list[str] | None
            All discovered URLs, or *None* if **no** links were found for
            the given category/date window.
        """
        self.report_links = []
        pbar = tqdm(desc="Fetching pages", unit="", leave=False)
        for page in count(self.start_page):
            # Build the search page URL
            page_url = self.page_template.format(page=page, **self.date_params)
            hrefs = self._get_report_href_values(page_url)
            if not hrefs:
                break
            self.report_links.extend(hrefs)
            pbar.update(1)
        pbar.close()

        logger.info("Total collected report links: %d", len(self.report_links))
        return self.report_links

    def scrape_reports(self) -> pd.DataFrame:
        """Execute a full scrape with the Class configuration.

        Workflow
        --------
        1. Call ``get_report_links``.
        2. Extract each report according to ``scraping_strategy``.
        3. Cache the final DataFrame to ``self.reports``.

        Returns
        -------
        pandas.DataFrame
            One row per report.  Column presence matches the ``include_*`` flags.
            The DataFrame is empty if nothing was scraped.

        Examples
        --------
        Scrape reports and inspect columns::

            df = scraper.scrape_reports()
            df.columns
        """
        # Check to see if get_report_links() has already been run; if not, run it.
        if not self.report_links:
            fetched_links = self.get_report_links()
            if fetched_links is None:
                self.reports = pd.DataFrame()
                return self.reports

        report_data = self._scrape_report_details(self.report_links)
        reports_df = pd.DataFrame(report_data)

        # Output the timestamp of scraping completion for each report, if enabled
        if self.include_date:
            reports_df = reports_df.sort_values(by=[self.COL_DATE], ascending=False)
        self.reports = reports_df.copy()

        return reports_df

    def run_llm_fallback(self, reports_df: pd.DataFrame | None = None) -> pd.DataFrame:
        """Ask the LLM to fill cells still set to ``self.NOT_FOUND_TEXT``.

        Only the missing fields requested via ``include_*`` flags are sent to
        the model, along with the report’s PDF bytes (when available).

        Parameters
        ----------
        reports_df : pandas.DataFrame | None
            DataFrame to process. Defaults to ``self.reports``.

        Returns
        -------
        pandas.DataFrame
            Same shape as ``reports_df``, updated in place and re-cached to
            ``self.reports``.

        Raises
        ------
        ValueError
            If no LLM client was supplied at construction time.

        Examples
        --------
        Run the fallback step after scraping::

            updated_df = scraper.run_llm_fallback()
        """
        # Make sure llm param is set
        if not self.llm:
            raise ValueError(
                "LLM client (self.llm) not provided. Cannot run LLM fallback."
            )

        if reports_df is None:
            if self.reports is None:
                raise ValueError(
                    "No scraped reports found (reports_df is None and self.reports is None). Please run scrape_reports() first or provide a suitable DataFrame."
                )
            current_reports_df = self.reports.copy()
        else:
            current_reports_df = reports_df.copy()
            
        # Delegate missing values to the language model
        updated_df = _run_llm_fallback(
            current_reports_df,
            llm=self.llm,
            pdf_extractor=self._pdf_extractor,
            llm_field_config=self._LLM_FIELD_CONFIG,
            llm_to_df_mapping=self._LLM_TO_DF_MAPPING,
            col_url=self.COL_URL,
            not_found_text=self.NOT_FOUND_TEXT,
            llm_key_date=self.LLM_KEY_DATE,
            verbose=self.verbose,
        )
        
        self.reports = updated_df.copy()
        return updated_df

    def top_up(
        self,
        old_reports: pd.DataFrame | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
        clean: bool = False,
    ) -> pd.DataFrame | None:
        """Check for and append new PFD reports within the current parameters.

        If new links are found they are scraped and appended to
        ``self.reports``. Any URL (or ID) already present in
        *old_reports* is skipped.

        Optionally, you can override the *start_date* and *end_date*
        parameters from ``self`` for this call only.

        Parameters
        ----------
        old_reports : pandas.DataFrame | None
            Existing DataFrame. Defaults to ``self.reports``.
        start_date, end_date : str | None
            Optionally override the scraper’s date window *for this call only*.
        clean : bool, optional
            When ``True``, run the ``Cleaner`` on the newly
            scraped rows before merging them with existing reports.

        Returns
        -------
        pandas.DataFrame | None
            Updated DataFrame if new reports were added; *None* if no new
            records were found **and** *old_reports* was *None*.

        Raises
        ------
        ValueError
            If *old_reports* lacks columns required for duplicate checks.

        Examples
        --------
        Add new reports to an existing DataFrame::

            updated = scraper.top_up(df, end_date="2023-01-01")
            len(updated) - len(df)  # number of new reports
        """
        logger.info("Attempting to 'top up' the existing reports with new data.")

        # Update date range for this top_up if new dates provided
        if start_date is not None or end_date is not None:
            new_start_date = (
                date_parser.parse(start_date)
                if start_date is not None
                else self.start_date
            )
            new_end_date = (
                date_parser.parse(end_date) if end_date is not None else self.end_date
            )
            if new_start_date > new_end_date:
                raise ValueError("start_date must be before end_date.")
            self.start_date = new_start_date
            self.end_date = new_end_date
            self.date_params.update(
                {
                    "after_day": self.start_date.day,
                    "after_month": self.start_date.month,
                    "after_year": self.start_date.year,
                    "before_day": self.end_date.day,
                    "before_month": self.end_date.month,
                    "before_year": self.end_date.year,
                }
            )

        # If provided, update provided DataFrame. Else, update the internal attribute
        base_df = old_reports if old_reports is not None else self.reports
        # Ensure base_df has required columns for duplicate checking
        if base_df is not None:
            required_columns = [
                col_name
                for include_flag, col_name in self._COLUMN_CONFIG
                if include_flag
            ]
            missing_cols = [
                col for col in required_columns if col not in base_df.columns
            ]
            if missing_cols:
                raise ValueError(
                    f"Required columns missing from the provided DataFrame: {missing_cols}"
                )

        # Determine unique key for identifying existing/duplicate reports: URL or ID
        if self.include_url:
            unique_key = self.COL_URL
        elif self.include_id:
            unique_key = self.COL_ID
        else:
            logger.error(
                "No unique identifier available for duplicate checking.\nEnsure include_url or include_id was set to True in instance initialisation."
            )
            return None
        existing_identifiers = (
            set(base_df[unique_key].tolist())
            if base_df is not None and unique_key in base_df.columns
            else set()
        )

        # Fetch updated list of report links within current date range
        updated_links = self.get_report_links()
        if updated_links is None:
            updated_links = []
        new_links = [link for link in updated_links if link not in existing_identifiers]
        logger.info(
            "Top-up: %d new report(s) found; %d duplicate(s) which won't be added",
            len(new_links),
            len(updated_links) - len(new_links),
        )
        if not new_links:
            return None if base_df is None and old_reports is None else base_df

        # Scrape details for new links using existing helpers
        new_records = self._scrape_report_details(new_links)
        if new_records:
            new_df = pd.DataFrame(new_records)
            # Apply LLM fallback if configured
            if self.llm_fallback and self.llm:
                new_df = self.run_llm_fallback(new_df)
            if clean:
                if not self.llm:
                    raise ValueError(
                        "LLM client (self.llm) required when clean=True."
                    )
                cleaner = Cleaner(
                    reports=new_df,
                    llm=self.llm,
                    include_coroner=self.include_coroner,
                    include_receiver=self.include_receiver,
                    include_area=self.include_area,
                    include_investigation=self.include_investigation,
                    include_circumstances=self.include_circumstances,
                    include_concerns=self.include_concerns,
                    verbose=self.verbose,
                )
                new_df = cleaner.clean_reports()
            updated_reports_df = (
                pd.concat([base_df, new_df], ignore_index=True)
                if base_df is not None
                else new_df
            )
        else:
            updated_reports_df = base_df if base_df is not None else pd.DataFrame()
        if self.include_date:
            updated_reports_df = updated_reports_df.sort_values(
                by=[self.COL_DATE], ascending=False
            )
        self.reports = updated_reports_df.copy()
        return updated_reports_df

    # ──────────────────────────────────────────────────────────────────────────
    # Initialisation validation & warnings
    # ─────────────────────────────────────────────────────────────────────────

    def _validate_init_params(self) -> None:
        """Validate initialisation parameters and raise errors for invalid configs."""
        if self.start_date > self.end_date:
            raise ValueError("start_date must be before end_date.")
        if self.llm_fallback and not self.llm:
            raise ValueError(
                "LLM Client must be provided if LLM fallback is enabled. \nPlease create an instance of the LLM class and pass this in the llm parameter. \nGet an API key from https://platform.openai.com/."
            )
        if self.max_workers <= 0:
            raise ValueError("max_workers must be a positive integer.")
        if self.max_requests <= 0:
            raise ValueError("max_requests must be a positive integer.")
        if (
            not isinstance(self.delay_range, tuple)
            or len(self.delay_range) != 2
            or not all(isinstance(i, (int, float)) for i in self.delay_range)
        ):
            raise ValueError(
                "delay_range must be a tuple of two numbers (int or float) - e.g. (1, 2) or (1.5, 2.5). If you are attempting to disable delays, set to (0,0)."
            )
        if self.delay_range[1] < self.delay_range[0]:
            raise ValueError(
                "Upper bound of delay_range must be greater than or equal to lower bound."
            )
        if not (self.html_scraping or self.pdf_fallback or self.llm_fallback):
            raise ValueError(
                "scraping_strategy disables all stages. Enable at least one of HTML, PDF or LLM."
            )
        if not any(
            [
                self.include_id,
                self.include_date,
                self.include_coroner,
                self.include_area,
                self.include_receiver,
                self.include_investigation,
                self.include_circumstances,
                self.include_concerns,
            ]
        ):
            raise ValueError(
                "At least one field must be included in the output. Please set one or more of the following to True:\n 'include_id', 'include_date', 'include_coroner', 'include_area', 'include_receiver', 'include_investigation', 'include_circumstances', 'include_concerns'.\n"
            )

    def _warn_if_suboptimal_config(self) -> None:
        """Log warnings for configurations that might lead to suboptimal scraping."""
        if self.html_scraping and not self.pdf_fallback and not self.llm_fallback:
            logger.warning(
                "Only HTML scraping is enabled. \nConsider enabling .pdf or LLM fallback for more complete data extraction.\n"
            )
        if not self.html_scraping and self.pdf_fallback and not self.llm_fallback:
            logger.warning(
                "Only .pdf fallback is enabled. \nConsider enabling HTML scraping or LLM fallback for more complete data extraction.\n"
            )
        if not self.html_scraping and not self.pdf_fallback and self.llm_fallback:
            logger.warning(
                "Only LLM fallback is enabled. \nWhile this is a high-performance option, large API costs may be incurred, especially for large requests. \nConsider enabling HTML scraping or .pdf fallback for more cost-effective data extraction.\n"
            )
        if self.max_workers > 50:
            logger.warning(
                "max_workers is set to a high value (>50). \nDepending on your system, this may cause performance issues. It could also trigger anti-scraping measures by the host, leading to temporary or permanent IP bans. \nWe recommend setting to between 10 and 50.\n"
            )
        if self.max_workers < 10:
            logger.warning(
                "max_workers is set to a low value (<10). \nThis may result in slower scraping speeds. Consider increasing the value for faster performance. \nWe recommend setting to between 10 and 50.\n"
            )
        if self.max_requests > 10:
            logger.warning(
                "max_requests is set to a high value (>10). \nThis may trigger anti-scraping measures by the host, leading to temporary or permanent IP bans. \nWe recommend setting to between 3 and 10.\n"
            )
        if self.max_requests < 3:
            logger.warning(
                "max_requests is set to a low value (<3). \nThis may result in slower scraping speeds. Consider increasing the value for faster performance. \nWe recommend setting to between 3 and 10.\n"
            )
        if self.delay_range == (0, 0):
            logger.warning(
                "delay_range has been disabled. \nThis will disable delays between requests. This may trigger anti-scraping measures by the host, leading to temporary or permanent IP bans. \nWe recommend setting to (1,2).\n"
            )
        elif self.delay_range[0] < 0.5 and self.delay_range[1] != 0:
            logger.warning(
                "delay_range is set to a low value (<0.5 seconds). \nThis may trigger anti-scraping measures by the host, leading to temporary or permanent IP bans. We recommend setting to between (1, 2).\n"
            )
        if self.delay_range[1] > 5:
            logger.warning(
                "delay_range is set to a high value (>5 seconds). \nThis may result in slower scraping speeds. Consider decreasing the value for faster performance. We recommend setting to between (1, 2).\n"
            )

    # ──────────────────────────────────────────────────────────────────────────
    # Link-fetching
    # ──────────────────────────────────────────────────────────────────────────
    def _get_report_href_values(self, url: str) -> list[str]:
        """
        Parses through a **single page** of PFD search results and extracts individual report URLs via href values.

        Applies a random delay and uses a semaphore to limit concurrent requests.

        :param url: The URL of the search results page to scrape.
        :return: A list of href strings, each being a URL to a PFD report page.
                 Returns an empty list if the page fetch fails or no links are found.
        """
        with self.cfg.domain_semaphore:
            self.cfg.apply_random_delay()
            try:
                response = self.cfg.session.get(url, timeout=self.timeout)
                response.raise_for_status()
                if self.verbose:
                    logger.debug(f"Fetched URL: {url} (Status: {response.status_code})")

            except requests.RequestException as e:
                if self.verbose:
                    logger.error("Failed to fetch page: %s; Error: %s", url, e)
                return []

        soup = BeautifulSoup(response.text, "html.parser")
        links = soup.find_all("a", class_="card__link")
        # Return the URLs found on this page
        return [link.get("href") for link in links if link.get("href")]


    # ──────────────────────────────────────────────────────────────────────────
    # Internal helpers for staged scraping
    # ──────────────────────────────────────────────────────────────────────────

    def _fetch_initial_record(self, url: str) -> dict[str, Any] | None:
        """Fetch the report page and initialise scraping fields."""
        fields: dict[str, str] = {
            "id": self.NOT_FOUND_TEXT,
            "date": self.NOT_FOUND_TEXT,
            "receiver": self.NOT_FOUND_TEXT,
            "coroner": self.NOT_FOUND_TEXT,
            "area": self.NOT_FOUND_TEXT,
            "investigation": self.NOT_FOUND_TEXT,
            "circumstances": self.NOT_FOUND_TEXT,
            "concerns": self.NOT_FOUND_TEXT,
        }

        soup = self._html_extractor.fetch_report_page(url)
        if soup is None:
            return None

        pdf_link = self._pdf_extractor.get_pdf_link(soup)
        if not pdf_link:
            logger.error("No .pdf links found on %s", url)
            return None

        return {"url": url, "pdf_link": pdf_link, "fields": fields, "soup": soup}

    def _apply_html_stage(self, record: dict[str, Any]) -> dict[str, Any]:
        """Extract fields from HTML if still missing."""
        soup = record.get("soup")
        if soup is None or not self.html_scraping:
            return record

        temp_fields = {key: self.NOT_FOUND_TEXT for key in record["fields"].keys()}
        self._html_extractor.extract_fields_from_html(soup, temp_fields, self._include_flags)
        for key, value in temp_fields.items():
            if (
                pd.isna(record["fields"][key])
                or record["fields"][key] is self.NOT_FOUND_TEXT
                or record["fields"][key] == ""
            ):
                record["fields"][key] = value
        return record

    def _apply_pdf_fallback_stage(self, record: dict[str, Any]) -> dict[str, Any]:
        """Apply PDF fallback extraction for a single record."""
        pdf_link = record.get("pdf_link")
        if not pdf_link:
            return record

        pdf_text = self._pdf_extractor.extract_text_from_pdf(pdf_link)
        fields = record["fields"]
        if pd.notna(pdf_text) and pdf_text != "N/A: Source file not PDF":
            if any(
                pd.isna(fields[key])
                for key in [
                    "coroner",
                    "area",
                    "receiver",
                    "investigation",
                    "circumstances",
                    "concerns",
                ]
            ):
                if self.verbose:
                    logger.debug(
                        f"Initiating .pdf fallback for URL: {record['url']} because one or more fields are missing."
                    )
                self._pdf_extractor.apply_pdf_fallback(pdf_text, fields, self._include_flags)
        return record

    def _apply_llm_stage(self, records: list[dict[str, Any]]) -> list[dict[str, Any]]:
        """Run the LLM fallback on the current records."""
        if not self.llm_fallback or not self.llm or not records:
            return records

        df = pd.DataFrame([self._assemble_report(r["url"], r["fields"]) for r in records])
        df_updated = self.run_llm_fallback(df)
        mapping = {
            "id": self.COL_ID,
            "date": self.COL_DATE,
            "coroner": self.COL_CORONER_NAME,
            "area": self.COL_AREA,
            "receiver": self.COL_RECEIVER,
            "investigation": self.COL_INVESTIGATION,
            "circumstances": self.COL_CIRCUMSTANCES,
            "concerns": self.COL_CONCERNS,
        }
        for rec, (_, row) in zip(records, df_updated.iterrows()):
            for key, col in mapping.items():
                if col in df_updated.columns:
                    rec["fields"][key] = row[col]
        return records


    # ──────────────────────────────────────────────────────────────────────────
    # Utilities: text-cleaning & assembly
    # ──────────────────────────────────────────────────────────────────────────

    def _scrape_report_details(self, urls: list[str]) -> list[dict[str, Any]]:
        """Scrape reports according to ``scraping_strategy``."""

        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            initial = list(
                tqdm(
                    executor.map(self._fetch_initial_record, urls),
                    total=len(urls),
                    desc="Fetching pages",
                    position=0,
                    leave=False,
                )
            )

        records = [r for r in initial if r is not None]

        for stage in self._scraping_order:
            if stage == "html":
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    records = list(
                        tqdm(
                            executor.map(self._apply_html_stage, records),
                            total=len(records),
                            desc="HTML scraping",
                            position=0,
                            leave=False,
                        )
                    )
            elif stage == "pdf":
                with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
                    records = list(
                        tqdm(
                            executor.map(self._apply_pdf_fallback_stage, records),
                            total=len(records),
                            desc=".pdf scraping",
                            position=0,
                            leave=False,
                        )
                    )
            elif stage == "llm":
                records = self._apply_llm_stage(records)

        return [self._assemble_report(r["url"], r["fields"]) for r in records]

    def _assemble_report(self, url: str, fields: dict[str, str]) -> dict[str, Any]:
        """Assemble a single report's data into a dictionary based on included fields."""
        # Prepare the final record using inclusion flags
        report: dict[str, Any] = {}
        if self.include_url:
            report[self.COL_URL] = url
        if self.include_id:
            report[self.COL_ID] = fields.get("id", self.NOT_FOUND_TEXT)
        if self.include_date:
            report[self.COL_DATE] = fields.get("date", self.NOT_FOUND_TEXT)
        if self.include_coroner:
            report[self.COL_CORONER_NAME] = fields.get("coroner", self.NOT_FOUND_TEXT)
        if self.include_area:
            report[self.COL_AREA] = fields.get("area", self.NOT_FOUND_TEXT)
        if self.include_receiver:
            report[self.COL_RECEIVER] = fields.get("receiver", self.NOT_FOUND_TEXT)
        if self.include_investigation:
            report[self.COL_INVESTIGATION] = fields.get(
                "investigation", self.NOT_FOUND_TEXT
            )
        if self.include_circumstances:
            report[self.COL_CIRCUMSTANCES] = fields.get(
                "circumstances", self.NOT_FOUND_TEXT
            )
        if self.include_concerns:
            report[self.COL_CONCERNS] = fields.get("concerns", self.NOT_FOUND_TEXT)
        if self.include_time_stamp:
            report[self.COL_DATE_SCRAPED] = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        return report

    def _extract_report_info(self, url: str) -> dict[str, Any] | None:
        """Extract full report information for a single URL."""

        record = self._fetch_initial_record(url)
        if record is None:
            return None

        records = [record]
        for stage in self._scraping_order:
            if stage == "html":
                records = [self._apply_html_stage(records[0])]
            elif stage == "pdf":
                records = [self._apply_pdf_fallback_stage(records[0])]
            elif stage == "llm":
                records = self._apply_llm_stage(records)

        record = records[0]
        return self._assemble_report(record["url"], record["fields"])
