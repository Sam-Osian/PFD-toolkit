from __future__ import annotations

from typing import Dict, List, Tuple, Any
import logging
from bs4 import BeautifulSoup
import pandas as pd
from dateutil import parser as date_parser
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime
from tqdm.auto import tqdm
import requests
from itertools import count

from .html_extractor import HtmlExtractor
from .pdf_extractor import PdfExtractor
from .text_utils import normalise_date

from .config import GeneralConfig, ScraperConfig

# -----------------------------------------------------------------------------
# Logging Configuration:
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)


class PFDScraper:
    """Scrape UK “Prevention of Future Death” (PFD) reports into a
    :class:`pandas.DataFrame`.

    The extractor runs in three cascading layers
    (`html → pdf → llm`), each independently switchable.

    1. **HTML scrape** – parse metadata and rich sections directly from
       the web page.
    2. **PDF fallback** – download the attached PDF and extract text with
       *PyMuPDF* for any missing fields.
    3. **LLM fallback** – delegate unresolved gaps to a Large Language
       Model supplied via *llm*.

    All three layers are optional and independently switchable.

    Parameters
    ----------
    llm : LLM | None
        Client implementing ``_call_llm_fallback()``; required only when
        *llm_fallback* is *True*.
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
    html_scraping, pdf_fallback, llm_fallback : bool
        Toggles for the three extraction layers.
    include_* : bool
        Flags that control which columns appear in the output DataFrame.
    verbose : bool
        Emit debug-level logs when *True*.

    Attributes
    ----------
    reports : pandas.DataFrame | None
        Cached result of the last call to :py:meth:`scrape_reports`
        or :py:meth:`top_up`.
    report_links : list[str]
        URLs discovered by :py:meth:`get_report_links`.
    NOT_FOUND_TEXT : str
        Placeholder value set when a field cannot be extracted.

    Examples
    --------
    >>> from pfd_toolkit import PFDScraper
    >>> scraper = PFDScraper(category="suicide",
    ...                      start_date="2020-01-01",
    ...                      end_date="2022-12-31",
    ...                      llm_fallback=True,
    ...                      llm=my_llm_client)         # Configured in LLM class
    >>> df = scraper.scrape_reports()          # full scrape
    >>> newer_df = scraper.top_up(df)             # later "top-up"
    >>> added_llm_df = scraper.run_llm_fallback(df)   # apply LLM retro-actively
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
        llm: "LLM" = None,
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
        html_scraping: bool = True,
        pdf_fallback: bool = True,
        llm_fallback: bool = False,
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

        # Store scraping strategy flags
        self.html_scraping = html_scraping
        self.pdf_fallback = pdf_fallback
        self.llm_fallback = llm_fallback
        self.llm = llm

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

        # Configuration for identifying missing fields for LLM fallback
        self._LLM_FIELD_CONFIG: List[Tuple[bool, str, str, str]] = [
            (
                self.include_date,
                self.COL_DATE,
                self.LLM_KEY_DATE,
                "[Date of the report, not the death]",
            ),
            (
                self.include_coroner,
                self.COL_CORONER_NAME,
                self.LLM_KEY_CORONER,
                "[Name of the coroner. Provide the name only.]",
            ),
            (
                self.include_area,
                self.COL_AREA,
                self.LLM_KEY_AREA,
                "[Area/location of the Coroner. Provide the location itself only.]",
            ),
            (
                self.include_receiver,
                self.COL_RECEIVER,
                self.LLM_KEY_RECEIVER,
                "[Name or names of the recipient(s) as provided in the report.]",
            ),
            (
                self.include_investigation,
                self.COL_INVESTIGATION,
                self.LLM_KEY_INVESTIGATION,
                "[The text from the Investigation/Inquest section.]",
            ),
            (
                self.include_circumstances,
                self.COL_CIRCUMSTANCES,
                self.LLM_KEY_CIRCUMSTANCES,
                "[The text from the Circumstances of Death section.]",
            ),
            (
                self.include_concerns,
                self.COL_CONCERNS,
                self.LLM_KEY_CONCERNS,
                "[The text from the Coroner's Concerns section.]",
            ),
        ]

        # Mapping from LLM response keys back to DataFrame column names
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
        pbar = tqdm(desc="Fetching pages", unit="page", leave=False)
        for page in count(self.start_page):
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
        1. Call :py:meth:`get_report_links`.
        2. Extract each report in parallel via
           :py:meth:`_extract_report_info`.
        3. Optionally invoke :py:meth:`run_llm_fallback`.
        4. Cache the final DataFrame to :pyattr:`self.reports`.

        Returns
        -------
        pandas.DataFrame
            One row per report.  Column presence matches the *include_* flags.
            The DataFrame is empty if nothing was scraped.

        Examples
        --------
        >>> df = scraper.scrape_reports()
        >>> df.columns
        Index(['URL', 'ID', 'Date', ...], dtype='object')
        """
        # Check to see if get_report_links() has already been run; if not, run it.
        if not self.report_links:
            fetched_links = self.get_report_links()
            if fetched_links is None:
                self.reports = pd.DataFrame()
                return self.reports

        report_data = self._scrape_report_details(self.report_links)
        reports_df = pd.DataFrame(report_data)

        # Run the LLM fallback if enabled
        if self.llm_fallback and self.llm:
            reports_df = self.run_llm_fallback(
                reports_df if not reports_df.empty else None
            )

        # Output the timestamp of scraping completion for each report, if enabled
        if self.include_date:
            reports_df = reports_df.sort_values(by=[self.COL_DATE], ascending=False)
        self.reports = reports_df.copy()

        return reports_df

    def run_llm_fallback(self, reports_df: pd.DataFrame | None = None) -> pd.DataFrame:
        """Ask the LLM to fill cells still set to :pyattr:`self.NOT_FOUND_TEXT`.

        Only the missing fields requested via *include_* flags are sent to
        the model, along with the report’s PDF bytes (when available).

        Parameters
        ----------
        reports_df : pandas.DataFrame | None
            DataFrame to process.  Defaults to :pyattr:`self.reports`.

        Returns
        -------
        pandas.DataFrame
            Same shape as *reports_df*, updated in place and re-cached to
            :pyattr:`self.reports`.

        Raises
        ------
        ValueError
            If no LLM client was supplied at construction time.

        Examples
        --------
        >>> updated_df = scraper.run_llm_fallback()
        """
        # Make sure llm param is set
        if not self.llm:
            raise ValueError(
                "LLM client (self.llm) not provided. Cannot run LLM fallback."
            )

        current_reports_df: pd.DataFrame
        if reports_df is None:
            if self.reports is None:
                raise ValueError(
                    "No scraped reports found (reports_df is None and self.reports is None). Please run scrape_reports() first or provide a suitable DataFrame."
                )
            current_reports_df = self.reports.copy()
        else:
            current_reports_df = reports_df.copy()
        if current_reports_df.empty:
            logger.info("Report DataFrame is empty. Skipping LLM fallback.")
            return current_reports_df

        # Helper function to process a single row for LLM fallback
        def _process_row(idx: int, row_data: pd.Series) -> tuple[int, dict[str, str]]:
            """Identifies missing fields for a given row and calls LLM for them."""
            missing_fields: dict[str, str] = {}

            # Build dictionary of fields needing LLM extraction based on _LLM_FIELD_CONFIG
            for (
                include_flag,
                df_col_name,
                llm_key,
                llm_prompt,
            ) in self._LLM_FIELD_CONFIG:
                if (
                    include_flag
                    and row_data.get(df_col_name, "") == self.NOT_FOUND_TEXT
                ):
                    missing_fields[llm_key] = llm_prompt
            if not missing_fields:
                return idx, {}
            pdf_bytes: bytes | None = None
            report_url = row_data.get(self.COL_URL)
            if report_url:
                pdf_bytes = self._pdf_extractor.fetch_pdf_bytes(report_url)
            if not pdf_bytes and self.verbose:
                logger.warning(
                    f"Could not obtain PDF bytes for URL {report_url} (row {idx}). LLM fallback for this row might be impaired."
                )

            # Call the LLM client's fallback method
            updates = self.llm._call_llm_fallback(
                pdf_bytes=pdf_bytes,
                missing_fields=missing_fields,
                report_url=str(report_url) if report_url else "N/A",
                verbose=self.verbose,
                tqdm_extra_kwargs={"disable": True},
            )
            return idx, updates if updates else {}

        # Process rows for LLM fallback
        results_map: Dict[int, Dict[str, str]] = {}
        use_parallel = (
            self.llm and hasattr(self.llm, "max_workers") and self.llm.max_workers > 1
        )
        if use_parallel:
            with ThreadPoolExecutor(max_workers=self.llm.max_workers) as executor:
                future_to_idx = {
                    executor.submit(_process_row, idx, row_series): idx
                    for idx, row_series in current_reports_df.iterrows()
                }
                for future in tqdm(
                    as_completed(future_to_idx),
                    total=len(future_to_idx),
                    desc="Running LLM fallback",
                    position=0,
                    leave=True,
                ):
                    idx = future_to_idx[future]
                    try:
                        _, updates = future.result()
                        results_map[idx] = updates
                    except Exception as e:
                        logger.error(f"LLM fallback failed for row index {idx}: {e}")
        else:
            for idx, row_series in tqdm(
                current_reports_df.iterrows(),
                total=len(current_reports_df),
                desc="LLM fallback (sequential processing)",
                position=0,
                leave=True,
            ):
                try:
                    _, updates = _process_row(idx, row_series)
                    results_map[idx] = updates
                except Exception as e:
                    logger.error(f"LLM fallback failed for row index {idx}: {e}")

        # Apply updates from LLM to the DataFrame
        for idx, updates_dict in results_map.items():
            if not updates_dict:
                continue
            for llm_key, value_from_llm in updates_dict.items():
                df_col_name = self._LLM_TO_DF_MAPPING.get(llm_key)
                if df_col_name:
                    if llm_key == self.LLM_KEY_DATE:
                        if value_from_llm != self.NOT_FOUND_TEXT:
                            current_reports_df.at[idx, df_col_name] = (
                                normalise_date(value_from_llm, verbose=self.verbose)
                            )
                    else:
                        current_reports_df.at[idx, df_col_name] = value_from_llm
        self.reports = current_reports_df.copy()
        return current_reports_df

    def top_up(
        self,
        old_reports: pd.DataFrame | None = None,
        start_date: str | None = None,
        end_date: str | None = None,
    ) -> pd.DataFrame | None:
        """Checks to see if there are any unscraped PFD reports within Class instance parameters.

        If so, it reruns the scraper and appends new reports to
        :pyattr:`self.reports` under Class instance parameters.

        Any URL (or ID) already present in *old_reports* is skipped.

        Optionally, you can override the *start_date* and *end_date*
        parameters from `self`.

        Parameters
        ----------
        old_reports : pandas.DataFrame | None
            Existing DataFrame.  Defaults to :pyattr:`self.reports`.
        start_date, end_date : str | None
            Optionally override the scraper’s date window *for this call only*.

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
        >>> updated = scraper.top_up(df, end_date="2023-01-01")
        >>> len(updated) - len(df)     # number of new reports
        3
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

        # Scrape details for new links
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            new_results = list(
                tqdm(
                    executor.map(self._extract_report_info, new_links),
                    total=len(new_links),
                    desc="Topping up reports",
                    position=0,
                    leave=True,
                )
            )
        new_records = [record for record in new_results if record is not None]
        if new_records:
            new_df = pd.DataFrame(new_records)
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
                "At least one of 'html_scraping', 'pdf_fallback', or 'llm_fallback' must be enabled."
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
        return [link.get("href") for link in links if link.get("href")]


    # ──────────────────────────────────────────────────────────────────────────
    # Utilities: text-cleaning & assembly
    # ──────────────────────────────────────────────────────────────────────────

    def _scrape_report_details(self, urls: list[str]) -> list[dict[str, Any]]:
        """Handles the mechanics of scraping PFD reports for all given URLs using multithreading,
        returning a list of result dicts."""
        with ThreadPoolExecutor(max_workers=self.max_workers) as executor:
            raw_results = list(
                tqdm(
                    executor.map(self._extract_report_info, urls),
                    total=len(urls),
                    desc="Scraping reports",
                    position=0,
                    leave=False,
                )
            )
        # filter out any None failures
        return [r for r in raw_results if r is not None]

    def _assemble_report(self, url: str, fields: dict[str, str]) -> dict[str, Any]:
        """Assemble a single report's data into a dictionary based on included fields."""
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
        """
        Extracts text from a single PFD report page (given by URL).

        The process involves:
          1. Fetching and parsing the HTML of the report page.
          2. Identifying and downloading the associated PDF report.
          3. Extracting all text from the PDF.
          4. If `html_scraping` is enabled, attempting to extract all configured fields from the HTML.
          5. If `pdf_fallback` is enabled and any fields are still missing, attempting to extract them
             from the PDF text using keyword-based section extraction.
          6. (LLM fallback is handled by `run_llm_fallback` method if enabled globally).

        :param url: The URL of the PFD report's HTML page.
        :return: A dictionary containing the extracted report information.
                 Returns None if the page fetch fails or essential components (like PDF link) are missing.
        """
        # Initialise all fields with default "not found" text
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
        # Fetch HTML page
        soup = self._html_extractor.fetch_report_page(url)
        if soup is None:
            return None
        # Find PDF download link
        pdf_link = self._pdf_extractor.get_pdf_link(soup)
        if not pdf_link:
            logger.error("No .pdf links found on %s", url)
            return None
        # Download and extract PDF text
        pdf_text = self._pdf_extractor.extract_text_from_pdf(pdf_link)
        # Extract fields from HTML if enabled
        if self.html_scraping:
            self._html_extractor.extract_fields_from_html(soup, fields, self._include_flags)
        # Use PDF fallback if enabled and PDF text is available
        if self.pdf_fallback and pdf_text not in (
            self.NOT_FOUND_TEXT,
            "N/A: Source file not PDF",
        ):
            if any(
                fields[key] == self.NOT_FOUND_TEXT
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
                        f"Initiating .pdf fallback for URL: {url} because one or more fields are missing."
                    )
                self._pdf_extractor.apply_pdf_fallback(pdf_text, fields, self._include_flags)
        # Assemble result dictionary
        report = self._assemble_report(url, fields)
        return report
