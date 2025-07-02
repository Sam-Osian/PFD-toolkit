from typing import Literal, Dict, List, Tuple, Any, Optional, Union
import logging
import warnings
from pydantic import BaseModel, Field, ConfigDict, create_model
import pandas as pd

from .config import GeneralConfig

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TopicMatch(BaseModel):
    """Pydantic model to structure the LLM's response for
    topic matching. Ensures the LLM returns either "Yes"
    or "No".
    """

    matches_topic: Literal["Yes", "No"] = Field(
        ...,
        description="Indicate whether the report text is relevant to the user's query. Must be Yes or No.",
    )

    model_config = ConfigDict(extra="forbid")


def _topic_model_with_spans() -> type[BaseModel]:
    """Return a TopicMatch model extended with a ``spans_matches_topic`` field."""
    return create_model(
        "TopicMatchWithSpans",
        spans_matches_topic=(
            str,
            Field(
                ...,
                description="Text spans supporting matches_topic",
                alias="spans_matches_topic",
            ),
        ),
        matches_topic=(
            Literal["Yes", "No"],
            Field(
                ...,
                description=(
                    "Indicate whether the report text is relevant to the user's query. Must be Yes or No."
                ),
                alias="matches_topic",
            ),
        ),
        __config__=ConfigDict(extra="forbid"),
    )


class Screener:
    """
    Classifies a list of report texts against a user-defined topic using an LLM.

    This class takes a DataFrame of reports, a search query, and various
    configuration options to classify whether each report matches the query.
    It can either filter the DataFrame to return only matching reports or
    add a classification column to the original DataFrame.

    Parameters
    ----------
    llm : LLM, optional
        An instance of the LLM class from `pfd_toolkit`.
    reports : pd.DataFrame, optional
        A DataFrame containing Prevention of Future Death reports.
    verbose : bool, optional
        If True, print more detailed logs. Defaults to False.
    include_date : bool, optional
        Flag to determine if the 'date' column is included. Defaults to False.
    include_coroner : bool, optional
        Flag to determine if the 'coroner' column is included. Defaults to False.
    include_area : bool, optional
        Flag to determine if the 'area' column is included. Defaults to False.
    include_receiver : bool, optional
        Flag to determine if the 'receiver' column is included. Defaults to False.
    include_investigation : bool, optional
        Flag to determine if the 'investigation' column is included. Defaults to True.
    include_circumstances : bool, optional
        Flag to determine if the 'circumstances' column is included. Defaults to True.
    include_concerns : bool, optional
        Flag to determine if the 'concerns' column is included. Defaults to True.

    Examples
    --------

        user_topic = "medication errors"
        llm_client = LLM()
        screener = Screener(llm=llm_client, reports=reports_df)
        screened_reports = screener.screen_reports(search_query=user_topic)
        print(f"Found {len(screened_reports)} report(s) on '{user_topic}'.")

    """

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

    def __init__(
        self,
        llm: Optional[Any] = None,
        reports: Optional[pd.DataFrame] = None,
        verbose: bool = False,
        include_date: bool = False,
        include_coroner: bool = False,
        include_area: bool = False,
        include_receiver: bool = False,
        include_investigation: bool = True,
        include_circumstances: bool = True,
        include_concerns: bool = True,
    ) -> None:
        self.llm = llm
        self.verbose = verbose
        self.produce_spans = False

        # Store column inclusion toggles
        self.include_date = include_date
        self.include_coroner = include_coroner
        self.include_area = include_area
        self.include_receiver = include_receiver
        self.include_investigation = include_investigation
        self.include_circumstances = include_circumstances
        self.include_concerns = include_concerns

        # Initialise reports; always copy!
        self.reports: pd.DataFrame = (
            reports.copy() if reports is not None else pd.DataFrame()
        )

        self.search_query: Optional[str] = None
        self.prompt_template: Optional[str] = None

        if self.verbose:
            logger.debug(
                "Screener initialised without a search query. Prompt will be built on first screen_reports call."
            )

        if self.verbose:
            if self.reports is not None:
                logger.debug(f"Initial reports DataFrame shape: {self.reports.shape}")

    def _build_prompt_template(self, current_search_query: str) -> str:
        """
        Constructs the prompt template based on the search query and match approach.
        """
        span_line = (
            "Text spans should be **extremely concise**, but always **verbatim** from the source. "
            "Wrap each text span in quotation marks. If multiple spans are found, separate them with semicolons (;).\n"
            if self.produce_spans
            else ""
        )

        base_prompt_template = (
    "You are an expert text classification assistant. Your task is to read "
    "the following excerpt from a Prevention of Future Death (PFD) report and "
    "decide whether it matches the user's query. \n\n"

    "**Instructions:** \n"
    "- Only respond 'Yes' if **all** elements of the user query are clearly present in the report. \n"
    "- If any required element is missing or there is not enough information, respond 'No'. \n"
    "- You may not infer or make judgements; the evidence must be clear."
    "- Make sure any user query related to the deceased is concerned with them *only*, not other persons.\n"
    "- Your response must be a JSON object in which 'matches_topic' can be either 'Yes' or 'No'. \n\n"
    f"{span_line}"
    f"**User query:** \n'{current_search_query}'"
)

        # Add the placeholder for the report text (adding right at the end should support caching!)
        full_template_text = (
            base_prompt_template
            + """
Here is the PFD report excerpt:

{report_excerpt}"""
        )

        if self.verbose:
            logger.debug(
                f"Building prompt template for search query: '{current_search_query}'."
            )
            logger.debug(
                f"Base prompt template created:\n{full_template_text.replace('{report_excerpt}', '[REPORT_TEXT_WILL_GO_HERE]')}"
            )
        return full_template_text

    def screen_reports(
        self,
        reports: Optional[pd.DataFrame] = None,
        search_query: Optional[str] = None,
        user_query: Optional[str] = None,
        filter_df: bool = True,
        result_col_name: str = "matches_query",
        produce_spans: bool = False,
        drop_spans: bool = False,
    ) -> pd.DataFrame:
        """
        Classifies reports in the DataFrame against the user-defined topic using the LLM.

        Parameters
        ----------
        reports : pd.DataFrame, optional
            If provided, this DataFrame will be used for screening, replacing any
            DataFrame stored in the instance for this call.
        search_query : str, optional
            Query string describing the reports you want to find. Overrides any
            query stored on the instance for this call. The prompt template will
            be rebuilt.
        user_query : str, optional
            Deprecated alias for ``search_query``. Will be removed in a future
            release.
        filter_df : bool, optional
            If ``True`` the returned DataFrame is filtered to only matching
            reports. Defaults to ``True``.
        result_col_name : str, optional
            Name of the boolean column added when ``filter_df`` is ``False``.
            Defaults to ``"matches_query"``.
        produce_spans : bool, optional
            When ``True`` a ``spans_matches_topic`` column is created containing
            the text snippet that justified the classification. Defaults to ``False``.
        drop_spans : bool, optional
            When ``True`` and ``produce_spans`` is also ``True``, the
            ``spans_`` column corresponding to ``result_col_name`` created during
            this call is removed from the returned DataFrame. Spans columns from
            other sources remain intact. Defaults to ``False``.

        Returns
        ----------
        pd.DataFrame
            Either a filtered DataFrame (if ``filter_df`` is ``True``), or the
            original DataFrame with an added classification column.

        Examples
        --------

            reports_df = pd.DataFrame(data)
            screener = Screener(LLM(), reports=reports_df)

            # Screen reports with the initial query
            filtered_df = screener.screen_reports(search_query="medication safety")

            # Screen the same reports with a new query and add a classification column
            classified_df = screener.screen_reports(search_query="tree safety", filter_df=False)
        """
        if search_query is None and user_query is not None:
            warnings.warn(
                "'user_query' is deprecated; please use 'search_query' instead. This parameter will be removed in a future update.",
                DeprecationWarning,
            )
            search_query = user_query
        elif search_query is not None and user_query is not None:
            warnings.warn(
                "'user_query' is deprecated and will be ignored as 'search_query' was also provided.",
                DeprecationWarning,
            )
        # Update produce_spans flag and prompt if needed
        if produce_spans != self.produce_spans:
            self.produce_spans = produce_spans
            self.prompt_template = None

        # Update reports if a new one is provided for this call
        if reports is not None:
            # Use a copy of the provided DataFrame for this operation
            current_reports = reports.copy()
            if self.verbose:
                logger.debug(
                    f"Using new DataFrame provided to screen_reports (shape: {current_reports.shape})."
                )
        else:
            # Use the instance's DataFrame (which is already a copy or an empty DF)
            current_reports = (
                self.reports.copy()
            )  # Ensure we work with a copy even of the instance's df for this run
            if self.verbose:
                logger.debug(
                    f"Using instance's DataFrame for screen_reports (shape: {current_reports.shape})."
                )

        # Determine the search query for this call
        active_search_query = search_query if search_query is not None else self.search_query

        if not active_search_query:
            logger.error("Search query is not set. Cannot screen reports.")
            raise ValueError(
                "Search query must be provided either at initialisation or to screen_reports."
            )

        # Rebuild prompt if the active query is different from the one used for the current template,
        # or if the template hasn't been built yet.
        if not self.prompt_template or (
            search_query is not None and search_query != self.search_query
        ):
            if (
                self.verbose
                and search_query is not None
                and search_query != self.search_query
            ):
                logger.debug(
                    f"New search query provided to screen_reports: '{search_query}'. Rebuilding prompt template."
                )
            elif self.verbose and not self.prompt_template:
                logger.debug(
                    f"Prompt template not yet built. Building for query: '{active_search_query}'."
                )
            self.prompt_template = self._build_prompt_template(active_search_query)
            self.search_query = active_search_query

        result_col = result_col_name

        span_col = f"spans_{result_col_name}"
        if self.produce_spans and span_col not in current_reports.columns:
            current_reports[span_col] = GeneralConfig.NOT_FOUND_TEXT

        # --- Pre-flight checks ---
        if self.llm is None:
            logger.error("LLM client is not initialised. Cannot screen reports.")

        if current_reports.empty:
            if self.verbose:
                logger.error("Reports DataFrame is empty. Nothing to screen.")

        if not self.prompt_template:  # (Should be built if active_search_query is valid)
            logger.error(
                "Prompt template not built. This should not happen if search_query is set. Cannot screen reports."
            )

        # --- Prepare prompts ---
        prompts_for_screening = []
        report_indices = []  # ...to map results back to original indices

        if self.verbose:
            logger.debug(
                f"Preparing prompts for {len(current_reports)} reports using classification column '{result_col}'."
            )

        for index, row in current_reports.iterrows():
            report_parts = []
            fields = [
                (self.include_date, self.COL_DATE, "The date of the report:"),
                (self.include_coroner, self.COL_CORONER_NAME, "The name of the coroner:"),
                (self.include_area, self.COL_AREA, "The area where the investigation took place:"),
                (self.include_receiver, self.COL_RECEIVER, "The recipients of the report:"),
                (
                    self.include_investigation,
                    self.COL_INVESTIGATION,
                    "The Investigation & Inquest section:\n",
                ),
                (
                    self.include_circumstances,
                    self.COL_CIRCUMSTANCES,
                    "The Circumstances of Death section:\n",
                ),
                (self.include_concerns, self.COL_CONCERNS, "The Matters of Concern section:"),
            ]

            for flag, col, label in fields:
                if not flag or col not in row:
                    continue
                val = row.get(col)
                if pd.notna(val):
                    report_parts.append(f"{label} {str(val)}".strip())

            report_text = "\n\n".join(report_parts).strip()
            report_text = " ".join(report_text.split())

            if not report_text:
                if self.verbose:
                    logger.debug(
                        f"Report at index {index} resulted in empty text after column selection."
                    )

            current_prompt = self.prompt_template.format(report_excerpt=report_text)
            prompts_for_screening.append(current_prompt)
            report_indices.append(index)

            if self.verbose and len(prompts_for_screening) <= 2:
                logger.debug(
                    f"Full prompt for report index {index}:\n{current_prompt}\n---"
                )

        if not prompts_for_screening:
            if self.verbose:
                logger.debug(
                    "No prompts generated (was the input DataFrame was empty?)."
                )

        # --- Call LLM ---
        if self.verbose:
            logger.debug(
                f"Sending {len(prompts_for_screening)} prompts to LLM.generate..."
            )

        response_model = _topic_model_with_spans() if self.produce_spans else TopicMatch
        llm_results = self.llm.generate(
            prompts=prompts_for_screening, response_format=response_model
        )

        if self.verbose:
            logger.debug(f"Received {len(llm_results)} results from LLM.")

        # --- Process results ---
        # Make a temporary pandas Series to hold classifications
        temp_classifications_series = pd.Series(index=report_indices, dtype=object)

        for i, result in enumerate(llm_results):
            original_report_index = report_indices[i]
            if isinstance(result, BaseModel):
                classification_value = result.matches_topic == "Yes"
                temp_classifications_series.loc[original_report_index] = classification_value
                if self.produce_spans:
                    span_val = getattr(result, "spans_matches_topic", "")
                    if not isinstance(span_val, str) or not span_val.strip():
                        span_val = GeneralConfig.NOT_FOUND_TEXT
                    current_reports.at[original_report_index, span_col] = span_val
                if self.verbose:
                    logger.debug(
                        f"Report original index {original_report_index}: LLM classified as '{result.matches_topic}' -> {classification_value}"
                    )
            elif isinstance(result, dict):
                classification_value = result.get("matches_topic") == "Yes"
                temp_classifications_series.loc[original_report_index] = classification_value
                if self.produce_spans:
                    span_val = result.get("spans_matches_topic", "")
                    if not isinstance(span_val, str) or not span_val.strip():
                        span_val = GeneralConfig.NOT_FOUND_TEXT
                    current_reports.at[original_report_index, span_col] = span_val
            else:
                logger.error(
                    f"Error classifying report at original index {original_report_index}: {result}"
                )
                temp_classifications_series.loc[original_report_index] = pd.NA

        # Add classification results to the DataFrame being processed for this call
        current_reports[result_col] = temp_classifications_series

        if (
            reports is None
        ):  # If we used the instance's reports as the base for current_reports...
            self.reports = (
                current_reports.copy()
            )  # ...update the instance's dataframe with the results

        if self.verbose:
            if result_col in current_reports.columns:
                logger.debug(
                    f"Added '{result_col}' column. Distribution:\n{current_reports[result_col].value_counts(dropna=False)}"
                )
            else:
                logger.warning(
                    f"'{result_col}' classification column was not added. This was unexpected!"
                )

        if drop_spans:
            if not self.produce_spans:
                logger.warning(
                    "drop_spans=True has no effect because produce_spans=False"
                )
            else:
                current_reports.drop(span_col, axis=1, inplace=True, errors="ignore")

        # --- Filter DataFrame if requested ---
        if filter_df:
            if result_col in current_reports.columns:
                mask = current_reports[result_col] == True
                filtered_df = current_reports[mask].copy()
                filtered_df.drop(
                    result_col, axis=1, inplace=True, errors="ignore"
                )

                if self.verbose:
                    logger.debug(
                        f"DataFrame screened. Original size for this run: {len(current_reports)}, Filtered size: {len(filtered_df)}"
                    )
                return filtered_df
            else:
                if self.verbose:
                    logger.warning(
                        f"Cannot screen DataFrame as '{result_col}' column is missing."
                    )
                return current_reports
        else:
            if self.verbose:
                logger.debug(
                    f"Returning DataFrame with '{result_col}' column (no filtering applied as filter_df is False)."
                )
            return current_reports
