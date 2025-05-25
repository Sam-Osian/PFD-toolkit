from typing import Literal, Dict, List, Tuple, Any, Optional, Union
import logging
from pydantic import BaseModel, Field
import pandas as pd

# Configure basic logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class TopicMatch(BaseModel):
    """Pydantic model to structure the LLM's response for
    topic matching. Ensures the LLM returns either "Yes"
    or "No".
    """
    matches_topic: Literal['Yes', 'No'] = Field(
        ...,
        description="Indicate whether the report text is relevant to the user's query. Must be Yes or No."
    )


class Filter:
    """
    Classifies a list of report texts against a user-defined topic using an LLM.

    This class takes a DataFrame of reports, a user query, and various
    configuration options to classify whether each report matches the query.
    It can either filter the DataFrame to return only matching reports or
    add a classification column to the original DataFrame.

    Parameters
    ----------
    llm : LLM, optional
        An instance of the LLM class from `pfd_toolkit`.
    reports : pd.DataFrame, optional
        A DataFrame containing Prevention of Future Death reports.
    user_query : str, optional
        The topic string provided by the user. If provided, the prompt template
        is built during initialisation.
    match_approach : str, optional
        Either 'strict' or 'liberal'. Determines the LLM's bias when in doubt.
        Defaults to 'strict'.
    filter_df : bool, optional
        If True, the returned DataFrame will be filtered. Defaults to True.
    verbose : bool, optional
        If True, print more detailed logs. Defaults to False.
    classification_column_name : str, optional
        The name for the classification column added to the DataFrame.
        Defaults to 'matches_query'.
    include_date : bool, optional
        Flag to determine if the 'Date' column is included. Defaults to False.
    include_coroner_name : bool, optional
        Flag to determine if the 'CoronerName' column is included. Defaults to False.
    include_area : bool, optional
        Flag to determine if the 'Area' column is included. Defaults to False.
    include_receiver : bool, optional
        Flag to determine if the 'Receiver' column is included. Defaults to False.
    include_investigation : bool, optional
        Flag to determine if the 'InvestigationAndInquest' column is included. Defaults to True.
    include_circumstances : bool, optional
        Flag to determine if the 'CircumstancesOfDeath' column is included. Defaults to True.
    include_concerns : bool, optional
        Flag to determine if the 'MattersOfConcern' column is included. Defaults to True.

    Attributes
    ----------

    llm : LLM
        The LLM client instance.
    reports : pd.DataFrame
        The DataFrame containing report data.
    user_query : Optional[str]
        The current user query string.
    prompt_template : Optional[str]
        The generated prompt template for the LLM.
    match_approach : str
        The matching strategy ('strict' or 'liberal').
    filter_df : bool
        Flag indicating if the DataFrame should be filtered or if a column should be added.
    classification_column_name : str
        Name of the column used for classification results.
    include_date : bool
        Whether to include the date column in the LLM prompt.
    include_coroner_name : bool
        Whether to include the coroner name column in the LLM prompt.
    include_area : bool
        Whether to include the area column in the LLM prompt.
    include_receiver : bool
        Whether to include the receiver column in the LLM prompt.
    include_investigation : bool
        Whether to include the investigation column in the LLM prompt.
    include_circumstances : bool
        Whether to include the circumstances column in the LLM prompt.
    include_concerns : bool
        Whether to include the concerns column in the LLM prompt.
    verbose : bool
        Flag for verbose logging.
    COL_URL : str
        Default column name for report URLs.
    COL_ID : str
        Default column name for report IDs.
    COL_DATE : str
        Default column name for report dates.
    COL_CORONER_NAME : str
        Default column name for coroner names.
    COL_AREA : str
        Default column name for coroner areas.
    COL_RECEIVER : str
        Default column name for report receivers.
    COL_INVESTIGATION : str
        Default column name for investigation/inquest details.
    COL_CIRCUMSTANCES : str
        Default column name for circumstances of death.
    COL_CONCERNS : str
        Default column name for matters of concern.


    Examples
    --------
    >>> user_topic = "medication errors"
    >>> llm_client = LLM()
    >>> report_filter = Filter(llm=llm_client, user_query=user_topic, reports=reports_df)
    >>> filtered_reports = report_filter.filter_reports()
    >>> print(f"Found {len(filtered_reports)} report(s) on '{user_topic}'.")
    >>> # Found 1 report(s) on 'medication errors'.

    """
    # DataFrame column names (as class attributes)
    COL_URL = "URL"
    COL_ID = "ID"
    COL_DATE = "Date"
    COL_CORONER_NAME = "CoronerName"
    COL_AREA = "Area"
    COL_RECEIVER = "Receiver"
    COL_INVESTIGATION = "InvestigationAndInquest"
    COL_CIRCUMSTANCES = "CircumstancesOfDeath"
    COL_CONCERNS = "MattersOfConcern"

    def __init__(
        self,
        llm: Optional[Any] = None,
        reports: Optional[pd.DataFrame] = None,
        user_query: Optional[str] = None,
        match_approach: str = 'strict',
        filter_df: bool = True,
        verbose: bool = False,
        classification_column_name: str = 'matches_query',
        include_date: bool = False,
        include_coroner_name: bool = False,
        include_area: bool = False,
        include_receiver: bool = False,
        include_investigation: bool = True,
        include_circumstances: bool = True,
        include_concerns: bool = True
    ) -> None:
        self.llm = llm
        self.match_approach = match_approach
        self.filter_df = filter_df
        self.verbose = verbose
        self.classification_column_name = classification_column_name

        # Store column inclusion toggles
        self.include_date = include_date
        self.include_coroner_name = include_coroner_name
        self.include_area = include_area
        self.include_receiver = include_receiver
        self.include_investigation = include_investigation
        self.include_circumstances = include_circumstances
        self.include_concerns = include_concerns

        # Initialise reports; always copy!
        self.reports: pd.DataFrame = reports.copy() if reports is not None else pd.DataFrame()
        
        self.user_query: Optional[str] = user_query
        self.prompt_template: Optional[str] = None 
        
        # If a query is provided at init, build the prompt
        if self.user_query:  
            self.prompt_template = self._build_prompt_template(self.user_query)
        elif self.verbose:
            logger.debug("Filter initialised without an initial user query. Prompt will be built on first filter_reports call with a query.")

        if self.verbose:
            if self.reports is not None:
                 logger.debug(f"Initial reports DataFrame shape: {self.reports.shape}")


    def _build_prompt_template(self, current_user_query: str) -> str:
        """
        Constructs the prompt template based on the user query and match approach.
        """
        base_prompt_template = f"""
            You are an expert text classification assistant. Your job is to read
            through the Prevention of Future Death (PFD) report excerpt at the
            bottom of this message, and decide whether or not it matches a user
            query.

            The user's query may be thematic, or it might pertain to a small or
            subtle inclusion in the report. The user query is:

            '{current_user_query}'

            If the report/excerpt matches this query, you must respond 'Yes'. Else,
            respond 'No'.

            Your response must be a JSON object in which "matches_topic" can be either
            "Yes" or "No".
            """
            
        # Add match approach instructions
        if self.match_approach == 'strict':
            base_prompt_template += """

                Your match should be strict.
                This means that if you are in reasonable doubt as to whether a report
                matches the user query, you should respond "No".
                """
        elif self.match_approach == 'liberal':
            base_prompt_template += """

                Your match should be liberal.
                This means that if you are in reasonable doubt as to whether a report
                matches the user query, you should respond "Yes".
                """
        # Add the placeholder for the report text
        full_template_text = base_prompt_template + """
        Here is the PFD report excerpt:

        {report_excerpt}"""

        if self.verbose:
            logger.debug(f"Building prompt template for user query: '{current_user_query}'. Match approach: {self.match_approach}.")
            logger.debug(f"Base prompt template created:\n{full_template_text.replace('{report_excerpt}', '[REPORT_TEXT_WILL_GO_HERE]')}")
        return full_template_text


    def filter_reports(self, reports: Optional[pd.DataFrame] = None, user_query: Optional[str] = None) -> pd.DataFrame:
        """
        Classifies reports in the DataFrame against the user-defined topic using the LLM.

        Parameters
        ----------
        reports : pd.DataFrame, optional
            If provided, this DataFrame will be used for filtering, replacing any
            DataFrame stored in the instance for this call.
        user_query : str, optional
            If provided, this query will be used, overriding any query stored
            in the instance for this call. The prompt template will be rebuilt.

        Returns
        ----------
        pd.DataFrame
            Either a filtered DataFrame (if self.filter_df is True), or the
            original DataFrame with an added classification column.

        Examples
        --------
        >>> reports_df = pd.DataFrame(data)
        >>> my_filter = Filter(LLM(), user_query="medication safety", filter_df=True, reports=reports_df)
        >>>
        >>> # Filter reports with the initial query
        >>> filtered_df = my_filter.filter_reports()
        >>>
        >>> # Filter the same reports with a new query and add classification column
        >>> my_filter.filter_df = False     # Modify filter behaviour
        >>> classified_df = my_filter.filter_reports(user_query="tree safety")
        """
        # Update reports if a new one is provided for this call
        if reports is not None:
            # Use a copy of the provided DataFrame for this operation
            current_reports = reports.copy()
            if self.verbose:
                logger.debug(f"Using new DataFrame provided to filter_reports (shape: {current_reports.shape}).")
        else:
            # Use the instance's DataFrame (which is already a copy or an empty DF)
            current_reports = self.reports.copy() # Ensure we work with a copy even of the instance's df for this run
            if self.verbose:
                logger.debug(f"Using instance's DataFrame for filter_reports (shape: {current_reports.shape}).")


        # Determine the user query for this call
        active_user_query = user_query if user_query is not None else self.user_query

        if not active_user_query:
            logger.error("User query is not set. Cannot filter reports.")
            raise ValueError("User query must be provided either at initialisation or to filter_reports.")

        # Rebuild prompt if the active query is different from the one used for the current template,
        # or if the template hasn't been built yet.
        if not self.prompt_template or (user_query is not None and user_query != self.user_query):
            if self.verbose and user_query is not None and user_query != self.user_query:
                logger.debug(f"New user query provided to filter_reports: '{user_query}'. Rebuilding prompt template.")
            elif self.verbose and not self.prompt_template:
                 logger.debug(f"Prompt template not yet built. Building for query: '{active_user_query}'.")
            self.prompt_template = self._build_prompt_template(active_user_query)
            self.user_query = active_user_query 

        # --- Pre-flight checks ---
        if self.llm is None:
            logger.error("LLM client is not initialised. Cannot filter reports.")

        if current_reports.empty:
            if self.verbose:
                logger.error("Reports DataFrame is empty. Nothing to filter.")

        if not self.prompt_template: # (Should be built if active_user_query is valid)
            logger.error("Prompt template not built. This should not happen if user_query is set. Cannot filter reports.")

        # --- Prepare prompts ---
        prompts_for_filtering = []
        report_indices = [] # ...to map results back to original indices

        if self.verbose:
            logger.debug(f"Preparing prompts for {len(current_reports)} reports using classification column '{self.classification_column_name}'.")

        for index, row in current_reports.iterrows():
            report_parts = []
            # Conditionally include column data based on toggles and existence
            if self.include_date and self.COL_DATE in row and pd.notna(row[self.COL_DATE]): report_parts.append(str(row[self.COL_DATE]))
            if self.include_coroner_name and self.COL_CORONER_NAME in row and pd.notna(row[self.COL_CORONER_NAME]): report_parts.append(str(row[self.COL_CORONER_NAME]))
            if self.include_area and self.COL_AREA in row and pd.notna(row[self.COL_AREA]): report_parts.append(str(row[self.COL_AREA]))
            if self.include_receiver and self.COL_RECEIVER in row and pd.notna(row[self.COL_RECEIVER]): report_parts.append(str(row[self.COL_RECEIVER]))
            if self.include_investigation and self.COL_INVESTIGATION in row and pd.notna(row[self.COL_INVESTIGATION]): report_parts.append(str(row[self.COL_INVESTIGATION]))
            if self.include_circumstances and self.COL_CIRCUMSTANCES in row and pd.notna(row[self.COL_CIRCUMSTANCES]): report_parts.append(str(row[self.COL_CIRCUMSTANCES]))
            if self.include_concerns and self.COL_CONCERNS in row and pd.notna(row[self.COL_CONCERNS]): report_parts.append(str(row[self.COL_CONCERNS]))

            report_text = ' '.join(report_parts).strip()

            if not report_text:
                if self.verbose:
                    logger.debug(f"Report at index {index} resulted in empty text after column selection. LLM will receive minimal context.")

            current_prompt = self.prompt_template.format(report_excerpt=report_text)
            prompts_for_filtering.append(current_prompt)
            report_indices.append(index)

            if self.verbose and len(prompts_for_filtering) <= 2 :
                 logger.debug(f"Full prompt for report index {index}:\n{current_prompt}\n---")

        if not prompts_for_filtering:
            if self.verbose:
                logger.debug("No prompts generated (was the input DataFrame was empty?).")

        # --- Call LLM ---
        if self.verbose:
            logger.debug(f"Sending {len(prompts_for_filtering)} prompts to LLM.generate_batch...")

        llm_results = self.llm.generate_batch(
            prompts=prompts_for_filtering,
            response_format=TopicMatch,
            temperature=0.0
        )

        if self.verbose:
            logger.debug(f"Received {len(llm_results)} results from LLM.")

        # --- Process results ---
        # Make a temporary pandas Series to hold classifications
        temp_classifications_series = pd.Series(index=report_indices, dtype=object)

        for i, result in enumerate(llm_results):
            original_report_index = report_indices[i]
            if isinstance(result, TopicMatch):
                classification_value = result.matches_topic == 'Yes'
                temp_classifications_series.loc[original_report_index] = classification_value
                if self.verbose:
                    logger.debug(f"Report original index {original_report_index}: LLM classified as '{result.matches_topic}' -> {classification_value}")
            else:
                logger.error(f"Error classifying report at original index {original_report_index}: {result}")
                temp_classifications_series.loc[original_report_index] = pd.NA

        # Add classification results to the DataFrame being processed for this call
        current_reports[self.classification_column_name] = temp_classifications_series
        
        if reports is None: # If we used the instance's reports as the base for current_reports...
            self.reports = current_reports.copy() # ...update the instance's dataframe with the results


        if self.verbose:
            if self.classification_column_name in current_reports.columns:
                logger.debug(f"Added '{self.classification_column_name}' column. Distribution:\n{current_reports[self.classification_column_name].value_counts(dropna=False)}")
            else:
                logger.warning(f"'{self.classification_column_name}' classification column was not added. This was unexpected!")

        # --- Filter DataFrame if requested ---
        if self.filter_df:
            if self.classification_column_name in current_reports.columns:
                mask = current_reports[self.classification_column_name] == True
                filtered_df = current_reports[mask].copy()
                filtered_df.drop(self.classification_column_name, axis=1, inplace=True, errors='ignore')
                
                if self.verbose:
                    logger.debug(f"DataFrame filtered. Original size for this run: {len(current_reports)}, Filtered size: {len(filtered_df)}")
                return filtered_df
            else:
                if self.verbose:
                    logger.warning(f"Cannot filter DataFrame as '{self.classification_column_name}' column is missing.")
                return current_reports
        else:
            if self.verbose:
                 logger.debug(f"Returning DataFrame with '{self.classification_column_name}' column (no filtering applied as filter_df is False).")
            return current_reports