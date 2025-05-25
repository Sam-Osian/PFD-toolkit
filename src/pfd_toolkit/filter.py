from typing import Literal, Dict, List, Tuple, Any, Optional, Union 
import logging
from pfd_toolkit import LLM 
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
    Classifies a list of report texts against a user-defined topic using the LLM.

    Parameters
    ----------
    llm : LLM
        An instance of the LLM class.
    reports : pd.DataFrame, optional
        A DataFrame containing Prevention of Future Death reports.
    user_query : str, optional
        The topic string provided by the user.
    match_approach : str, optional
        Either 'strict' or 'liberal'. Determines the LLM's bias when in doubt.
        Defaults to 'strict'.
    filter_df : bool, optional
        If True, the returned DataFrame will be filtered to include only
        matching reports. If False, a classification column will be added.
        Defaults to True.
    verbose : bool, optional
        If True, print more detailed logs. Defaults to False.
    """

    def __init__(
        self,
        llm: Optional[Any] = None, 
        reports: Optional[pd.DataFrame] = None,
        user_query: Optional[str] = None,
        match_approach: str = 'strict',
        filter_df: bool = True,
        verbose: bool = False
    ) -> None:

        self.llm = llm
        self.reports = reports.copy()
        self.user_query = user_query
        self.match_approach = match_approach
        self.filter_df = filter_df
        self.verbose = verbose

        if not self.user_query:
            raise ValueError("User query must be provided")

        # --- Prompt Template Construction ---

        base_prompt_template = f"""
        You are an expert text classification assistant. Your job is to read
        through the Prevention of Future Death (PFD) report excerpt at the
        bottom of this message, which may be the full report or just an excerpt.

        The user's query may be thematic, or it might pertain to a small or
        subtle inclusion in the report. The user query is:

        '{self.user_query}'

        If the report/excerpt matches this query, you must respond 'Yes'. Else,
        respond 'No'.

        Your response must be a JSON object where "matching_topic" can be either
        "Yes" or "No".
        """
    
        # Add match approach instructions
        if self.match_approach == 'strict':
            base_prompt_template += """\n\n
                Your match should be strict.
                This means that if you are in reasonable doubt in whether a report
                matches the user query, you should respond "No".
                """
                
        elif self.match_approach == 'liberal':
            base_prompt_template += """\n\n
                Your match should be liberal.
                This means that if you are in reasonable doubt in whether a report
                matches the user query, you should respond "Yes".
                """
        # Add the placeholder for the report text
        self.prompt_template = base_prompt_template + """

        Here is the PFD report excerpt:

        {report_excerpt}""" # Placeholder for individual report text

        if self.verbose:
            logger.info(f"Filter initialized. User query: '{self.user_query}'. Match approach: {self.match_approach}.")
            logger.info(f"Base prompt template created:\n{self.prompt_template.replace('{report_excerpt}', '[REPORT_TEXT_WILL_GO_HERE]')}")


    def filter_reports(self) -> pd.DataFrame:
        """
        Classifies reports in the DataFrame against the user-defined topic using the LLM.

        Returns
        ----------
        pd.DataFrame
            Either a filtered DataFrame (if self.filter_df is True), or the
            original DataFrame with an added classification column ('matches_query').
        """
        if self.llm is None:
            logger.error("LLM client is not initialised. Cannot filter reports.")

        if self.reports.empty:
            logger.error("Reports DataFrame is empty. Nothing to filter.")

        if not self.user_query:
            logger.error("User query is not set. Cannot filter reports.")

        prompts_for_filtering = []
        report_indices = []

        if self.verbose:
            logger.info(f"Preparing prompts for {len(self.reports)} reports...")

        for index, row in self.reports.iterrows():
            # Concatenate all columns into a single string for each given report (row)
            report_text = ' '.join(row.astype(str).values)

            # Add the LLM prompt to the beginning
            current_prompt = self.prompt_template.format(report_excerpt=report_text)
            prompts_for_filtering.append(current_prompt)
            report_indices.append(index) # ...store the original index

            if self.verbose and index < 2 : # Log first couple of full prompts for verification
                 logger.debug(f"Full prompt for report index {index}:\n{current_prompt}\n---")


        if not prompts_for_filtering:
            if self.verbose:
                logger.info("No prompts generated. Returning original DataFrame.")
            return self.reports

        if self.verbose:
            logger.info(f"Sending {len(prompts_for_filtering)} prompts to LLM.generate_batch...")

        # Call generate_batch
        llm_results = self.llm.generate_batch(
            prompts=prompts_for_filtering,
            response_format=TopicMatch,
            temperature=0.0 
        )

        if self.verbose:
            logger.info(f"Received {len(llm_results)} rows from LLM.")

        classifications = []
        for i, result in enumerate(llm_results):
            original_report_index = report_indices[i]
            if isinstance(result, TopicMatch):
                classifications.append(result.matches_topic == 'Yes')
                if self.verbose:
                    logger.debug(f"Report index {original_report_index}: LLM classified as '{result.matches_topic}'")
            else:
                # Handle cases where the LLM failed or returned an error string
                logger.error(f"Error classifying report at original index {original_report_index}: {result}")
                classifications.append(pd.NA) 

        # Add classification results to the DataFrame
        # Ensure classifications list aligns with self.reports.index if reports were modified
        if len(classifications) == len(self.reports):
            self.reports['matches_query'] = classifications
        else:
            logger.error(f"Mismatch in length of classifications ({len(classifications)}) and reports ({len(self.reports)}). Cannot assign column.")


        if self.verbose:
            if 'matches_query' in self.reports.columns:
                logger.info(f"Added 'matches_query' column. Distribution:\n{self.reports['matches_query'].value_counts(dropna=False)}")
            else:
                logger.warning("'matches_query' column was not added.")

        # Filter DataFrame if requested
        if self.filter_df and 'matches_query' in self.reports.columns:
            filtered_df = self.reports[self.reports['matches_query'] == True].copy()
            filtered_df.drop('matches_query', axis=1, inplace=True)
            
            if self.verbose:
                logger.info(f"DataFrame filtered. Original size: {len(self.reports)}, Filtered size: {len(filtered_df)}")
            return filtered_df
        
        else:
            if self.verbose and 'matches_query' not in self.reports.columns:
                 logger.warning("Cannot filter DataFrame as 'matches_query' column is missing.")
            elif self.verbose:
                 logger.info("Returning DataFrame with 'matches_query' column (no filtering applied as filter_df is False or column missing).")
            return self.reports