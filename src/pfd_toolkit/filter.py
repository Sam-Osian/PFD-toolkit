from typing import Literal, Dict, List, Tuple, Any
import logging
from pfd_toolkit import LLM
from pydantic import BaseModel, Field
import pandas as pd

class TopicMatch(BaseModel):
    """Pydantic model to structure the LLM's response for 
    topic matching. Ensure the LLM returns either "Yes"
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
        llm: An instance of the LLM class.
        reports: A DataFrame containing Prevention of Future Death reports.
        user_query: The topic string provided by the user.
        
    """
    
    def __init__(
        self,
        llm: "LLM" = None,
        reports = pd.DataFrame,
        user_query: str = None,
        match_approach: str = 'strict',
        filter_df: bool = True,
        verbose: bool = False
    ) -> None:
        
        self.llm = llm
        self.reports = reports
        self.user_query = user_query
        self.match_approach = match_approach
        self.filter_df = filter_df
        self.verbose = verbose
        
        self.prompt = f"""
        You are an expert text classification assistant. Your job is to read
        through the Prevention of Future Death (PFD) report exerpt at the 
        bottom of this message, which may be the full report or just an excerpt.\n
        
        The user's query may be thematic, or it might pertain to a small or
        subtle inclusion in the report. The user query is:\n
        
        '{self.user_query}'\n
        
        If the report/exerpt matches this query, you must respond 'Yes'. Else, 
        respond 'No'. 
        
        Your response must be a JSON object matching the following schema:
        {{ "matches_topic": "Yes" }} or {{ "matches_topic": "No" }}
        """
        
        if self.match_approach=='strict':
            self.prompt =+ """Your match should be strict. 
            This means that if you are in reasonable doubt in whether a report 
            matches the user query, you should respond "No".
            """
        
        if self.match_approach=='liberal':
            self.prompt =+ """Your match should be liberal.
            This means that if you are in reasonable doubt in whether a report
            matches the user query, you should respond "Yes".
            """
        
        self.prompt =+ """Here is the PFD report exerpt:\n\n
        
        {report}"""
    
    def filter_report(self):
        """
        Classifies a list of report texts against a user-defined topic using the LLM.

        Returns
        ----------
            Either a filtered dataframe, or the exact same dataframe with an added
            classification column (True or False).
        """
        
        # Convert each report (row) to a string, convert to list, and add prompt to start
        prompts_for_filtering = [self.prompt + s for s in self.reports.astype(str).agg(' '.join, axis=1)]

        filtered_results = self.llm.generate_batch(prompts=prompts_for_filtering)