import pandas as pd
from pydantic import BaseModel
from pfd_toolkit import LLM
class Categoriser:
    def __init__(self,
        reports: pd.DataFrame,
        llm: LLM,
        schema: BaseModel,
        third_tier_as_theme: bool = False,
        multi_assignment: bool = True,
        include_date: bool = False,
        include_circumstances: bool = True,
        include_coroner_name: bool = False,
        include_receiver: bool = False
    ):

        self.reports = reports
        self.llm = llm
        self.schema = schema

        self.third_tier_as_theme = third_tier_as_theme
        self.multi_assignment = multi_assignment

        self.include_date = include_date
        self.include_cirsumstances = include_circumstances
        self.include_coroner_name = include_coroner_name
        self.include_receiver = include_receiver

        

    def _removed_unwanted_columns(self, df: pd.DataFrame):
        # Remove unwanted columns as required
        if not self.include_date and 'Date' in df.columns:
            df.drop('Date', axis=1)
        
        if not self.include_cirsumstances and 'CircumstancesOfDeath' in df.columns:
            df.drop('CircumstancesOfDeath', axis=1)

        if not self.include_coroner_name and 'CoronerName' in df.columns:
            df.drop('CoronerName', axis=1)
        
        if not self.include_receiver and 'Receiver' in df.columns:
            df.drop('Receiver', axis=1)

    def categorise_reports(self, n: int = 1):
        df = self.reports.copy() # for testing, or we can keep it so that we don't wreck the OG .reports attribute, you choose
        
        df = self._remove_unwanted_columns(df=df)
