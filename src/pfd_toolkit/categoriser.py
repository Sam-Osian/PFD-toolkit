import pandas as pd
from pydantic import BaseModel
from pfd_toolkit import LLM
class Categoriser:
    def __init__(self, reports: pd.DataFrame, llm: LLM, schema: BaseModel, third_tier_as_theme: bool = False, multi_assignment: bool = True, include_date: bool = False, include_circumstances: bool = True, include_coroner_name: bool = False, include_reciever: bool = False):
