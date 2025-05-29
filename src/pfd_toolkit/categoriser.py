import pandas as pd
from pydantic import BaseModel
import numpy as np
from typing import Dict, Any, List

from pfd_toolkit import LLM


class Categoriser:
    def __init__(
        self,
        reports: pd.DataFrame,
        llm: LLM,
        json_schema: Dict[str, Any],
        third_tier_as_theme: bool = False,
        multi_assignment: bool = True,
        include_date: bool = False,
        include_circumstances: bool = True,
        include_coroner_name: bool = False,
        include_receiver: bool = False,
    ):

        self.reports = reports
        self.llm = llm
        self.json_schema = json_schema

        self.third_tier_as_theme = third_tier_as_theme
        self.multi_assignment = multi_assignment

        self.include_date = include_date
        self.include_cirsumstances = include_circumstances
        self.include_coroner_name = include_coroner_name
        self.include_receiver = include_receiver

    def _remove_unwanted_columns(self, df: pd.DataFrame) -> pd.DataFrame:
        # Remove unwanted columns as required
        if not self.include_date and "Date" in df.columns:
            df = df.drop("Date", axis=1)

        if not self.include_cirsumstances and "CircumstancesOfDeath" in df.columns:
            df = df.drop("CircumstancesOfDeath", axis=1)

        if not self.include_coroner_name and "CoronerName" in df.columns:
            df = df.drop("CoronerName", axis=1)

        if not self.include_receiver and "Receiver" in df.columns:
            df = df.drop("Receiver", axis=1)
        return df

    def categorise_reports(self, n: int = 1):
        self.copy_df = (
            self.reports.copy()
        )  # for testing, or we can keep it so that we don't wreck the OG .reports attribute, you choose
        self.copy_df = self._remove_unwanted_columns(df=self.copy_df)
        new_col_names = self._extract_column_names(json_schema=self.json_schema)
        for col in new_col_names:
            self.copy_df[col] = np.nan

        return self.copy_df

    def _extract_column_names(
        self, json_schema: dict | list, parent_key="", recursion_depth: int = 0
    ) -> List[str]:
        """Recurse through the given json schema to extract column names to use for classifications.
        Args:
            json_schema (dict | list): The json schema to use, starts as a dictiionary but at the end of the tree it becomes a list.
            parent_key (str): Prefix for new column names, used internally within recursion, humans should not set this.
            recursion_depth (int): Recursion depth safeguard to disallow json category schemas that are too deep. Currently we are hardcoding the limit to 10

        Returns:
            column_names (List[str])

        Raises:
            RecursionError: Raised manually as a safeguard.
        """

        column_names = []
        if recursion_depth > 10:
            raise RecursionError(
                "PFD Toolkit has gracefully stopped to avoid a potential recurrsion error. Your json schema for categorisation is likely too nested."
            )

        if isinstance(json_schema, dict):
            for key, value in json_schema.items():
                full_key = f"{parent_key}-{key}" if parent_key else key
                column_names.append(full_key)
                recursion_depth += 1

                # Go another level deep and when you get out, extend the column names list with the ones you pull out... and so on
                column_names.extend(
                    self._extract_column_names(
                        json_schema=value,
                        parent_key=full_key,
                        recursion_depth=recursion_depth,
                    )
                )

        elif isinstance(json_schema, list):
            # End of json nesting, just append the column names from the root list of categories.
            prefix = parent_key + "-" if parent_key else ""
            column_names.extend([prefix + item for item in json_schema])

        return column_names

    def traverse_category_tree(
        self, json_schema: dict | list, final_node: bool = False
    ):
        if not final_node:
            keys = list(json_schema.keys())  # all parent nodes are dicts
        else:
            keys = json_schema  # the last node is just a list

        model = get_categorisation_model(keys)
        result = self.llm.generate_batch(prompts=["prompt"], response_format=model)
        # TODO: Figure out the column name. Assign 'Yes' to that column for that row.
        # TODO: Why am I even here what's going on...
        # TODO: Track the column name by passing it in, each time adding the next suffix to it !!!
        if not final_node:
            if isinstance(json_schema[result], dict):
                # traverse
                self.traverse_category_tree(json_schema)
            else:
                assert isinstance(json_schema[result], list)  # sanity check
                self.traverse_category_tree(json_schema[result], final_node=True)
        else:
            # we've traversed to the maximum depth of this tree
            pass


def get_categorisation_model(categories: list[str]) -> BaseModel:
    class Categorisation(BaseModel):
        category: Literal[tuple(categories)]  # type: ignore

    return Categorisation
