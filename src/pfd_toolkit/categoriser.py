import pandas as pd
from pydantic import BaseModel
import numpy as np
from typing import Dict, Any, List, Literal
from copy import deepcopy
from pfd_toolkit import LLM
import logging

logger = logging.getLogger(__name__)

CATEGORISATION_PROMPT = """You are an expert in assigning labels to text extracts.
Your task is to assign a class to the provided text based on the options given.
In some cases where specified by the given json schema, you are allowed to assign multiple categories to the same extract.
If you do not think ANY of the classes provided reflect the text, then you must always default to the None category.
Here is the text:
{text}"""


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
        include_concerns: bool = True,
        include_coroner_name: bool = False,
        include_receiver: bool = False,
        debug: bool = False,
    ):

        self.reports = reports
        self.llm = llm
        self.json_schema = json_schema
        self.debug = debug

        self.third_tier_as_theme = third_tier_as_theme
        self.multi_assignment = multi_assignment

        self.include_date = include_date
        self.include_circumstances = include_circumstances
        self.include_concerns = include_concerns
        self.include_coroner_name = include_coroner_name
        self.include_receiver = include_receiver

    def categorise_reports(self):
        """Categorise PFD Reports."""
        self.categorised_reports = (
            self.reports.copy()
        )  # for testing, or we can keep it so that we don't wreck the OG .reports attribute, you choose

        new_col_names = self._extract_column_names(json_schema=self.json_schema)
        if self.debug:
            logger.info(
                f"Created {len(new_col_names)} for all of the categories extracted from the given json_schema."
            )
        for col in new_col_names:
            self.categorised_reports[col] = "No"

        for row_idx, _ in self.categorised_reports.iterrows():
            self._traverse_category_tree(
                json_schema=self.json_schema,
                colnames=list(self.json_schema.keys()),
                row_idx=row_idx,
                final_node=False,
            )  # never set final node to true unless your schema is just a List of categories.

        return self.categorised_reports

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
                # Construct the name of the column for each key in the json schema at this level. This will include the prior parent key prefix (eg: suicide-access_to_means)
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
            # End of json nesting, just append the column names from the root list of categories with the parent key.
            prefix = parent_key + "-" if parent_key else ""
            column_names.extend([prefix + item for item in json_schema])

        return column_names

    def _traverse_category_tree(
        self,
        json_schema: dict | list,
        colnames: List[str],
        row_idx: int,
        final_node: bool = False,
    ):
        if not final_node:
            keys = list(json_schema.keys())  # all parent nodes are dicts
        else:
            keys = deepcopy(json_schema)  # the last node is already a list[str]
            # LEARNING POINT:
            # it's important to note here that deepcopy is very important here.
            # When the LIST is modified, even as passed across functions/methods (the list of keys passed to classes in get_categorisation_model),
            # these changes are within an OBJECT (the OG json schema) and propagate back up to self.json_schema.
            # The result here is that suddenly our root node lists now have None in it (only if it was traversed into at the categorisation stage)
            # So, when we pass the json_schema (which here is a list but memory mapped to where its already stored inside self.json_schema) and modify it - we are changing self.json_schema.
            # We don't want to do this, as we are only appending None (in get_categorisation_model) to allow the LLM to opt to not select a category if it pleases.

        model = _get_categorisation_model(
            classes=keys, multiclass=self.multi_assignment
        )
        if self.debug:
            logger.info(f"Generating response_format model for classes: {keys}")

        # TODO: Sort out a proper prompt...
        result = self.llm.generate_batch(
            prompts=[self._construct_prompt(row_idx=row_idx)],
            response_format=model,
        )[
            0
        ]  # selecting element zero because I am just hijacking generate batch for now...

        categories: List[str] = (
            result.categories if self.multi_assignment else [result.category]
        )

        for idx, category in enumerate(
            categories
        ):  # this is where we could potentially introduce multi-threading - but oh my god be careful because we are in a recursive state here xD

            if category:
                col_idx = keys.index(
                    category
                )  # figure out the index of the category in the original keys
                self.categorised_reports.at[row_idx, colnames[col_idx]] = (
                    "Yes"  # update the respective column name that was classified
                )
                if self.debug:
                    logger.info(
                        f"LLM assigned {colnames[col_idx]} category to report index {row_idx}"
                    )

                if (
                    not final_node
                ):  # we only want to traverse down again if there are children below the current level
                    if isinstance(json_schema[category], dict):
                        # adjust column names, select the column for the next traversal based on the current iteration of selected categories.
                        new_colnames = [
                            colnames[col_idx] + "-" + child
                            for child in list(json_schema[category].keys())
                        ]

                        # traverse deeper
                        self._traverse_category_tree(
                            json_schema=json_schema[category],
                            colnames=new_colnames,
                            row_idx=row_idx,
                            final_node=False,
                        )
                    else:
                        assert isinstance(
                            json_schema[category], list
                        ), f"Expected root of json schema to be a list but got {type(json_schema[category])}"  # sanity check
                        # adjust column names
                        new_colnames = [
                            colnames[col_idx] + "-" + child
                            for child in json_schema[category]
                        ]
                        if self.debug:
                            logger.info(
                                f"LLM is now at the root of the json_schema with the following categories: {json_schema[category]}"
                            )

                        # traverse deeper - next iteration is a final node because we've reached a list
                        self._traverse_category_tree(
                            json_schema[category],
                            colnames=new_colnames,
                            row_idx=row_idx,
                            final_node=True,
                        )

    def _construct_prompt(self, row_idx) -> str:
        text = ""
        if self.include_date:
            text += f"This report was written: {self.categorised_reports.loc[row_idx, 'Date']}\n"
        if self.include_coroner_name:
            text += (
                f"Written by {self.categorised_reports.loc[row_idx, 'CoronerName']}\n"
            )
        if self.include_receiver:
            text += f"Received by {self.categorised_reports.loc[row_idx, 'Receiver']}\n"
        if self.include_circumstances:
            text += f"The circumstances of the subjects death are detailed as follows:\n{self.categorised_reports.loc[row_idx, 'CircumstancesOfDeath']}\n\n"
        if self.include_concerns:
            text += f"The concerns raised by the coroner are as follows:\n{self.categorised_reports.loc[row_idx, 'MattersOfConcern']}"
        prompt = CATEGORISATION_PROMPT.format(text=text)
        if self.debug:
            logger.info(
                f"--- START OF LLM PROMPT ---\n{prompt}\n--- END OF LLM PROMPT ---\n\n"
            )
        return prompt


def _get_categorisation_model(classes: list[str], multiclass: bool = True) -> BaseModel:
    classes.append(None)
    if not multiclass:

        class Categorisation(BaseModel):
            category: Literal[tuple(classes)]  # type: ignore - have to type ignore these

    else:

        class Categorisation(BaseModel):
            categories: List[Literal[tuple(classes)]]  # type: ignore

    return Categorisation
