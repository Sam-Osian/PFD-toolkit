import logging
import warnings

import pandas as pd
from pydantic import BaseModel, Field, ConfigDict, field_validator
from tqdm import tqdm
from tqdm import TqdmWarning

from pfd_toolkit.llm import LLM
from pfd_toolkit.config import GeneralConfig

# -----------------------------------------------------------------------------
# Logging Configuration:
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, force=True)

# Set the log level for the 'httpx' library to WARNING to reduce verbosity
logging.getLogger("httpx").setLevel(logging.WARNING)
# Silence the OpenAI client’s info-level logs (as in llm.py)
logging.getLogger("openai").setLevel(logging.WARNING)

warnings.filterwarnings("ignore", category=TqdmWarning)


# ---------------------------------------------------------------------------
# Area validation model
# ---------------------------------------------------------------------------


class AreaModel(BaseModel):
    """Pydantic model restricting the area field."""

    area: str = Field(..., description="Name of the coroner area")

    model_config = ConfigDict(extra="forbid")

    @field_validator("area", mode="before")
    @classmethod
    def apply_synonyms(cls, v: str) -> str:
        """Normalise location names using :class:`Cleaner` synonyms."""
        return Cleaner.map_area_synonym(v)

    @field_validator("area")
    @classmethod
    def validate_area(cls, v: str) -> str:
        """Ensure the area is one of the allowed values."""
        if v not in GeneralConfig.ALLOWED_AREAS:
            # If the provided area is not recognised, default to "Other"
            return "Other"
        return v


class Cleaner:
    """Batch-clean PFD report fields with an LLM.

    The cleaner loops over selected columns, builds field-specific prompts and
    writes the returned text back into a copy of the DataFrame.

    Parameters
    ----------
    reports : pandas.DataFrame
        Input DataFrame to clean.
    llm : LLM
        Instance of the ``LLM`` helper used for prompting.
    include_coroner : bool, optional
        Clean the ``coroner`` column. Defaults to ``True``.
    include_receiver : bool, optional
        Clean the ``receiver`` column. Defaults to ``True``.
    include_area : bool, optional
        Clean the ``area`` column. Defaults to ``True``.
    include_investigation : bool, optional
        Clean the ``investigation`` column. Defaults to ``True``.
    include_circumstances : bool, optional
        Clean the ``circumstances`` column. Defaults to ``True``.
    include_concerns : bool, optional
        Clean the ``concerns`` column. Defaults to ``True``.
    coroner_prompt : str or None, optional
        Custom prompt for the coroner field. Defaults to ``None``.
    area_prompt : str or None, optional
        Custom prompt for the area field. Defaults to ``None``.
    receiver_prompt : str or None, optional
        Custom prompt for the receiver field. Defaults to ``None``.
    investigation_prompt : str or None, optional
        Custom prompt for the investigation field. Defaults to ``None``.
    circumstances_prompt : str or None, optional
        Custom prompt for the circumstances field. Defaults to ``None``.
    concerns_prompt : str or None, optional
        Custom prompt for the concerns field. Defaults to ``None``.
    verbose : bool, optional
        Emit info-level logs for each batch when ``True``. Defaults to ``False``.

    Attributes
    ----------
    cleaned_reports : pandas.DataFrame
        Result of the last call to ``clean_reports``.
    coroner_prompt_template, area_prompt_template, ... : str
        Finalised prompt strings actually sent to the model.

    Examples
    --------

        cleaner = Cleaner(df, llm, include_coroner=False, verbose=True)
        cleaned_df = cleaner.clean_reports()
        cleaned_df.head()
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

    @classmethod
    def map_area_synonym(cls, area: str) -> str:
        """Return canonical name for an area synonym."""
        return GeneralConfig.AREA_SYNONYMS.get(area, area)

    # Base prompt template used for all cleaning operations
    CLEANER_BASE_PROMPT = (
        "You are an expert in extracting and cleaning specific information from UK Coronal Prevention of Future Deaths (PFD) reports.\n\n"
        "Task:\n"
        "1. **Extract** only the information related to {field_description}.\n"
        "2. **Clean** the input text by fixing typos and removing clearly spurious characters (e.g. rogue numbers, stray punctuation, HTML tags). Do not delete any valid sentences or shorten the text.\n"
        "3. **Correct** any misspellings, ensuring the text is in sentence-case **British English**. Keep any existing acronyms if used; do not expand them.\n"
        "4. **Return** exactly and only the cleaned data for {field_contents_and_rules}. You must only return the cleaned string, without adding additional commentary, summarisation, or headings.\n"
        f"5. **If extraction fails**, return only and exactly: {GeneralConfig.NOT_FOUND_TEXT}\n"
        "6. **Do not** remove or summarise any of the original content other than the minimal fixes described above.\n\n"
        "Extra instructions:\n"
        "{extra_instructions}\n\n"
        "Input Text:\n"
        '"""'
    )

    # Field-specific configuration for prompt substitution
    CLEANER_PROMPT_CONFIG = {
        "Coroner": {
            "field_description": "the name of the Coroner who presided over the inquest",
            "field_contents_and_rules": "this name of the Coroner - nothing else",
            "extra_instructions": (
                "Remove all reference to titles & middle name(s), if present, and replace the first name with an initial. "
                'For example, if the string is "Mr. Joe E Bloggs", return "J. Bloggs". '
                'If the string is "Joe Bloggs Senior Coroner for West London", return "J. Bloggs". '
                'If the string is "J. Bloggs", just return "J. Bloggs" (no modification). '
            ),
        },
        "Area": {
            "field_description": "the area where the coroner's inquest took place",
            "field_contents_and_rules": "only the name of the coroner's area -- nothing else",
            "extra_instructions": (
                'For example, if the string is "Area: West London", return "London West". '
            ),
        },
        "Receiver": {
            "field_description": "the name(s)/organisation(s) of the receiver(s) of the report",
            "field_contents_and_rules": "only the name(s)/organisation(s) and, if given, their job title(s) -- nothing else",
            "extra_instructions": (
                "Separate multiple recipients with semicolons (;). "
                "Do not use a numbered list. "
                "Remove reference to family altogether. "
                "Remove address(es) if given (i.e. just include the recipient). "
            ),
        },
        "InvestigationAndInquest": {
            "field_description": "the details of the investigation and inquest",
            "field_contents_and_rules": "the entire text",
            "extra_instructions": (
                "If the string appears to need no cleaning, return it as is. "
                "If a date is used, put it in numerical form (e.g. '1 January 2024'). "
                "Keep any existing paragraph formatting (e.g. spacing). "
                "Do not summarise or shorten the text."
            ),
        },
        "CircumstancesOfDeath": {
            "field_description": "the circumstances of death",
            "field_contents_and_rules": "the entire text",
            "extra_instructions": (
                "If the string appears to need no cleaning, return it as is. "
                "If a date is used, put it in numerical form (e.g. '1 January 2024'). "
                "Keep any existing paragraph formatting (e.g. spacing). "
                "Do not summarise or shorten the text."
            ),
        },
        "MattersOfConcern": {
            "field_description": "the matters of concern",
            "field_contents_and_rules": "the entire text",
            "extra_instructions": (
                'Remove reference to boilerplate text, if any occurs. This is usually 1 or 2 non-specific sentences at the start of the string often ending with "...The Matters of Concern are as follows:" (which should also be removed). '
                "If the string appears to need no cleaning, return it as is. "
                "If a date is used, put it in numerical form (e.g. '1 January 2024'). "
                "Keep any existing paragraph formatting (e.g. spacing). "
                "Do not summarise or shorten the text."
            ),
        },
    }

    def __init__(
        self,
        # Input DataFrame containing PFD reports
        reports: pd.DataFrame,
        llm: LLM,
        # Fields to clean
        include_coroner: bool = True,
        include_receiver: bool = True,
        include_area: bool = True,
        include_investigation: bool = True,
        include_circumstances: bool = True,
        include_concerns: bool = True,
        # Custom prompts for each field; defaults to None
        coroner_prompt: str = None,
        area_prompt: str = None,
        receiver_prompt: str = None,
        investigation_prompt: str = None,
        circumstances_prompt: str = None,
        concerns_prompt: str = None,
        verbose: bool = False,
    ) -> None:
        self.reports = reports
        self.llm = llm

        # Flags for which fields to clean
        self.include_coroner = include_coroner
        self.include_receiver = include_receiver
        self.include_area = include_area
        self.include_investigation = include_investigation
        self.include_circumstances = include_circumstances
        self.include_concerns = include_concerns

        # Prompt templates
        self.coroner_prompt_template = coroner_prompt or self._get_prompt_for_field(
            "Coroner"
        )
        self.area_prompt_template = area_prompt or self._get_prompt_for_field("Area")
        self.receiver_prompt_template = receiver_prompt or self._get_prompt_for_field(
            "Receiver"
        )
        self.investigation_prompt_template = (
            investigation_prompt
            or self._get_prompt_for_field("InvestigationAndInquest")
        )
        self.circumstances_prompt_template = (
            circumstances_prompt or self._get_prompt_for_field("CircumstancesOfDeath")
        )
        self.concerns_prompt_template = concerns_prompt or self._get_prompt_for_field(
            "MattersOfConcern"
        )

        self.verbose = verbose

        # -----------------------------------------------------------------------------
        # Error and Warning Handling for Initialisation Parameters
        # -----------------------------------------------------------------------------

        ### Errors
        # If the reports parameter is not a DataFrame
        if not isinstance(reports, pd.DataFrame):
            raise TypeError("The 'reports' parameter must be a pandas DataFrame.")

        # If the input DataFrame does not contain the necessary columns
        required_df_columns = []
        if self.include_coroner:
            required_df_columns.append(self.COL_CORONER_NAME)
        if self.include_area:
            required_df_columns.append(self.COL_AREA)
        if self.include_receiver:
            required_df_columns.append(self.COL_RECEIVER)
        if self.include_investigation:
            required_df_columns.append(self.COL_INVESTIGATION)
        if self.include_circumstances:
            required_df_columns.append(self.COL_CIRCUMSTANCES)
        if self.include_concerns:
            required_df_columns.append(self.COL_CONCERNS)

        # Get unique column names in case user mapped multiple flags to the same df column
        required_df_columns = list(set(required_df_columns))

        missing_columns = [
            col for col in required_df_columns if col not in self.reports.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Cleaner could not find the following DataFrame columns: {missing_columns}."
            )

    def _get_prompt_for_field(self, field_name: str) -> str:
        """Generates a complete prompt template for a given PFD report field."""
        # Access prompt configuration stored on this class
        config = self.CLEANER_PROMPT_CONFIG[field_name]
        return self.CLEANER_BASE_PROMPT.format(
            field_description=config["field_description"],
            field_contents_and_rules=config["field_contents_and_rules"],
            extra_instructions=config["extra_instructions"],
        )

    def generate_prompt_template(self) -> dict[str, str]:
        """Return the prompt templates used for each field.

        The returned dictionary maps DataFrame column names to the full prompt
        text with a ``[TEXT]`` placeholder appended to illustrate how the
        prompt will look during ``clean_reports``.
        """

        return {
            self.COL_CORONER_NAME: f"{self.coroner_prompt_template}\n[TEXT]",
            self.COL_AREA: f"{self.area_prompt_template}\n[TEXT]",
            self.COL_RECEIVER: f"{self.receiver_prompt_template}\n[TEXT]",
            self.COL_INVESTIGATION: f"{self.investigation_prompt_template}\n[TEXT]",
            self.COL_CIRCUMSTANCES: f"{self.circumstances_prompt_template}\n[TEXT]",
            self.COL_CONCERNS: f"{self.concerns_prompt_template}\n[TEXT]",
        }

    def clean_reports(self, anonymise: bool = False) -> pd.DataFrame:
        """Run LLM-based cleaning for the configured columns.

        The method operates **in place on a copy** of ``self.reports`` so the
        original DataFrame is never mutated.

        Returns
        -------
        pandas.DataFrame
            A new DataFrame in which the selected columns have been
            replaced by the LLM output (or left unchanged when the model
            returns an error marker).

        Parameters
        ----------
        anonymise : bool, optional
            When ``True`` append an instruction to anonymise names and pronouns
            in the investigation, circumstances and concerns fields. Defaults to
            ``False``.

        Examples
        --------
            cleaner = Cleaner(llm=llm_client, reports=reports)
            cleaned = cleaner.clean_reports()

        """
        cleaned_df = self.reports.copy()  # Work on a copy

        # Optional anonymisation instruction
        anonymise_instruction = (
            "Replace all personal names and pronouns with they/them/their."
        )

        investigation_prompt = self.investigation_prompt_template
        circumstances_prompt = self.circumstances_prompt_template
        concerns_prompt = self.concerns_prompt_template

        if anonymise:
            # Insert the instruction just before the input text portion so the
            # LLM treats it as guidance rather than part of the text to clean
            insertion_point = "\n\nInput Text:"
            investigation_prompt = investigation_prompt.replace(
                insertion_point,
                f"\n{anonymise_instruction}{insertion_point}",
            )
            circumstances_prompt = circumstances_prompt.replace(
                insertion_point,
                f"\n{anonymise_instruction}{insertion_point}",
            )
            concerns_prompt = concerns_prompt.replace(
                insertion_point,
                f"\n{anonymise_instruction}{insertion_point}",
            )

        # Define fields to process: (Config Key, Process Flag, DF Column Name, Prompt Template)
        field_processing_config = [
            (
                "Coroner",
                self.include_coroner,
                self.COL_CORONER_NAME,
                self.coroner_prompt_template,
            ),
            ("Area", self.include_area, self.COL_AREA, self.area_prompt_template),
            (
                "Receiver",
                self.include_receiver,
                self.COL_RECEIVER,
                self.receiver_prompt_template,
            ),
            (
                "InvestigationAndInquest",
                self.include_investigation,
                self.COL_INVESTIGATION,
                investigation_prompt,
            ),
            (
                "CircumstancesOfDeath",
                self.include_circumstances,
                self.COL_CIRCUMSTANCES,
                circumstances_prompt,
            ),
            (
                "MattersOfConcern",
                self.include_concerns,
                self.COL_CONCERNS,
                concerns_prompt,
            ),
        ]

        # Use tqdm for the outer loop over fields
        for config_key, process_flag, column_name, prompt_template in tqdm(
            field_processing_config, desc="Processing Fields", position=0, leave=True
        ):
            if not process_flag:
                continue

            if column_name not in cleaned_df.columns:
                # This case should ideally be caught by __init__ checks, but good to have defence here
                logger.warning(
                    f"Column '{column_name}' for field '{config_key}' not found at cleaning time. Skipping."
                )
                continue
            if self.verbose:
                logger.info(
                    f"Preparing batch for column: '{column_name}' (Field: {config_key})"
                )

            # Ensure column is treated as string for processing
            # Handle cases where column might be all NaNs or mixed type before attempting string operations
            if cleaned_df[column_name].notna().any():
                if not pd.api.types.is_string_dtype(cleaned_df[column_name]):
                    cleaned_df[column_name] = cleaned_df[column_name].astype(str)
            else:
                logger.info(
                    f"Column '{column_name}' contains all NaN values. No text to clean."
                )
                continue  # Skip to next field if column is all NaN

            # Select non-null texts to clean and their original indices
            # Ensure we are working with string representations for LLM processing
            texts_to_clean_series = cleaned_df[column_name][
                cleaned_df[column_name].notna()
            ].astype(str)

            if texts_to_clean_series.empty:
                logger.info(
                    f"No actual text data to clean in column '{column_name}' after filtering NaNs. Skipping."
                )
                continue

            original_indices = texts_to_clean_series.index
            original_texts_list = texts_to_clean_series.tolist()

            # Construct prompts for the batch
            # Each prompt consists of the field-specific template followed by the actual text
            prompts_for_batch = [
                f"{prompt_template}\n{text}" for text in original_texts_list
            ]

            if (
                not prompts_for_batch
            ):  # Should not happen if texts_to_clean_series was not empty
                logger.info(
                    f"No prompts generated for column '{column_name}'. Skipping LLM call."
                )
                continue

            if self.verbose:
                logger.info(
                    f"First prompt for '{column_name}' batch: {prompts_for_batch[0][:250]}..."
                )  # Log snippet of first prompt

            # Call LLM in batch
            if self.verbose:
                logger.info(
                    f"Sending {len(prompts_for_batch)} text items to LLM for column '{column_name}'."
                )

            inner_tqdm_config = {
                "desc": f"LLM: Cleaning {config_key}",
                "position": 1,
                "leave": False,
            }

            response_model = AreaModel if config_key == "Area" else None
            cleaned_results_batch = self.llm.generate(
                prompts=prompts_for_batch,
                tqdm_extra_kwargs=inner_tqdm_config,
                response_format=response_model,
            )

            if len(cleaned_results_batch) != len(prompts_for_batch):
                logger.error(
                    f"Mismatch in results count for '{column_name}'. "
                    f"Expected {len(prompts_for_batch)}, got {len(cleaned_results_batch)}. "
                    "Skipping update for this column to prevent data corruption."
                )
                continue  # Skip if counts don't match

            # Process results and update DataFrame
            modifications_count = 0
            for i, cleaned_text_output in enumerate(cleaned_results_batch):
                original_text = original_texts_list[i]
                df_index = original_indices[i]

                if isinstance(cleaned_text_output, BaseModel):
                    cleaned_text_output = getattr(cleaned_text_output, "area", "")

                final_text_to_write = cleaned_text_output  # Assume success initially

                # Logic to revert to original if cleaning "failed" or LLM indicated "N/A"
                if isinstance(cleaned_text_output, str) or pd.isna(cleaned_text_output):
                    if (
                        pd.isna(cleaned_text_output)
                        or cleaned_text_output.startswith("Error:")
                        or cleaned_text_output.startswith("N/A: LLM Error")
                        or cleaned_text_output.startswith("N/A: Unexpected LLM output")
                    ):
                        if self.verbose:
                            logger.info(
                                f"Reverting to original for column '{column_name}', index {df_index}. LLM output: '{cleaned_text_output}'"
                            )
                        final_text_to_write = original_text
                    elif cleaned_text_output != original_text:
                        modifications_count += 1
                elif cleaned_text_output is None and original_text is not None:
                    logger.warning(
                        f"LLM returned None for non-null original text (index {df_index}, col '{column_name}'). Reverting to original."
                    )
                    final_text_to_write = original_text

                cleaned_df.loc[df_index, column_name] = final_text_to_write

            if self.verbose:
                logger.info(
                    f"Finished batch cleaning for '{column_name}'. {modifications_count} entries were actively modified by the LLM."
                )

        self.cleaned_reports = cleaned_df
        return cleaned_df
