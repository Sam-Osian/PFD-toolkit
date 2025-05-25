import logging
import pandas as pd
from tqdm import tqdm

from pfd_toolkit.llm import LLM

# -----------------------------------------------------------------------------
# Logging Configuration:
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, force=True)

# Set the log level for the 'httpx' library to WARNING to reduce verbosity
logging.getLogger("httpx").setLevel(logging.WARNING)
# Silence the OpenAI client’s info-level logs (as in llm.py)
logging.getLogger("openai").setLevel(logging.WARNING)


class Cleaner:
    """Batch-clean PFD report fields with an LLM.

    The cleaner loops over selected columns, builds field-specific prompts,
    calls :pyattr:`llm.generate_batch`, and writes the returned text back into
    a copy of the DataFrame.

    Parameters
    ----------
    reports : pandas.DataFrame
        Input DataFrame returned by :pyclass:`~pfd_toolkit.scraper.PFDScraper`
        or similar.
    llm : LLM
        Client that implements ``generate_batch`` and exposes
        ``CLEANER_BASE_PROMPT`` plus ``CLEANER_PROMPT_CONFIG``.
    coroner, receiver, area, investigation_and_inquest,\
    circumstances_of_death, matters_of_concern : bool, optional
        Flags that decide which columns are processed.
    *_field : str, optional
        Column names for each section; defaults follow the PFDScraper
        convention.
    *_prompt : str or None, optional
        Custom prompt templates.  If *None*, a default is assembled from the
        LLM’s prompt config.
    verbose : bool, optional
        Emit info-level logs for each batch when ``True``.

    Attributes
    ----------
    cleaned_reports : pandas.DataFrame
        Result of the last call to :py:meth:`clean_reports`.
    coroner_prompt_template, area_prompt_template, ... : str
        Finalised prompt strings actually sent to the model.

    Examples
    --------
    >>> cleaner = Cleaner(df, llm, coroner=False, verbose=True)
    >>> cleaned_df = cleaner.clean_reports()
    >>> cleaned_df.head()
    """

    def __init__(
        self,
        # Input DataFrame containing PFD reports
        reports: pd.DataFrame,
        llm: LLM,
        
        # Fields to clean
        coroner: bool = True,
        receiver: bool = True,
        area: bool = True,
        investigation_and_inquest: bool = True,
        circumstances_of_death: bool = True,
        matters_of_concern: bool = True,
        
        # Name of column for each report section; default to the names provided by PFDScraper
        coroner_field: str = "CoronerName",
        area_field: str = "Area",
        receiver_field: str = "Receiver",
        investigation_field: str = "InvestigationAndInquest",
        circumstances_field: str = "CircumstancesOfDeath",
        concerns_field: str = "MattersOfConcern",
        
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
        self.process_coroner = coroner
        self.process_receiver = receiver
        self.process_area = area
        self.process_investigation_and_inquest = investigation_and_inquest
        self.process_circumstances_of_death = circumstances_of_death
        self.process_matters_of_concern = matters_of_concern

        # DataFrame column names
        self.coroner_col_name = coroner_field
        self.area_col_name = area_field
        self.receiver_col_name = receiver_field
        self.investigation_col_name = investigation_field
        self.circumstances_col_name = circumstances_field
        self.concerns_col_name = concerns_field
        
        # Prompt templates
        self.coroner_prompt_template = coroner_prompt or self._get_prompt_for_field("Coroner")
        self.area_prompt_template = area_prompt or self._get_prompt_for_field("Area")
        self.receiver_prompt_template = receiver_prompt or self._get_prompt_for_field("Receiver")
        self.investigation_prompt_template = investigation_prompt or self._get_prompt_for_field("InvestigationAndInquest")
        self.circumstances_prompt_template = circumstances_prompt or self._get_prompt_for_field("CircumstancesOfDeath")
        self.concerns_prompt_template = concerns_prompt or self._get_prompt_for_field("MattersOfConcern")
        
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
        if self.process_coroner: required_df_columns.append(self.coroner_col_name)
        if self.process_area: required_df_columns.append(self.area_col_name)
        if self.process_receiver: required_df_columns.append(self.receiver_col_name)
        if self.process_investigation_and_inquest: required_df_columns.append(self.investigation_col_name)
        if self.process_circumstances_of_death: required_df_columns.append(self.circumstances_col_name)
        if self.process_matters_of_concern: required_df_columns.append(self.concerns_col_name)
        
        # Get unique column names in case user mapped multiple flags to the same df column
        required_df_columns = list(set(required_df_columns))
        
        missing_columns = [col for col in required_df_columns if col not in self.reports.columns]
        if missing_columns:
            raise ValueError(f"Cleaner could not find the following DataFrame columns: {missing_columns}.")
    
    def _get_prompt_for_field(self, field_name: str) -> str:
        """Generates a complete prompt template for a given PFD report field."""
        # Access PROMPT_CONFIG and BASE_PROMPT from the llm instance
        config = self.llm.CLEANER_PROMPT_CONFIG[field_name] 
        return self.llm.CLEANER_BASE_PROMPT.format( 
            field_description=config["field_description"],
            field_contents_and_rules=config["field_contents_and_rules"],
            extra_instructions=config["extra_instructions"],
        )

    def clean_reports(self) -> pd.DataFrame:
        """Run LLM-based cleaning for the configured columns.

        The method operates **in place on a copy** of
        :pyattr:`self.reports`, so the original DataFrame is never mutated.

        Returns
        -------
        pandas.DataFrame
            A new DataFrame in which the selected columns have been
            replaced by the LLM output (or left unchanged when the model
            returns an error marker).

        Examples
        --------
        >>> cleaned = cleaner.clean_reports()
        >>> cleaned.equals(df)
        False
        """
        cleaned_df = self.reports.copy() # Work on a copy
        
        # Define fields to process: (Config Key, Process Flag, DF Column Name, Prompt Template)
        field_processing_config = [
            ("Coroner", self.process_coroner, self.coroner_col_name, self.coroner_prompt_template),
            ("Area", self.process_area, self.area_col_name, self.area_prompt_template),
            ("Receiver", self.process_receiver, self.receiver_col_name, self.receiver_prompt_template),
            ("InvestigationAndInquest", self.process_investigation_and_inquest, self.investigation_col_name, self.investigation_prompt_template),
            ("CircumstancesOfDeath", self.process_circumstances_of_death, self.circumstances_col_name, self.circumstances_prompt_template),
            ("MattersOfConcern", self.process_matters_of_concern, self.concerns_col_name, self.concerns_prompt_template),
        ]

        # Use tqdm for the outer loop over fields
        for config_key, process_flag, column_name, prompt_template in tqdm(field_processing_config, desc="Processing Fields"):
            if not process_flag:
                continue

            if column_name not in cleaned_df.columns:
                # This case should ideally be caught by __init__ checks, but good to have defence here
                logger.warning(f"Column '{column_name}' for field '{config_key}' not found at cleaning time. Skipping.")
                continue
            if self.verbose:
                logger.info(f"Preparing batch for column: '{column_name}' (Field: {config_key})")

            # Ensure column is treated as string for processing
            # Handle cases where column might be all NaNs or mixed type before attempting string operations
            if cleaned_df[column_name].notna().any():
                 if not pd.api.types.is_string_dtype(cleaned_df[column_name]):
                    cleaned_df[column_name] = cleaned_df[column_name].astype(str)
            else:
                logger.info(f"Column '{column_name}' contains all NaN values. No text to clean.")
                continue # Skip to next field if column is all NaN

            # Select non-null texts to clean and their original indices
            # Ensure we are working with string representations for LLM processing
            texts_to_clean_series = cleaned_df[column_name][cleaned_df[column_name].notna()].astype(str)
            
            if texts_to_clean_series.empty:
                logger.info(f"No actual text data to clean in column '{column_name}' after filtering NaNs. Skipping.")
                continue

            original_indices = texts_to_clean_series.index
            original_texts_list = texts_to_clean_series.tolist()

            # Construct prompts for the batch
            # Each prompt consists of the field-specific template followed by the actual text
            prompts_for_batch = [f"{prompt_template}\n{text}" for text in original_texts_list]
            
            if not prompts_for_batch: # Should not happen if texts_to_clean_series was not empty
                logger.info(f"No prompts generated for column '{column_name}'. Skipping LLM call.")
                continue

            if self.verbose:
                logger.info(f"First prompt for '{column_name}' batch: {prompts_for_batch[0][:250]}...") # Log snippet of first prompt

            # Call LLM in batch
            if self.verbose:
                logger.info(f"Sending {len(prompts_for_batch)} text items to LLM for column '{column_name}'.")
            
            cleaned_results_batch = self.llm.generate_batch(prompts=prompts_for_batch) 

            if len(cleaned_results_batch) != len(prompts_for_batch):
                logger.error(
                    f"Mismatch in results count for '{column_name}'. "
                    f"Expected {len(prompts_for_batch)}, got {len(cleaned_results_batch)}. "
                    "Skipping update for this column to prevent data corruption."
                )
                continue # Skip if counts don't match

            # Process results and update DataFrame
            modifications_count = 0
            for i, cleaned_text_output in enumerate(cleaned_results_batch):
                original_text = original_texts_list[i]
                df_index = original_indices[i]

                final_text_to_write = cleaned_text_output # Assume success initially

                # Logic to revert to original if cleaning "failed" or LLM indicated "N/A"
                if isinstance(cleaned_text_output, str):
                    if cleaned_text_output == "N/A: Not found" or \
                       cleaned_text_output.startswith("Error:") or \
                       cleaned_text_output.startswith("N/A: LLM Error") or \
                       cleaned_text_output.startswith("N/A: Unexpected LLM output"): # Match potential error strings from llm.py
                        if self.verbose:
                            logger.info(f"Reverting to original for column '{column_name}', index {df_index}. LLM output: '{cleaned_text_output}'")
                        final_text_to_write = original_text # Revert to original
                    elif cleaned_text_output != original_text:
                        modifications_count += 1
                elif cleaned_text_output is None and original_text is not None: # If LLM returned None for actual text
                    logger.warning(f"LLM returned None for non-null original text (index {df_index}, col '{column_name}'). Reverting to original.")
                    final_text_to_write = original_text # Revert
                
                cleaned_df.loc[df_index, column_name] = final_text_to_write
            
            if self.verbose:
                logger.info(f"Finished batch cleaning for '{column_name}'. {modifications_count} entries were actively modified by the LLM.")
        
        self.cleaned_reports = cleaned_df 
        return cleaned_df