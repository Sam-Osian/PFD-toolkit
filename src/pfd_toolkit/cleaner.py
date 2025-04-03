import logging
import pandas as pd
from dotenv import load_dotenv
import os
from tqdm import tqdm

from pfd_toolkit.llm import LLM, BASE_PROMPT, PROMPT_CONFIG

tqdm.pandas()  # ...initialising tqdm's pandas integration

# -----------------------------------------------------------------------------
# Logging Configuration:
# - Sets up logging for the module. The logger is used to record events,
#   debugging messages, and error messages.
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, force=True)

# Set the log level for the 'httpx' library to WARNING to reduce verbosity
logging.getLogger("httpx").setLevel(logging.WARNING)


class Cleaner:
    """
    LLM-powered cleaning agent for Prevention of Future Death Reports.
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
        """Create Cleaner object.

        Args:
            reports (pd.DataFrame): Input DataFrame containing PFD reports.
            llm (LLM): The LLM client to use for text cleaning.
            coroner (bool): Whether or not to clean the coroner field.
            receiver (bool): Whether or not to clean the receiver field.
            area (bool): Whether or not to clean the area field.
            investigation_and_inquest (bool): Whether or not to clean the InvestigationAndInquest field?.
            circumstances_of_death (bool): Whether or not to clean the CircumstancesOfDeath field.
            matters_of_concern (bool): Whether or not to clean the MattersOfConcern field.
            coroner_field (str): Name of the coroner field, defaults to 'CoronerName'.
            area_field (str): Name of the area field, defaults to 'Area'.
            receiver_field (str): Name of the receiver field, defaults to 'Receiver'.
            investigation_field (str): Name of the ingestigation field, defaults to 'InvestigationAndInquest'.
            circumstances_field (str): Name of the  circumstances of death field, defaults to 'CircumstancesOfDeath'.
            concerns_field (str): Name of the concerns field, defaults to MattersOfConcern.
            coroner_prompt (str): Coroner prompt override. Defaults to hardcoded version.
            area_prompt (str): Area prompt override. Defaults to hardcoded version.
            receiver_prompt (str): Receiver prompt override. Defaults to hardcoded version.
            investigation_prompt (str): Investigation prompt override. Defaults to hardcoded version.
            circumstances_prompt (str): Circumstances prompt override. Defaults to hardcoded version.
            concerns_prompt (str): Concerns prompt override. Defaults to hardcoded version.
            verbose (bool): Whether or not to verbosely run pfd-toolkit. Defaults to False.
        """

        self.reports = reports
        self.llm = llm

        self.coroner = coroner
        self.receiver = receiver
        self.area = area
        self.investigation_and_inquest = investigation_and_inquest
        self.circumstances_of_death = circumstances_of_death
        self.matters_of_concern = matters_of_concern

        self.coroner_field = coroner_field
        self.area_field = area_field
        self.receiver_field = receiver_field
        self.investigation_field = investigation_field
        self.circumstances_field = circumstances_field
        self.concerns_field = concerns_field

        # The below makes it so that the class instance uses the user-set prompts if provided, or the default ones if not.
        self.coroner_prompt = (
            coroner_prompt
            if coroner_prompt is not None
            else Cleaner._get_prompt_for_field("Coroner")
        )
        self.area_prompt = (
            area_prompt
            if area_prompt is not None
            else Cleaner._get_prompt_for_field("Area")
        )
        self.receiver_prompt = (
            receiver_prompt
            if receiver_prompt is not None
            else Cleaner._get_prompt_for_field("Receiver")
        )
        self.investigation_prompt = (
            investigation_prompt
            if investigation_prompt is not None
            else Cleaner._get_prompt_for_field("InvestigationAndInquest")
        )
        self.circumstances_prompt = (
            circumstances_prompt
            if circumstances_prompt is not None
            else Cleaner._get_prompt_for_field("CircumstancesOfDeath")
        )
        self.concerns_prompt = (
            concerns_prompt
            if concerns_prompt is not None
            else Cleaner._get_prompt_for_field("MattersOfConcern")
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
        required_columns = []
        if self.coroner:
            required_columns.append("CoronerName")
        if self.receiver:
            required_columns.append("Receiver")
        if self.investigation_and_inquest:
            required_columns.append("InvestigationAndInquest")
        if self.circumstances_of_death:
            required_columns.append("CircumstancesOfDeath")
        if self.matters_of_concern:
            required_columns.append("MattersOfConcern")
        missing_columns = [
            col for col in required_columns if col not in reports.columns
        ]
        if missing_columns:
            raise ValueError(
                f"Cleaner could not find the following fields in your input DataFrame: {missing_columns}."
            )

    @staticmethod
    def _get_prompt_for_field(field_name: str) -> str:
        """Generates a complete prompt for a given PFD report field based on the BASE_PROMPT and configuration."""
        config = PROMPT_CONFIG[field_name]
        return BASE_PROMPT.format(
            field_description=config["field_description"],
            field_contents_and_rules=config["field_contents_and_rules"],
            extra_instructions=config["extra_instructions"],
        )

    def _clean_coroner(self, text: str) -> str:
        """Clean the coroner field using the generated prompt."""
        prompt = self.coroner_prompt + "\n" + text
        return self.llm.generate(prompt)

    def _clean_area(self, text: str) -> str:
        """Clean the area field using the generated prompt."""
        prompt = self.area_prompt + "\n" + text
        return self.llm.generate(prompt)

    def _clean_receiver(self, text: str) -> str:
        """Clean the receiver field using the generated prompt."""
        prompt = self.receiver_prompt + "\n" + text
        return self.llm.generate(prompt)

    def _clean_investigation(self, text: str) -> str:
        """Clean the investigation field using the generated prompt."""
        prompt = self.investigation_prompt + "\n" + text
        return self.llm.generate(prompt)

    def _clean_circumstances(self, text: str) -> str:
        """Clean the circumstances of death field using the generated prompt."""
        prompt = self.circumstances_prompt + "\n" + text
        return self.llm.generate(prompt)

    def _clean_concerns(self, text: str) -> str:
        """Clean the matters of concern field using the generated prompt."""
        prompt = self.concerns_prompt + "\n" + text
        return self.llm.generate(prompt)

    def _apply_cleaning(self, text: str, cleaning_func) -> str:
        """
        Helper function to apply the cleaning function to a cell's text, and if the cleaned result
        is "N/A: Not found", return the original text instead.
        """
        if pd.isnull(text):
            return text
        cleaned = cleaning_func(text)
        if cleaned == "N/A: Not found":
            return text
        return cleaned

    # Main public function
    def clean_reports(self) -> pd.DataFrame:
        """
        Clean the text fields in a DataFrame of Prevention of Future Death Reports.

        Returns:
            A new DataFrame with the cleaned fields.
        """
        # Make a copy for non-destructive cleaning
        cleaned_df = self.reports.copy()

        # Mapping of field configuration...
        # 1. Field Name
        # 2. Field Boolean (whether to clean)
        # 3. Column Name in input dataframe,
        # 4. Cleaning Function
        fields_to_clean = [
            ("Coroner", self.coroner, self.coroner_field, self._clean_coroner),
            ("Area", self.area, self.area_field, self._clean_area),
            ("Receiver", self.receiver, self.receiver_field, self._clean_receiver),
            (
                "InvestigationAndInquest",
                self.investigation_and_inquest,
                self.investigation_field,
                self._clean_investigation,
            ),
            (
                "CircumstancesOfDeath",
                self.circumstances_of_death,
                self.circumstances_field,
                self._clean_circumstances,
            ),
            (
                "MattersOfConcern",
                self.matters_of_concern,
                self.concerns_field,
                self._clean_concerns,
            ),
        ]
        # Loop over each field and clean it if the corresponding flag is set to True
        for field_name, flag, column_name, cleaning_func in fields_to_clean:
            if flag:
                logger.info(f"Cleaning field: {column_name}")
                cleaned_df[column_name] = cleaned_df[
                    column_name
                ].progress_apply(  # ...apply changed to .progress_apply for tqdm integration
                    lambda text: self._apply_cleaning(text, cleaning_func)
                )

        # Save as internal attribute in case user forgets to assign the output to a variable
        self.cleaned_reports = cleaned_df
        return cleaned_df


# Attach the base prompt and field-specific configurations to the class for easy access
Cleaner.BASE_PROMPT = BASE_PROMPT
Cleaner.PROMPT_CONFIG = PROMPT_CONFIG


# Load OpenAI API key
load_dotenv("api.env")
openai_api_key = os.getenv("OPENAI_API_KEY")

# Create instance of LLM class as required
llm = LLM(api_key=openai_api_key)

# Read reports data frame
reports = pd.read_csv("../data/testreports.csv")
cleaner = Cleaner(
    llm=llm,
    reports=reports,
    coroner=False,
    receiver=False,
    area=False,
    investigation_and_inquest=False,
    circumstances_of_death=False,
    matters_of_concern=True,
)
clean_reports = cleaner.clean_reports()
clean_reports
