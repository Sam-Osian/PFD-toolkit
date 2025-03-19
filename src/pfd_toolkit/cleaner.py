import logging
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os
from tqdm import tqdm

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

# Base prompt template that all prompts will share, with placeholders for field-specific information.
BASE_PROMPT = """\
You are an expert in extracting and cleaning specific information from UK Coronial Prevention of Future Death Reports.

Task:
1. **Extract** only the information related to {field_description}.
2. **Clean** the input text by removing extraneous details such as rogue numbers, punctuation, HTML tags, or redundant content.
3. **Correct** any misspellings, ensuring the text is in **British English**.
4. **Return** exactly and only the cleaned data for {field_contents_and_rules}. You must only return the cleaned string, without adding additional commentary, summarisation, or headings.
5. **If extraction fails**, return exactly: N/A: Not found (without any additional commentary).

Extra instructions:
{extra_instructions}

Input Text:
"""


# Dictionary holding field-specific configurations for the prompt
# The placeholders for the above `BASE_PROMPT` will be 'filled in' using the values below...
PROMPT_CONFIG = {
    "Coroner": {
        "field_description": "the name of the Coroner who presided over the inquest",
        "field_contents_and_rules": "the name of the Coroner and nothing else",
        "extra_instructions": (
            'For example, if the string is "Coroner: Mr. Joe Bloggs", return "Joe Bloggs".\n'
            'If the string is "Joe Bloggs Senior Coroner for West London", return "Joe Bloggs".\n'
            'If the string is "Joe Bloggs", just return "Joe Bloggs" (no modification).'
        ),
    },
    "Area": {
        "field_description": "the area where the inquest took place",
        "field_contents_and_rules": "only the name of the area and nothing else",
        "extra_instructions": (
            'For example, if the string is "Area: West London", return "West London".\n'
            'If the string is "Hampshire, Portsmouth and Southampton", return it as is.'
        ),
    },
    "Receiver": {
        "field_description": "the name(s)/organisation(s) of the receiver(s) of the report",
        "field_contents_and_rules": "only the name(s)/organisation(s) and, if given, their job title(s) and nothing else",
        "extra_instructions": (
            "Separate multiple names/organisations with semicolons (;)."
            "Do not use a numbered list."
            "Do not separate information with commas or new lines."
        ),
    },
    "InvestigationAndInquest": {
        "field_description": "the details of the investigation and inquest",
        "field_contents_and_rules": "only the details of the investigation and inquest—nothing else",
        "extra_instructions": (
            "If the string appears to need no cleaning, return it as is."
            "Change all dates to the format 'YYYY-MM-DD'"
        ),
    },
    "CircumstancesOfDeath": {
        "field_description": "the circumstances of death",
        "field_contents_and_rules": "only the circumstances of death—nothing else",
        "extra_instructions": (
            "If the string appears to need no cleaning, return it as is."
            "Change all dates to the format 'YYYY-MM-DD'"
        ),
    },
    "MattersOfConcern": {
        "field_description": "the matters of concern",
        "field_contents_and_rules": "only the matters of concern—nothing else",
        "extra_instructions": (
            "Remove reference to boiletplate text, if any occurs. This is usually 1 or 2 non-specific sentences at the start of the string ending with '...The Matters of Concern are as follows:',"
            "If the string appears to need no cleaning, return it as is."
            "Change all dates to the format 'YYYY-MM-DD'"
        ),
    },
}


class Cleaner:
    """
    LLM-powered cleaning agent for Prevention of Future Death Reports.
    """

    def __init__(
        self,
        # Input DataFrame containing PFD reports
        reports: pd.DataFrame,
        # LLM configuration
        llm_model: str = "gpt-4o-mini",
        openai_api_key: str = None,
        openai_client: OpenAI = None,
        # Fields to clean
        Coroner: bool = True,
        Receiver: bool = True,
        Area: bool = True,
        InvestigationAndInquest: bool = True,
        CircumstancesOfDeath: bool = True,
        MattersOfConcern: bool = True,
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
        self.llm_model = llm_model
        self.openai_api_key = openai_api_key

        # Use injected LLM client if provided; otherwise, create one from the API key.
        # This allows the user to pass in their own OpenAI client instance as an alternative to supplying an API key.
        if openai_client is not None:
            self.openai_client = openai_client
        elif openai_api_key is not None:
            self.openai_client = OpenAI(api_key=openai_api_key)
        else:
            raise ValueError("Either openai_client or openai_api_key must be provided.")

        self.Coroner = Coroner
        self.Receiver = Receiver
        self.Area = Area
        self.InvestigationAndInquest = InvestigationAndInquest
        self.CircumstancesOfDeath = CircumstancesOfDeath
        self.MattersOfConcern = MattersOfConcern

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
        if self.Coroner:
            required_columns.append("CoronerName")
        if self.Receiver:
            required_columns.append("Receiver")
        if self.InvestigationAndInquest:
            required_columns.append("InvestigationAndInquest")
        if self.CircumstancesOfDeath:
            required_columns.append("CircumstancesOfDeath")
        if self.MattersOfConcern:
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

    def _call_llm(self, prompt: str) -> str:
        """Call the OpenAI API to generate a cleaned string based on the given prompt."""

        try:
            response = self.openai_client.chat.completions.create(
                model=self.llm_model,
                messages=[{"role": "system", "content": prompt},],
                temperature=0.0,
            )
            # Extract the cleaned string from the response
            cleaned_string = response.choices[0].message.content.strip()
            return cleaned_string

        except Exception as e:
            logger.error(f"An error occurred while calling the LLM model: {e}")
            return "Error: Could not clean string"

    def _clean_coroner(self, text: str) -> str:
        """Clean the coroner field using the generated prompt."""
        prompt = self.coroner_prompt + "\n" + text
        return self._call_llm(prompt)

    def _clean_area(self, text: str) -> str:
        """Clean the area field using the generated prompt."""
        prompt = self.area_prompt + "\n" + text
        return self._call_llm(prompt)

    def _clean_receiver(self, text: str) -> str:
        """Clean the receiver field using the generated prompt."""
        prompt = self.receiver_prompt + "\n" + text
        return self._call_llm(prompt)

    def _clean_investigation(self, text: str) -> str:
        """Clean the investigation field using the generated prompt."""
        prompt = self.investigation_prompt + "\n" + text
        return self._call_llm(prompt)

    def _clean_circumstances(self, text: str) -> str:
        """Clean the circumstances of death field using the generated prompt."""
        prompt = self.circumstances_prompt + "\n" + text
        return self._call_llm(prompt)

    def _clean_concerns(self, text: str) -> str:
        """Clean the matters of concern field using the generated prompt."""
        prompt = self.concerns_prompt + "\n" + text
        return self._call_llm(prompt)

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
            ("Coroner", self.Coroner, self.coroner_field, self._clean_coroner),
            ("Area", self.Area, self.area_field, self._clean_area),
            ("Receiver", self.Receiver, self.receiver_field, self._clean_receiver),
            (
                "InvestigationAndInquest",
                self.InvestigationAndInquest,
                self.investigation_field,
                self._clean_investigation,
            ),
            (
                "CircumstancesOfDeath",
                self.CircumstancesOfDeath,
                self.circumstances_field,
                self._clean_circumstances,
            ),
            (
                "MattersOfConcern",
                self.MattersOfConcern,
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

# Read reports data frame
reports = pd.read_csv("../../data/testreports.csv")
cleaner = Cleaner(
    reports=reports,
    openai_api_key=openai_api_key,
    Coroner=False,
    Receiver=False,
    Area=False,
    InvestigationAndInquest=False,
    CircumstancesOfDeath=False,
    MattersOfConcern=True,
)
clean_reports = cleaner.clean_reports()
clean_reports
