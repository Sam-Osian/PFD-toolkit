import logging
import pandas as pd
from openai import OpenAI
from dotenv import load_dotenv
import os

# -----------------------------------------------------------------------------
# Logging Configuration:
# - Sets up logging for the module. The logger is used to record events,
#   debugging messages, and error messages.
# -----------------------------------------------------------------------------
logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, force=True)


# Base prompt template that all prompts will share, with placeholders for field-specific information.
BASE_PROMPT = """\
You are an expert in extracting and cleaning specific information from UK Coronial Prevention of Future Death Reports.

Task:
1. **Extract** only the information related to {field_description}.
2. **Clean** the input text by removing extraneous details such as rogue numbers, punctuation, HTML tags, or redundant content.
3. **Correct** any misspellings, ensuring the text is in **British English**.
4. **Return** exactly and only the cleaned data for {field_contents_and_rules}.
5. **If extraction fails**, return exactly: "N/A: Not found" (without any additional commentary).

Extra instructions:
{extra_instructions}

Input Text:
"""


# Dictionary holding field-specific configurations for the prompt
# Each field has a description, rules for the field contents, and extra_instructions.
PROMPT_CONFIG = {
    "Coroner": {
        "field_description": "the name of the Coroner who presided over the inquest",
        "field_contents_and_rules": "the name of the Coroner and nothing else",
        "extra_instructions": (
            "For example, if the string is \"Coroner: Mr. Joe Bloggs\", return \"Joe Bloggs\".\n"
            "If the string is \"Joe Bloggs Senior Coroner for West London\", return \"Joe Bloggs\".\n"
            "If the string is \"Joe Bloggs\", just return \"Joe Bloggs\" (no modification)."
        ),
    },
    "Area": {
        "field_description": "the area where the inquest took place",
        "field_contents_and_rules": "only the name of the area and nothing else",
        "extra_instructions": (
            "For example, if the string is \"Area: West London\", return \"West London\".\n"
            "If the string is \"Hampshire, Portsmouth and Southampton\", return it as is."
        ),
    },
    "Receiver": {
        "field_description": "the name(s)/organisation(s) of the receiver(s) of the report",
        "field_contents_and_rules": "only the name(s)/organisation(s) and, if given, their job title(s) and nothing else",
        "extra_instructions": (
            "Separate multiple names/organisations with semicolons (;).\n"
            "Do not use a numbered list.\n"
            "Do not separate information with commas or new lines."
        ),
    },
    "InvestigationAndInquest": {
        "field_description": "the details of the investigation and inquest",
        "field_contents_and_rules": "only the details of the investigation and inquest—nothing else",
        "extra_instructions": (
            "If the string appears to need no cleaning, return it as is."
        ),
    },
    "CircumstancesOfDeath": {
        "field_description": "the circumstances of death",
        "field_contents_and_rules": "only the circumstances of death—nothing else",
        "extra_instructions": (
            "If the string appears to need no cleaning, return it as is."
        ),
    },
    "MattersOfConcern": {
        "field_description": "the matters of concern",
        "field_contents_and_rules": "only the matters of concern—nothing else",
        "extra_instructions": (
            "Remove reference to boiletplate text, if any occurs. This is usually 1 or 2 non-specific sentences ending with '...The Matters of Concern are as follows:',"
            "If the string appears to need no cleaning, return it as is."
        ),
    },
}

def _get_prompt_for_field(field_name: str) -> str:
    """Generates a complete prompt for a given PFD report field based on the BASE_PROMPT and configuration."""
    config = PROMPT_CONFIG[field_name]
    return BASE_PROMPT.format(
        field_description=config["field_description"],
        field_contents_and_rules=config["field_contents_and_rules"],
        extra_instructions=config["extra_instructions"],
    )

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
        
        # Fields to clean
        Coroner: bool = True,
        Receiver: bool = True,
        Area: bool=True,
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
        self.coroner_prompt = coroner_prompt if coroner_prompt is not None else _get_prompt_for_field("Coroner")
        self.area_prompt = area_prompt if area_prompt is not None else _get_prompt_for_field("Area")
        self.receiver_prompt = receiver_prompt if receiver_prompt is not None else _get_prompt_for_field("Receiver")
        self.investigation_prompt = investigation_prompt if investigation_prompt is not None else _get_prompt_for_field("InvestigationAndInquest")
        self.circumstances_prompt = circumstances_prompt if circumstances_prompt is not None else _get_prompt_for_field("CircumstancesOfDeath")
        self.concerns_prompt = concerns_prompt if concerns_prompt is not None else _get_prompt_for_field("MattersOfConcern")
        
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
        missing_columns = [col for col in required_columns if col not in reports.columns]
        if missing_columns:
            raise ValueError(f"Cleaner could not find the following fields in your input DataFrame: {missing_columns}.")
    
    
    def _call_llm(self, prompt: str) -> str:
        """Call the OpenAI API to generate a cleaned string based on the given prompt."""
        
        try:
            response = client.chat.completions.create(
                model=self.llm_model,
                messages=[
                    {"role": "system", "content": prompt},
                ], temperature=0.0,
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

        if self.Coroner:
            logger.info(f"Cleaning field: {self.coroner_field}")
            cleaned_df[self.coroner_field] = cleaned_df[self.coroner_field].apply(
                lambda text: self._apply_cleaning(text, self._clean_coroner)
            )

        if self.Area:
            logger.info(f"Cleaning field: {self.area_field}")
            cleaned_df[self.area_field] = cleaned_df[self.area_field].apply(
                lambda text: self._apply_cleaning(text, self._clean_area)
            )
        
        if self.Receiver:
            logger.info(f"Cleaning field: {self.receiver_field}")
            cleaned_df[self.receiver_field] = cleaned_df[self.receiver_field].apply(
                lambda text: self._apply_cleaning(text, self._clean_receiver)
            )

        if self.InvestigationAndInquest:
            logger.info(f"Cleaning field: {self.investigation_field}")
            cleaned_df[self.investigation_field] = cleaned_df[self.investigation_field].apply(
                lambda text: self._apply_cleaning(text, self._clean_investigation)
            )

        if self.CircumstancesOfDeath:
            logger.info(f"Cleaning field: {self.circumstances_field}")
            cleaned_df[self.circumstances_field] = cleaned_df[self.circumstances_field].apply(
                lambda text: self._apply_cleaning(text, self._clean_circumstances)
            )

        if self.MattersOfConcern:
            logger.info(f"Cleaning field: {self.concerns_field}")
            cleaned_df[self.concerns_field] = cleaned_df[self.concerns_field].apply(
                lambda text: self._apply_cleaning(text, self._clean_concerns)
            )
        
        # Save as internal attribute in case user forgets to assign the output to a variable
        self.cleaned_reports = cleaned_df
        return cleaned_df


# Load OpenAI API key
load_dotenv('api.env')
openai_api_key = os.getenv('OPENAI_API_KEY')
client = OpenAI(api_key=openai_api_key)


# Read reports data frame
reports = pd.read_csv('../../data/testreports.csv')
cleaner = Cleaner(reports=reports, 
                  Coroner=False, 
                  Receiver=False, 
                  Area=False, 
                  InvestigationAndInquest=True, 
                  CircumstancesOfDeath=False, 
                  MattersOfConcern=False)
clean_reports = cleaner.clean_reports()
clean_reports

cleaner.cleaned_reports