# Base prompt template that all prompts will share, with placeholders for field-specific information.
BASE_PROMPT = """\
You are an expert in cleaning string data from UK Coronial Prevention of Future Death Reports.
You will be given a string from just one report that should contain {field_description}.
It may also contain other superfluous information from the web scraping process, such as rogue numbers, punctuation, HTML, or content from a different section.
Your task is to clean the string so that it only contains {field_contents_and_rules}.
{examples}
If for some reason you are unable to complete this task, return: N/A: Not found.
Here is the string:
"""

# Dictionary holding field-specific configurations for the prompt
# Each field has a description, rules for the field contents, and examples.
PROMPT_CONFIG = {
    "Coroner": {
        "field_description": "the name of the Coroner who presided over the inquest",
        "field_contents_and_rules": "only the name of the Coroner and nothing else",
        "examples": (
            "For example, if the string is \"Coroner: Mr. Joe Bloggs\", return \"Joe Bloggs\".\n"
            "If the string is \"Joe Bloggs Senior Coroner for West London\", return \"Joe Bloggs\".\n"
            "If the string is \"Joe Bloggs\", just return \"Joe Bloggs\" (no modification)."
        ),
    },
    "Area": {
        "field_description": "the area where the inquest took place",
        "field_contents_and_rules": "only the name of the area and nothing else",
        "examples": (
            "For example, if the string is \"Area: West London\", return \"West London\".\n"
            "If the string is \"Hampshire, Portsmouth and Southampton\", return it as is."
        ),
    },
    "Receiver": {
        "field_description": "the name(s)/organisation(s) of the receiver(s) of the report",
        "field_contents_and_rules": "only the name(s)/organisation(s) and, if given, their job title(s)—nothing else",
        "examples": (
            "Separate multiple names/organisations with semicolons (;).\n"
            "Do not use a numbered list.\n"
            "Do not separate information with commas or new lines."
        ),
    },
    "InvestigationAndInquest": {
        "field_description": "the details of the investigation and inquest",
        "field_contents_and_rules": "only the details of the investigation and inquest—nothing else",
        "examples": (
            "Correct spelling mistakes and always use British English.\n"
            "If the string appears to need no cleaning, return it as is."
        ),
    },
    "CircumstancesofDeath": {
        "field_description": "the circumstances of death",
        "field_contents_and_rules": "only the circumstances of death—nothing else",
        "examples": (
            "Correct spelling mistakes and always use British English.\n"
            "If the string appears to need no cleaning, return it as is."
        ),
    },
    "MattersOfConcern": {
        "field_description": "the matters of concern",
        "field_contents_and_rules": "only the matters of concern—nothing else",
        "examples": (
            "Correct spelling mistakes and always use British English.\n"
            "If the string appears to need no cleaning, return it as is."
        ),
    },
}

def get_prompt_for_field(field_name: str) -> str:
    """Generates a complete prompt for a given PFD report field based on the BASE_PROMPT and configuration."""
    config = PROMPT_CONFIG[field_name]
    return BASE_PROMPT.format(
        field_description=config["field_description"],
        field_contents_and_rules=config["field_contents_and_rules"],
        examples=config["examples"],
    )

class Cleaner:
    """
    LLM-powered cleaning agent for Prevention of Future Death Reports.
    """
    def __init__(
        self,
        
        # LLM configuration
        llm_model: str = "gpt-4o-mini",
        openai_api_key: str = None,
        
        # Fields to clean
        Coroner: bool = True,
        Receiver: bool = True,
        InvestigationAndInquest: bool = True,
        CircumstancesofDeath: bool = True,
        MattersOfConcern: bool = True,
        
        # Custom prompts for each field
        coroner_prompt: str = None,
        area_prompt: str = None,
        receiver_prompt: str = None,
        investigation_prompt: str = None,
        circumstances_prompt: str = None,
        concerns_prompt: str = None,
        
        verbose: bool = False,
        
    ) -> None:
        
        self.llm_model = llm_model
        self.Coroner = Coroner
        self.Receiver = Receiver
        self.InvestigationAndInquest = InvestigationAndInquest
        self.CircumstancesofDeath = CircumstancesofDeath
        self.MattersOfConcern = MattersOfConcern
        self.verbose = verbose
        
        # The below makes it so that the class instance uses the user-set prompts if provided, or the default ones if not.
        self.coroner_prompt = coroner_prompt if coroner_prompt is not None else get_prompt_for_field("Coroner")
        self.area_prompt = area_prompt if area_prompt is not None else get_prompt_for_field("Area")
        self.receiver_prompt = receiver_prompt if receiver_prompt is not None else get_prompt_for_field("Receiver")
        self.investigation_prompt = investigation_prompt if investigation_prompt is not None else get_prompt_for_field("InvestigationAndInquest")
        self.circumstances_prompt = circumstances_prompt if circumstances_prompt is not None else get_prompt_for_field("CircumstancesofDeath")
        self.concerns_prompt = concerns_prompt if concerns_prompt is not None else get_prompt_for_field("MattersOfConcern")
    
    def clean_coroner(self, text: str) -> str:
        """Clean the coroner field using the generated prompt."""
        prompt = self.coroner_prompt + "\n" + text
        # [here we will call the LLM model]
        #
        return "Placeholder result for coroner cleaning"
    
    def clean_area(self, text: str) -> str:
        """Clean the area field using the generated prompt."""
        prompt = self.area_prompt + "\n" + text
        #
        #
        return "Placeholder result for area cleaning"
    
    def clean_receiver(self, text: str) -> str:
        """Clean the receiver field using the generated prompt."""
        prompt = self.receiver_prompt + "\n" + text
        #
        #
        return "Placeholder result for receiver cleaning"
    
    def clean_investigation(self, text: str) -> str:
        """Clean the investigation field using the generated prompt."""
        prompt = self.investigation_prompt + "\n" + text
        #
        #
        return "Placeholder result for investigation cleaning"
    
    def clean_circumstances(self, text: str) -> str:
        """Clean the circumstances of death field using the generated prompt."""
        prompt = self.circumstances_prompt + "\n" + text
        #
        #
        return "Placeholder result for circumstances cleaning"
    
    def clean_concerns(self, text: str) -> str:
        """Clean the matters of concern field using the generated prompt."""
        prompt = self.concerns_prompt + "\n" + text
        #
        #
        return "Placeholder result for concerns cleaning"