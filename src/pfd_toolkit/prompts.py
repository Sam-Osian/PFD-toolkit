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
