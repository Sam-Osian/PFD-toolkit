import openai
import logging
from typing import List, Optional
from pydantic import BaseModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, force=True)

# Set the log level for the 'httpx' library to WARNING to reduce verbosity
logging.getLogger("httpx").setLevel(logging.WARNING)


class LLM:
    def __init__(self, api_key: str, model: str = "gpt-4o-mini", base_url: str = None):
        """Create an LLM object for use within PFD_Toolkit

        Args:
            api_key (str): api key for whatever openai sdk llm service you are using.
            model (str): Model name. Defaults to gpt-4o-mini.
            base_url (str): Set this to redirect openai sdk to a different api service. For example, Fireworks.ai, Groq, Ollama, Text-Generation-Inference. Defaults to None (openai).
        """

        self.api_key = api_key
        self.model = model

        # Set the base_url, if non is provided then we just set self.base_url to the openai default (OpenAI themselves).
        if base_url:
            self.base_url = base_url
        else:
            self.base_url = openai.base_url
        self.client = openai.Client(api_key=self.api_key, base_url=base_url)

    def generate(
        self,
        prompt: str,
        images: Optional[List[bytes]] = None,
        response_format: Optional[BaseModel] = None,
        temperature: float = 0.0,
    ) -> str | BaseModel:
        """Generate response to given input prompt

        Args:
            prompt (str): The prompt to pass to the LLM.
            images (Optional[List[bytes]]): Byes for images to pass to the LLM. Defaults to None.
            response_format (BaseModel): Pass a class name that inherits from pydantic BaseModel if you wish to use guided outputs. Defaults to None.
            temperature (float): The temperature to use.
        """

        messages = [{"role": "user", "content": prompt}]
        if images:
            # Add images to messages if given.
            for b64_img in images:
                messages.append(
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"},
                    }
                )
        if response_format:
            try:
                # For guided outputs, you have to use beta.chat.completions.parse to use response_format.
                response = self.client.beta.chat.completions.parse(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                    response_format=response_format,
                )
                return response_format.model_validate_json(
                    response.choices[0].message.content
                )
            except Exception as e:
                logger.error(
                    f"LLM was unable to complete generation request, or incorrect response format was produced: {e}"
                )
                return "Error: LLM Failed."
        else:
            try:
                # Normal text generation.
                response = self.client.chat.completions.create(
                    model=self.model, messages=messages, temperature=temperature,
                )
                # Extract the cleaned string from the response
                response_str = response.choices[0].message.content.strip()
                return response_str

            except Exception as e:
                logger.error(f"An error occurred while calling the LLM model: {e}")
                return "Error: LLM Failed."


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
