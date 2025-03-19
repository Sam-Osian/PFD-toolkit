import openai
import logging
from pydantic import BaseModel

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, force=True)

# Set the log level for the 'httpx' library to WARNING to reduce verbosity
logging.getLogger("httpx").setLevel(logging.WARNING)


class LLM:
    def __init__(self, api_key: str, model: str, base_url: str = None):
        self.api_key = api_key
        self.model = model
        if base_url:
            self.base_url = base_url
        else:
            self.base_url = openai.base_url
        self.client = openai.Client(api_key=self.api_key, base_url=base_url)

    def generate(self, prompt: str, response_format: BaseModel = None) -> str:
        messages = [{"role": "user", "content": prompt}]
        if response_format:
            try:
                response = self.client.beta.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.0,
                    response_format=response_format,
                )
                return response
            except Exception as e:
                logger.error(
                    f"LLM was unable to complete generation request with response format: {e}"
                )
                return "Error: LLM Failed."
        else:
            try:
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=0.0,
                )
                # Extract the cleaned string from the response
                response_str = response.choices[0].message.content.strip()
                return response_str
            except Exception as e:
                logger.error(f"An error occurred while calling the LLM model: {e}")
                return "Error: LLM Failed."
