import openai
import logging
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
        if base_url:
            self.base_url = base_url
        else:
            self.base_url = openai.base_url
        self.client = openai.Client(api_key=self.api_key, base_url=base_url)

    def generate(
        self, prompt: str, response_format: BaseModel = None, temperature: float = 0.0
    ) -> str | BaseModel:
        """Generate response to given input prompt

        Args:
            prompt (str): The prompt to pass to the LLM.
            response_format (BaseModel): Pass a class name that inherits from pydantic BaseModel if you wish to use guided outputs. Defaults to None.
            temperature (float): The temperature to use.
        """

        messages = [{"role": "user", "content": prompt}]
        if response_format:
            try:
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
                response = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                )
                # Extract the cleaned string from the response
                response_str = response.choices[0].message.content.strip()
                return response_str
            except Exception as e:
                logger.error(f"An error occurred while calling the LLM model: {e}")
                return "Error: LLM Failed."
