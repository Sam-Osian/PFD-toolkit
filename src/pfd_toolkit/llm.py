import openai
import logging
import base64
from typing import List, Optional, Dict, Tuple, Type, Union
from pydantic import BaseModel, create_model
import pymupdf
from ratelimit import limits, RateLimitException
from ratelimit import sleep_and_retry
from concurrent.futures import ThreadPoolExecutor, as_completed


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, force=True)

# Set the log level for the 'httpx' library to WARNING to reduce verbosity
logging.getLogger("httpx").setLevel(logging.WARNING)


class LLM:
    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        parallelise: bool=False,
        rpm_limit: Optional[int] = 300,
        max_workers: Optional[int] = None
        ):
        """Create an LLM object for use within PFD_Toolkit

        Args:
            api_key (str): api key for whatever openai sdk llm service you are using.
            model (str): Model name. Defaults to gpt-4o-mini.
            base_url (str): Set this to redirect openai sdk to a different api service. For example, Fireworks.ai, Groq, Ollama, Text-Generation-Inference. Defaults to None (openai).
            rpm_limit: Requests per minute (overrides defaults).
        """

        self.api_key = api_key
        self.model = model
        self.base_url = base_url or openai.base_url
        self.client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        self.parallelise = parallelise
        self.rpm_limit = rpm_limit
        
        # If user did not supply max_workers, calculate a sensible default
        # (requests/sec * avg_latency)
        avg_latency = 0.5  # seconds per call, tune as you measure
        auto_workers = max(int(self.rpm_limit / 60 * avg_latency), 1)
        self.max_workers = max_workers or auto_workers

        # Build a rate-limited version of the raw generate
        limiter = limits(calls=self.rpm_limit, period=60)
        self._safe_generate_impl = sleep_and_retry(limiter)(self._raw_generate)


    def _raw_generate(
        self,
        messages: List[Dict],
        temperature: float = 0.0
    ) -> str:
        """
        Low-level single request to the OpenAI chat endpoint.
        """
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )
        return resp.choices[0].message.content.strip()

    def _pdf_bytes_to_base64_images(self, pdf_bytes: bytes, dpi: int = 200) -> list[str]:
        """
        Convert PDF bytes into base64‑encoded JPEGs at the given DPI.
        """
        # Open the PDF
        doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")

        zoom = dpi / 72
        mat = pymupdf.Matrix(zoom, zoom)

        imgs: list[str] = []
        for page in doc:
            pix = page.get_pixmap(matrix=mat, alpha=False)
            img_bytes = pix.tobytes("jpeg")
            b64 = base64.b64encode(img_bytes).decode("utf-8")
            imgs.append(b64)

        doc.close()
        return imgs
    
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
        content = [{"type": 'text', "text": prompt}]
        if images:
            for b64_img in images:
                content.append({'type': "image_url", "image_url": {"url": f"data:image/jpeg;base64,{b64_img}"}})
        messages = [{"role": "user", "content": content}]
        
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
                # Normal text generation via rate-limited helper
                return self._safe_generate_impl(messages, temperature)
            except RateLimitException as e:
                logger.error(f"Rate limit reached: {e}")
                raise
            except Exception as e:
                logger.error(f"An error occurred while calling the LLM model: {e}")
                return "Error: LLM Failed."


    def generate_batch(
            self,
            prompts: List[str],
            images_list: Optional[List[List[bytes]]] = None,
            response_format: Optional[Type[BaseModel]] = None,
            max_workers: Optional[int] = None
        ) -> List[Union[str, BaseModel]]:
            """
            Parallel generation of a list of prompts, returning results in order.
            Each thread goes through the same RPM limiter.
            """
            workers = max_workers or self.max_workers or len(prompts)
            results: List[Optional[str]] = [None] * len(prompts)

            def worker(idx, prompt):
                imgs = images_list[idx] if images_list else None
                if response_format:
                    return idx, self.generate(prompt, images=imgs, response_format=response_format)
                else:
                    return idx, self.generate(prompt, images=imgs)

            with ThreadPoolExecutor(max_workers=workers) as executor:
                futures = [executor.submit(worker, i, p) for i, p in enumerate(prompts)]
                for fut in as_completed(futures):
                    i, txt = fut.result()
                    results[i] = txt

            return results


    # Main method for calling the LLM for missing fields in the Scraper module
    def call_llm_fallback(
        self,
        pdf_bytes: Optional[bytes],
        missing_fields: Dict[str, str],
        report_url: Optional[str] = None,
        verbose: bool = False,
    ) -> Dict[str, str]:
        """
        Use the LLM to extract text from PDF images for missing fields.

        Args:
            pdf_bytes (bytes): Raw PDF bytes.
            missing_fields (dict): Mapping of field names to prompt instructions.
            report_url (str, optional): URL of the report, for logging.
            verbose (bool): If True, log prompt and output.

        Returns:
            dict: Extracted values keyed by the original field names.
        """
        # 1) Convert PDF bytes to base64 images
        base64_images: List[str] = []
        if pdf_bytes:
            try:
                base64_images = self._pdf_bytes_to_base64_images(pdf_bytes, dpi=200)
            except Exception as e:
                logger.error(f"Error converting PDF to images with PyMuPDF: {e}")

        # 2) Build the prompt
        prompt = (
            "Your goal is to transcribe the **exact** text from this report, presented as images.\n\n"
            "Please extract the following section(s):\n"
        )
        response_fields: List[str] = []
        for field, instruction in missing_fields.items():
            response_fields.append(field)
            prompt += f"\n{field}: {instruction}\n"
        prompt += (
            "\nRespond with nothing else whatsoever. You must not respond in your own 'voice'...\n"
            'If you are unable to identify the text for any section, respond exactly: "N/A: Not found".\n'
            "Transcribe redactions as '[REDACTED]'.\n"
            "Do *not* change section titles. Respond in the specified format.\n"
        )

        # 3) Create a dynamic pydantic model for the expected keys
        schema = {fld: (str, ...) for fld in response_fields}
        MissingModel = create_model("MissingFields", **schema)

        if verbose:
            logger.info("LLM fallback prompt for %s:\n%s", report_url, prompt)

        # 4) Invoke the LLM
        try:
            output = self.generate(
                prompt=prompt,
                images=base64_images,
                response_format=MissingModel,
            )
        except Exception as e:
            logger.error(f"LLM fallback call failed: {e}")
            return {}

        # 5) Normalize output to dict
        if isinstance(output, BaseModel):
            out_json = output.model_dump()
        elif isinstance(output, dict):
            out_json = output
        else:
            logger.error(f"Unexpected LLM fallback output type: {type(output)}")
            return {}

        if verbose:
            logger.info("LLM fallback output for %s: %s", report_url, out_json)

        # 6) Build fallback_updates
        updates: Dict[str, str] = {}
        for fld in response_fields:
            val = out_json.get(fld)
            updates[fld] = val if val is not None else "LLM Fallback failed"
        return updates



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
