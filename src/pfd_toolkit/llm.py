import openai
from openai import RateLimitError
import logging
import base64
from typing import List, Optional, Dict, Type
from pydantic import BaseModel, create_model
import pymupdf
from concurrent.futures import ThreadPoolExecutor, as_completed
import backoff
from threading import Semaphore

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, force=True)

# Set the log level for the 'httpx' library to WARNING to reduce verbosity
logging.getLogger("httpx").setLevel(logging.WARNING)

# Silence the ratelimit package
#logging.getLogger("ratelimit").setLevel(logging.WARNING)

# Disable all logging calls from pfd_toolkit.llm
#logging.getLogger("pfd_toolkit.llm").disabled = True

# Silence the OpenAI client’s info-level logs
#logging.getLogger("openai").setLevel(logging.WARNING)


class LLM:
    
    # Base prompt template that all prompts will share, with placeholders for field-specific information.
    CLEANER_BASE_PROMPT = """\
    You are an expert in extracting and cleaning specific information from UK Coronial Prevention of Future Death Reports.

    Task:
    1. **Extract** only the information related to {field_description}.
    2. **Clean** the input text by removing extraneous details such as rogue numbers, punctuation, HTML tags, or redundant content.
    3. **Correct** any misspellings, ensuring the text is in **British English**.
    4. **Return** exactly and only the cleaned data for {field_contents_and_rules}. You must only return the cleaned string, without adding additional commentary, summarisation, or headings.
    5. **If extraction fails**, return exactly: N/A: Not found (without any additional commentary).
    6. **Do not** change any content of the string unless it explicitly relates to the above. Do not ever summarise, *nor* edit for conciseness or flow.

    Extra instructions:
    {extra_instructions}

    Input Text:
    """

    # Dictionary holding field-specific configurations for the prompt
    # The placeholders for the above `BASE_PROMPT` will be 'filled in' using the values below...
    CLEANER_PROMPT_CONFIG = {
        "Coroner": {
            "field_description": "the name of the Coroner who presided over the inquest",
            "field_contents_and_rules": "this name of the Coroner and nothing else",
            "extra_instructions": (
                'Remove all reference to titles & middle name(s), if present, and replace the first name with an initial.'
                'For example, if the string is "Mr. Joe E Bloggs", return "J. Bloggs".\n'
                'If the string is "Joe Bloggs Senior Coroner for West London", return "J. Bloggs".\n'
                'If the string is "J. Bloggs", just return "J. Bloggs" (no modification).'
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
                "Remove reference to boilerplate text, if any occurs. This is usually 1 or 2 non-specific sentences at the start of the string often ending with '...The Matters of Concern are as follows:' (which should also be removed),"
                "If the string appears to need no cleaning, return it as is."
            ),
        },
    }

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4.1-mini",
        base_url: Optional[str] = None,
        max_workers: int = 1 
    ):
        """Create an LLM object for use within pfd_toolkit

        Args:
            api_key (str): api key for whatever openai sdk llm service you are using.
            model (str): Model name. Defaults to gpt-4o-mini.
            base_url (str): Redirect OpenAI SDK to a different API service.
            max_workers (int): Maximum number of parallel workers for API calls. Defaults to 1.
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url or openai.base_url
        self.client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        
        # Ensure max_workers is at least 1
        self.max_workers = max(1, max_workers)

        # Global semaphore to throttle calls based on max_workers
        self._sem = Semaphore(self.max_workers)

        # Backoff for raw generate calls, handles OpenAI's RateLimitError
        @backoff.on_exception(backoff.expo, RateLimitError, max_time=60)
        def _generate_with_backoff(messages: List[Dict], temperature: float = 0.0) -> str:
            with self._sem:
                return self._raw_generate(messages, temperature)
        self._safe_generate_impl = _generate_with_backoff

        # Backoff for parse endpoint, handles OpenAI's RateLimitError
        @backoff.on_exception(backoff.expo, RateLimitError, max_time=60)
        def _parse_with_backoff(**kwargs):
            with self._sem:
                # Call the client's parse method directly
                return self.client.beta.chat.completions.parse(**kwargs)
        self._parse_with_backoff = _parse_with_backoff

    def _raw_generate(
        self,
        messages: List[Dict],
        temperature: float = 0.0
    ) -> str:
        # Make the API call
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )

        # Record actual usage (for future tuning/logging)
        try:
            used = resp.usage.total_tokens
            logger.debug(f"Actual tokens used: {used}")
        except Exception:
            pass

        # Return content
        return resp.choices[0].message.content.strip()

    def _pdf_bytes_to_base64_images(self, pdf_bytes: bytes, dpi: int = 200) -> list[str]:
        """
        Convert PDF bytes into base64-encoded JPEGs at the given DPI.
        """
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

    #  -- LLM Method for the cleaner.py module --
    def generate_batch(
        self,
        prompts: List[str],
        images_list: Optional[List[List[bytes]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        temperature: float = 0.0,
        max_workers: Optional[int] = None
    ) -> List[BaseModel | str]:
        """
        Manages parallel (or sequential) generation of a list of prompts, returning
        either raw strings or validated BaseModel instances in the same order.
        """
        def _build_messages(prompt: str, imgs: Optional[List[bytes]]):
            content = [{"type": "text", "text": prompt}]
            if imgs:
                for b64_img_data in imgs: 
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64_img_data}"}
                    })
            return [{"role": "user", "content": content}]

        # Determine effective worker count for this batch
        if max_workers is not None and max_workers > 0:
            effective_workers = max_workers
        else:
            effective_workers = self.max_workers 

        # Sequential execution if only one worker is designated
        if effective_workers <= 1:
            results: List[BaseModel | str] = []
            for idx, prompt in enumerate(prompts):
                current_images = images_list[idx] if images_list and idx < len(images_list) else None
                messages = _build_messages(prompt, current_images)

                if response_format:
                    try:
                        resp = self._parse_with_backoff(
                            model=self.model,
                            messages=messages,
                            temperature=temperature,
                            response_format=response_format,
                        )
                        validated = response_format.model_validate_json(
                            resp.choices[0].message.content
                        )
                        results.append(validated)
                    except Exception as e:
                        logger.error(f"Batch pydantic parse failed for item {idx}: {e}")
                        results.append(f"Error: {e}")
                else:
                    txt = self._safe_generate_impl(messages, temperature)
                    results.append(txt)
            return results

        # Parallel execution
        results: List[BaseModel | str] = [None] * len(prompts)

        def _worker(idx: int, prompt_text: str):
            current_images = images_list[idx] if images_list and idx < len(images_list) else None
            messages = _build_messages(prompt_text, current_images)

            if response_format:
                try:
                    resp = self._parse_with_backoff(
                        model=self.model,
                        messages=messages,
                        temperature=temperature,
                        response_format=response_format,
                    )
                    validated = response_format.model_validate_json(
                        resp.choices[0].message.content
                    )
                    return idx, validated
                except Exception as e:
                    logger.error(f"Batch pydantic parse failed for item {idx}: {e}")
                    return idx, f"Error: {e}"
            else:
                txt = self._safe_generate_impl(messages, temperature)
                return idx, txt

        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = [executor.submit(_worker, i, p) for i, p in enumerate(prompts)]
            for fut in as_completed(futures):
                i, out = fut.result()
                results[i] = out

        return results

    # -- LLM method for scraper.py module --
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
        base64_images_list: List[str] = [] # This will be a list of base64 strings
        if pdf_bytes:
            try:
                base64_images_list = self._pdf_bytes_to_base64_images(pdf_bytes, dpi=200)
            except Exception as e:
                logger.error(f"Error converting PDF to images with PyMuPDF: {e}")

        images_for_batch: Optional[List[List[str]]] = None
        if base64_images_list:
            images_for_batch = [base64_images_list]


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
            "'If you are unable to identify the text for any section, respond exactly: \"N/A: Not found\".\n"
            "Transcribe redactions as '[REDACTED]'.\n"
            "Do *not* change section titles. Respond in the specified format.\n"
        )

        schema = {fld: (str, ...) for fld in response_fields}
        MissingModel = create_model("MissingFields", **schema)

        if verbose:
            logger.info("LLM fallback prompt for %s:\n%s", report_url, prompt)

        try:
            # Casting images_for_batch to any to satisfy the potentially problematic type hint
            # without changing the underlying data which is List[List[str]]
            result_list = self.generate_batch(
                prompts=[prompt],
                images_list=images_for_batch, # type: ignore 
                response_format=MissingModel,
                temperature=0.0
            )
            output = result_list[0]
        except Exception as e:
            logger.error(f"LLM fallback call failed: {e}")
            return {}

        if isinstance(output, BaseModel):
            out_json = output.model_dump()
        elif isinstance(output, dict): # Fallback if error string was returned as dict by mistake
            out_json = output
        elif isinstance(output, str) and output.startswith("Error:"): # Handle error string
             logger.error(f"LLM fallback returned an error string: {output}")
             return {fld: "LLM Fallback error" for fld in response_fields}
        else:
            logger.error(f"Unexpected LLM fallback output type: {type(output)}, value: {output}")
            return {fld: "LLM Fallback failed - unexpected type" for fld in response_fields}


        if verbose:
            logger.info("LLM fallback output for %s: %s", report_url, out_json)

        updates: Dict[str, str] = {}
        for fld in response_fields:
            val = out_json.get(fld) # out_json might not be a dict if error occurred above
            if val is None and isinstance(out_json, dict): # Check if out_json is a dict
                 updates[fld] = "N/A: Not found in LLM response" # Field was expected but not in output
            elif val is not None:
                 updates[fld] = str(val) # Ensure value is string
            else: # out_json was not a dict or other issue
                 updates[fld] = "LLM Fallback processing error"
        return updates