import openai
from openai import RateLimitError
import logging
import base64
import time
from typing import List, Optional, Dict, Type
from pydantic import BaseModel, create_model
import pymupdf
from concurrent.futures import ThreadPoolExecutor, as_completed
from ratelimit import limits, RateLimitException, sleep_and_retry
import backoff
from threading import Semaphore, Lock

# Helper to estimate token usage

def estimate_tokens(messages: List[Dict]) -> int:
    """Rough heuristic for token count: 1 token per 4 characters of text."""
    total_chars = 0
    for msg in messages:
        content = msg.get("content", [])
        if isinstance(content, list):
            for part in content:
                if part.get("type") == "text":
                    total_chars += len(part.get("text", ""))
        else:
            total_chars += len(str(content))
    return max(1, total_chars // 4)


logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, force=True)

# Set the log level for the 'httpx' library to WARNING to reduce verbosity
logging.getLogger("httpx").setLevel(logging.WARNING)

# Silence the ratelimit package
logging.getLogger("ratelimit").setLevel(logging.WARNING)

# Disable all logging calls from pfd_toolkit.llm
logging.getLogger("pfd_toolkit.llm").disabled = True

# Silence the OpenAI client’s info-level logs
logging.getLogger("openai").setLevel(logging.WARNING)


class LLM:
    
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

    def __init__(
        self,
        api_key: str,
        model: str = "gpt-4o-mini",
        base_url: Optional[str] = None,
        parallelise: bool = False,
        rpm_limit: Optional[int] = 300,
        tpm_limit: Optional[int] = 40000,
        max_workers: Optional[int] = None
    ):
        """Create an LLM object for use within pdf_toolkit

        Args:
            api_key (str): api key for whatever openai sdk llm service you are using.
            model (str): Model name. Defaults to gpt-4o-mini.
            base_url (str): Redirect OpenAI SDK to a different API service.
            parallelise (bool): Whether to enable parallel API calls.
            rpm_limit (int): Requests per minute (rate limit).
            tpm_limit (int): Tokens per minute (rate limit).
        """
        self.api_key = api_key
        self.model = model
        self.base_url = base_url or openai.base_url
        self.client = openai.Client(api_key=self.api_key, base_url=self.base_url)
        self.parallelise = parallelise
        self.rpm_limit = rpm_limit
        self.tpm_limit = tpm_limit

        # Thread-safe token bucket for TPM limiting
        self._bucket_capacity = float(self.tpm_limit)
        self._bucket_tokens = float(self.tpm_limit)
        self._bucket_fill_rate = float(self.tpm_limit) / 60.0  # tokens per second
        self._bucket_last = time.monotonic()
        self._bucket_lock = Lock()

        # Compute sensible default for workers (dynamic token estimate)
        avg_latency = 0.5
        sample_msg = [{"role": "user", "content": [{"type": "text", "text": ""}]}]
        est_tokens_per_call = estimate_tokens(sample_msg)
        calls_per_min_by_tpm = self.tpm_limit / est_tokens_per_call
        calls_per_sec_allowed = min(self.rpm_limit, calls_per_min_by_tpm) / 60.0
        # auto_workers = max(int(calls_per_sec_allowed * avg_latency), 1)
        
        # Mke concurrency more conservative by applying a "safety factor"
        safety_factor = 0.5 # ...this halves the max workers from the 'sensible default'
        auto_workers = max(int(calls_per_sec_allowed * avg_latency * safety_factor), 1)
        
        if not self.parallelise:
            self.max_workers = 1
        else:
            # cap at auto_workers even if user requests higher
            self.max_workers = min(max_workers or auto_workers, auto_workers)

        # Global semaphore to throttle both generate & parse
        self._sem = Semaphore(self.max_workers)

        # Rate-limit + backoff the raw generate calls
        gen_limiter = limits(calls=self.rpm_limit, period=60)
        @backoff.on_exception(backoff.expo, RateLimitException, max_time=60)
        @sleep_and_retry(gen_limiter)
        def _generate_with_backoff(messages: List[Dict], temperature: float = 0.0) -> str:
            with self._sem:
                return self._raw_generate(messages, temperature)
        self._safe_generate_impl = _generate_with_backoff

        # Rate-limit the raw parse endpoint
        parse_limiter = limits(calls=self.rpm_limit, period=60)
        self._safe_parse = sleep_and_retry(parse_limiter)(
            self.client.beta.chat.completions.parse
        )

        # Wrap parse in backoff + semaphore
        @backoff.on_exception(backoff.expo, RateLimitError, max_time=60)
        def _parse_with_backoff(**kwargs):
            with self._sem:
                return self._safe_parse(**kwargs)
        self._parse_with_backoff = _parse_with_backoff

    def _replenish_bucket(self):
        with self._bucket_lock:
            now = time.monotonic()
            elapsed = now - self._bucket_last
            self._bucket_last = now
            self._bucket_tokens = min(
                self._bucket_capacity,
                self._bucket_tokens + elapsed * self._bucket_fill_rate
            )

    def _consume_tokens(self, count: int):
        # Thread-safe consume
        self._replenish_bucket()
        with self._bucket_lock:
            if count > self._bucket_tokens:
                deficit = count - self._bucket_tokens
                wait_time = deficit / self._bucket_fill_rate
                logger.debug(f"TPM limit reached, sleeping for {wait_time:.2f}s")
                time.sleep(wait_time)
                # refill after sleep
                self._replenish_bucket()
            self._bucket_tokens -= count

    def _raw_generate(
        self,
        messages: List[Dict],
        temperature: float = 0.0
    ) -> str:
        # 1) Estimate token usage
        estimated = estimate_tokens(messages) * 1.1
        self._consume_tokens(int(estimated))

        # 2) Make the call
        resp = self.client.chat.completions.create(
            model=self.model,
            messages=messages,
            temperature=temperature,
        )

        # 3) Record actual usage (for future tuning/logging)
        try:
            used = resp.usage.total_tokens
            logger.debug(f"Actual tokens used: {used}, estimated: {int(estimated)}")
        except Exception:
            pass

        # 4) Return content
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
                for b64 in imgs:
                    content.append({
                        "type": "image_url",
                        "image_url": {"url": f"data:image/jpeg;base64,{b64}"}
                    })
            return [{"role": "user", "content": content}]

        # Determine worker count
        workers = max_workers or self.max_workers or len(prompts)

        # Sequential
        if not self.parallelise:
            results: List[BaseModel | str] = []
            for idx, prompt in enumerate(prompts):
                imgs = images_list[idx] if images_list else None
                messages = _build_messages(prompt, imgs)

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

        # Parallel
        results: List[BaseModel | str] = [None] * len(prompts)

        def _worker(idx: int, prompt: str):
            imgs = images_list[idx] if images_list else None
            messages = _build_messages(prompt, imgs)

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

        with ThreadPoolExecutor(max_workers=workers) as executor:
            futures = [executor.submit(_worker, i, p) for i, p in enumerate(prompts)]
            for fut in as_completed(futures):
                i, out = fut.result()
                results[i] = out

        return results

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
        base64_images: List[str] = []
        if pdf_bytes:
            try:
                base64_images = self._pdf_bytes_to_base64_images(pdf_bytes, dpi=200)
            except Exception as e:
                logger.error(f"Error converting PDF to images with PyMuPDF: {e}")

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
            result_list = self.generate_batch(
                prompts=[prompt],
                images_list=[base64_images],
                response_format=MissingModel,
                temperature=0.0
            )
            output = result_list[0]
        except Exception as e:
            logger.error(f"LLM fallback call failed: {e}")
            return {}

        if isinstance(output, BaseModel):
            out_json = output.model_dump()
        elif isinstance(output, dict):
            out_json = output
        else:
            logger.error(f"Unexpected LLM fallback output type: {type(output)}")
            return {}

        if verbose:
            logger.info("LLM fallback output for %s: %s", report_url, out_json)

        updates: Dict[str, str] = {}
        for fld in response_fields:
            val = out_json.get(fld)
            updates[fld] = val if val is not None else "LLM Fallback failed"
        return updates


