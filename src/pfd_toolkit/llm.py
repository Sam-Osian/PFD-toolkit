import openai
from openai import RateLimitError, APIConnectionError, APITimeoutError
import logging
import base64
from typing import List, Optional, Dict, Type, Any
from pydantic import BaseModel, create_model
import pymupdf
from concurrent.futures import ThreadPoolExecutor, as_completed
import backoff
from threading import Semaphore
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, force=True)

# Set the log level for the 'httpx' library to WARNING to reduce verbosity
logging.getLogger("httpx").setLevel(logging.WARNING)

# Silence the ratelimit package
# logging.getLogger("ratelimit").setLevel(logging.WARNING)

# Disable all logging calls from pfd_toolkit.llm
# logging.getLogger("pfd_toolkit.llm").disabled = True

# Silence the OpenAI client’s info-level logs
# logging.getLogger("openai").setLevel(logging.WARNING)


class LLM:
    """Wrapper around the OpenAI Python SDK for batch prompting and PDF
    vision fallback.

    The helper provides:

    * A generic :py:meth:`self.generate_batch()` that optionally supports vision
      inputs and pydantic validation.
    * A PDF-to-image utility used by
      :py:meth:`self._call_llm_fallback()` - the method the scraper invokes when
      HTML and PDF heuristics fail.
    * Built-in back-off and host-wide throttling via a semaphore.

    Parameters
    ----------
    api_key : str, optional
        OpenAI (or proxy) API key.
    model : str, optional
        Chat model name; defaults to ``"gpt-4.1-mini"``.
    base_url : str or None, optional
        Override the OpenAI endpoint (for Azure/OpenRouter etc.).
    max_workers : int, optional
        Maximum parallel workers for batch calls and for the global
        semaphore.

    Attributes
    ----------
    CLEANER_BASE_PROMPT : str
        The shared template used by *Cleaner* to build field-specific
        prompts.
    CLEANER_PROMPT_CONFIG : dict
        Field-level substitution values for the cleaner prompt.
    _sem : threading.Semaphore
        Global semaphore that limits concurrent requests to *max_workers*.
    client : openai.Client
        Low-level SDK client configured with key and base URL.

    Examples
    --------
    >>> llm = LLM(api_key="sk-...", model="gpt-4o-mini", max_workers=4)
    >>> out = llm.generate_batch(["Hello world"])
    >>> out[0]
    'Hello! How can I assist you today?'
    """

    # Base prompt template that all prompts will share, with placeholders for field-specific information.
    CLEANER_BASE_PROMPT = """\
    You are an expert in extracting and cleaning specific information from UK Coronial Prevention of Future Death Reports.

    Task:
    1. **Extract** only the information related to {field_description}.
    2. **Clean** the input text by removing extraneous details such as rogue numbers, punctuation, HTML tags, or redundant content, if any occurs.
    3. **Correct** any misspellings, ensuring the text is in sentence-case **British English**. Do not replace any acronyms.
    4. **Return** exactly and only the cleaned data for {field_contents_and_rules}. You must only return the cleaned string, without adding additional commentary, summarisation, or headings.
    5. **If extraction fails**, return only and exactly: N/A: Not found
    6. **Do not** change any content of the string unless it explicitly relates to the instructions above or below. Do not ever summarise, *nor* edit for conciseness or flow.

    Extra instructions:
    {extra_instructions}

    Input Text:
    """

    # Dictionary holding field-specific configurations for the prompt
    # The placeholders for the above `BASE_PROMPT` will be 'filled in' using the values below...
    CLEANER_PROMPT_CONFIG = {
        "Coroner": {
            "field_description": "the name of the Coroner who presided over the inquest",
            "field_contents_and_rules": "this name of the Coroner -- nothing else",
            "extra_instructions": (
                "Remove all reference to titles & middle name(s), if present, and replace the first name with an initial. "
                'For example, if the string is "Mr. Joe E Bloggs", return "J. Bloggs". '
                'If the string is "Joe Bloggs Senior Coroner for West London", return "J. Bloggs". '
                'If the string is "J. Bloggs", just return "J. Bloggs" (no modification). '
            ),
        },
        "Area": {
            "field_description": "the area where the inquest took place",
            "field_contents_and_rules": "only the name of the area -- nothing else",
            "extra_instructions": (
                'For example, if the string is "Area: West London", return "West London". '
                'If the string is "Hampshire, Portsmouth and Southampton", return it as is.'
            ),
        },
        "Receiver": {
            "field_description": "the name(s)/organisation(s) of the receiver(s) of the report",
            "field_contents_and_rules": "only the name(s)/organisation(s) and, if given, their job title(s) -- nothing else",
            "extra_instructions": (
                "Separate multiple names/organisations with semicolons (;). "
                "Do not use a numbered list. "
                "Do not separate information with commas or new lines. "
            ),
        },
        "InvestigationAndInquest": {
            "field_description": "the details of the investigation and inquest",
            "field_contents_and_rules": "only the details of the investigation and inquest -- nothing else",
            "extra_instructions": (
                "If the string appears to need no cleaning, return it as is. "
                "If a date is used, convert it into British long format. "
            ),
        },
        "CircumstancesOfDeath": {
            "field_description": "the circumstances of death",
            "field_contents_and_rules": "only the circumstances of death -- nothing else",
            "extra_instructions": (
                "If the string appears to need no cleaning, return it as is. "
                "If a date is used, convert it into British long format. "
            ),
        },
        "MattersOfConcern": {
            "field_description": "the matters of concern",
            "field_contents_and_rules": "only the matters of concern, nothing else",
            "extra_instructions": (
                'Remove reference to boilerplate text, if any occurs. This is usually 1 or 2 non-specific sentences at the start of the string often ending with "...The Matters of Concern are as follows:" (which should also be removed). '
                "If the string appears to need no cleaning, return it as is. "
                "If a date is used, convert it into British long format. "
            ),
        },
    }

    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4.1-mini",
        base_url: Optional[str] = None,
        max_workers: int = 8,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url or openai.base_url
        self.client = openai.Client(api_key=self.api_key, base_url=self.base_url)

        # Ensure max_workers is at least 1
        self.max_workers = max(1, max_workers)

        # Global semaphore to throttle calls based on max_workers
        self._sem = Semaphore(self.max_workers)

        # Backoff for parse endpoint, handles OpenAI connection errors
        # Adding jitter avoids thundering-herd retries
        @backoff.on_exception(
            backoff.expo,
            (RateLimitError, APIConnectionError, APITimeoutError),
            max_time=60,
            jitter=backoff.full_jitter,
        )
        def _parse_with_backoff(**kwargs):
            with self._sem:
                # Call the client's parse method directly
                return self.client.beta.chat.completions.parse(**kwargs)

        self._parse_with_backoff = _parse_with_backoff

    def _pdf_bytes_to_base64_images(
        self, pdf_bytes: bytes, dpi: int = 200
    ) -> list[str]:
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

    # Main LLM method for other modules
    def generate_batch(
        self,
        prompts: List[str],
        images_list: Optional[List[List[bytes]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        temperature: float = 0.0,
        max_workers: Optional[int] = None,
        tqdm_extra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[BaseModel | str]:
        """Run many prompts either sequentially or in parallel.

        Parameters
        ----------
        prompts : list[str]
            List of user prompts. One prompt per model call.

        images_list : list[list[bytes]] or None, optional
            For vision models: a parallel list where each inner list
            holds **base64-encoded** JPEG pages for that prompt.  Use
            *None* to send no images.

        response_format : type[pydantic.BaseModel] or None, optional
            If provided, each response is parsed into that model via the
            *beta/parse* endpoint; otherwise a raw string is returned.

        temperature : float, optional
            Sampling temperature.  Defaults to *0.0* (deterministic).

        max_workers : int or None, optional
            Thread count just for this batch.  When *None*, fall back to
            the instance-wide :pyattr:`max_workers`.

        Returns
        -------
        list[Union[pydantic.BaseModel, str]]
            Results in the same order as `prompts`.

        Raises
        ------
        openai.RateLimitError
            Raised only if the exponential back-off exhausts all retries.
        openai.APIConnectionError
            Raised if network issues persist beyond the retry window.
        openai.APITimeoutError
            Raised if the API repeatedly times out.

        Examples
        --------
        >>> msgs = ["Summarise:\\n" + txt for txt in docs]
        >>> summaries = llm.generate_batch(msgs, temperature=0.2, max_workers=8)
        """
        tqdm_kwargs = dict(tqdm_extra_kwargs or {})
        if len(prompts) == 1:
            tqdm_kwargs.setdefault("disable", True)

        def _build_messages(prompt: str, imgs: Optional[List[bytes]]):
            content = [{"type": "text", "text": prompt}]
            if imgs:
                for b64_img_data in imgs:
                    content.append(
                        {
                            "type": "image_url",
                            "image_url": {
                                "url": f"data:image/jpeg;base64,{b64_img_data}"
                            },
                        }
                    )
            return [{"role": "user", "content": content}]

        # Determine effective worker count for this batch
        if max_workers is not None and max_workers > 0:
            effective_workers = max_workers
        else:
            effective_workers = self.max_workers

        @backoff.on_exception(
            backoff.expo,
            (RateLimitError, APIConnectionError, APITimeoutError),
            max_time=60,
            jitter=backoff.full_jitter,
        )
        def _call_llm(messages: List[Dict]) -> str:
            with self._sem:
                resp = self.client.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    temperature=temperature,
                )
            try:
                used = resp.usage.total_tokens
                logger.debug(f"Actual tokens used: {used}")
            except Exception:
                pass
            return resp.choices[0].message.content.strip()

        results: List[BaseModel | str] = [None] * len(prompts)

        def _worker(idx: int, prompt_text: str):
            current_images = (
                images_list[idx] if images_list and idx < len(images_list) else None
            )
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
                txt = _call_llm(messages)
                return idx, txt

        with ThreadPoolExecutor(max_workers=effective_workers) as executor:
            futures = [executor.submit(_worker, i, p) for i, p in enumerate(prompts)]
            bar_kwargs = dict(tqdm_kwargs)
            current_desc = bar_kwargs.pop(
                "desc", "Sending requests to the LLM"
            )
            for fut in tqdm(
                as_completed(futures),
                total=len(prompts),
                desc=current_desc,
                **bar_kwargs,
            ):
                i, out = fut.result()
                results[i] = out

        return results

    # LLM method for scraper.py module
    def _call_llm_fallback(
        self,
        pdf_bytes: Optional[bytes],
        missing_fields: Dict[str, str],
        report_url: Optional[str] = None,
        verbose: bool = False,
        tqdm_extra_kwargs: Optional[Dict[str, Any]] = None,
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
        base64_images_list: List[str] = []  # This will be a list of base64 strings
        if pdf_bytes:
            try:
                base64_images_list = self._pdf_bytes_to_base64_images(
                    pdf_bytes, dpi=200
                )
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
            '\'If you are unable to identify the text for any section, respond exactly: "N/A: Not found".\n'
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
                images_list=images_for_batch,  # type: ignore
                response_format=MissingModel,
                temperature=0.0,
                tqdm_extra_kwargs=tqdm_extra_kwargs,
            )
            output = result_list[0]
        except Exception as e:
            logger.error(f"LLM fallback call failed: {e}")
            return {}

        if isinstance(output, BaseModel):
            out_json = output.model_dump()
        elif isinstance(
            output, dict
        ):  # Fallback if error string was returned as dict by mistake
            out_json = output
        elif isinstance(output, str) and output.startswith(
            "Error:"
        ):  # Handle error string
            logger.error(f"LLM fallback returned an error string: {output}")
            return {fld: "LLM Fallback error" for fld in response_fields}
        else:
            logger.error(
                f"Unexpected LLM fallback output type: {type(output)}, value: {output}"
            )
            return {
                fld: "LLM Fallback failed - unexpected type" for fld in response_fields
            }

        if verbose:
            logger.info("LLM fallback output for %s: %s", report_url, out_json)

        updates: Dict[str, str] = {}
        for fld in response_fields:
            val = out_json.get(
                fld
            )  # out_json might not be a dict if error occurred above
            if val is None and isinstance(
                out_json, dict
            ):  # Check if out_json is a dict
                updates[fld] = (
                    "N/A: Not found in LLM response"  # Field was expected but not in output
                )
            elif val is not None:
                updates[fld] = str(val)  # Ensure value is string
            else:  # out_json was not a dict or other issue
                updates[fld] = "LLM Fallback processing error"
        return updates
