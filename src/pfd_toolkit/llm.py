import openai
from openai import RateLimitError, APIConnectionError, APITimeoutError
import httpx
import tiktoken
import logging
import base64
import re
from typing import List, Optional, Dict, Type, Any
from pydantic import BaseModel, create_model, ConfigDict
import pymupdf
from concurrent.futures import ThreadPoolExecutor, as_completed
import backoff
from threading import Semaphore
from tqdm import tqdm

from .config import GeneralConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, force=True)

# Set the log level for the 'httpx' library to WARNING to reduce verbosity
logging.getLogger("httpx").setLevel(logging.WARNING)


def _strip_json_markdown(text: str) -> str:
    """Return ``text`` with any surrounding markdown code fences removed.

    Providers such as OpenRouter occasionally return JSON wrapped in `````"
    blocks (with or without a ``json`` language hint) which causes
    ``pydantic`` to raise ``json_invalid`` errors.  This helper takes a very
    permissive approach: if any triple-backtick block is found, the contents of
    the first block are returned.  All other fences are stripped as well.  The
    function also handles stray BOM characters or spaces around the fences.
    """

    text = (text or "").strip().lstrip("\ufeff")
    if not text:
        return text

    if "```" not in text:
        return text

    # Split on fences and grab the first non-empty chunk after the opening
    # fence.  This avoids fragile regular expressions when providers add extra
    # newlines or spaces.
    parts = text.split("```")
    if len(parts) < 3:
        # Something odd – drop all fences just in case
        return text.replace("```", "").strip()

    # parts[1] may contain a language spec like ``json``; remove the first word
    inner = parts[1]
    inner = re.sub(r"^json\s*", "", inner, flags=re.IGNORECASE)
    cleaned = inner.strip()

    if cleaned:
        return cleaned

    # Fallback: remove all fences globally
    return text.replace("```", "").strip()


class LLM:
    """Wrapper around the OpenAI Python SDK for batch prompting.

    The helper provides:

    * ``generate`` for plain or vision-enabled prompts with optional pydantic
      validation.
    * ``_call_llm_fallback`` used by the scraper when HTML and PDF heuristics
      fail.
    * Built-in back-off and host-wide throttling via a semaphore.

    Parameters
    ----------
    api_key : str, optional
        OpenAI (or proxy) API key. Defaults to ``None`` which expects the
        environment variable to be set.
    model : str, optional
        Chat model name. Defaults to ``"gpt-4.1"``.
    base_url : str or None, optional
        Override the OpenAI endpoint. Defaults to ``None``.
    max_workers : int, optional
        Maximum parallel workers for batch calls and for the global semaphore.
        Defaults to ``8``.
    temperature : float, optional
        Sampling temperature used for all requests. Defaults to ``0.0``.
    seed : int or None, optional
        Deterministic seed value passed to the API. Defaults to ``None``.
    validation_attempts : int, optional
        Number of times to retry parsing LLM output into a pydantic model.
        Defaults to ``2``.
    timeout : float | httpx.Timeout | None, optional
        Override the HTTP timeout in seconds. ``None`` uses the OpenAI client
        default of 600 seconds.

    Attributes
    ----------
    _sem : threading.Semaphore
        Global semaphore that limits concurrent requests to *max_workers*.
    client : openai.Client
        Low-level SDK client configured with key and base URL.

    Examples
    --------

        llm_client = LLM(api_key="sk-...", model="gpt-4o-mini", temperature=0.2,
                  timeout=600)
    """


    def __init__(
        self,
        api_key: Optional[str] = None,
        model: str = "gpt-4.1",
        base_url: Optional[str] = None,
        max_workers: int = 8,
        temperature: float = 0.0,
        seed: Optional[int] = None,
        validation_attempts: int = 2,
        timeout: float | httpx.Timeout = 120,
    ):
        self.api_key = api_key
        self.model = model
        self.base_url = base_url or openai.base_url
        self.timeout = timeout
        self.client = openai.Client(
            api_key=self.api_key,
            base_url=self.base_url,
            timeout=self.timeout,
        )

        self.temperature = float(temperature)
        self.seed = seed
        self.validation_attempts = max(1, validation_attempts)

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

    def estimate_tokens(
        self, texts: List[str] | str, model: Optional[str] = None
    ) -> List[int]:
        """Return token counts for text using ``tiktoken``.

        Parameters
        ----------
        texts : list[str] | str
            Input strings to tokenise.
        model : str, optional
            Model name for selecting the encoding. Defaults to
            ``self.model``.

        Returns
        -------
        list[int]
            Token counts in the same order as ``texts``.
        """

        if isinstance(texts, str):
            texts = [texts]

        enc_model = model or self.model
        try:
            try:
                enc = tiktoken.encoding_for_model(enc_model)
            except KeyError:
                enc = tiktoken.get_encoding("cl100k_base")
            counts = [len(enc.encode(t or "")) for t in texts]
        except Exception as e:  # pragma: no cover - network or other failure
            logger.warning("tiktoken failed (%s); using fallback estimate", e)
            counts = [len((t or "").split()) for t in texts]

        return counts

    # Main LLM method for other modules
    def generate(
        self,
        prompts: List[str],
        images_list: Optional[List[List[bytes]]] = None,
        response_format: Optional[Type[BaseModel]] = None,
        max_workers: Optional[int] = None,
        tqdm_extra_kwargs: Optional[Dict[str, Any]] = None,
    ) -> List[BaseModel | str]:
        """Run many prompts either sequentially or in parallel.

        Parameters:
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

        max_workers : int or None, optional
            Thread count just for this batch. ``None`` uses the instance-wide
            ``max_workers`` value. Defaults to ``None``.

        Returns:
        -------
        list[Union[pydantic.BaseModel, str]]
            Results in the same order as `prompts`.

        Raises:
        ------
        openai.RateLimitError
            Raised only if the exponential back-off exhausts all retries.
        openai.APIConnectionError
            Raised if network issues persist beyond the retry window.
        openai.APITimeoutError
            Raised if the API repeatedly times out.

        Examples:
        --------
            msgs = ["Summarise:\n" + txt for txt in docs]
            summaries = llm.generate(msgs)
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
                    temperature=self.temperature,
                    **({"seed": self.seed} if self.seed is not None else {}),
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
                for attempt in range(self.validation_attempts):
                    try:
                        resp = self._parse_with_backoff(
                            model=self.model,
                            messages=messages,
                            temperature=self.temperature,
                            response_format=response_format,
                            **({"seed": self.seed} if self.seed is not None else {}),
                        )
                        raw = resp.choices[0].message.content
                        cleaned = _strip_json_markdown(raw)
                        validated = response_format.model_validate_json(
                            cleaned,
                            strict=True,
                        )
                        return idx, validated
                    except Exception as e:
                        if attempt == self.validation_attempts - 1:
                            logger.error(
                                f"Batch pydantic parse failed for item {idx}: {e}"
                            )
                            return idx, f"Error: {e}"
                        logger.debug(
                            "Validation attempt %s failed for item %s: %s",
                            attempt + 1,
                            idx,
                            e,
                        )
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
        """Use the LLM to extract text from PDF images for missing fields.

        Parameters
        ----------
        pdf_bytes : bytes or None
            Raw PDF data. If ``None`` no images are sent.
        missing_fields : dict
            Mapping of field names to prompt instructions.
        report_url : str, optional
            URL of the report for logging. Defaults to ``None``.
        verbose : bool, optional
            When ``True`` log prompt and output. Defaults to ``False``.
        tqdm_extra_kwargs : dict or None, optional
            Extra keyword arguments passed to ``tqdm``. Defaults to ``None``.

        Returns
        -------
        dict
            Extracted values keyed by the original field names.
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
            "You will be presented with screenshots of a Prevention of Future Deaths (PFD) report. \n\n"

            "Your goal is to transcribe verbatim text from this report. \n\n"

            "Please extract the following report elements: \n\n"
        )
        response_fields: List[str] = []
        for field, instruction in missing_fields.items():
            response_fields.append(field)
            prompt += f"\n{field}: {instruction}\n"
        prompt += (
            "\n\nFurther instructions:\n\n - You must not respond in your own 'voice'; output verbatim text from the reports **only**.\n"
            f" - If you are unable to identify the text for any section, respond exactly: {GeneralConfig.NOT_FOUND_TEXT}.\n"
            " - Transcribe redacted text (black rectangles) as '[REDACTED]'.\n"
            " - Confirm the PDF is the coroner's PFD report and not a response document. If it is a response document, return "
            f"{GeneralConfig.NOT_FOUND_TEXT} for all sections.\n"
            " - You must extract the *full* and verbatim text for each given section - no shortening or partial extractions.\n"
        )

        schema = {fld: (str, ...) for fld in response_fields}
        MissingModel = create_model(
            "MissingFields", **schema, __config__=ConfigDict(extra="forbid")
        )

        if verbose:
            logger.info("LLM fallback prompt for %s:\n%s", report_url, prompt)

        try:
            result_list = self.generate(
                prompts=[prompt],
                images_list=images_for_batch,  # type: ignore
                response_format=MissingModel,
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
                    f"{GeneralConfig.NOT_FOUND_TEXT} in LLM response"  # Field was expected but not in output
                )
            elif val is not None:
                updates[fld] = str(val)  # Ensure value is string
            else:  # out_json was not a dict or other issue
                updates[fld] = "LLM Fallback processing error"
        return updates
