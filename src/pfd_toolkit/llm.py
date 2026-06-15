import openai
from openai import BadRequestError, RateLimitError, APIConnectionError, APITimeoutError
import httpx
import tiktoken
import logging
import base64
import re
import json
import time
from typing import Callable, List, Optional, Dict, Type, Any
from pydantic import BaseModel, create_model, ConfigDict
import pymupdf
from concurrent.futures import ThreadPoolExecutor, wait, FIRST_COMPLETED
import backoff
from threading import Semaphore, Event
from tqdm import tqdm

from .config import GeneralConfig

logger = logging.getLogger(__name__)
logging.basicConfig(level=logging.INFO, force=True)

# Set the log level for the 'httpx' library to WARNING to reduce verbosity
logging.getLogger("httpx").setLevel(logging.WARNING)


def _is_insufficient_quota_error(exc: Exception) -> bool:
    """Return True when an API error indicates provider quota is exhausted."""
    if exc is None:
        return False

    message = str(exc).strip().lower()
    if "insufficient_quota" in message or "insufficient quota" in message:
        return True

    body = getattr(exc, "body", None)
    if isinstance(body, dict):
        err = body.get("error", body)
        if isinstance(err, dict):
            code = str(err.get("code") or "").strip().lower()
            err_type = str(err.get("type") or "").strip().lower()
            err_message = str(err.get("message") or "").strip().lower()
            if code == "insufficient_quota":
                return True
            if "insufficient_quota" in err_type or "insufficient quota" in err_type:
                return True
            if "insufficient_quota" in err_message or "insufficient quota" in err_message:
                return True

    response = getattr(exc, "response", None)
    if response is not None:
        try:
            payload = response.json()
        except Exception:
            payload = None
        if isinstance(payload, dict):
            err = payload.get("error", payload)
            if isinstance(err, dict):
                code = str(err.get("code") or "").strip().lower()
                err_type = str(err.get("type") or "").strip().lower()
                err_message = str(err.get("message") or "").strip().lower()
                if code == "insufficient_quota":
                    return True
                if "insufficient_quota" in err_type or "insufficient quota" in err_type:
                    return True
                if "insufficient_quota" in err_message or "insufficient quota" in err_message:
                    return True

    return False


def _is_unsupported_parameter_error(exc: Exception, parameter_name: str) -> bool:
    """Return True when an API error indicates the parameter is unsupported."""
    if exc is None or not isinstance(exc, BadRequestError):
        return False

    param = str(parameter_name or "").strip()
    if not param:
        return False

    payload = getattr(exc, "body", None)
    if isinstance(payload, dict):
        err = payload.get("error", payload)
    else:
        err = None

    if err is None:
        response = getattr(exc, "response", None)
        if response is not None:
            try:
                payload = response.json()
            except Exception:
                payload = None
            if isinstance(payload, dict):
                err = payload.get("error", payload)

    if isinstance(err, dict):
        error_param = str(err.get("param") or "").strip()
        if error_param == param:
            return True
        message = str(err.get("message") or "").strip().lower()
        code = str(err.get("code") or "").strip().lower()
        if (
            (f"'{param}'" in message or param in message)
            and ("unsupported" in message or "does not support" in message)
        ):
            return True
        if code in {"unsupported_parameter", "unsupported_value"} and (
            f"'{param}'" in message or param in message
        ):
            return True

    message = str(exc).strip().lower()
    return (f"'{param}'" in message or param in message) and (
        "unsupported parameter" in message or "does not support" in message
    )


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


def _normalise_yes_no(value: Any) -> str | None:
    """Map common provider outputs to canonical 'Yes'/'No'."""
    if isinstance(value, bool):
        return "Yes" if value else "No"
    if isinstance(value, (int, float)):
        if value == 1:
            return "Yes"
        if value == 0:
            return "No"
    text = str(value or "").strip().lower()
    if text in {"yes", "y", "true", "1"}:
        return "Yes"
    if text in {"no", "n", "false", "0"}:
        return "No"
    return None


def _extract_text_from_message_content(content: Any) -> str:
    """Handle SDK variants where message.content may be str or content parts."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        text_parts: list[str] = []
        for part in content:
            if isinstance(part, dict):
                txt = part.get("text")
                if txt is not None:
                    text_parts.append(str(txt))
                continue
            txt = getattr(part, "text", None)
            if txt is not None:
                text_parts.append(str(txt))
        if text_parts:
            return "\n".join(text_parts)
    return str(content or "")


def _coerce_structured_response_from_text(
    *,
    response_format: Type[BaseModel],
    raw_text: str,
) -> BaseModel:
    """Best-effort parsing for non-OpenAI-compatible structured responses."""
    cleaned = _strip_json_markdown(raw_text or "")
    if not cleaned:
        raise ValueError("Empty structured response.")

    # First try strict JSON validation as originally intended.
    try:
        return response_format.model_validate_json(cleaned, strict=True)
    except Exception:
        pass

    # Some providers emit valid JSON that fails strict mode due type coercion.
    try:
        payload = json.loads(cleaned)
    except Exception:
        payload = None
    if isinstance(payload, dict):
        if "matches_topic" in payload:
            normalized = _normalise_yes_no(payload.get("matches_topic"))
            if normalized is not None:
                payload = dict(payload)
                payload["matches_topic"] = normalized
        try:
            return response_format.model_validate(payload)
        except Exception:
            pass

    # Final heuristic for topic classification models that sometimes emit plain
    # "Yes"/"No" text instead of JSON.
    lowered = cleaned.lower()
    inferred_answer = None
    if re.search(r"\byes\b", lowered):
        inferred_answer = "Yes"
    elif re.search(r"\bno\b", lowered):
        inferred_answer = "No"
    if inferred_answer is None:
        raise ValueError(f"Unable to parse structured response: {cleaned[:200]}")

    fields = getattr(response_format, "model_fields", {})
    heuristic_payload: dict[str, str] = {}
    if "matches_topic" in fields:
        heuristic_payload["matches_topic"] = inferred_answer
    if "spans_matches_topic" in fields:
        heuristic_payload["spans_matches_topic"] = GeneralConfig.NOT_FOUND_TEXT
    if not heuristic_payload:
        raise ValueError(f"Unable to map heuristic answer into model: {response_format}")

    return response_format.model_validate(heuristic_payload)


class GenerationCancelledError(RuntimeError):
    """Raised when an LLM batch is cancelled before completion."""


class ReasoningControlUnsupportedError(RuntimeError):
    """Raised when model/provider does not support requested reasoning control."""


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
    reasoning_effort : str or None, optional
        Optional reasoning effort hint for reasoning-capable models.
        Common values include ``none``, ``low``, ``medium`` and ``high``.
        Defaults to ``None``.
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
        reasoning_effort: Optional[str] = None,
        seed: Optional[int] = None,
        validation_attempts: int = 2,
        timeout: float | httpx.Timeout = 120,
        per_report_timeout_s: Optional[float] = None,
        strict_reasoning_effort: bool = False,
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
        self.reasoning_effort = str(reasoning_effort).strip() if reasoning_effort else None
        self.seed = seed
        self.validation_attempts = max(1, validation_attempts)
        self.strict_reasoning_effort = bool(strict_reasoning_effort)
        self.per_report_timeout_s = (
            float(per_report_timeout_s)
            if per_report_timeout_s is not None and float(per_report_timeout_s) > 0
            else None
        )

        # Ensure max_workers is at least 1
        self.max_workers = max(1, max_workers)

        # Global semaphore to throttle calls based on max_workers
        self._sem = Semaphore(self.max_workers)
        self._cancel_requested = Event()

        # Backoff for parse endpoint, handles OpenAI connection errors
        # Adding jitter avoids thundering-herd retries
        @backoff.on_exception(
            backoff.expo,
            (RateLimitError, APIConnectionError, APITimeoutError),
            max_time=60,
            giveup=lambda exc: _is_insufficient_quota_error(exc) or self._cancel_requested.is_set(),
            jitter=backoff.full_jitter,
        )
        def _parse_with_backoff(**kwargs):
            if self._cancel_requested.is_set():
                raise GenerationCancelledError("LLM generation cancelled.")
            with self._sem:
                if self._cancel_requested.is_set():
                    raise GenerationCancelledError("LLM generation cancelled.")
                return self._chat_parse_with_compat_retry(**kwargs)

        self._parse_with_backoff = _parse_with_backoff

    def request_cancellation(self) -> None:
        """Request cancellation for current/future generation calls."""
        self._cancel_requested.set()
        close_client = getattr(self.client, "close", None)
        if callable(close_client):
            try:
                close_client()
            except Exception:
                # Best effort: caller is already cancelling.
                pass

    def reset_cancellation(self) -> None:
        """Clear any prior cancellation request."""
        self._cancel_requested.clear()

    def _chat_create_with_compat_retry(self, **request_kwargs):
        """Retry once without unsupported optional parameters."""
        attempted_removals: set[str] = set()
        while True:
            try:
                return self.client.chat.completions.create(**request_kwargs)
            except Exception as exc:
                removable = None
                for candidate in ("reasoning_effort", "temperature", "seed"):
                    if candidate in request_kwargs and candidate not in attempted_removals:
                        if _is_unsupported_parameter_error(exc, candidate):
                            if candidate == "reasoning_effort" and self.strict_reasoning_effort:
                                raise ReasoningControlUnsupportedError(
                                    "Model does not support enforced reasoning_effort control."
                                ) from exc
                            removable = candidate
                            break
                if removable is None:
                    raise
                attempted_removals.add(removable)
                request_kwargs = dict(request_kwargs)
                request_kwargs.pop(removable, None)
                logger.warning(
                    "Model '%s' rejected parameter '%s'; retrying without it.",
                    self.model,
                    removable,
                )

    def _chat_parse_with_compat_retry(self, **request_kwargs):
        """Retry parse requests without unsupported optional parameters."""
        attempted_removals: set[str] = set()
        while True:
            try:
                return self.client.beta.chat.completions.parse(**request_kwargs)
            except Exception as exc:
                removable = None
                for candidate in ("reasoning_effort", "temperature", "seed"):
                    if candidate in request_kwargs and candidate not in attempted_removals:
                        if _is_unsupported_parameter_error(exc, candidate):
                            if candidate == "reasoning_effort" and self.strict_reasoning_effort:
                                raise ReasoningControlUnsupportedError(
                                    "Model does not support enforced reasoning_effort control."
                                ) from exc
                            removable = candidate
                            break
                if removable is None:
                    raise
                attempted_removals.add(removable)
                request_kwargs = dict(request_kwargs)
                request_kwargs.pop(removable, None)
                logger.warning(
                    "Model '%s' rejected parameter '%s' for parse call; retrying without it.",
                    self.model,
                    removable,
                )

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
        progress_callback: Optional[Callable[[int, int, str], None]] = None,
        cancellation_check: Optional[Callable[[], bool]] = None,
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

        cancellation_check : callable or None, optional
            Optional callback returning ``True`` when cancellation is requested.
            Pending tasks are cancelled immediately and in-flight requests are
            interrupted best-effort.

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
        self.reset_cancellation()

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

        def _is_cancelled() -> bool:
            if self._cancel_requested.is_set():
                return True
            if cancellation_check is None:
                return False
            if cancellation_check():
                self.request_cancellation()
                return True
            return False

        def _raise_if_cancelled() -> None:
            if _is_cancelled():
                raise GenerationCancelledError("LLM generation cancelled.")

        @backoff.on_exception(
            backoff.expo,
            (RateLimitError, APIConnectionError, APITimeoutError),
            max_time=60,
            giveup=lambda exc: _is_insufficient_quota_error(exc) or self._cancel_requested.is_set(),
            jitter=backoff.full_jitter,
        )
        def _call_llm(messages: List[Dict]) -> str:
            _raise_if_cancelled()
            with self._sem:
                _raise_if_cancelled()
                request_kwargs = {
                    "model": self.model,
                    "messages": messages,
                }
                request_kwargs["temperature"] = self.temperature
                if self.seed is not None:
                    request_kwargs["seed"] = self.seed
                if self.reasoning_effort:
                    request_kwargs["reasoning_effort"] = self.reasoning_effort
                resp = self._chat_create_with_compat_retry(**request_kwargs)
            try:
                used = resp.usage.total_tokens
                logger.debug(f"Actual tokens used: {used}")
            except Exception:
                pass
            content = _extract_text_from_message_content(resp.choices[0].message.content)
            return content.strip()

        results: List[BaseModel | str] = [None] * len(prompts)

        def _worker(idx: int, prompt_text: str):
            _raise_if_cancelled()
            current_images = (
                images_list[idx] if images_list and idx < len(images_list) else None
            )
            messages = _build_messages(prompt_text, current_images)

            if response_format:
                for attempt in range(self.validation_attempts):
                    _raise_if_cancelled()
                    try:
                        parse_kwargs = {
                            "model": self.model,
                            "messages": messages,
                            "temperature": self.temperature,
                            "response_format": response_format,
                        }
                        if self.seed is not None:
                            parse_kwargs["seed"] = self.seed
                        if self.reasoning_effort:
                            parse_kwargs["reasoning_effort"] = self.reasoning_effort
                        resp = self._parse_with_backoff(**parse_kwargs)
                        message = resp.choices[0].message
                        parsed = getattr(message, "parsed", None)
                        if parsed is not None:
                            if isinstance(parsed, response_format):
                                return idx, parsed
                            validated = response_format.model_validate(parsed, strict=True)
                            return idx, validated

                        raw = _extract_text_from_message_content(message.content)
                        validated = _coerce_structured_response_from_text(
                            response_format=response_format,
                            raw_text=raw,
                        )
                        return idx, validated
                    except GenerationCancelledError:
                        raise
                    except Exception as e:
                        if _is_insufficient_quota_error(e):
                            raise
                        if isinstance(e, APITimeoutError) or "timed out" in str(e).lower():
                            # Fail fast on timeout so callers can abort the whole model run.
                            raise
                        fallback_error: Exception = e
                        # Fallback for providers that do not fully support parse-mode.
                        try:
                            fallback_text = _call_llm(messages)
                            validated = _coerce_structured_response_from_text(
                                response_format=response_format,
                                raw_text=fallback_text,
                            )
                            logger.debug(
                                "Structured parse fallback succeeded for item %s after parse failure: %s",
                                idx,
                                e,
                            )
                            return idx, validated
                        except GenerationCancelledError:
                            raise
                        except Exception as fallback_parse_exc:
                            if _is_insufficient_quota_error(fallback_parse_exc):
                                raise
                            if isinstance(fallback_parse_exc, APITimeoutError) or "timed out" in str(fallback_parse_exc).lower():
                                raise
                            fallback_error = fallback_parse_exc
                        if attempt == self.validation_attempts - 1:
                            logger.error(
                                "Batch pydantic parse failed for item %s: parse error=%s, fallback error=%s",
                                idx,
                                e,
                                fallback_error,
                            )
                            return idx, f"Error: {fallback_error}"
                        logger.debug(
                            "Validation attempt %s failed for item %s: %s",
                            attempt + 1,
                            idx,
                            fallback_error,
                        )
            else:
                txt = _call_llm(messages)
                return idx, txt

        if not prompts:
            return results

        bar_kwargs = dict(tqdm_kwargs)
        current_desc = bar_kwargs.pop("desc", "Sending requests to the LLM")
        progress_bar = tqdm(total=len(prompts), desc=current_desc, **bar_kwargs)
        executor = ThreadPoolExecutor(max_workers=effective_workers)
        futures: dict = {}
        future_started_at: dict = {}
        next_idx = 0
        completed_count = 0
        cancelled = False
        try:
            while next_idx < len(prompts) and len(futures) < effective_workers:
                _raise_if_cancelled()
                fut = executor.submit(_worker, next_idx, prompts[next_idx])
                futures[fut] = next_idx
                future_started_at[fut] = time.monotonic()
                next_idx += 1

            while futures:
                _raise_if_cancelled()
                if self.per_report_timeout_s is not None:
                    now = time.monotonic()
                    for fut in list(futures.keys()):
                        started = future_started_at.get(fut, now)
                        if (now - started) > self.per_report_timeout_s:
                            self.request_cancellation()
                            raise TimeoutError(
                                f"Per-report wall-clock timeout exceeded "
                                f"({self.per_report_timeout_s:.0f}s)"
                            )
                done, _ = wait(
                    set(futures.keys()),
                    timeout=0.1,
                    return_when=FIRST_COMPLETED,
                )
                if not done:
                    continue

                for fut in done:
                    futures.pop(fut, None)
                    future_started_at.pop(fut, None)
                    try:
                        i, out = fut.result()
                    except GenerationCancelledError:
                        cancelled = True
                        self.request_cancellation()
                        raise
                    results[i] = out
                    completed_count += 1
                    progress_bar.update(1)
                    if progress_callback is not None:
                        progress_callback(completed_count, len(prompts), current_desc)

                while next_idx < len(prompts) and len(futures) < effective_workers:
                    _raise_if_cancelled()
                    fut = executor.submit(_worker, next_idx, prompts[next_idx])
                    futures[fut] = next_idx
                    future_started_at[fut] = time.monotonic()
                    next_idx += 1
        except GenerationCancelledError:
            cancelled = True
            for fut in list(futures.keys()):
                fut.cancel()
            futures.clear()
            future_started_at.clear()
            raise
        finally:
            progress_bar.close()
            if cancelled:
                self.request_cancellation()
                executor.shutdown(wait=False, cancel_futures=True)
            else:
                executor.shutdown(wait=True)

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
