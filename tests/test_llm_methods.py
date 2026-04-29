import sys
import types
import pytest
import backoff
import time
import threading
from pydantic import BaseModel

# Provide a minimal openai stub before importing LLM
class DummyClient:
    def __init__(self, *args, **kwargs):
        # Minimal structure matching openai.Client
        self.beta = types.SimpleNamespace(
            chat=types.SimpleNamespace(
                completions=types.SimpleNamespace(parse=lambda **kwargs: None)
            )
        )
        self.chat = types.SimpleNamespace(
            completions=types.SimpleNamespace(create=lambda **kwargs: None)
        )

class DummyRateLimitError(Exception):
    pass

class DummyAPIConnectionError(Exception):
    pass

class DummyAPITimeoutError(DummyAPIConnectionError):
    pass

class DummyBadRequestError(Exception):
    pass

dummy_openai = types.SimpleNamespace(
    Client=DummyClient,
    BadRequestError=DummyBadRequestError,
    RateLimitError=DummyRateLimitError,
    APIConnectionError=DummyAPIConnectionError,
    APITimeoutError=DummyAPITimeoutError,
    base_url="http://example.com",
)
sys.modules['openai'] = dummy_openai

import importlib
import pfd_toolkit.llm as llm_module
importlib.reload(llm_module)
from pfd_toolkit.llm import (
    LLM,
    GenerationCancelledError,
    _is_insufficient_quota_error,
    _strip_json_markdown,
)


def test_generate_sequential(monkeypatch):
    llm = LLM(api_key="test", max_workers=1, timeout=1)

    def fake_create(model, messages, temperature=0.0, seed=None, reasoning_effort=None):
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content=messages[0]["content"][0]["text"].upper())
                )
            ],
            usage=types.SimpleNamespace(total_tokens=0),
        )

    monkeypatch.setattr(llm.client.chat.completions, "create", fake_create)

    results = llm.generate(["one", "two"], max_workers=1)
    assert results == ["ONE", "TWO"]


def test_generate_parallel(monkeypatch):
    llm = LLM(api_key="test", max_workers=4, timeout=1)

    def fake_create(model, messages, temperature=0.0, seed=None, reasoning_effort=None):
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content=messages[0]["content"][0]["text"].upper())
                )
            ],
            usage=types.SimpleNamespace(total_tokens=0),
        )

    monkeypatch.setattr(llm.client.chat.completions, "create", fake_create)

    results = llm.generate(["a", "b", "c"], max_workers=3)
    assert results == ["A", "B", "C"]


def test_generate_markdown_wrapped_json(monkeypatch):
    """Responses wrapped in markdown code fences should be parsed correctly."""

    class TopicMatch(BaseModel):
        matches_topic: str

    llm = LLM(api_key="test")

    def fake_parse(**kwargs):
        content = "```json\n{\n  \"matches_topic\": \"Yes\"\n}\n```"
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(message=types.SimpleNamespace(content=content))
            ]
        )

    monkeypatch.setattr(llm, "_parse_with_backoff", fake_parse)

    result = llm.generate(["prompt"], response_format=TopicMatch)
    assert result[0].matches_topic == "Yes"


def test_generate_retries_without_unsupported_temperature(monkeypatch):
    llm = LLM(api_key="test", model="gpt-5.4", max_workers=1, timeout=1, reasoning_effort="none")
    calls = []

    def fake_create(**kwargs):
        calls.append(dict(kwargs))
        if len(calls) == 1:
            exc = dummy_openai.BadRequestError("unsupported temperature")
            exc.body = {
                "error": {
                    "message": "Unsupported parameter: 'temperature' is not supported with this model.",
                    "param": "temperature",
                    "code": "unsupported_parameter",
                }
            }
            raise exc
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="OK"))],
            usage=types.SimpleNamespace(total_tokens=0),
        )

    monkeypatch.setattr(llm.client.chat.completions, "create", fake_create)

    result = llm.generate(["prompt"], max_workers=1)
    assert result == ["OK"]
    assert "temperature" in calls[0]
    assert "temperature" not in calls[1]
    assert calls[1].get("reasoning_effort") == "none"


def test_generate_retries_parse_without_unsupported_reasoning_effort(monkeypatch):
    class TopicMatch(BaseModel):
        matches_topic: str

    llm = LLM(api_key="test", model="gpt-5.4", max_workers=1, timeout=1, reasoning_effort="none")
    calls = []

    def fake_parse(**kwargs):
        calls.append(dict(kwargs))
        if len(calls) == 1:
            exc = dummy_openai.BadRequestError("unsupported reasoning_effort")
            exc.body = {
                "error": {
                    "message": "Unsupported parameter: 'reasoning_effort' is not supported with this model.",
                    "param": "reasoning_effort",
                    "code": "unsupported_parameter",
                }
            }
            raise exc
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content='{"matches_topic":"Yes"}'))]
        )

    monkeypatch.setattr(llm.client.beta.chat.completions, "parse", fake_parse)

    result = llm.generate(["prompt"], response_format=TopicMatch, max_workers=1)
    assert result[0].matches_topic == "Yes"
    assert "reasoning_effort" in calls[0]
    assert "reasoning_effort" not in calls[1]


@pytest.mark.parametrize(
    "wrapped",
    [
        "```json\n{\"a\":1}\n```",
        "```JSON\n{\n  \"a\": 1\n}\n```",
        "text before```json\n{\"a\":1}\n```text after",
        "\ufeff```json\n{\"a\":1}\n```",
        "```\n{\"a\":1}\n```",
    ],
)
def test_strip_json_markdown_variants(wrapped):
    import json

    cleaned = _strip_json_markdown(wrapped)
    assert json.loads(cleaned) == {"a": 1}


def test_parse_with_backoff_retries(monkeypatch):
    # Avoid actual sleeping during backoff
    def zero_wait(*args, **kwargs):
        for _ in range(2):
            yield 0
    monkeypatch.setattr(backoff, "expo", zero_wait)
    # Patch sleep in both time module and backoff._sync to avoid real delays
    monkeypatch.setattr(time, "sleep", lambda s: None)
    import backoff._sync as backoff_sync
    monkeypatch.setattr(backoff_sync.time, "sleep", lambda s: None)

    llm = LLM(api_key="test", timeout=1)

    call_counter = {"n": 0}

    def flaky_parse(**kwargs):
        call_counter["n"] += 1
        if call_counter["n"] == 1:
            raise dummy_openai.APIConnectionError("boom")
        return types.SimpleNamespace(
            choices=[types.SimpleNamespace(message=types.SimpleNamespace(content="ok"))]
        )

    monkeypatch.setattr(llm.client.beta.chat.completions, "parse", flaky_parse)

    resp = llm._parse_with_backoff(model="gpt", messages=[], temperature=0)

    assert resp.choices[0].message.content == "ok"
    assert call_counter["n"] == 2


def test_estimate_tokens():
    llm = LLM(api_key="test")
    counts = llm.estimate_tokens(["hello"])
    assert counts == [len("hello".split())]


def test_call_llm_fallback_success(monkeypatch):
    llm = LLM(api_key="test")

    # Pretend PDF conversion succeeded
    monkeypatch.setattr(llm, "_pdf_bytes_to_base64_images", lambda b, dpi=200: ["img"])  # noqa: E501

    def fake_generate(prompts, images_list=None, response_format=None, **kwargs):
        assert images_list == [["img"]]
        return [response_format(foo="BAR")]

    monkeypatch.setattr(llm, "generate", fake_generate)

    out = llm._call_llm_fallback(b"pdf", {"foo": "prompt"})
    assert out == {"foo": "BAR"}


def test_call_llm_fallback_error_string(monkeypatch):
    llm = LLM(api_key="test")

    monkeypatch.setattr(llm, "_pdf_bytes_to_base64_images", lambda b, dpi=200: [])
    monkeypatch.setattr(llm, "generate", lambda **kwargs: ["Error: boom"])

    out = llm._call_llm_fallback(b"pdf", {"foo": "prompt"})
    assert out == {"foo": "LLM Fallback error"}


def test_call_llm_fallback_missing_field(monkeypatch):
    llm = LLM(api_key="test")

    monkeypatch.setattr(llm, "_pdf_bytes_to_base64_images", lambda b, dpi=200: [])
    monkeypatch.setattr(llm, "generate", lambda **kwargs: [{}])

    out = llm._call_llm_fallback(None, {"foo": "prompt"})
    assert out == {"foo": str(llm_module.GeneralConfig.NOT_FOUND_TEXT) + " in LLM response"}


def test_generate_can_cancel_before_submission(monkeypatch):
    llm = LLM(api_key="test", max_workers=1, timeout=1)
    call_counter = {"n": 0}

    def fake_create(model, messages, temperature=0.0, seed=None, reasoning_effort=None):
        call_counter["n"] += 1
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content=messages[0]["content"][0]["text"].upper())
                )
            ],
            usage=types.SimpleNamespace(total_tokens=0),
        )

    monkeypatch.setattr(llm.client.chat.completions, "create", fake_create)

    with pytest.raises(GenerationCancelledError):
        llm.generate(
            ["one", "two"],
            cancellation_check=lambda: True,
        )

    assert call_counter["n"] == 0


def test_generate_stops_submitting_after_cancellation(monkeypatch):
    llm = LLM(api_key="test", max_workers=1, timeout=1)
    call_counter = {"n": 0}
    cancel_state = {"value": False}

    def fake_create(model, messages, temperature=0.0, seed=None, reasoning_effort=None):
        call_counter["n"] += 1
        return types.SimpleNamespace(
            choices=[
                types.SimpleNamespace(
                    message=types.SimpleNamespace(content=messages[0]["content"][0]["text"].upper())
                )
            ],
            usage=types.SimpleNamespace(total_tokens=0),
        )

    def progress_callback(done, total, desc):
        if done >= 1:
            cancel_state["value"] = True

    monkeypatch.setattr(llm.client.chat.completions, "create", fake_create)

    with pytest.raises(GenerationCancelledError):
        llm.generate(
            ["one", "two", "three"],
            max_workers=1,
            progress_callback=progress_callback,
            cancellation_check=lambda: cancel_state["value"],
        )

    assert call_counter["n"] == 1


def test_request_cancellation_marks_state_and_closes_client():
    llm = LLM(api_key="test", max_workers=1, timeout=1)
    closed = threading.Event()

    def fake_close():
        closed.set()

    llm.client.close = fake_close
    llm.request_cancellation()

    assert llm._cancel_requested.is_set()
    assert closed.is_set()


def test_is_insufficient_quota_error_from_message():
    exc = Exception("RateLimitError: code=insufficient_quota")
    assert _is_insufficient_quota_error(exc) is True


def test_is_insufficient_quota_error_from_body():
    exc = Exception("rate limit")
    exc.body = {"error": {"code": "insufficient_quota", "message": "quota exhausted"}}
    assert _is_insufficient_quota_error(exc) is True
