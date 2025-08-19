import sys
import types
import pytest
import backoff
import time
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

dummy_openai = types.SimpleNamespace(
    Client=DummyClient,
    RateLimitError=DummyRateLimitError,
    APIConnectionError=DummyAPIConnectionError,
    APITimeoutError=DummyAPITimeoutError,
    base_url="http://example.com",
)
sys.modules['openai'] = dummy_openai

import importlib
import pfd_toolkit.llm as llm_module
importlib.reload(llm_module)
from pfd_toolkit.llm import LLM, _strip_json_markdown


def test_generate_sequential(monkeypatch):
    llm = LLM(api_key="test", max_workers=1, timeout=1)

    def fake_create(model, messages, temperature=0.0, seed=None):
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

    def fake_create(model, messages, temperature=0.0, seed=None):
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
