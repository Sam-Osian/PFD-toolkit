import sys
import types
import pytest

# Provide a minimal openai stub before importing LLM
class DummyClient:
    def __init__(self, *args, **kwargs):
        pass

class DummyRateLimitError(Exception):
    pass

dummy_openai = types.SimpleNamespace(Client=DummyClient, RateLimitError=DummyRateLimitError, base_url="http://example.com")
sys.modules['openai'] = dummy_openai

from pfd_toolkit.llm import LLM


def test_generate_batch_sequential(monkeypatch):
    llm = LLM(api_key="test", max_workers=1)

    def fake_safe(messages, temperature=0.0):
        return messages[0]["content"][0]["text"].upper()

    monkeypatch.setattr(llm, "_safe_generate_impl", fake_safe)
    results = llm.generate_batch(["one", "two"], max_workers=1)
    assert results == ["ONE", "TWO"]


def test_generate_batch_parallel(monkeypatch):
    llm = LLM(api_key="test", max_workers=4)

    def fake_safe(messages, temperature=0.0):
        return messages[0]["content"][0]["text"].upper()

    monkeypatch.setattr(llm, "_safe_generate_impl", fake_safe)
    results = llm.generate_batch(["a", "b", "c"], max_workers=3)
    assert results == ["A", "B", "C"]
