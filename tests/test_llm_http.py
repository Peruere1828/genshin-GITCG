"""HTTP-level behaviour of the LLM client (no network).

Pins the robustness rules that are easy to regress: ``User-Agent`` (some CDNs
reject urllib's default), ``429``/``5xx`` retries that respect ``Retry-After``,
empty/whitespace content treated as a failure, and cache/cost accounting.
"""

from __future__ import annotations

import io
import json
import urllib.error

import pytest

import common.llm as llm
from common.llm import LLMClient, LLMConfig


class _FakeResponse(io.BytesIO):
    def __enter__(self):
        return self

    def __exit__(self, *exc):
        return False


def _config(**overrides) -> LLMConfig:
    base = dict(
        base_url="https://example.test/v1",
        model="deepseek-v4.1-flash",
        api_key="oc_sk_test_key_1234567890",
        timeout=5.0,
    )
    base.update(overrides)
    return LLMConfig(**base)


def _body(content: str, *, cached: int = 0, cost: float | None = 0.0, prompt: int = 10,
          completion: int = 3) -> _FakeResponse:
    usage = {
        "prompt_tokens": prompt,
        "completion_tokens": completion,
        "prompt_tokens_details": {"cached_tokens": cached},
    }
    if cost is not None:
        usage["cost"] = cost
    return _FakeResponse(json.dumps({
        "choices": [{"message": {"content": content}}],
        "usage": usage,
    }).encode())


def test_chat_sends_user_agent(monkeypatch):
    seen: list = []

    def fake_urlopen(request, timeout=None):
        seen.append(request)
        return _body('{"choice": 1}')

    monkeypatch.setattr(llm.urllib.request, "urlopen", fake_urlopen)
    client = LLMClient(_config())
    assert client.chat([{"role": "user", "content": "hi"}]) == '{"choice": 1}'

    request = seen[0]
    headers = {key.lower(): value for key, value in request.header_items()}
    assert headers["user-agent"] == "genshin-GITCG/1.0"
    assert headers["authorization"].startswith("Bearer oc_sk_")
    payload = json.loads(request.data.decode())
    assert payload["model"] == "deepseek-v4.1-flash"
    assert payload["messages"] == [{"role": "user", "content": "hi"}]


def test_chat_treats_empty_content_as_failure(monkeypatch):
    monkeypatch.setattr(llm.urllib.request, "urlopen", lambda *a, **k: _body(""))
    client = LLMClient(_config())
    assert client.chat([{"role": "user", "content": "hi"}]) is None
    assert client.usage.errors == 1


def test_chat_treats_whitespace_content_as_failure(monkeypatch):
    monkeypatch.setattr(llm.urllib.request, "urlopen", lambda *a, **k: _body("  \n\t "))
    client = LLMClient(_config())
    assert client.chat([{"role": "user", "content": "hi"}]) is None
    assert client.usage.errors == 1


def test_chat_records_cache_and_cost(monkeypatch):
    monkeypatch.setattr(
        llm.urllib.request, "urlopen",
        lambda *a, **k: _body('{"choice": 0}', cached=800, cost=0.00002, prompt=1000),
    )
    client = LLMClient(_config())
    client.chat([{"role": "user", "content": "hi"}])
    usage = client.usage.as_dict()
    assert usage["cached_tokens"] == 800
    assert usage["cost_usd"] == pytest.approx(0.00002)
    assert usage["prompt_tokens"] == 1000


def test_chat_skips_cost_when_endpoint_omits_it(monkeypatch):
    monkeypatch.setattr(
        llm.urllib.request, "urlopen",
        lambda *a, **k: _body('{"choice": 0}', cost=None, prompt=7),
    )
    client = LLMClient(_config())
    assert client.chat([{"role": "user", "content": "hi"}]) == '{"choice": 0}'
    usage = client.usage.as_dict()
    assert usage["cost_usd"] == 0.0  # never guessed
    assert usage["prompt_tokens"] == 7


def test_chat_honours_retry_after_and_retries_once(monkeypatch):
    calls = {"n": 0}

    def flaky(request, timeout=None):
        calls["n"] += 1
        if calls["n"] == 1:
            raise urllib.error.HTTPError(
                request.full_url, 429, "rate limited",
                {"Retry-After": "5"}, io.BytesIO(b'{"error":"rate"}'),
            )
        return _body('{"choice": 2}')

    sleeps: list = []
    monkeypatch.setattr(llm.urllib.request, "urlopen", flaky)
    monkeypatch.setattr(llm.time, "sleep", lambda s: sleeps.append(s))
    client = LLMClient(_config())
    assert client.chat([{"role": "user", "content": "hi"}]) == '{"choice": 2}'
    assert calls["n"] == 2
    assert sleeps == [5.0]  # Retry-After honoured over the 2**attempt default
    assert client.usage.errors == 0


def test_chat_does_not_retry_client_errors(monkeypatch):
    calls = {"n": 0}

    def bad_request(request, timeout=None):
        calls["n"] += 1
        raise urllib.error.HTTPError(
            request.full_url, 400, "bad request",
            {}, io.BytesIO(b'{"error":"bad"}'),
        )

    monkeypatch.setattr(llm.urllib.request, "urlopen", bad_request)
    monkeypatch.setattr(llm.time, "sleep", lambda _s: None)
    client = LLMClient(_config())
    assert client.chat([{"role": "user", "content": "hi"}]) is None
    assert calls["n"] == 1
    assert client.usage.errors == 1


def test_chat_retries_then_fails_after_three_attempts(monkeypatch):
    calls = {"n": 0}

    def always_503(request, timeout=None):
        calls["n"] += 1
        raise urllib.error.HTTPError(
            request.full_url, 503, "unavailable",
            {}, io.BytesIO(b'{"error":"down"}'),
        )

    monkeypatch.setattr(llm.urllib.request, "urlopen", always_503)
    monkeypatch.setattr(llm.time, "sleep", lambda _s: None)
    client = LLMClient(_config())
    assert client.chat([{"role": "user", "content": "hi"}]) is None
    assert calls["n"] == 3
    assert client.usage.errors == 1  # counted once, on final failure


def test_redact_covers_sk_and_opencode_prefixes():
    text = "key sk-abcdef1234567890 and oc_sk_abcdef1234567890 done"
    redacted = llm.redact(text)
    assert "sk-abcdef1234567890" not in redacted
    assert "oc_sk_abcdef1234567890" not in redacted
    assert redacted.count("sk-***") == 2
