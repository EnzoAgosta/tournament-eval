"""Tests for OllamaClient, driving the real ollama.AsyncClient over httpx.MockTransport."""

from typing import Any

import httpx
import pytest
from ollama import AsyncClient

from tournament_eval.llm.ollama import OllamaClient

_HttpMock = Any


def _client(transport: httpx.MockTransport, **cfg: Any) -> OllamaClient:
    sdk = AsyncClient(host="http://t.local", transport=transport)
    cfg.setdefault("model_id", "llama")
    return OllamaClient(sdk, **cfg)


def _response(content: str, thinking: str | None = None) -> dict[str, Any]:
    message: dict[str, Any] = {"role": "assistant", "content": content}
    if thinking is not None:
        message["thinking"] = thinking
    return {
        "model": "llama",
        "created_at": "2024-01-01T00:00:00.000000Z",
        "message": message,
        "done": True,
        "done_reason": "stop",
    }


class TestOllamaClient:
    async def test_generate(self, http_mock: _HttpMock) -> None:
        transport, rec = http_mock(_response("bonjour"))
        client = _client(transport, system_prompt="sys", temperature=0.5, max_tokens=20, seed=3)
        response = await client.generate("hi")
        assert (response.text, response.reasoning) == ("bonjour", None)
        assert client.name == "llama"
        assert rec.json["model"] == "llama"
        assert rec.json["messages"][0] == {"role": "system", "content": "sys"}
        assert rec.json["options"] == {"temperature": 0.5, "num_predict": 20, "seed": 3}

    async def test_captures_thinking_as_reasoning(self, http_mock: _HttpMock) -> None:
        transport, _ = http_mock(_response("bonjour", thinking="let me think"))
        response = await _client(transport).generate("hi")
        assert (response.text, response.reasoning) == ("bonjour", "let me think")

    async def test_empty_content_returns_empty_string(self, http_mock: _HttpMock) -> None:
        transport, _ = http_mock(_response(""))
        assert (await _client(transport).generate("hi")).text == ""

    async def test_default_options_only_temperature(self, http_mock: _HttpMock) -> None:
        transport, rec = http_mock(_response("x"))
        await _client(transport).generate("hi")
        assert rec.json["options"] == {"temperature": 1.0}

    async def test_structured_parses_and_sends_format(self, http_mock: _HttpMock) -> None:
        transport, rec = http_mock(_response('{"x": 1}'))
        response = await _client(transport).generate_structured("hi", {"type": "object"})
        assert response.data == {"x": 1}
        assert rec.json["format"] == {"type": "object"}

    async def test_structured_non_object_raises(self, http_mock: _HttpMock) -> None:
        transport, _ = http_mock(_response("[1]"))
        with pytest.raises(ValueError, match="Expected JSON object"):
            await _client(transport).generate_structured("hi", {})
