"""Tests for AnthropicClient, driving the real AsyncAnthropic SDK over httpx.MockTransport."""

from typing import Any

import httpx
import pytest
from anthropic import AsyncAnthropic

from tournament_eval.llm.anthropic import AnthropicClient

_HttpMock = Any


def _client(transport: httpx.MockTransport, **cfg: Any) -> AnthropicClient:
    sdk = AsyncAnthropic(api_key="test", http_client=httpx.AsyncClient(transport=transport))
    cfg.setdefault("model_id", "claude")
    return AnthropicClient(sdk, **cfg)


def _message(blocks: list[dict[str, Any]], stop_reason: str = "end_turn") -> dict[str, Any]:
    return {
        "id": "msg_1",
        "type": "message",
        "role": "assistant",
        "model": "claude",
        "content": blocks,
        "stop_reason": stop_reason,
        "stop_sequence": None,
        "usage": {"input_tokens": 1, "output_tokens": 1},
    }


def _text(text: str) -> dict[str, Any]:
    return _message([{"type": "text", "text": text}])


class TestAnthropicClient:
    async def test_generate(self, http_mock: _HttpMock) -> None:
        transport, rec = http_mock(_text("bonjour"))
        client = _client(transport, system_prompt="sys", max_tokens=10, temperature=0.5)
        assert await client.generate("hi") == "bonjour"
        assert client.name == "claude"
        assert rec.json["model"] == "claude"
        assert rec.json["max_tokens"] == 10
        assert rec.json["system"] == "sys"
        assert rec.json["messages"] == [{"role": "user", "content": "hi"}]

    async def test_generate_without_system_omits_it(self, http_mock: _HttpMock) -> None:
        transport, rec = http_mock(_text("x"))
        client = _client(transport)
        await client.generate("hi")
        assert "system" not in rec.json  # omit sentinel, not null

    async def test_generate_refusal_raises(self, http_mock: _HttpMock) -> None:
        transport, _ = http_mock(_message([{"type": "text", "text": ""}], stop_reason="refusal"))
        client = _client(transport)
        with pytest.raises(ValueError, match="refused"):
            await client.generate("hi")

    async def test_generate_no_text_block_raises(self, http_mock: _HttpMock) -> None:
        body = _message([{"type": "tool_use", "id": "t", "name": "x", "input": {}}], stop_reason="tool_use")
        transport, _ = http_mock(body)
        client = _client(transport)
        with pytest.raises(ValueError, match="no text block"):
            await client.generate("hi")

    async def test_structured_parses_and_sends_output_config(self, http_mock: _HttpMock) -> None:
        transport, rec = http_mock(_text('{"x": 1}'))
        client = _client(transport)
        response = await client.generate_structured("hi", {"type": "object"})
        assert response.data == {"x": 1}
        assert rec.json["output_config"] == {"format": {"type": "json_schema", "schema": {"type": "object"}}}

    async def test_structured_non_object_raises(self, http_mock: _HttpMock) -> None:
        transport, _ = http_mock(_text("[1]"))
        client = _client(transport)
        with pytest.raises(ValueError, match="Expected JSON object"):
            await client.generate_structured("hi", {})
