"""Tests for OpenAIClient, driving the real AsyncOpenAI SDK over httpx.MockTransport.

Canned JSON is parsed by the real Responses-API models, so these exercise the SDK's
request serialization and our response/refusal handling end to end.
"""

from typing import Any

import httpx
import pytest
from openai import AsyncOpenAI

from tournament_eval.llm.openai import OpenAIClient

_HttpMock = Any


def _client(transport: httpx.MockTransport, **cfg: Any) -> OpenAIClient:
    sdk = AsyncOpenAI(api_key="test", http_client=httpx.AsyncClient(transport=transport))
    cfg.setdefault("model_id", "gpt-x")
    return OpenAIClient(sdk, **cfg)


def _response(text: str) -> dict[str, Any]:
    """A minimal Responses-API object whose output_text is ``text``."""
    return {
        "id": "resp_1",
        "object": "response",
        "created_at": 0,
        "status": "completed",
        "model": "gpt-x",
        "output": [
            {
                "type": "message",
                "id": "msg_1",
                "status": "completed",
                "role": "assistant",
                "content": [{"type": "output_text", "text": text, "annotations": []}],
            }
        ],
        "parallel_tool_calls": False,
        "tool_choice": "auto",
        "tools": [],
        "temperature": 1.0,
        "top_p": 1.0,
        "usage": None,
        "metadata": {},
    }


def _refusal() -> dict[str, Any]:
    body = _response("")
    body["output"][0]["content"] = [{"type": "refusal", "refusal": "policy"}]
    return body


def _with_reasoning(text: str, summary: str) -> dict[str, Any]:
    body = _response(text)
    body["output"].insert(
        0, {"type": "reasoning", "id": "rs_1", "summary": [{"type": "summary_text", "text": summary}]}
    )
    return body


class TestOpenAIClient:
    async def test_generate(self, http_mock: _HttpMock) -> None:
        transport, rec = http_mock(_response("bonjour"))
        client = _client(transport, system_prompt="sys", max_tokens=20)
        assert (await client.generate("hi")).text == "bonjour"
        assert client.name == "gpt-x"
        assert rec.json["model"] == "gpt-x"
        assert rec.json["input"] == "hi"
        assert rec.json["instructions"] == "sys"
        assert rec.json["max_output_tokens"] == 20

    async def test_generate_refusal_raises(self, http_mock: _HttpMock) -> None:
        transport, _ = http_mock(_refusal())
        client = _client(transport)
        with pytest.raises(ValueError, match="refused"):
            await client.generate("hi")

    async def test_structured_parses_and_sends_schema(self, http_mock: _HttpMock) -> None:
        transport, rec = http_mock(_response('{"x": 1}'))
        client = _client(transport)
        response = await client.generate_structured("hi", {"type": "object"})
        assert response.data == {"x": 1}
        assert rec.json["text"]["format"]["type"] == "json_schema"
        assert rec.json["text"]["format"]["schema"] == {"type": "object"}

    async def test_structured_refusal_raises(self, http_mock: _HttpMock) -> None:
        transport, _ = http_mock(_refusal())
        client = _client(transport)
        with pytest.raises(ValueError, match="refused"):
            await client.generate_structured("hi", {})

    async def test_structured_non_object_raises(self, http_mock: _HttpMock) -> None:
        transport, _ = http_mock(_response("[1]"))
        client = _client(transport)
        with pytest.raises(ValueError, match="Expected JSON object"):
            await client.generate_structured("hi", {})

    async def test_reasoning_effort_sends_param_and_captures_summary(self, http_mock: _HttpMock) -> None:
        transport, rec = http_mock(_with_reasoning("answer", "thinking hard"))
        client = _client(transport, reasoning_effort="max")
        response = await client.generate("hi")
        assert (response.text, response.reasoning) == ("answer", "thinking hard")
        assert rec.json["reasoning"] == {"effort": "xhigh", "summary": "auto"}  # max clamps to xhigh

    async def test_reasoning_unset_sends_null(self, http_mock: _HttpMock) -> None:
        # Like instructions / max_output_tokens, the SDK serializes the unset
        # reasoning config as an explicit null rather than omitting it.
        transport, rec = http_mock(_response("x"))
        await _client(transport).generate("hi")
        assert rec.json["reasoning"] is None
