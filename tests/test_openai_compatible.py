"""Tests for the built-in httpx OpenAICompatibleClient, via httpx.MockTransport."""

import asyncio
from typing import Any

import httpx
import pytest

from tournament_eval.llm.openai_compatible import OpenAICompatibleClient

_HttpMock = Any


def _client(transport: httpx.MockTransport | None = None, **cfg: Any) -> OpenAICompatibleClient:
    cfg.setdefault("model_id", "m")
    cfg.setdefault("base_url", "http://t.local/v1")
    cfg.setdefault("api_key", "k")
    client = OpenAICompatibleClient(**cfg)
    client._transport = transport
    return client


def _chat(content: str) -> dict[str, Any]:
    return {"choices": [{"message": {"content": content}}]}


async def _noop(*_args: object, **_kwargs: object) -> None:
    return None


class TestEndpointAndHeaders:
    def test_endpoint_appends_chat_completions(self) -> None:
        assert _client()._endpoint_url == "http://t.local/v1/chat/completions"

    def test_custom_base_url_builds_endpoint(self) -> None:
        client = OpenAICompatibleClient(model_id="m", base_url="http://x/v1", api_key="k")
        assert client._endpoint_url == "http://x/v1/chat/completions"

    def test_headers_with_key(self) -> None:
        assert OpenAICompatibleClient(model_id="m", base_url="b", api_key="sk")._build_headers()["Authorization"] == (
            "Bearer sk"
        )

    def test_headers_without_key(self) -> None:
        headers = OpenAICompatibleClient(model_id="m", base_url="b", api_key="")._build_headers()
        assert "Authorization" not in headers

    def test_name_defaults_to_model_id(self) -> None:
        assert OpenAICompatibleClient(model_id="gpt-x", base_url="b", api_key="k").name == "gpt-x"

    def test_name_uses_explicit_name(self) -> None:
        assert OpenAICompatibleClient(model_id="gpt-x", base_url="b", api_key="k", name="local").name == "local"


class TestGenerate:
    async def test_returns_content_and_sends_payload(self, http_mock: _HttpMock) -> None:
        transport, rec = http_mock(_chat("bonjour"))
        async with _client(transport) as client:
            assert await client.generate("hi") == "bonjour"
        assert rec.json["model"] == "m"
        assert rec.json["messages"] == [{"role": "user", "content": "hi"}]
        assert rec.json["stream"] is False
        assert rec.json["temperature"] == 1.0

    async def test_system_prompt_prepended(self, http_mock: _HttpMock) -> None:
        transport, rec = http_mock(_chat("x"))
        async with _client(transport, system_prompt="be terse") as client:
            await client.generate("hi")
        assert rec.json["messages"][0] == {"role": "system", "content": "be terse"}

    async def test_max_tokens_and_seed_sent(self, http_mock: _HttpMock) -> None:
        transport, rec = http_mock(_chat("x"))
        async with _client(transport, max_tokens=10, seed=7) as client:
            await client.generate("hi")
        assert rec.json["max_tokens"] == 10
        assert rec.json["seed"] == 7

    async def test_defaults_omit_max_tokens_and_seed(self, http_mock: _HttpMock) -> None:
        transport, rec = http_mock(_chat("x"))
        async with _client(transport) as client:
            await client.generate("hi")
        assert "max_tokens" not in rec.json
        assert "seed" not in rec.json


class TestGenerateStructured:
    async def test_parses_object_and_sends_schema(self, http_mock: _HttpMock) -> None:
        transport, rec = http_mock(_chat('{"a": 1}'))
        async with _client(transport) as client:
            response = await client.generate_structured("hi", {"type": "object"})
        assert response.data == {"a": 1}
        assert response.raw == '{"a": 1}'
        fmt = rec.json["response_format"]
        assert fmt["type"] == "json_schema"
        assert fmt["json_schema"]["schema"] == {"type": "object"}

    async def test_non_object_json_raises(self, http_mock: _HttpMock) -> None:
        transport, _ = http_mock(_chat("[1, 2]"))
        async with _client(transport) as client:
            with pytest.raises(ValueError, match="Expected JSON object"):
                await client.generate_structured("hi", {})


class TestExtractionErrors:
    @pytest.mark.parametrize(
        ("body", "match"),
        [
            ({}, "non-empty 'choices'"),
            ({"choices": []}, "non-empty 'choices'"),
            ({"choices": ["x"]}, "not an object"),
            ({"choices": [{}]}, "missing 'message'"),
            ({"choices": [{"message": {"refusal": "no"}}]}, "refused"),
            ({"choices": [{"message": {"content": 5}}]}, "not a string"),
        ],
    )
    async def test_malformed_bodies_raise(self, http_mock: _HttpMock, body: dict[str, Any], match: str) -> None:
        transport, _ = http_mock(body)
        async with _client(transport) as client:
            with pytest.raises(ValueError, match=match):
                await client.generate("hi")


class TestRetries:
    async def test_retries_then_succeeds(self, http_mock: _HttpMock, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("tournament_eval.llm.openai_compatible.asyncio.sleep", _noop)
        transport, rec = http_mock((500, {}), (200, _chat("ok")))
        async with _client(transport) as client:
            assert await client.generate("hi") == "ok"
        assert rec.calls == 2

    async def test_exhausts_and_raises(self, http_mock: _HttpMock, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.setattr("tournament_eval.llm.openai_compatible.asyncio.sleep", _noop)
        transport, rec = http_mock((500, {}))
        async with _client(transport, retry_count=2) as client:
            with pytest.raises(httpx.HTTPStatusError):
                await client.generate("hi")
        assert rec.calls == 2


class TestLifecycleAndConcurrency:
    async def test_generate_outside_context_raises(self) -> None:
        client = OpenAICompatibleClient(model_id="m", base_url="b", api_key="k")
        with pytest.raises(RuntimeError, match="async context manager"):
            await client.generate("hi")

    async def test_aclose_without_open_is_noop(self) -> None:
        await OpenAICompatibleClient(model_id="m", base_url="b", api_key="k").__aexit__()

    async def test_double_enter_keeps_one_pool(self) -> None:
        client = OpenAICompatibleClient(model_id="m", base_url="b", api_key="k")
        async with client:
            first = client._http_client
            await client.__aenter__()
            assert client._http_client is first

    async def test_max_concurrency_caps_in_flight(
        self, tracking_transport: tuple[httpx.MockTransport, dict[str, int]]
    ) -> None:
        transport, state = tracking_transport
        async with _client(transport, max_concurrency=2) as client:
            await asyncio.gather(*(client.generate("hi") for _ in range(6)))
        assert state["peak"] <= 2

    async def test_unbounded_runs_concurrently(
        self, tracking_transport: tuple[httpx.MockTransport, dict[str, int]]
    ) -> None:
        transport, state = tracking_transport
        async with _client(transport) as client:
            await asyncio.gather(*(client.generate("hi") for _ in range(5)))
        assert state["peak"] >= 2
