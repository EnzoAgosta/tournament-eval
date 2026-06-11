"""Tests for the OpenAI-compatible LLM client."""

import json
import os
import unittest
import unittest.mock

import httpx
import pytest

from tournament_eval.llm import OpenAILLMClient, OpenAIModelConfig


def _chat_response(content: str) -> dict[str, object]:
    """Build a minimal OpenAI chat-completions response body."""
    return {"choices": [{"message": {"role": "assistant", "content": content}}]}


class TestOpenAIClient:
    def test_name_returns_config_model_name(self) -> None:
        client = OpenAILLMClient(OpenAIModelConfig(model_name="gpt-4o"))
        assert client.name == "gpt-4o"


class TestOpenAIUrl:
    def test_default_base_url(self) -> None:
        client = OpenAILLMClient(OpenAIModelConfig(model_name="gpt-4o"))
        assert client._url == "https://api.openai.com/v1/chat/completions"

    def test_custom_base_url_trailing_slash_stripped(self) -> None:
        client = OpenAILLMClient(
            OpenAIModelConfig(model_name="ft", base_url="http://localhost:8080/v1/")
        )
        assert client._url == "http://localhost:8080/v1/chat/completions"


class TestOpenAIHeaders:
    def test_explicit_api_key(self) -> None:
        client = OpenAILLMClient(OpenAIModelConfig(model_name="gpt-4o", api_key="sk-x"))
        assert client._headers["Authorization"] == "Bearer sk-x"

    def test_env_api_key(self) -> None:
        with unittest.mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-env"}):
            client = OpenAILLMClient(OpenAIModelConfig(model_name="gpt-4o"))
            assert client._headers["Authorization"] == "Bearer sk-env"

    def test_no_api_key(self) -> None:
        client = OpenAILLMClient(OpenAIModelConfig(model_name="ft"))
        assert "Authorization" not in client._headers


class TestOpenAIPayload:
    def test_basic_payload(self) -> None:
        client = OpenAILLMClient(OpenAIModelConfig(model_name="gpt-4o"))
        payload = client._payload("hello")
        assert payload["model"] == "gpt-4o"
        assert payload["messages"] == [{"role": "user", "content": "hello"}]
        assert payload["stream"] is False
        assert "response_format" not in payload
        assert "max_tokens" not in payload
        assert "seed" not in payload

    def test_temperature_included(self) -> None:
        client = OpenAILLMClient(
            OpenAIModelConfig(model_name="gpt-4o", temperature=0.5)
        )
        assert client._payload("hi")["temperature"] == 0.5

    def test_max_tokens_mapped(self) -> None:
        client = OpenAILLMClient(OpenAIModelConfig(model_name="gpt-4o", max_tokens=128))
        assert client._payload("hi")["max_tokens"] == 128

    def test_seed_included(self) -> None:
        client = OpenAILLMClient(OpenAIModelConfig(model_name="gpt-4o", seed=23012023))
        assert client._payload("hi")["seed"] == 23012023

    def test_system_prompt_prepended(self) -> None:
        client = OpenAILLMClient(
            OpenAIModelConfig(model_name="gpt-4o", system_prompt="Be terse")
        )
        assert client._payload("hi")["messages"] == [
            {"role": "system", "content": "Be terse"},
            {"role": "user", "content": "hi"},
        ]

    def test_response_format_included(self) -> None:
        client = OpenAILLMClient(OpenAIModelConfig(model_name="gpt-4o"))
        rf: dict[str, object] = {"type": "json_schema"}
        assert client._payload("hi", response_format=rf)["response_format"] == rf


class TestOpenAIRequest:
    async def test_success_on_first_attempt(self) -> None:
        called = 0

        def handler(_request: httpx.Request) -> httpx.Response:
            nonlocal called
            called += 1
            return httpx.Response(200, json=_chat_response("hi back"))

        client = OpenAILLMClient(OpenAIModelConfig(model_name="gpt-4o"))
        client._transport = httpx.MockTransport(handler)
        body = await client._request("hi")
        assert called == 1
        assert body == _chat_response("hi back")

    async def test_retries_on_failure(self) -> None:
        called = 0

        def handler(_request: httpx.Request) -> httpx.Response:
            nonlocal called
            called += 1
            if called < 2:
                return httpx.Response(500, text="boom")
            return httpx.Response(200, json=_chat_response("ok"))

        client = OpenAILLMClient(OpenAIModelConfig(model_name="gpt-4o", retry_count=3))
        client._transport = httpx.MockTransport(handler)
        body = await client._request("hi")
        assert called == 2
        assert body == _chat_response("ok")

    async def test_all_retries_exhausted(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="always")

        client = OpenAILLMClient(OpenAIModelConfig(model_name="gpt-4o", retry_count=2))
        client._transport = httpx.MockTransport(handler)
        with pytest.raises(httpx.HTTPStatusError):
            await client._request("hi")


class TestOpenAIGenerate:
    async def test_returns_content(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_chat_response("リモートコントロール"))

        client = OpenAILLMClient(OpenAIModelConfig(model_name="ft"))
        client._transport = httpx.MockTransport(handler)
        assert await client.generate("translate") == "リモートコントロール"


class TestOpenAIContentErrors:
    @pytest.mark.parametrize(
        ("body", "match"),
        [
            ({}, "non-empty 'choices'"),
            ({"choices": []}, "non-empty 'choices'"),
            ({"choices": ["x"]}, "not an object"),
            ({"choices": [{}]}, "missing 'message'"),
            ({"choices": [{"message": {"refusal": "nope"}}]}, "refused"),
            ({"choices": [{"message": {"content": 123}}]}, "not a string"),
        ],
    )
    async def test_content_error_paths(
        self, body: dict[str, object], match: str
    ) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=body)

        client = OpenAILLMClient(OpenAIModelConfig(model_name="ft"))
        client._transport = httpx.MockTransport(handler)
        with pytest.raises(ValueError, match=match):
            await client.generate("x")


class TestOpenAIGenerateStructured:
    async def test_parses_json_and_sends_schema(self) -> None:
        captured: dict[str, object] = {}

        def handler(request: httpx.Request) -> httpx.Response:
            captured.update(json.loads(request.content))
            return httpx.Response(200, json=_chat_response('{"ranking": ["A", "B"]}'))

        client = OpenAILLMClient(OpenAIModelConfig(model_name="gpt-4o"))
        client._transport = httpx.MockTransport(handler)
        schema: dict[str, object] = {"type": "object"}
        result = await client.generate_structured("rank", schema)

        assert result.data == {"ranking": ["A", "B"]}
        assert result.raw == '{"ranking": ["A", "B"]}'
        rf = captured["response_format"]
        assert isinstance(rf, dict)
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["schema"] == schema
        assert rf["json_schema"]["strict"] is True

    async def test_rejects_non_dict_json(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json=_chat_response("[1, 2, 3]"))

        client = OpenAILLMClient(OpenAIModelConfig(model_name="gpt-4o"))
        client._transport = httpx.MockTransport(handler)
        with pytest.raises(ValueError, match="Expected JSON object"):
            await client.generate_structured("rank", {})
