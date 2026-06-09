"""Tests for the Ollama LLM client."""

import httpx
import pytest

from tournament_eval.llm import OllamaLLMClient, OllamaModelConfig


class TestOllamaClient:
    def test_name_returns_config_model_name(self) -> None:
        cfg = OllamaModelConfig(model_name="llama3")
        client = OllamaLLMClient(cfg)
        assert client.name == "llama3"


class TestOllamaPayload:
    def test_basic_payload(self) -> None:
        cfg = OllamaModelConfig(model_name="llama3")
        client = OllamaLLMClient(cfg)
        payload = client._payload("hello")

        assert payload["model"] == "llama3"
        assert payload["prompt"] == "hello"
        assert payload["stream"] is False
        assert "format" not in payload

    def test_temperature_included(self) -> None:
        cfg = OllamaModelConfig(model_name="llama3", temperature=0.5)
        client = OllamaLLMClient(cfg)
        payload = client._payload("hello")

        assert payload["options"]["temperature"] == 0.5

    def test_max_tokens_mapped(self) -> None:
        cfg = OllamaModelConfig(model_name="llama3", max_tokens=100)
        client = OllamaLLMClient(cfg)
        payload = client._payload("hello")

        assert payload["options"]["num_predict"] == 100

    def test_system_prompt_mapped(self) -> None:
        cfg = OllamaModelConfig(model_name="llama3", system_prompt="Be helpful")
        client = OllamaLLMClient(cfg)
        payload = client._payload("hello")

        assert payload["system"] == "Be helpful"

    def test_format_json(self) -> None:
        cfg = OllamaModelConfig(model_name="llama3")
        client = OllamaLLMClient(cfg)
        payload = client._payload("hello", format_json=True)

        assert payload["format"] == "json"

    def test_none_max_tokens_excluded(self) -> None:
        cfg = OllamaModelConfig(model_name="llama3")
        client = OllamaLLMClient(cfg)
        payload = client._payload("hello")

        assert "num_predict" not in payload["options"]

    def test_none_system_prompt_excluded(self) -> None:
        cfg = OllamaModelConfig(model_name="llama3")
        client = OllamaLLMClient(cfg)
        payload = client._payload("hello")

        assert "system" not in payload

    def test_full_payload(self) -> None:
        cfg = OllamaModelConfig(
            model_name="llama3",
            temperature=0.7,
            max_tokens=512,
            system_prompt="Be concise",
        )
        client = OllamaLLMClient(cfg)
        payload = client._payload("hello", format_json=True)

        assert payload["model"] == "llama3"
        assert payload["prompt"] == "hello"
        assert payload["stream"] is False
        assert payload["format"] == "json"
        assert payload["options"]["temperature"] == 0.7
        assert payload["options"]["num_predict"] == 512
        assert payload["system"] == "Be concise"


class TestOllamaRequest:
    async def test_success_on_first_attempt(self) -> None:
        called = 0

        def handler(_request: httpx.Request) -> httpx.Response:
            nonlocal called
            called += 1
            return httpx.Response(200, json={"response": "hello back"})

        cfg = OllamaModelConfig(model_name="llama3")
        client = OllamaLLMClient(cfg)
        client._transport = httpx.MockTransport(handler)

        result = await client._request("hello")

        assert called == 1
        assert result == {"response": "hello back"}

    async def test_retries_on_failure(self) -> None:
        called = 0

        def handler(_request: httpx.Request) -> httpx.Response:
            nonlocal called
            called += 1
            if called < 2:
                return httpx.Response(500, text="boom")
            return httpx.Response(200, json={"response": "hello back"})

        cfg = OllamaModelConfig(model_name="llama3", retry_count=3)
        client = OllamaLLMClient(cfg)
        client._transport = httpx.MockTransport(handler)

        result = await client._request("hello")

        assert called == 2
        assert result == {"response": "hello back"}

    async def test_all_retries_exhausted(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(500, text="always fails")

        cfg = OllamaModelConfig(model_name="llama3", retry_count=2)
        client = OllamaLLMClient(cfg)
        client._transport = httpx.MockTransport(handler)

        with pytest.raises(httpx.HTTPStatusError):
            await client._request("hello")


class TestOllamaGenerate:
    async def test_returns_response_text(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"response": "world"})

        cfg = OllamaModelConfig(model_name="llama3")
        client = OllamaLLMClient(cfg)
        client._transport = httpx.MockTransport(handler)

        result = await client.generate("hello")

        assert result == "world"


class TestOllamaGenerateStructured:
    async def test_parses_json_response(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"response": '{"ranking": ["A"]}'})

        cfg = OllamaModelConfig(model_name="llama3")
        client = OllamaLLMClient(cfg)
        client._transport = httpx.MockTransport(handler)

        result = await client.generate_structured("rank this", {})

        assert result.data == {"ranking": ["A"]}
        assert result.raw == '{"ranking": ["A"]}'

    async def test_rejects_non_dict_json(self) -> None:
        def handler(_request: httpx.Request) -> httpx.Response:
            return httpx.Response(200, json={"response": "[1, 2, 3]"})

        cfg = OllamaModelConfig(model_name="llama3")
        client = OllamaLLMClient(cfg)
        client._transport = httpx.MockTransport(handler)

        with pytest.raises(ValueError, match="Expected JSON object"):
            await client.generate_structured("rank this", {})
