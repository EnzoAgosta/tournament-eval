"""Tests for the OpenAI-compatible client's protocol hooks.

Only the four hooks that distinguish this provider are tested here; the shared
request/template/lifecycle machinery lives in ``test_http_client.py``.
"""

import os
import unittest
import unittest.mock
from collections.abc import Callable

import pytest

_MakeClient = Callable[..., object]


class TestEndpointUrl:
    def test_default_base_url(self, make_openai_client: _MakeClient) -> None:
        client = make_openai_client()
        assert client._endpoint_url == "https://api.openai.com/v1/chat/completions"

    def test_custom_base_url_trailing_slash_stripped(
        self, make_openai_client: _MakeClient
    ) -> None:
        client = make_openai_client(base_url="http://localhost:8080/v1/")
        assert client._endpoint_url == "http://localhost:8080/v1/chat/completions"


class TestHeaders:
    def test_explicit_api_key(self, make_openai_client: _MakeClient) -> None:
        client = make_openai_client(api_key="sk-x")
        assert client._headers["Authorization"] == "Bearer sk-x"

    def test_env_api_key(self, make_openai_client: _MakeClient) -> None:
        with unittest.mock.patch.dict(os.environ, {"OPENAI_API_KEY": "sk-env"}):
            client = make_openai_client()
            assert client._headers["Authorization"] == "Bearer sk-env"

    def test_no_api_key(self, make_openai_client: _MakeClient) -> None:
        client = make_openai_client()
        assert "Authorization" not in client._headers


class TestBuildPayload:
    def test_basic_payload(self, make_openai_client: _MakeClient) -> None:
        payload = make_openai_client()._build_payload("hello")
        assert payload["model"] == "gpt-4o"
        assert payload["messages"] == [{"role": "user", "content": "hello"}]
        assert payload["stream"] is False
        assert "response_format" not in payload
        assert "max_tokens" not in payload
        assert "seed" not in payload

    def test_temperature_included(self, make_openai_client: _MakeClient) -> None:
        client = make_openai_client(temperature=0.5)
        assert client._build_payload("hi")["temperature"] == 0.5

    def test_max_tokens_mapped(self, make_openai_client: _MakeClient) -> None:
        client = make_openai_client(max_tokens=128)
        assert client._build_payload("hi")["max_tokens"] == 128

    def test_seed_included(self, make_openai_client: _MakeClient) -> None:
        client = make_openai_client(seed=23012023)
        assert client._build_payload("hi")["seed"] == 23012023

    def test_system_prompt_prepended(self, make_openai_client: _MakeClient) -> None:
        client = make_openai_client(system_prompt="Be terse")
        assert client._build_payload("hi")["messages"] == [
            {"role": "system", "content": "Be terse"},
            {"role": "user", "content": "hi"},
        ]

    def test_schema_becomes_strict_json_schema(
        self, make_openai_client: _MakeClient
    ) -> None:
        schema: dict[str, object] = {"type": "object"}
        rf = make_openai_client()._build_payload("hi", schema=schema)["response_format"]
        assert isinstance(rf, dict)
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["schema"] == schema
        assert rf["json_schema"]["strict"] is True


class TestExtractText:
    def test_returns_message_content(self, make_openai_client: _MakeClient) -> None:
        body = {"choices": [{"message": {"content": "リモートコントロール"}}]}
        assert make_openai_client()._extract_text(body) == "リモートコントロール"

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
    def test_extract_errors(
        self, make_openai_client: _MakeClient, body: dict[str, object], match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            make_openai_client()._extract_text(body)
