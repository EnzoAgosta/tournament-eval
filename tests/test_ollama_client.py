"""Tests for the Ollama client's protocol hooks.

Only the four hooks that distinguish this provider are tested here; the shared
request/template/lifecycle machinery lives in ``test_http_client.py``.
"""

from collections.abc import Callable

import pytest

_MakeClient = Callable[..., object]


class TestEndpointUrl:
    def test_default_base_url_targets_native_api_chat(
        self, make_ollama_client: _MakeClient
    ) -> None:
        assert make_ollama_client()._endpoint_url == "http://localhost:11434/api/chat"

    def test_base_url_is_overridable(self, make_ollama_client: _MakeClient) -> None:
        client = make_ollama_client(base_url="http://box:11434/")
        assert client._endpoint_url == "http://box:11434/api/chat"


class TestHeaders:
    def test_headers_are_plain_json(self, make_ollama_client: _MakeClient) -> None:
        assert make_ollama_client()._headers == {"Content-Type": "application/json"}


class TestBuildPayload:
    def test_basic_payload(self, make_ollama_client: _MakeClient) -> None:
        payload = make_ollama_client()._build_payload("hello")
        assert payload["model"] == "llama3"
        assert payload["messages"] == [{"role": "user", "content": "hello"}]
        assert payload["stream"] is False
        assert payload["options"] == {"temperature": 1.0}
        assert "format" not in payload

    def test_options_mapped(self, make_ollama_client: _MakeClient) -> None:
        client = make_ollama_client(temperature=0.5, max_tokens=256, seed=7)
        options = client._build_payload("hi")["options"]
        assert options == {"temperature": 0.5, "num_predict": 256, "seed": 7}

    def test_system_prompt_prepended(self, make_ollama_client: _MakeClient) -> None:
        client = make_ollama_client(system_prompt="Be terse")
        assert client._build_payload("hi")["messages"] == [
            {"role": "system", "content": "Be terse"},
            {"role": "user", "content": "hi"},
        ]

    def test_schema_passed_through_as_native_format(
        self, make_ollama_client: _MakeClient
    ) -> None:
        schema: dict[str, object] = {"type": "object", "required": ["ranking"]}
        assert make_ollama_client()._build_payload("rank", schema=schema)["format"] == (
            schema
        )


class TestExtractText:
    def test_returns_message_content(self, make_ollama_client: _MakeClient) -> None:
        body = {"message": {"role": "assistant", "content": "bonjour"}, "done": True}
        assert make_ollama_client()._extract_text(body) == "bonjour"

    @pytest.mark.parametrize(
        ("body", "match"),
        [
            ({}, "missing 'message'"),
            ({"message": {"content": 123}}, "not a string"),
        ],
    )
    def test_extract_errors(
        self, make_ollama_client: _MakeClient, body: dict[str, object], match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            make_ollama_client()._extract_text(body)
