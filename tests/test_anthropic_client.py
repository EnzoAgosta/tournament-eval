"""Tests for the Anthropic client's protocol hooks.

Only the four hooks that distinguish this provider are tested here; the shared
request/template/lifecycle machinery lives in ``test_http_client.py``.
"""

import json
import os
import unittest
import unittest.mock
from collections.abc import Callable

import pytest

_MakeClient = Callable[..., object]


class TestEndpointUrl:
    def test_default_base_url(self, make_anthropic_client: _MakeClient) -> None:
        client = make_anthropic_client()
        assert client._endpoint_url == "https://api.anthropic.com/v1/messages"

    def test_custom_base_url_trailing_slash_stripped(
        self, make_anthropic_client: _MakeClient
    ) -> None:
        client = make_anthropic_client(base_url="http://localhost:8080/")
        assert client._endpoint_url == "http://localhost:8080/v1/messages"


class TestHeaders:
    def test_explicit_api_key(self, make_anthropic_client: _MakeClient) -> None:
        headers = make_anthropic_client(api_key="sk-ant-x")._headers
        assert headers["x-api-key"] == "sk-ant-x"
        assert headers["anthropic-version"] == "2023-06-01"
        assert "Authorization" not in headers  # Anthropic uses x-api-key, not Bearer

    def test_env_api_key(self, make_anthropic_client: _MakeClient) -> None:
        with unittest.mock.patch.dict(os.environ, {"ANTHROPIC_API_KEY": "sk-ant-env"}):
            client = make_anthropic_client()
            assert client._headers["x-api-key"] == "sk-ant-env"

    def test_no_api_key(self, make_anthropic_client: _MakeClient) -> None:
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            assert "x-api-key" not in make_anthropic_client()._headers


class TestBuildPayload:
    def test_basic_payload(self, make_anthropic_client: _MakeClient) -> None:
        payload = make_anthropic_client()._build_payload("hello")
        assert payload["model"] == "claude-opus-4-8"
        assert payload["messages"] == [{"role": "user", "content": "hello"}]
        assert payload["max_tokens"] == 4096  # required by the API — falls back
        assert payload["temperature"] == 1.0
        assert "system" not in payload
        assert "thinking" not in payload
        assert "output_config" not in payload
        assert "tools" not in payload
        assert "tool_choice" not in payload

    def test_max_tokens_overrides_default(
        self, make_anthropic_client: _MakeClient
    ) -> None:
        client = make_anthropic_client(max_tokens=512)
        assert client._build_payload("hi")["max_tokens"] == 512

    def test_temperature_included(self, make_anthropic_client: _MakeClient) -> None:
        client = make_anthropic_client(temperature=0.3)
        assert client._build_payload("hi")["temperature"] == 0.3

    def test_system_prompt_is_top_level(
        self, make_anthropic_client: _MakeClient
    ) -> None:
        client = make_anthropic_client(system_prompt="Be terse")
        payload = client._build_payload("hi")
        assert payload["system"] == "Be terse"
        # The system prompt must NOT leak into the messages turns.
        assert payload["messages"] == [{"role": "user", "content": "hi"}]

    def test_effort_becomes_output_config(
        self, make_anthropic_client: _MakeClient
    ) -> None:
        client = make_anthropic_client(effort="high")
        assert client._build_payload("hi")["output_config"] == {"effort": "high"}

    def test_adaptive_thinking(self, make_anthropic_client: _MakeClient) -> None:
        client = make_anthropic_client(thinking="adaptive")
        assert client._build_payload("hi")["thinking"] == {"type": "adaptive"}

    def test_adaptive_thinking_with_json_schema(
        self, make_anthropic_client: _MakeClient
    ) -> None:
        # The whole point: adaptive thinking coexists with schema-enforced output.
        client = make_anthropic_client(thinking="adaptive")
        payload = client._build_payload("rank", schema={"type": "object"})
        assert payload["thinking"] == {"type": "adaptive"}
        assert payload["output_config"] == {
            "format": {"type": "json_schema", "schema": {"type": "object"}}
        }

    def test_manual_thinking_budget(self, make_anthropic_client: _MakeClient) -> None:
        client = make_anthropic_client(thinking=2048)
        assert client._build_payload("hi")["thinking"] == {
            "type": "enabled",
            "budget_tokens": 2048,
        }

    def test_schema_defaults_to_json_schema_output_config(
        self, make_anthropic_client: _MakeClient
    ) -> None:
        # Default mode relies on the API to enforce the schema (output_config.format).
        schema: dict[str, object] = {"type": "object", "required": ["ranking"]}
        payload = make_anthropic_client()._build_payload("rank", schema=schema)
        assert payload["output_config"] == {
            "format": {"type": "json_schema", "schema": schema}
        }
        assert "tools" not in payload
        assert "tool_choice" not in payload

    def test_json_schema_merges_with_effort(
        self, make_anthropic_client: _MakeClient
    ) -> None:
        # effort and format share one output_config — neither clobbers the other.
        schema: dict[str, object] = {"type": "object"}
        client = make_anthropic_client(effort="high")
        assert client._build_payload("rank", schema=schema)["output_config"] == {
            "effort": "high",
            "format": {"type": "json_schema", "schema": schema},
        }

    def test_schema_tool_use_fallback(self, make_anthropic_client: _MakeClient) -> None:
        schema: dict[str, object] = {"type": "object", "required": ["ranking"]}
        payload = make_anthropic_client(structured_output="tool_use")._build_payload(
            "rank", schema=schema
        )
        tools = payload["tools"]
        assert isinstance(tools, list)
        assert tools[0]["name"] == "structured_response"
        assert tools[0]["input_schema"] == schema
        assert payload["tool_choice"] == {
            "type": "tool",
            "name": "structured_response",
        }
        assert "output_config" not in payload  # no effort set, no format in this mode

    def test_tool_use_fallback_merges_with_effort(
        self, make_anthropic_client: _MakeClient
    ) -> None:
        # In tool_use mode the schema goes to tools; effort still rides output_config.
        client = make_anthropic_client(structured_output="tool_use", effort="max")
        payload = client._build_payload("rank", schema={"type": "object"})
        assert payload["output_config"] == {"effort": "max"}
        assert payload["tool_choice"] == {"type": "tool", "name": "structured_response"}


class TestExtractText:
    def test_returns_text_block(self, make_anthropic_client: _MakeClient) -> None:
        body = {"content": [{"type": "text", "text": "リモートコントロール"}]}
        assert make_anthropic_client()._extract_text(body) == "リモートコントロール"

    def test_skips_leading_thinking_block(
        self, make_anthropic_client: _MakeClient
    ) -> None:
        body = {
            "content": [
                {"type": "thinking", "thinking": "hmm"},
                {"type": "text", "text": "bonjour"},
            ]
        }
        assert make_anthropic_client()._extract_text(body) == "bonjour"

    def test_skips_non_dict_block(self, make_anthropic_client: _MakeClient) -> None:
        body = {"content": ["junk", {"type": "text", "text": "ok"}]}
        assert make_anthropic_client()._extract_text(body) == "ok"

    def test_returns_tool_use_input_as_json(
        self, make_anthropic_client: _MakeClient
    ) -> None:
        body = {
            "content": [{"type": "tool_use", "input": {"ranking": ["A", "B"]}}],
        }
        raw = make_anthropic_client()._extract_text(body)
        assert json.loads(raw) == {"ranking": ["A", "B"]}

    @pytest.mark.parametrize(
        ("body", "match"),
        [
            ({"stop_reason": "refusal", "content": []}, "refused"),
            ({}, "non-empty 'content'"),
            ({"content": []}, "non-empty 'content'"),
            ({"content": [{"type": "text", "text": 123}]}, "not a string"),
            ({"content": [{"type": "tool_use", "input": "nope"}]}, "not an object"),
            ({"content": [{"type": "thinking", "thinking": "x"}]}, "no 'text'"),
        ],
    )
    def test_extract_errors(
        self, make_anthropic_client: _MakeClient, body: dict[str, object], match: str
    ) -> None:
        with pytest.raises(ValueError, match=match):
            make_anthropic_client()._extract_text(body)
