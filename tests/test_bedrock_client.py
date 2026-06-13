"""Tests for the Bedrock client.

Bedrock speaks the OpenAI ``/chat/completions`` protocol, so ``BedrockLLMClient``
is a thin subclass of ``OpenAICompatibleLLMClient`` — the payload/extraction logic
is covered in ``test_openai_client.py``.  Only the two things it overrides are
tested here: the ``bedrock-mantle`` endpoint URL and the ``AWS_BEARER_TOKEN_BEDROCK``
auth fallback.  One smoke test confirms the inherited OpenAI payload still applies.
"""

import os
import unittest
import unittest.mock
from collections.abc import Callable

_MakeClient = Callable[..., object]


class TestEndpointUrl:
    def test_default_targets_region_mantle_url(
        self, make_bedrock_client: _MakeClient
    ) -> None:
        client = make_bedrock_client()  # default region us-east-1
        assert client._endpoint_url == (
            "https://bedrock-mantle.us-east-1.api.aws/v1/chat/completions"
        )

    def test_region_is_used(self, make_bedrock_client: _MakeClient) -> None:
        client = make_bedrock_client(region="eu-west-1")
        assert client._endpoint_url == (
            "https://bedrock-mantle.eu-west-1.api.aws/v1/chat/completions"
        )

    def test_explicit_base_url_overrides_region(
        self, make_bedrock_client: _MakeClient
    ) -> None:
        # An explicit base_url (e.g. bedrock-runtime or a gateway) wins over region.
        client = make_bedrock_client(
            region="us-east-1",
            base_url="https://bedrock-runtime.us-west-2.amazonaws.com/v1/",
        )
        assert client._endpoint_url == (
            "https://bedrock-runtime.us-west-2.amazonaws.com/v1/chat/completions"
        )


class TestHeaders:
    def test_explicit_api_key(self, make_bedrock_client: _MakeClient) -> None:
        headers = make_bedrock_client(api_key="bedrock-key")._headers
        assert headers["Authorization"] == "Bearer bedrock-key"

    def test_env_api_key(self, make_bedrock_client: _MakeClient) -> None:
        with unittest.mock.patch.dict(
            os.environ, {"AWS_BEARER_TOKEN_BEDROCK": "bedrock-env"}
        ):
            client = make_bedrock_client()
            assert client._headers["Authorization"] == "Bearer bedrock-env"

    def test_no_api_key(self, make_bedrock_client: _MakeClient) -> None:
        with unittest.mock.patch.dict(os.environ, {}, clear=True):
            assert "Authorization" not in make_bedrock_client()._headers


class TestInheritedPayload:
    def test_openai_protocol_payload_is_inherited(
        self, make_bedrock_client: _MakeClient
    ) -> None:
        # Sanity check: the OpenAI-compatible payload (model id + strict json_schema
        # structured output) flows through unchanged for a Bedrock model.
        schema: dict[str, object] = {"type": "object"}
        payload = make_bedrock_client()._build_payload("rank", schema=schema)
        assert payload["model"] == "us.anthropic.claude-sonnet-4-6"
        assert payload["messages"] == [{"role": "user", "content": "rank"}]
        rf = payload["response_format"]
        assert isinstance(rf, dict)
        assert rf["type"] == "json_schema"
        assert rf["json_schema"]["schema"] == schema
