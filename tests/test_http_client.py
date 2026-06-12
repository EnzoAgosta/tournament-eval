"""Tests for the generic client machinery.

Exercises :class:`HTTPLLMClient` (the build → POST/retry → extract → parse
template, ``_messages``) through the fake client in ``conftest``, so none of it
depends on a real provider's wire format.
"""

from collections.abc import Callable

import httpx
import pytest

_MakeClient = Callable[..., object]
_MakeHandler = Callable[..., tuple[object, dict[str, object]]]


class TestName:
    def test_name_returns_config_model_name(
        self, make_http_client: _MakeClient
    ) -> None:
        assert make_http_client(model_name="my-model").name == "my-model"


class TestRequest:
    async def test_success_on_first_attempt(
        self, make_http_client: _MakeClient, make_handler: _MakeHandler
    ) -> None:
        handler, record = make_handler("hi")
        client = make_http_client(handler)
        body = await client._request(client._build_payload("hi"))
        assert record["calls"] == 1
        assert body == {"text": "hi"}

    async def test_retries_then_succeeds(
        self, make_http_client: _MakeClient, make_handler: _MakeHandler
    ) -> None:
        handler, record = make_handler((500, "boom"), "ok")
        client = make_http_client(handler, retry_count=3)
        body = await client._request(client._build_payload("hi"))
        assert record["calls"] == 2
        assert body == {"text": "ok"}

    async def test_all_retries_exhausted(
        self, make_http_client: _MakeClient, make_handler: _MakeHandler
    ) -> None:
        handler, _ = make_handler((500, "always"))
        client = make_http_client(handler, retry_count=2)
        with pytest.raises(httpx.HTTPStatusError):
            await client._request(client._build_payload("hi"))


class TestMessages:
    def test_user_only(self, make_http_client: _MakeClient) -> None:
        client = make_http_client()
        assert client._messages("hi") == [{"role": "user", "content": "hi"}]

    def test_system_prompt_prepended(self, make_http_client: _MakeClient) -> None:
        client = make_http_client(system_prompt="Be terse")
        assert client._messages("hi") == [
            {"role": "system", "content": "Be terse"},
            {"role": "user", "content": "hi"},
        ]


class TestGenerate:
    async def test_returns_extracted_text(
        self, make_http_client: _MakeClient, make_handler: _MakeHandler
    ) -> None:
        handler, _ = make_handler("bonjour")
        client = make_http_client(handler)
        assert await client.generate("hi") == "bonjour"


class TestGenerateStructured:
    async def test_passes_schema_and_parses(
        self, make_http_client: _MakeClient, make_handler: _MakeHandler
    ) -> None:
        handler, record = make_handler('{"ranking": ["A"]}')
        client = make_http_client(handler)
        schema: dict[str, object] = {"type": "object"}
        result = await client.generate_structured("rank", schema)

        assert result.data == {"ranking": ["A"]}
        assert result.raw == '{"ranking": ["A"]}'
        captured = record["captured"]
        assert isinstance(captured, dict)
        assert captured["schema"] == schema  # template forwarded schema to payload

    async def test_rejects_non_dict_json(
        self, make_http_client: _MakeClient, make_handler: _MakeHandler
    ) -> None:
        handler, _ = make_handler("[1, 2, 3]")
        client = make_http_client(handler)
        with pytest.raises(ValueError, match="Expected JSON object"):
            await client.generate_structured("rank", {})
