"""Tests for the generic client machinery.

Exercises :class:`HTTPLLMClient` (the build → POST/retry → extract → parse
template, ``_messages``) and the :class:`LLMClient` base (async-context
lifecycle, connection pooling, the concurrency limit) through the fake client in
``conftest``, so none of it depends on a real provider's wire format.
"""

import asyncio
from collections.abc import Callable

import httpx
import pytest

from tournament_eval.llm import AnthropicLLMClient, AnthropicModelConfig

_MakeClient = Callable[..., object]
_MakeHandler = Callable[..., tuple[object, dict[str, object]]]


class TestName:
    def test_name_returns_config_model_name(
        self, make_http_client: _MakeClient
    ) -> None:
        assert make_http_client(model_name="my-model").name == "my-model"


class TestContextRequirement:
    async def test_request_outside_context_raises(
        self, make_http_client: _MakeClient
    ) -> None:
        client = make_http_client()
        with pytest.raises(RuntimeError, match="async context manager"):
            await client.generate("hello")

    async def test_enter_is_idempotent(self, make_http_client: _MakeClient) -> None:
        client = make_http_client()
        async with client:
            first = client._http
            assert first is not None
            await client.__aenter__()  # second enter must not rebuild the pool
            assert client._http is first


class TestLifecycle:
    async def test_context_opens_and_closes_pool(
        self, make_http_client: _MakeClient
    ) -> None:
        client = make_http_client()
        assert client._http is None
        async with client:
            assert client._http is not None
        assert client._http is None

    async def test_aclose_is_idempotent(self, make_http_client: _MakeClient) -> None:
        client = make_http_client()
        await client.aclose()  # never opened — must be a no-op
        assert client._http is None

    async def test_pool_reused_across_requests(
        self, make_http_client: _MakeClient, make_handler: _MakeHandler
    ) -> None:
        handler, _ = make_handler("hi")
        client = make_http_client(handler)
        async with client:
            pool = client._http
            await client.generate("a")
            await client.generate("b")
            assert client._http is pool


class TestConcurrency:
    async def test_unbounded_runs_all_at_once(
        self,
        make_http_client: _MakeClient,
        tracking_handler: tuple[object, dict[str, int]],
    ) -> None:
        handler, state = tracking_handler
        client = make_http_client(handler)
        async with client:
            await asyncio.gather(*(client.generate("p") for _ in range(6)))
        assert state["peak"] == 6

    async def test_max_concurrency_bounds_inflight(
        self,
        make_http_client: _MakeClient,
        tracking_handler: tuple[object, dict[str, int]],
    ) -> None:
        handler, state = tracking_handler
        client = make_http_client(handler, max_concurrency=2)
        async with client:
            await asyncio.gather(*(client.generate("p") for _ in range(6)))
        assert state["peak"] == 2

    async def test_shared_semaphore_caps_across_clients(
        self,
        make_http_client: _MakeClient,
        tracking_handler: tuple[object, dict[str, int]],
    ) -> None:
        # One handler/state shared so peak counts in-flight across both clients.
        handler, state = tracking_handler
        sem = asyncio.Semaphore(1)
        a = make_http_client(handler, model_name="a", semaphore=sem)
        b = make_http_client(handler, model_name="b", semaphore=sem)

        async with a, b:
            await asyncio.gather(
                a.generate("p"), a.generate("p"), b.generate("p"), b.generate("p")
            )
        assert state["peak"] == 1


class TestRequest:
    async def test_success_on_first_attempt(
        self, make_http_client: _MakeClient, make_handler: _MakeHandler
    ) -> None:
        handler, record = make_handler("hi")
        client = make_http_client(handler)
        async with client:
            body = await client._request(client._build_payload("hi"))
        assert record["calls"] == 1
        assert body == {"text": "hi"}

    async def test_retries_then_succeeds(
        self, make_http_client: _MakeClient, make_handler: _MakeHandler
    ) -> None:
        handler, record = make_handler((500, "boom"), "ok")
        client = make_http_client(handler, retry_count=3)
        async with client:
            body = await client._request(client._build_payload("hi"))
        assert record["calls"] == 2
        assert body == {"text": "ok"}

    async def test_all_retries_exhausted(
        self, make_http_client: _MakeClient, make_handler: _MakeHandler
    ) -> None:
        handler, _ = make_handler((500, "always"))
        client = make_http_client(handler, retry_count=2)
        async with client:
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
        async with client:
            assert await client.generate("hi") == "bonjour"


class TestGenerateStructured:
    async def test_passes_schema_and_parses(
        self, make_http_client: _MakeClient, make_handler: _MakeHandler
    ) -> None:
        handler, record = make_handler('{"ranking": ["A"]}')
        client = make_http_client(handler)
        schema: dict[str, object] = {"type": "object"}
        async with client:
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
        async with client:
            with pytest.raises(ValueError, match="Expected JSON object"):
                await client.generate_structured("rank", {})


class TestBaseDefaultTimeout:
    async def test_non_http_provider_uses_base_default(self) -> None:
        # AnthropicLLMClient subclasses LLMClient directly (not HTTPLLMClient) and
        # its config carries no timeout, so it falls back to the base default.
        client = AnthropicLLMClient(AnthropicModelConfig(model_name="claude"))
        async with client:
            assert client._http is not None
            assert client._http_timeout == 300.0
