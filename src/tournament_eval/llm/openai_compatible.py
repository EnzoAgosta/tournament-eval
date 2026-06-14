"""The built-in client: any OpenAI-compatible ``/chat/completions`` server.

Zero extra dependencies — httpx is core — so this is what you get out of the
box.  Point ``base_url`` at a local server you run (``mlx_lm.server``, vLLM,
llama.cpp, ...) or at the OpenAI API itself.  For the official OpenAI SDK
instead (retries, streaming helpers, typed models), install the ``openai`` extra
and use :class:`~tournament_eval.llm.openai.OpenAIClient`.

This is a single concrete class, not a template: it owns its httpx connection
pool, retries failed requests, optionally bounds its own concurrency, and asks
for structured output via ``response_format``'s strict ``json_schema``.
"""

import asyncio
from typing import NotRequired, Required, Self, Unpack

import httpx
import orjson

from tournament_eval.llm.base import (
    GenerationConfig,
    StructuredResponse,
    concurrency_guard,
    resolve_semaphore,
)


class OpenAICompatibleClientConfig(GenerationConfig):
    """Configuration for any OpenAI-compatible ``/chat/completions`` server.

    Extends the shared :class:`GenerationConfig` with the transport/auth this
    built-in client owns — it speaks HTTP directly rather than wrapping an SDK.
    """

    base_url: Required[str]
    """Server base URL.
    Note that this should not contain the ``/chat/completions`` path, for example
    for a local server you control (e.g. ``mlx_lm.server``), this should
    point to ``http://localhost:8080/v1``."""
    api_key: Required[str]
    """API key. Local servers usually ignore it so can be set to any non-empty string."""
    retry_count: NotRequired[int]
    """Total attempts for a failed request (default 3)."""
    timeout: NotRequired[float]
    """Request timeout in seconds (default 5 minutes)."""
    seed: NotRequired[int]
    """Optional seed for deterministic sampling."""


class OpenAICompatibleClient:
    """Client for any server speaking the OpenAI ``/chat/completions`` protocol.

    It owns an httpx connection pool, so it's an async context manager — open it
    with ``async with`` before calling it directly::

        async with OpenAICompatibleClient(model_id="m", base_url="http://...", api_key="k") as client:
            text = await client.generate("hello")

    Handed to :func:`~tournament_eval.orchestration.generate_all` /
    :func:`~tournament_eval.orchestration.rank_all`, the lifecycle is managed for you.
    """

    def __init__(
        self,
        *,
        semaphore: asyncio.Semaphore | None = None,
        **kwargs: Unpack[OpenAICompatibleClientConfig],
    ) -> None:
        self._model_id = kwargs["model_id"]
        self._base_url = kwargs["base_url"]
        self._api_key = kwargs["api_key"]
        self._temperature = kwargs.get("temperature", 1.0)
        self._max_tokens = kwargs.get("max_tokens")
        self._system_prompt = kwargs.get("system_prompt")
        self._retry_count = kwargs.get("retry_count", 3)
        self._name = kwargs.get("name")
        self._timeout = kwargs.get("timeout", 300.0)
        self._seed = kwargs.get("seed")
        # Set by tests to inject a mock transport; not a public API.
        self._transport: httpx.AsyncBaseTransport | None = None
        self._http_client: httpx.AsyncClient | None = None
        self._sem = resolve_semaphore(semaphore, kwargs.get("max_concurrency"))

    @property
    def name(self) -> str:
        return self._name or self._model_id

    @property
    def _endpoint_url(self) -> str:
        return f"{self._base_url}/chat/completions"

    async def __aenter__(self) -> Self:
        if self._http_client is None:
            self._http_client = httpx.AsyncClient(
                timeout=httpx.Timeout(self._timeout),
                transport=self._transport,
            )
        return self

    async def __aexit__(self, *_exc: object) -> None:
        if self._http_client is not None:
            await self._http_client.aclose()
            self._http_client = None

    def _require_http(self) -> httpx.AsyncClient:
        if self._http_client is None:
            raise RuntimeError(
                f"{type(self).__name__} must be used as an async context "
                "manager: `async with Client(...) as client: ...`"
            )
        return self._http_client

    def _build_headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        if self._api_key:
            headers["Authorization"] = f"Bearer {self._api_key}"
        return headers

    def _build_messages(self, prompt: str) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if self._system_prompt is not None:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _build_payload(self, prompt: str, *, schema: dict[str, object] | None = None) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self._model_id,
            "messages": self._build_messages(prompt),
            "stream": False,
            "temperature": self._temperature,
        }
        if self._max_tokens is not None:
            payload["max_tokens"] = self._max_tokens
        if self._seed is not None:
            payload["seed"] = self._seed
        if schema is not None:
            payload["response_format"] = {
                "type": "json_schema",
                "json_schema": {
                    "name": "structured_response",
                    "schema": schema,
                    "strict": True,
                },
            }
        return payload

    async def _make_request(self, headers: dict[str, str], body: dict[str, object]) -> dict[str, object]:
        http = self._require_http()

        last_err: Exception | None = None
        for attempt in range(self._retry_count):
            try:
                async with concurrency_guard(self._sem):
                    response = await http.post(self._endpoint_url, json=body, headers=headers)
                    response.raise_for_status()
                    response_body: dict[str, object] = response.json()
                    return response_body
            except Exception as exc:
                last_err = exc
                if attempt < self._retry_count - 1:
                    await asyncio.sleep(2**attempt)

        assert last_err is not None, "retry loop ended without an exception"
        raise last_err

    def _extract_text(self, body: dict[str, object]) -> str:
        choices = body.get("choices")
        if not isinstance(choices, list) or not choices:
            raise ValueError("OpenAI response missing a non-empty 'choices' list")
        first = choices[0]
        if not isinstance(first, dict):
            raise ValueError("OpenAI 'choices[0]' is not an object")
        message = first.get("message")
        if not isinstance(message, dict):
            raise ValueError("OpenAI response missing 'message'")
        refusal = message.get("refusal")
        if isinstance(refusal, str):
            raise ValueError(f"Model refused to respond: {refusal}")
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("OpenAI response 'content' is not a string")
        return content

    async def generate(self, prompt: str) -> str:
        headers = self._build_headers()
        body = self._build_payload(prompt)
        response = await self._make_request(headers, body)
        return self._extract_text(response)

    async def generate_structured(self, prompt: str, schema: dict[str, object]) -> StructuredResponse:
        headers = self._build_headers()
        body = self._build_payload(prompt, schema=schema)
        response = await self._make_request(headers, body)
        raw = self._extract_text(response)
        parsed = orjson.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected JSON object, got {type(parsed).__name__}")
        return StructuredResponse(data=parsed, raw=raw)
