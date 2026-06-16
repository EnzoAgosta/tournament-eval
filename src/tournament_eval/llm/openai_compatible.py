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
    GenerationResponse,
    ReasoningEffort,
    StructuredResponse,
    concurrency_guard,
    resolve_semaphore,
)

# The chat-completions ``reasoning_effort`` field tops out at "high", so the
# Anthropic-only levels map down to it.
_REASONING_EFFORT: dict[ReasoningEffort, str] = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "high",
    "max": "high",
}


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
    api_key: NotRequired[str]
    """API key. Local servers usually ignore it so not required."""
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
            text = (await client.generate("hello")).text

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
        self._api_key = kwargs.get("api_key")
        self._temperature = kwargs.get("temperature", 1.0)
        self._max_tokens = kwargs.get("max_tokens")
        self._system_prompt = kwargs.get("system_prompt")
        self._retry_count = kwargs.get("retry_count", 3)
        self._name = kwargs.get("name")
        self._timeout = kwargs.get("timeout", 300.0)
        self._seed = kwargs.get("seed")
        self._reasoning_effort = kwargs.get("reasoning_effort")
        # Set by tests to inject a mock transport; not a public API.
        self._transport: httpx.AsyncBaseTransport | None = None
        self._http_client: httpx.AsyncClient | None = None
        self._sem = resolve_semaphore(semaphore, kwargs.get("max_concurrency"))

    @property
    def name(self) -> str:
        return self._name or self._model_id

    @property
    def _endpoint_url(self) -> str:
        return f"{self._base_url.rstrip('/')}/chat/completions"

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
        if self._reasoning_effort is not None:
            payload["reasoning_effort"] = _REASONING_EFFORT[self._reasoning_effort]
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
        retry_count = max(self._retry_count, 1)  # to ensure we at least make one attempt

        for attempt in range(retry_count):
            try:
                async with concurrency_guard(self._sem):
                    response = await http.post(self._endpoint_url, json=body, headers=headers)
                    response.raise_for_status()
                    response_body: dict[str, object] = response.json()
                    return response_body
            # Catch only transient failures; anything else (parse errors, etc.) bubbles
            # up — it would fail identically on retry. 4xx other than 429 is a client
            # error, so it isn't retryable either. This matches the SDK clients' policy.
            except (httpx.TransportError, httpx.HTTPStatusError) as exc:
                retryable = (
                    isinstance(exc, httpx.TransportError)
                    or exc.response.status_code == 429
                    or exc.response.status_code >= 500
                )
                if not retryable or attempt == retry_count - 1:
                    raise
                await asyncio.sleep(2**attempt)

        raise RuntimeError("You somehow bypassed the request retry loop... How?")  # pragma: no cover

    def _message(self, body: dict[str, object]) -> dict[str, object]:
        choices = body.get("choices")
        if choices is None:
            raise ValueError("OpenAI response missing a 'choices' list")
        if not isinstance(choices, list):
            raise ValueError("OpenAI response 'choices' is not a list")
        if not choices:
            raise ValueError("OpenAI response 'choices' is empty")
        first = choices[0]
        if not isinstance(first, dict):
            raise ValueError("OpenAI response 'choices[0]' is not an object")
        message = first.get("message")
        if message is None:
            raise ValueError("OpenAI response missing 'message'")
        if not isinstance(message, dict):
            raise ValueError("OpenAI response 'message' is not an object")
        return message

    def _extract_response(self, message: dict[str, object]) -> str:
        refusal = message.get("refusal")
        if refusal:
            raise ValueError(f"Model refused to respond: {refusal}")
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("OpenAI response 'content' is not a string")
        return content

    def _extract_reasoning(self, message: dict[str, object]) -> str | None:
        # Not part of the /chat/completions contract, but reasoning-capable
        # OpenAI-compatible servers (vLLM, SGLang, ...) return the trace in
        # ``message.reasoning_content``. Read it when present; best-effort.
        reasoning = message.get("reasoning_content")
        return reasoning if isinstance(reasoning, str) else None

    async def generate(self, prompt: str) -> GenerationResponse:
        headers = self._build_headers()
        body = self._build_payload(prompt)
        message = self._message(await self._make_request(headers, body))
        return GenerationResponse(text=self._extract_response(message), reasoning=self._extract_reasoning(message))

    async def generate_structured(self, prompt: str, schema: dict[str, object]) -> StructuredResponse:
        headers = self._build_headers()
        body = self._build_payload(prompt, schema=schema)
        message = self._message(await self._make_request(headers, body))
        raw = self._extract_response(message)
        return StructuredResponse(data=orjson.loads(raw), raw=raw)
