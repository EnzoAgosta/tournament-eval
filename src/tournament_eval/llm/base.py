"""Provider-agnostic LLM client base and the shared HTTP+JSON template.

Layering
--------
* :class:`LLMClient` — provider-agnostic base: connection-pool lifecycle,
  concurrency limiting, and the abstract ``generate`` / ``generate_structured``
  contract.  Knows nothing about any wire protocol.
* :class:`HTTPLLMClient` — the shared template for any HTTP+JSON chat API:
  build a payload, POST it with retries, pull the text back out, and (for
  structured calls) parse JSON.  Subclasses fill in four small hooks.

Concrete providers live in sibling modules (``openai``, ``bedrock``,
``anthropic``, ``ollama``); adding one is implementing four hooks on
:class:`HTTPLLMClient` — ``_endpoint_url``, ``_headers``, ``_build_payload`` and
``_extract_text``.
"""

import abc
import asyncio
import contextlib
import dataclasses
import json
from typing import Self

import httpx


@dataclasses.dataclass(frozen=True, slots=True)
class StructuredResponse:
    """Wrapper returned by :meth:`LLMClient.generate_structured`.

    Preserves both the parsed dict and the exact raw string for audit.
    """

    data: dict[str, object]
    """The parsed JSON object."""
    raw: str
    """The exact raw string returned by the LLM before parsing."""


@dataclasses.dataclass(frozen=True)
class ModelConfig:
    """Base configuration shared by all LLM providers."""

    model_name: str
    """Provider-specific model identifier (e.g. ``"gpt-4o"``, ``"claude-sonnet"``)."""
    temperature: float = 1.0
    """Sampling temperature."""
    max_tokens: int | None = None
    """Maximum tokens to generate, or ``None`` for provider default."""
    system_prompt: str | None = None
    """Optional system-level prompt prepended to each request."""
    retry_count: int = 3
    """Total attempts for a failed request."""
    max_concurrency: int | None = None
    """Maximum number of in-flight requests for this client.

    ``None`` (the default) means **unbounded** — every request fires as soon as
    it is scheduled.  This is fine for a local server you control (e.g. Ollama on
    a workstation that can happily serve many small-model requests at once), but
    against a rate-limited hosted API you almost certainly want a conservative
    value.  See :class:`LLMClient` for how to also share one limit across several
    clients.
    """


@dataclasses.dataclass(frozen=True)
class HTTPModelConfig(ModelConfig):
    """Configuration shared by every HTTP+JSON provider (see :class:`HTTPLLMClient`)."""

    base_url: str | None = None
    """Server base URL.  ``None`` lets the client fall back to its own default."""
    timeout: float = 300.0
    """Request timeout in seconds (default 5 minutes)."""


class LLMClient[C: ModelConfig](abc.ABC):
    """Abstract base class for LLM provider wrappers.

    Provider-agnostic on purpose: it owns the connection-pool lifecycle and the
    concurrency limit, and declares the ``generate`` / ``generate_structured``
    contract — but knows nothing about any wire protocol.  HTTP+JSON providers
    should subclass :class:`HTTPLLMClient` rather than this directly.

    Lifecycle
    ---------
    A client holds a long-lived ``httpx.AsyncClient`` (one connection pool,
    reused across every request) and **must be used as an async context
    manager** so that pool is opened on a running event loop and closed
    afterwards::

        async with OllamaLLMClient(config) as client:
            text = await client.generate("hello")

    Calling :meth:`generate` / :meth:`generate_structured` outside an
    ``async with`` block raises :class:`RuntimeError`.  The same client may be
    reused across both the generation and ranking phases of a tournament.

    Concurrency
    -----------
    Each client throttles its own in-flight requests, because rate limits live
    with the provider, not the orchestration.  Two ways to set the limit:

    * **Per-client** — set ``max_concurrency`` on the config; the client builds
      its own semaphore.
    * **Shared/global** — pass the *same* :class:`asyncio.Semaphore` instance as
      ``semaphore=`` to several clients so they draw from one budget.

    The injected ``semaphore`` wins over ``config.max_concurrency``.  If neither
    is set the client is unbounded.

    Generic over its config type ``C`` so each layer of the hierarchy sees
    ``self._config`` at its most specific type — a subclass parameterised with a
    narrower :class:`ModelConfig` reads its own fields off ``self._config``
    directly, with no per-class re-aliasing.
    """

    def __init__(
        self,
        config: C,
        *,
        semaphore: asyncio.Semaphore | None = None,
    ) -> None:
        self._config = config
        # Transport is deliberately not set here, it _can_ be set by users
        # at runtime, but it's not a public API and really only used by
        # tests.
        self._transport: httpx.AsyncBaseTransport | None = None
        self._http: httpx.AsyncClient | None = None
        self._sem: asyncio.Semaphore | None = semaphore or (
            asyncio.Semaphore(config.max_concurrency)
            if config.max_concurrency is not None
            else None
        )

    @property
    def name(self) -> str:
        """Return the model identifier used as ``author`` in results."""
        return self._config.model_name

    @property
    def _http_timeout(self) -> float:
        """Request timeout in seconds.

        Overridden by providers whose config carries a ``timeout`` field.
        """
        return 300.0

    def _require_http(self) -> httpx.AsyncClient:
        """Return the live ``httpx.AsyncClient`` or explain how to open one."""
        if self._http is None:
            raise RuntimeError(
                f"{type(self).__name__} must be used as an async context "
                "manager: `async with Client(config) as client: ...`"
            )
        return self._http

    def _concurrency_guard(self) -> contextlib.AbstractAsyncContextManager[None]:
        """Acquire the concurrency permit, or a no-op when unbounded."""
        return self._sem if self._sem is not None else contextlib.nullcontext()

    async def __aenter__(self) -> Self:
        if self._http is None:
            self._http = httpx.AsyncClient(
                timeout=httpx.Timeout(self._http_timeout),
                transport=self._transport,
            )
        return self

    async def __aexit__(self, *_exc: object) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        """Close the underlying connection pool."""
        if self._http is not None:
            await self._http.aclose()
            self._http = None

    @abc.abstractmethod
    async def generate(self, prompt: str) -> str:
        """Send a plain-text prompt and return the raw response."""
        ...

    @abc.abstractmethod
    async def generate_structured(
        self,
        prompt: str,
        schema: dict[str, object],
    ) -> StructuredResponse:
        """Send a prompt and return a structured response."""
        ...


class HTTPLLMClient[C: HTTPModelConfig](LLMClient[C]):
    """Shared template for any HTTP+JSON chat API.

    Implements the full ``generate`` / ``generate_structured`` flow — build a
    payload, POST it with retries under the concurrency limit, extract the text,
    and (for structured calls) parse it as JSON — in terms of four hooks a
    subclass fills in for its wire protocol:

    * :meth:`_endpoint_url` — where to POST.
    * :meth:`_headers` — request headers (auth, content type).
    * :meth:`_build_payload` — the request body; ``schema`` is non-``None`` for
      structured calls so the subclass can request constrained/JSON output.
    * :meth:`_extract_text` — pull the assistant text out of the response body.
    """

    @property
    def _http_timeout(self) -> float:
        return self._config.timeout

    @property
    @abc.abstractmethod
    def _endpoint_url(self) -> str:
        """The full URL to POST requests to."""
        ...

    @property
    @abc.abstractmethod
    def _headers(self) -> dict[str, str]:
        """Headers to send with each request."""
        ...

    @abc.abstractmethod
    def _build_payload(
        self,
        prompt: str,
        *,
        schema: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Build the request body.

        ``schema`` is ``None`` for plain generation and the desired JSON schema
        for structured generation; subclasses use it to request constrained or
        JSON-mode output in whatever dialect their server speaks.
        """
        ...

    @abc.abstractmethod
    def _extract_text(self, body: dict[str, object]) -> str:
        """Extract the assistant's text from a parsed response body."""
        ...

    def _messages(self, prompt: str) -> list[dict[str, str]]:
        """Build a ``[{"role", "content"}]`` chat array, system turn first if set.

        An *opt-in* helper, not one of the hooks: it encodes the role/content
        convention shared by OpenAI-compatible servers and Ollama, so their
        ``_build_payload`` implementations can just call it.  Providers that
        diverge should build their payload directly instead — e.g. Anthropic
        takes ``system`` as a top-level parameter rather than a turn, and Gemini
        uses a different ``contents``/``parts`` shape entirely.
        """
        messages: list[dict[str, str]] = []
        if self._config.system_prompt is not None:
            messages.append({"role": "system", "content": self._config.system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    async def _request(self, payload: dict[str, object]) -> dict[str, object]:
        """POST ``payload`` with retries and return the parsed JSON body."""
        http = self._require_http()

        last_err: Exception | None = None
        for attempt in range(self._config.retry_count):
            try:
                async with self._concurrency_guard():
                    response = await http.post(
                        self._endpoint_url, json=payload, headers=self._headers
                    )
                    response.raise_for_status()
                    body: dict[str, object] = response.json()
                    return body
            except Exception as exc:
                last_err = exc
                if attempt < self._config.retry_count - 1:
                    await asyncio.sleep(2**attempt)

        assert last_err is not None, "Somehow never returned with no exception"
        raise last_err

    async def generate(self, prompt: str) -> str:
        """Send a plain-text prompt and return the response text."""
        body = await self._request(self._build_payload(prompt))
        return self._extract_text(body)

    async def generate_structured(
        self,
        prompt: str,
        schema: dict[str, object],
    ) -> StructuredResponse:
        """Send a prompt and return a structured (JSON) response.

        The ``schema`` is handed to :meth:`_build_payload`; the client itself
        only guarantees the response parses to a JSON object.
        """
        body = await self._request(self._build_payload(prompt, schema=schema))
        raw = self._extract_text(body)
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected JSON object, got {type(parsed).__name__}")
        return StructuredResponse(data=parsed, raw=raw)
