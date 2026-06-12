"""LLM client abstractions and provider implementations.

Layering
--------
* :class:`LLMClient` — provider-agnostic base: the abstract ``generate`` /
  ``generate_structured`` contract.  Knows nothing about any wire protocol.
* :class:`HTTPLLMClient` — the shared template for any HTTP+JSON chat API:
  build a payload, POST it with retries, pull the text back out, and (for
  structured calls) parse JSON.  Subclasses fill in four small hooks.
* :class:`OpenAICompatibleLLMClient` / :class:`OllamaLLMClient` — sibling
  implementations of those hooks for two different wire protocols.

Adding a provider is implementing four hooks on :class:`HTTPLLMClient`:
``_endpoint_url``, ``_headers``, ``_build_payload`` and ``_extract_text``.
"""

import abc
import asyncio
import dataclasses
import json
import os

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


@dataclasses.dataclass(frozen=True)
class HTTPModelConfig(ModelConfig):
    """Configuration shared by every HTTP+JSON provider (see :class:`HTTPLLMClient`)."""

    base_url: str | None = None
    """Server base URL.  ``None`` lets the client fall back to its own default."""
    timeout: float = 300.0
    """Request timeout in seconds (default 5 minutes)."""


@dataclasses.dataclass(frozen=True)
class OpenAICompatibleModelConfig(HTTPModelConfig):
    """Configuration for any OpenAI-compatible ``/chat/completions`` server.

    Covers the official OpenAI API and the growing field of servers that speak
    the same protocol (``mlx_lm.server``, vLLM, llama.cpp, ...).  Point
    ``base_url`` at the server's ``/v1`` root.
    """

    api_key: str | None = None
    """API key.  Falls back to the ``OPENAI_API_KEY`` environment variable if
    ``None``.  Local servers usually ignore it."""
    seed: int | None = None
    """Optional seed for deterministic sampling."""


@dataclasses.dataclass(frozen=True)
class AnthropicModelConfig(ModelConfig):
    """Anthropic-specific configuration. Stub only until client is wired up."""

    api_key: str | None = None
    """Anthropic API key.  Falls back to environment variable if ``None``."""
    thinking_budget: int | None = None
    """Optional extended thinking budget for Claude 3.7+."""


@dataclasses.dataclass(frozen=True)
class OllamaModelConfig(HTTPModelConfig):
    """Ollama configuration, using its native ``/api/chat`` endpoint.

    The native API accepts a full JSON schema in its ``format`` field, giving
    grammar-constrained structured output.
    """

    base_url: str | None = "http://localhost:11434"
    """Ollama server base URL.  Defaults to the local default port."""
    seed: int | None = None
    """Optional seed for deterministic sampling."""


class LLMClient(abc.ABC):
    """Abstract base class for LLM provider wrappers.

    Provider-agnostic on purpose: it declares the ``generate`` /
    ``generate_structured`` contract but knows nothing about any wire protocol.
    HTTP+JSON providers should subclass :class:`HTTPLLMClient` rather than this
    directly.
    """

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
        # Transport is deliberately not set here, it _can_ be set by users
        # at runtime, but it's not a public API and really only used by
        # tests.
        self._transport: httpx.AsyncBaseTransport | None = None

    @property
    def name(self) -> str:
        """Return the model identifier used as ``author`` in results."""
        return self._config.model_name

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


class HTTPLLMClient(LLMClient):
    """Shared template for any HTTP+JSON chat API.

    Implements the full ``generate`` / ``generate_structured`` flow — build a
    payload, POST it with retries, extract the text, and (for structured calls)
    parse it as JSON — in terms of four hooks a subclass fills in for its wire
    protocol:

    * :meth:`_endpoint_url` — where to POST.
    * :meth:`_headers` — request headers (auth, content type).
    * :meth:`_build_payload` — the request body; ``schema`` is non-``None`` for
      structured calls so the subclass can request constrained/JSON output.
    * :meth:`_extract_text` — pull the assistant text out of the response body.
    """

    def __init__(self, config: HTTPModelConfig) -> None:
        super().__init__(config)
        self._http_config = config

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
        timeout = httpx.Timeout(self._http_config.timeout)

        last_err: Exception | None = None
        for attempt in range(self._config.retry_count):
            try:
                async with httpx.AsyncClient(
                    timeout=timeout, transport=self._transport
                ) as http:
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


class OpenAICompatibleLLMClient(HTTPLLMClient):
    """Client for any OpenAI-compatible ``/chat/completions`` endpoint.

    Talks to the official OpenAI API or any server that speaks the same protocol
    (``mlx_lm.server``, vLLM, ...): point ``base_url`` at the server's ``/v1``
    root.  Structured output uses ``response_format`` with a strict
    ``json_schema``.
    """

    _DEFAULT_BASE_URL = "https://api.openai.com/v1"

    def __init__(self, config: OpenAICompatibleModelConfig) -> None:
        super().__init__(config)
        self._compat_config = config

    @property
    def _endpoint_url(self) -> str:
        base = self._compat_config.base_url or self._DEFAULT_BASE_URL
        return f"{base.rstrip('/')}/chat/completions"

    @property
    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        key = self._compat_config.api_key or os.environ.get("OPENAI_API_KEY")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def _build_payload(
        self,
        prompt: str,
        *,
        schema: dict[str, object] | None = None,
    ) -> dict[str, object]:
        payload: dict[str, object] = {
            "model": self._compat_config.model_name,
            "messages": self._messages(prompt),
            "stream": False,
        }
        if self._compat_config.temperature is not None:
            payload["temperature"] = self._compat_config.temperature
        if self._compat_config.max_tokens is not None:
            payload["max_tokens"] = self._compat_config.max_tokens
        if self._compat_config.seed is not None:
            payload["seed"] = self._compat_config.seed
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

    def _extract_text(self, body: dict[str, object]) -> str:
        """Extract the assistant message text from a chat-completions body."""
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


class AnthropicLLMClient(LLMClient):
    """Concrete Anthropic client implementation (not yet wired up)."""

    def __init__(self, config: AnthropicModelConfig) -> None:
        super().__init__(config)  # pragma: no cover

    async def generate(self, prompt: str) -> str:
        raise NotImplementedError  # pragma: no cover

    async def generate_structured(
        self,
        prompt: str,
        schema: dict[str, object],
    ) -> StructuredResponse:
        raise NotImplementedError  # pragma: no cover


class OllamaLLMClient(HTTPLLMClient):
    """Ollama, via its native ``/api/chat`` endpoint.

    A sibling of :class:`OpenAICompatibleLLMClient`, not a subclass: Ollama
    speaks its own wire protocol.  The payoff is structured output via the
    native ``format`` field, which accepts a full JSON schema and so constrains
    decoding to it — stronger than the OpenAI-compatible endpoint, which ignores
    ``json_schema``.
    """

    _DEFAULT_BASE_URL = "http://localhost:11434"

    def __init__(self, config: OllamaModelConfig) -> None:
        super().__init__(config)
        self._ollama_config = config

    @property
    def _endpoint_url(self) -> str:
        base = self._ollama_config.base_url or self._DEFAULT_BASE_URL
        return f"{base.rstrip('/')}/api/chat"

    @property
    def _headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json"}

    def _build_payload(
        self,
        prompt: str,
        *,
        schema: dict[str, object] | None = None,
    ) -> dict[str, object]:
        options: dict[str, object] = {}
        if self._ollama_config.temperature is not None:
            options["temperature"] = self._ollama_config.temperature
        if self._ollama_config.max_tokens is not None:
            options["num_predict"] = self._ollama_config.max_tokens
        if self._ollama_config.seed is not None:
            options["seed"] = self._ollama_config.seed

        payload: dict[str, object] = {
            "model": self._ollama_config.model_name,
            "messages": self._messages(prompt),
            "stream": False,
            "options": options,
        }
        # Ollama's native `format` takes a JSON schema directly and constrains
        # decoding to it; the orchestration layer also restates the shape in the
        # prompt as a belt-and-braces measure.
        if schema is not None:
            payload["format"] = schema
        return payload

    def _extract_text(self, body: dict[str, object]) -> str:
        """Extract the assistant text from an ``/api/chat`` response body."""
        message = body.get("message")
        if not isinstance(message, dict):
            raise ValueError("Ollama response missing 'message'")
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("Ollama response 'content' is not a string")
        return content
