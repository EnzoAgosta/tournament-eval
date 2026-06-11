"""LLM client abstractions and provider implementations."""

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
class OpenAIModelConfig(ModelConfig):
    """OpenAI-specific configuration."""

    api_key: str | None = None
    """OpenAI API key.  Falls back to environment variable if ``None``."""
    base_url: str | None = None
    """Optional custom base URL (e.g. for proxies or Azure)."""
    seed: int | None = None
    """Optional seed for deterministic sampling."""
    timeout: float = 300.0
    """Request timeout in seconds (default 5 minutes)."""


@dataclasses.dataclass(frozen=True)
class AnthropicModelConfig(ModelConfig):
    """Anthropic-specific configuration."""

    api_key: str | None = None
    """Anthropic API key.  Falls back to environment variable if ``None``."""
    thinking_budget: int | None = None
    """Optional extended thinking budget for Claude 3.7+."""


@dataclasses.dataclass(frozen=True)
class OllamaModelConfig(ModelConfig):
    """Ollama-specific configuration."""

    base_url: str = "http://localhost:11434"
    """Ollama server base URL.  Defaults to the local default port."""
    timeout: float = 300.0
    """Request timeout in seconds (default 5 minutes)."""


class LLMClient(abc.ABC):
    """Abstract base class for LLM provider wrappers.

    Each concrete subclass is paired with a specific
    :class:`ModelConfig` subclass (e.g. :class:`OpenAIClient` with
    :class:`OpenAIModelConfig`).
    """

    def __init__(self, config: ModelConfig) -> None:
        self._config = config
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


class OpenAILLMClient(LLMClient):
    """Concrete OpenAI-compatible client.

    Talks to any OpenAI-compatible ``/chat/completions`` endpoint: the official
    OpenAI API, or a local server such as ``mlx_lm.server`` (point ``base_url``
    at its ``/v1`` root).  ``generate`` returns plain text; ``generate_structured``
    uses native Structured Outputs (``response_format`` with a ``json_schema`` and
    ``strict: true``).
    """

    _DEFAULT_BASE_URL = "https://api.openai.com/v1"

    def __init__(self, config: OpenAIModelConfig) -> None:
        super().__init__(config)
        self._openai_config = config
        self._transport: httpx.AsyncBaseTransport | None = None

    @property
    def _url(self) -> str:
        base = self._openai_config.base_url or self._DEFAULT_BASE_URL
        return f"{base.rstrip('/')}/chat/completions"

    @property
    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        key = self._openai_config.api_key or os.environ.get("OPENAI_API_KEY")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers

    def _messages(self, prompt: str) -> list[dict[str, str]]:
        """Build the chat ``messages`` array, prepending a system turn if set."""
        messages: list[dict[str, str]] = []
        if self._openai_config.system_prompt is not None:
            messages.append(
                {"role": "system", "content": self._openai_config.system_prompt}
            )
        messages.append({"role": "user", "content": prompt})
        return messages

    def _payload(
        self,
        prompt: str,
        *,
        response_format: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Build the JSON body for a ``/chat/completions`` request."""
        payload: dict[str, object] = {
            "model": self._openai_config.model_name,
            "messages": self._messages(prompt),
            "stream": False,
        }
        if self._openai_config.temperature is not None:
            payload["temperature"] = self._openai_config.temperature
        if self._openai_config.max_tokens is not None:
            payload["max_tokens"] = self._openai_config.max_tokens
        if self._openai_config.seed is not None:
            payload["seed"] = self._openai_config.seed
        if response_format is not None:
            payload["response_format"] = response_format
        return payload

    async def _request(
        self,
        prompt: str,
        *,
        response_format: dict[str, object] | None = None,
    ) -> dict[str, object]:
        """Send a request with retries and return the parsed JSON body."""
        payload = self._payload(prompt, response_format=response_format)
        timeout = httpx.Timeout(self._openai_config.timeout)

        last_err: Exception | None = None
        for attempt in range(self._openai_config.retry_count):
            try:
                async with httpx.AsyncClient(
                    timeout=timeout, transport=self._transport
                ) as client:
                    response = await client.post(
                        self._url, json=payload, headers=self._headers
                    )
                    response.raise_for_status()
                    body: dict[str, object] = response.json()
                    return body
            except Exception as exc:
                last_err = exc
                if attempt < self._openai_config.retry_count - 1:
                    await asyncio.sleep(2**attempt)

        assert last_err is not None, "Somehow never returned with no exception"
        raise last_err

    @staticmethod
    def _content(body: dict[str, object]) -> str:
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

    async def generate(self, prompt: str) -> str:
        """Send a plain-text prompt and return the response text."""
        body = await self._request(prompt)
        return self._content(body)

    async def generate_structured(
        self,
        prompt: str,
        schema: dict[str, object],
    ) -> StructuredResponse:
        """Send a prompt and return a structured (JSON) response.

        The ``schema`` is passed as a ``response_format`` ``json_schema`` with
        ``strict: true``; a schema-enforcing endpoint then guarantees conformance
        at the API boundary.  Only syntactic validity is the client's concern.
        """
        response_format: dict[str, object] = {
            "type": "json_schema",
            "json_schema": {
                "name": "structured_response",
                "schema": schema,
                "strict": True,
            },
        }
        body = await self._request(prompt, response_format=response_format)
        raw = self._content(body)
        parsed = json.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected JSON object, got {type(parsed).__name__}")
        return StructuredResponse(data=parsed, raw=raw)


class AnthropicLLMClient(LLMClient):
    """Concrete Anthropic client implementation."""

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


class OllamaLLMClient(LLMClient):
    """Concrete Ollama client implementation.

    Talks to a local (or remote) Ollama server via its native ``/api/generate``
    endpoint.  Supports both plain text and JSON-structured generation.
    """

    def __init__(self, config: OllamaModelConfig) -> None:
        super().__init__(config)
        self._ollama_config = config
        self._transport: httpx.AsyncBaseTransport | None = None

    @property
    def _url(self) -> str:
        return f"{self._ollama_config.base_url}/api/generate"

    def _payload(self, prompt: str, *, format_json: bool = False) -> dict[str, object]:
        """Build the JSON body for an ``/api/generate`` request."""
        payload: dict[str, object] = {
            "model": self._ollama_config.model_name,
            "prompt": prompt,
            "stream": False,
            "options": {},
        }
        if self._ollama_config.temperature is not None:
            payload["options"]["temperature"] = self._ollama_config.temperature  # type: ignore[index]
        if self._ollama_config.max_tokens is not None:
            payload["options"]["num_predict"] = self._ollama_config.max_tokens  # type: ignore[index]
        if self._ollama_config.system_prompt is not None:
            payload["system"] = self._ollama_config.system_prompt
        if format_json:
            payload["format"] = "json"
        return payload

    async def _request(
        self,
        prompt: str,
        *,
        format_json: bool = False,
    ) -> dict[str, object]:
        """Send a request to Ollama with retries and return the parsed JSON body."""
        payload = self._payload(prompt, format_json=format_json)
        timeout = httpx.Timeout(self._ollama_config.timeout)

        last_err: Exception | None = None
        for attempt in range(self._ollama_config.retry_count):
            try:
                async with httpx.AsyncClient(
                    timeout=timeout, transport=self._transport
                ) as client:
                    response = await client.post(self._url, json=payload)
                    response.raise_for_status()
                    body: dict[str, object] = response.json()
                    return body
            except Exception as exc:
                last_err = exc
                if attempt < self._ollama_config.retry_count - 1:
                    await asyncio.sleep(2**attempt)

        assert last_err is not None
        raise last_err

    async def generate(self, prompt: str) -> str:
        """Send a plain-text prompt and return the raw response string."""
        data = await self._request(prompt)
        return str(data["response"])

    async def generate_structured(
        self,
        prompt: str,
        _schema: dict[str, object],
    ) -> StructuredResponse:
        """Send a prompt and return a structured (JSON) response.

        Uses Ollama's ``format: "json"`` mode.  The schema is embedded in the
        prompt text by the orchestration layer; this method only enforces
        valid JSON output.
        """
        data = await self._request(prompt, format_json=True)
        raw_text = str(data["response"])
        parsed = json.loads(raw_text)
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected JSON object, got {type(parsed).__name__}")
        return StructuredResponse(data=parsed, raw=raw_text)
