"""LLM client abstractions and provider implementations.

Layering
--------
* :class:`LLMClient` — provider-agnostic base: connection-pool lifecycle,
  concurrency limiting, and the abstract ``generate`` / ``generate_structured``
  contract.  Knows nothing about any wire protocol.
* :class:`HTTPLLMClient` — the shared template for any HTTP+JSON chat API:
  build a payload, POST it with retries, pull the text back out, and (for
  structured calls) parse JSON.  Subclasses fill in four small hooks.
* :class:`OpenAICompatibleLLMClient` / :class:`OllamaLLMClient` /
  :class:`AnthropicLLMClient` — sibling implementations of those hooks for three
  different wire protocols.

Adding a provider is implementing four hooks on :class:`HTTPLLMClient`:
``_endpoint_url``, ``_headers``, ``_build_payload`` and ``_extract_text``.
"""

import abc
import asyncio
import contextlib
import dataclasses
import json
import os
from typing import Literal, Self

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
class BedrockModelConfig(OpenAICompatibleModelConfig):
    """Configuration for Amazon Bedrock via its OpenAI-compatible endpoint.

    Bedrock fronts *every* model it hosts — Claude, GPT, and open-weight — behind
    the OpenAI ``/chat/completions`` protocol, so :class:`BedrockLLMClient` is just
    a thin :class:`OpenAICompatibleLLMClient` pointed at the ``bedrock-mantle``
    endpoint with a Bedrock API key as the bearer token.  This is why a Claude
    model on Bedrock uses *this* config, not :class:`AnthropicModelConfig` — the
    wire protocol is OpenAI's, not Anthropic's native Messages API.
    """

    region: str = "us-east-1"
    """AWS region, used to build the default ``bedrock-mantle`` base URL
    (``https://bedrock-mantle.{region}.api.aws/v1``).  Ignored when ``base_url`` is
    set explicitly — e.g. to target ``bedrock-runtime`` or a self-hosted gateway."""
    api_key: str | None = None
    """Amazon Bedrock API key, sent as an ``Authorization: Bearer`` token.  Falls
    back to the ``AWS_BEARER_TOKEN_BEDROCK`` environment variable if ``None``.

    SigV4 (raw AWS credentials) is **not** supported yet — only the long- or
    short-lived bearer API key.  SigV4 can be added later as a config-selected auth
    mode on this client without disturbing the shared HTTP template."""


@dataclasses.dataclass(frozen=True)
class AnthropicModelConfig(HTTPModelConfig):
    """Configuration for Anthropic's native Messages API.

    See :class:`AnthropicLLMClient`.
    """

    base_url: str | None = "https://api.anthropic.com"
    """Anthropic API base URL.  Defaults to the hosted API."""
    api_key: str | None = None
    """API key.  Falls back to the ``ANTHROPIC_API_KEY`` environment variable if
    ``None``.  Sent in the ``x-api-key`` header (Anthropic does not use a Bearer
    token)."""
    effort: str | None = None
    """Optional reasoning effort, sent as ``output_config={"effort": ...}``.

    One of ``"low"``, ``"medium"``, ``"high"``, ``"xhigh"`` (Opus 4.7+) or
    ``"max"`` (Opus 4.5+).  A GA control on the current Opus / Sonnet 4.6 / Fable
    generation that tunes reasoning depth and overall token spend; **older models
    (Sonnet 4.5, Haiku 4.5, 3.x) reject it**, so leave it ``None`` for those.

    This is the recommended quality knob for this client: unlike :attr:`thinking`
    (below) it is compatible with the forced-tool-use structured-output path, so it
    applies to both ``generate`` and ``generate_structured``."""
    thinking: Literal["adaptive"] | int | None = None
    """The model's reasoning mode.

    * ``None`` (default) — no ``thinking`` parameter; the model answers directly.
    * ``"adaptive"`` — adaptive thinking (``thinking={"type": "adaptive"}``); the
      model decides when and how much to think.  The modern path (Opus 4.6+, Sonnet
      4.6, Fable 5) and the recommended setting for reasoning-heavy work.
    * an ``int`` — a *manual* extended-thinking budget in tokens
      (``thinking={"type": "enabled", "budget_tokens": N}``), for older models that
      still support it (Sonnet 4.5, 3.x).  The current Opus/Fable generation rejects
      a fixed budget — use ``"adaptive"`` there.

    Either thinking mode is **incompatible with forced tool use**, so it cannot be
    combined with :attr:`structured_output` = ``"tool_use"`` (which pins
    ``tool_choice``).  It *is* compatible with the ``"json_schema"`` path — so on the
    current models, adaptive thinking and schema-enforced output work together."""
    structured_output: Literal["json_schema", "tool_use"] = "json_schema"
    """How :meth:`AnthropicLLMClient.generate_structured` constrains output.

    * ``"json_schema"`` (default) — hand the schema to the API via
      ``output_config={"format": {"type": "json_schema", "schema": ...}}`` and let
      it enforce the shape, the same API-enforced path the OpenAI and Ollama
      clients use.  Supported on Opus 4.1/4.5/4.8, Sonnet 4.6, Haiku 4.5, and Fable
      5, and — unlike forced tool use — compatible with :attr:`thinking` (adaptive
      or manual).
    * ``"tool_use"`` — express the schema as a *forced tool call* (``tool_choice``
      pinned to a tool whose ``input_schema`` is the schema).  Works on every
      Claude model, so it's the fallback for older models that don't support
      ``output_config.format``.
    """


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


class OpenAICompatibleLLMClient[C: OpenAICompatibleModelConfig](HTTPLLMClient[C]):
    """Client for any OpenAI-compatible ``/chat/completions`` endpoint.

    Talks to the official OpenAI API or any server that speaks the same protocol
    (``mlx_lm.server``, vLLM, ...): point ``base_url`` at the server's ``/v1``
    root.  Structured output uses ``response_format`` with a strict
    ``json_schema``.

    Generic over its config so a subclass (e.g. :class:`BedrockLLMClient`) can
    re-bind ``C`` to a narrower config and still read its own fields off
    ``self._config``.  Instantiated directly, ``C`` is inferred from the config
    argument.
    """

    _DEFAULT_BASE_URL = "https://api.openai.com/v1"

    @property
    def _endpoint_url(self) -> str:
        base = self._config.base_url or self._DEFAULT_BASE_URL
        return f"{base.rstrip('/')}/chat/completions"

    @property
    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        key = self._config.api_key or os.environ.get("OPENAI_API_KEY")
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
            "model": self._config.model_name,
            "messages": self._messages(prompt),
            "stream": False,
        }
        if self._config.temperature is not None:
            payload["temperature"] = self._config.temperature
        if self._config.max_tokens is not None:
            payload["max_tokens"] = self._config.max_tokens
        if self._config.seed is not None:
            payload["seed"] = self._config.seed
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


class BedrockLLMClient(OpenAICompatibleLLMClient[BedrockModelConfig]):
    """Amazon Bedrock, via its OpenAI-compatible ``/chat/completions`` endpoint.

    A *thin* subclass of :class:`OpenAICompatibleLLMClient`: Bedrock speaks the
    OpenAI protocol for every model it hosts, so all the payload/extraction logic
    is inherited.  Only two things differ, both small:

    * **Endpoint** — defaults to the ``bedrock-mantle`` URL built from
      :attr:`BedrockModelConfig.region`; an explicit ``base_url`` still wins (to
      target ``bedrock-runtime`` or a gateway).
    * **Auth** — the bearer token falls back to the ``AWS_BEARER_TOKEN_BEDROCK``
      environment variable rather than ``OPENAI_API_KEY``.

    Because the protocol is OpenAI's, **Claude on Bedrock uses this client, not**
    :class:`AnthropicLLMClient` — the latter targets the first-party Messages API.

    Caveat: structured output is inherited as ``response_format`` with a strict
    ``json_schema``.  Whether Bedrock *enforces* that for every model (notably
    Claude) is not yet verified end-to-end — if a model ignores it, output falls
    back to the prompt-restated schema, same as any OpenAI-compatible endpoint.

    Only the bearer-token API key is supported today; SigV4 can be added later as a
    config-selected auth mode (via an ``httpx`` auth object, which sees the request
    body it needs to sign) without changing the shared template.
    """

    @property
    def _endpoint_url(self) -> str:
        base = self._config.base_url or (
            f"https://bedrock-mantle.{self._config.region}.api.aws/v1"
        )
        return f"{base.rstrip('/')}/chat/completions"

    @property
    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        key = self._config.api_key or os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers


class AnthropicLLMClient(HTTPLLMClient[AnthropicModelConfig]):
    """Anthropic, via its native ``/v1/messages`` endpoint.

    A sibling of :class:`OpenAICompatibleLLMClient`, not a subclass: the Messages
    API diverges from the OpenAI protocol in ways the shared template can't paper
    over — so this client builds its own payload rather than reusing
    :meth:`HTTPLLMClient._messages`:

    * The **system prompt is a top-level ``system`` parameter**, not a message
      turn, so the messages array carries user/assistant turns only.
    * **``max_tokens`` is required** by the API; when the config leaves it unset
      the client falls back to ``_DEFAULT_MAX_TOKENS`` (4096).
    * Structured output has **two strategies**, chosen by
      :attr:`AnthropicModelConfig.structured_output`.  ``"json_schema"`` (default)
      hands the schema to the API via ``output_config.format`` — the API-enforced
      path, returned as a normal text block — while ``"tool_use"`` expresses it as
      a forced tool call for older models that lack ``output_config.format``.
      Either way :meth:`_extract_text` yields a JSON string the template parses.

    Authentication is the ``x-api-key`` header plus ``anthropic-version`` — **not**
    a Bearer token.
    """

    _DEFAULT_BASE_URL = "https://api.anthropic.com"
    _ANTHROPIC_VERSION = "2023-06-01"
    _DEFAULT_MAX_TOKENS = 4096
    _STRUCTURED_TOOL_NAME = "structured_response"

    @property
    def _endpoint_url(self) -> str:
        base = self._config.base_url or self._DEFAULT_BASE_URL
        return f"{base.rstrip('/')}/v1/messages"

    @property
    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": self._ANTHROPIC_VERSION,
        }
        key = self._config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if key:
            headers["x-api-key"] = key
        return headers

    def _build_payload(
        self,
        prompt: str,
        *,
        schema: dict[str, object] | None = None,
    ) -> dict[str, object]:
        # `system` is a top-level parameter, not a turn, so the messages array is
        # user/assistant only — don't reuse HTTPLLMClient._messages here.
        payload: dict[str, object] = {
            "model": self._config.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self._config.max_tokens or self._DEFAULT_MAX_TOKENS,
        }
        if self._config.system_prompt is not None:
            payload["system"] = self._config.system_prompt
        if self._config.temperature is not None:
            payload["temperature"] = self._config.temperature
        thinking = self._config.thinking
        if thinking == "adaptive":
            payload["thinking"] = {"type": "adaptive"}
        elif isinstance(thinking, int):
            payload["thinking"] = {"type": "enabled", "budget_tokens": thinking}

        json_schema_mode = (
            schema is not None and self._config.structured_output == "json_schema"
        )

        # `effort` and `format` both live under `output_config`; build it once so
        # they coexist rather than clobbering each other.
        output_config: dict[str, object] = {}
        if self._config.effort is not None:
            output_config["effort"] = self._config.effort
        if json_schema_mode:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        if output_config:
            payload["output_config"] = output_config

        # The `tool_use` fallback: express the schema as a forced tool call — the
        # schema is the tool's input_schema and tool_choice pins the model to it.
        # Works on every Claude model, where `output_config.format` is newer-only.
        if schema is not None and not json_schema_mode:
            payload["tools"] = [
                {
                    "name": self._STRUCTURED_TOOL_NAME,
                    "description": "Return the structured response.",
                    "input_schema": schema,
                }
            ]
            payload["tool_choice"] = {
                "type": "tool",
                "name": self._STRUCTURED_TOOL_NAME,
            }
        return payload

    def _extract_text(self, body: dict[str, object]) -> str:
        """Extract the assistant text — or, for a forced tool call, its JSON input.

        The response ``content`` is a list of blocks; the first ``text`` block is
        returned verbatim (this also covers ``json_schema`` structured output,
        which the API returns as a text block of JSON), while a ``tool_use`` block
        (the forced-tool structured path) has its ``input`` re-serialised to JSON
        so the template's ``generate_structured`` can parse it back.  Any
        ``thinking`` blocks (e.g. from adaptive thinking) are skipped.
        """
        if body.get("stop_reason") == "refusal":
            raise ValueError("Model refused to respond")
        content = body.get("content")
        if not isinstance(content, list) or not content:
            raise ValueError("Anthropic response missing a non-empty 'content' list")
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                text = block.get("text")
                if not isinstance(text, str):
                    raise ValueError("Anthropic 'text' block content is not a string")
                return text
            if block.get("type") == "tool_use":
                tool_input = block.get("input")
                if not isinstance(tool_input, dict):
                    raise ValueError("Anthropic 'tool_use' 'input' is not an object")
                return json.dumps(tool_input)
        raise ValueError("Anthropic response had no 'text' or 'tool_use' block")


class OllamaLLMClient(HTTPLLMClient[OllamaModelConfig]):
    """Ollama, via its native ``/api/chat`` endpoint.

    A sibling of :class:`OpenAICompatibleLLMClient`, not a subclass: Ollama
    speaks its own wire protocol.  The payoff is structured output via the
    native ``format`` field, which accepts a full JSON schema and so constrains
    decoding to it — stronger than the OpenAI-compatible endpoint, which ignores
    ``json_schema``.
    """

    _DEFAULT_BASE_URL = "http://localhost:11434"

    @property
    def _endpoint_url(self) -> str:
        base = self._config.base_url or self._DEFAULT_BASE_URL
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
        if self._config.temperature is not None:
            options["temperature"] = self._config.temperature
        if self._config.max_tokens is not None:
            options["num_predict"] = self._config.max_tokens
        if self._config.seed is not None:
            options["seed"] = self._config.seed

        payload: dict[str, object] = {
            "model": self._config.model_name,
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
