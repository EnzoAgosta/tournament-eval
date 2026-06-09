"""LLM client abstractions and provider implementations."""

import dataclasses
import abc


@dataclasses.dataclass(frozen=True, slots=True)
class StructuredResponse:
    """Wrapper returned by :meth:`LLMClient.generate_structured`.

    Preserves both the parsed dict and the exact raw string for audit.
    """

    data: dict
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


@dataclasses.dataclass(frozen=True)
class AnthropicModelConfig(ModelConfig):
    """Anthropic-specific configuration."""

    api_key: str | None = None
    """Anthropic API key.  Falls back to environment variable if ``None``."""
    thinking_budget: int | None = None
    """Optional extended thinking budget for Claude 3.7+."""


class LLMClient(abc.ABC):
    """Abstract base class for LLM provider wrappers.

    Each concrete subclass is paired with a specific
    :class:`ModelConfig` subclass (e.g. :class:`OpenAIClient` with
    :class:`OpenAIModelConfig`).
    """

    def __init__(self, config: ModelConfig) -> None:
        self._config = config

    @property
    def name(self) -> str:
        """Return the model identifier used as ``author`` in results."""
        return self._config.model_name

    @abc.abstractmethod
    async def generate(self, prompt: str) -> str:
        """Send a plain-text prompt and return the raw response."""
        ...

    @abc.abstractmethod
    async def generate_structured(self, prompt: str, schema: dict) -> StructuredResponse:
        """Send a prompt and return a structured response."""
        ...


class OpenAILLMClient(LLMClient):
    """Concrete OpenAI client implementation."""

    def __init__(self, config: OpenAIModelConfig) -> None:
        super().__init__(config)

    async def generate(self, prompt: str) -> str:
        raise NotImplementedError

    async def generate_structured(self, prompt: str, schema: dict) -> StructuredResponse:
        raise NotImplementedError


class AnthropicLLMClient(LLMClient):
    """Concrete Anthropic client implementation."""

    def __init__(self, config: AnthropicModelConfig) -> None:
        super().__init__(config)

    async def generate(self, prompt: str) -> str:
        raise NotImplementedError

    async def generate_structured(self, prompt: str, schema: dict) -> StructuredResponse:
        raise NotImplementedError
