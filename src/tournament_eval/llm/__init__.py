"""LLM client abstractions and provider implementations.

The public surface is flat — import everything from :mod:`tournament_eval.llm`
regardless of which sibling module it lives in:

* :mod:`~tournament_eval.llm.base` — the provider-agnostic :class:`LLMClient`,
  the shared :class:`HTTPLLMClient` template, the base configs, and
  :class:`StructuredResponse`.
* :mod:`~tournament_eval.llm.openai` / :mod:`~tournament_eval.llm.bedrock` /
  :mod:`~tournament_eval.llm.anthropic` / :mod:`~tournament_eval.llm.ollama` —
  the concrete providers.

Adding a provider is implementing four hooks on :class:`HTTPLLMClient`
(``_endpoint_url``, ``_headers``, ``_build_payload``, ``_extract_text``) in a new
sibling module, then re-exporting it here.
"""

from tournament_eval.llm.anthropic import AnthropicLLMClient, AnthropicModelConfig
from tournament_eval.llm.base import (
    HTTPLLMClient,
    HTTPModelConfig,
    LLMClient,
    ModelConfig,
    StructuredResponse,
)
from tournament_eval.llm.bedrock import BedrockLLMClient, BedrockModelConfig
from tournament_eval.llm.ollama import OllamaLLMClient, OllamaModelConfig
from tournament_eval.llm.openai import (
    OpenAICompatibleLLMClient,
    OpenAICompatibleModelConfig,
)

__all__ = [
    "AnthropicLLMClient",
    "AnthropicModelConfig",
    "BedrockLLMClient",
    "BedrockModelConfig",
    "HTTPLLMClient",
    "HTTPModelConfig",
    "LLMClient",
    "ModelConfig",
    "OllamaLLMClient",
    "OllamaModelConfig",
    "OpenAICompatibleLLMClient",
    "OpenAICompatibleModelConfig",
    "StructuredResponse",
]
