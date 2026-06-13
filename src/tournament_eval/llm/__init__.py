"""LLM client contract and the built-in, dependency-free client.

Only the core surface is imported here: the :class:`LLMClient` Protocol,
:class:`StructuredResponse`, and the httpx-based :class:`OpenAICompatibleClient`.

The SDK-backed clients live in sibling modules gated behind extras
(:mod:`~tournament_eval.llm.openai`, ``.anthropic``, ``.ollama``, ``.bedrock``)
and are intentionally *not* imported here — importing them requires their extra
installed.  Reach them via the lazy top-level names (``from tournament_eval
import BedrockClient``) or import the submodule directly.
"""

from tournament_eval.llm.base import LLMClient, StructuredResponse, concurrency_guard
from tournament_eval.llm.openai_compatible import OpenAICompatibleClient

__all__ = [
    "LLMClient",
    "OpenAICompatibleClient",
    "StructuredResponse",
    "concurrency_guard",
]
