"""The client contract: a structural Protocol plus shared bits.

``LLMClient`` is a :class:`typing.Protocol`, not a base class — anything with a
``name`` and async ``generate`` / ``generate_structured`` satisfies it, no
subclassing required.  The built-in
:class:`~tournament_eval.llm.openai_compatible.OpenAICompatibleClient` (httpx, no
extra deps) implements it, and so do the optional SDK-backed clients behind the
``openai`` / ``anthropic`` / ``ollama`` / ``bedrock`` extras.

Authorship note: ``name`` is load-bearing — the orchestration stamps it as each
result's ``author`` and uses it (with the task id) as the resume key, so two
clients in one run must have distinct names.
"""

import asyncio
import contextlib
import dataclasses
from contextlib import AbstractAsyncContextManager
from typing import NotRequired, Protocol, Required, TypedDict


@dataclasses.dataclass(frozen=True, slots=True)
class StructuredResponse:
    """Wrapper returned by :meth:`LLMClient.generate_structured`.

    Preserves both the parsed dict and the exact raw string for audit.
    """

    data: dict[str, object]
    """The parsed JSON object."""
    raw: str
    """The exact raw string the model returned before parsing."""


class GenerationConfig(TypedDict):
    """Generation knobs shared by every client (built-in or SDK-backed).

    The provider-agnostic slice of configuration — *what* to generate, not *how*
    to reach the provider.  Transport/auth (base URLs, API keys, regions, the AWS
    credential chain) is each client's own concern: the built-in client carries it
    on its own config, while the SDK-backed clients take an already-constructed
    SDK client instead of re-exposing it here.
    """

    model_id: Required[str]
    """Model identifier; also the default ``author`` when ``name`` is unset."""
    name: NotRequired[str]
    """Optional author label; falls back to ``model_id``."""
    temperature: NotRequired[float]
    """Sampling temperature (clients default to 1.0)."""
    max_tokens: NotRequired[int]
    """Cap on generated tokens; omitted (provider default) when unset.  Note this
    also bounds reasoning budget, so a low value can truncate output."""
    system_prompt: NotRequired[str]
    """Optional system-level prompt."""
    max_concurrency: NotRequired[int]
    """Max in-flight requests for this client; unbounded when unset."""


class LLMClient(Protocol):
    """The minimal contract every client (built-in or SDK-backed) satisfies."""

    name: str
    """Model identifier, used as the ``author`` on every result it produces."""

    async def generate(self, prompt: str) -> str:
        """Send a plain-text prompt and return the response text."""
        ...

    async def generate_structured(self, prompt: str, schema: dict[str, object]) -> StructuredResponse:
        """Send a prompt and return a JSON response constrained to ``schema``."""
        ...


def resolve_semaphore(
    semaphore: asyncio.Semaphore | None,
    max_concurrency: int | None,
) -> asyncio.Semaphore | None:
    """Pick a client's throttle: an explicit ``semaphore`` wins, else build one
    from ``max_concurrency``, else ``None`` (unbounded).

    The companion to :func:`concurrency_guard`: clients don't share a base class,
    so this gives each constructor the same "shared semaphore, or a private one
    sized to my cap, or nothing" resolution without inheritance.
    """
    return semaphore or (asyncio.Semaphore(max_concurrency) if max_concurrency is not None else None)


def concurrency_guard(
    semaphore: asyncio.Semaphore | None,
) -> AbstractAsyncContextManager[None]:
    """Acquire ``semaphore`` if set, else a no-op — the shared throttle helper.

    Clients don't share a base class, so this small function gives each one the
    same "bound my in-flight requests, or don't" behaviour without inheritance.
    """
    return semaphore if semaphore is not None else contextlib.nullcontext()
