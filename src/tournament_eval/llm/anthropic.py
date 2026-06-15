"""Anthropic client over the official SDK's Messages API.  Requires the ``anthropic`` extra.

Wraps an :class:`anthropic.AsyncAnthropic` you construct and own — auth, base URL,
timeouts and retries are configured the SDK's way, so this adapter is just the
shared :class:`~tournament_eval.llm.base.GenerationConfig` knobs mapped onto two
``messages.create`` calls.  Nothing to open or close: the SDK client's lifecycle
is yours (it's itself an async context manager).

Structured output uses the API's native json_schema structured outputs
(``output_config={"format": {"type": "json_schema", ...}}``), returned as an
ordinary text block.  Only models that support it work (the current generation:
Opus 4.1 / 4.5 / 4.8, Sonnet 4.6, Haiku 4.5, Fable); older ones are not supported
yet — a deliberate keep-it-simple choice.
"""

import asyncio
from typing import Unpack

import orjson
from anthropic import AsyncAnthropic, Omit, omit
from anthropic.types import (
    Message,
    MessageParam,
    OutputConfigParam,
    TextBlock,
    ThinkingBlock,
    ThinkingConfigAdaptiveParam,
)

from tournament_eval.llm.base import (
    GenerationConfig,
    GenerationResponse,
    ReasoningEffort,
    StructuredResponse,
    concurrency_guard,
    resolve_semaphore,
)

# The Messages API requires max_tokens; GenerationConfig leaves it optional.
_DEFAULT_MAX_TOKENS = 4096


class AnthropicClient:
    """Adapter over a caller-provided :class:`anthropic.AsyncAnthropic`."""

    def __init__(
        self,
        client: AsyncAnthropic,
        *,
        semaphore: asyncio.Semaphore | None = None,
        **kwargs: Unpack[GenerationConfig],
    ) -> None:
        self._client = client
        self._model_id = kwargs["model_id"]
        self._name = kwargs.get("name")
        # Send temperature only when the caller set it explicitly. The current
        # models (Opus 4.7+/Fable) reject temperature outright, so a default of
        # 1.0 would 400 every request; omitting it lets the model use its own
        # default. A caller that sets it on one of those models owns the 400.
        temperature = kwargs.get("temperature")
        self._temperature: float | Omit = temperature if temperature is not None else omit
        self._max_tokens = kwargs.get("max_tokens", _DEFAULT_MAX_TOKENS)
        self._system_prompt = kwargs.get("system_prompt")
        self._reasoning_effort: ReasoningEffort | None = kwargs.get("reasoning_effort")
        self._sem = resolve_semaphore(semaphore, kwargs.get("max_concurrency"))

    @property
    def name(self) -> str:
        return self._name or self._model_id

    @property
    def _system(self) -> str | Omit:
        return self._system_prompt if self._system_prompt is not None else omit

    @property
    def _thinking(self) -> ThinkingConfigAdaptiveParam | Omit:
        # Adaptive thinking is off unless requested; "summarized" is what makes the
        # returned ThinkingBlock carry text (the raw chain is never exposed).
        if self._reasoning_effort is None:
            return omit
        return {"type": "adaptive", "display": "summarized"}

    def _output_config(self, schema: dict[str, object] | None) -> OutputConfigParam | Omit:
        # Anthropic's effort levels are exactly ReasoningEffort, so it passes through.
        config: OutputConfigParam = {}
        if schema is not None:
            config["format"] = {"type": "json_schema", "schema": schema}
        if self._reasoning_effort is not None:
            config["effort"] = self._reasoning_effort
        return config or omit

    def _text(self, message: Message) -> str:
        if message.stop_reason == "refusal":
            raise ValueError("Model refused to respond")
        for block in message.content:
            if isinstance(block, TextBlock):
                return block.text
        raise ValueError("Anthropic response had no text block")

    def _reasoning(self, message: Message) -> str | None:
        for block in message.content:
            if isinstance(block, ThinkingBlock):
                return block.thinking
        return None

    async def generate(self, prompt: str) -> GenerationResponse:
        async with concurrency_guard(self._sem):
            message = await self._client.messages.create(
                model=self._model_id,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                system=self._system,
                thinking=self._thinking,
                output_config=self._output_config(None),
                messages=[MessageParam(role="user", content=prompt)],
            )
        return GenerationResponse(text=self._text(message), reasoning=self._reasoning(message))

    async def generate_structured(self, prompt: str, schema: dict[str, object]) -> StructuredResponse:
        async with concurrency_guard(self._sem):
            message = await self._client.messages.create(
                model=self._model_id,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                system=self._system,
                thinking=self._thinking,
                output_config=self._output_config(schema),
                messages=[MessageParam(role="user", content=prompt)],
            )
        raw = self._text(message)
        parsed = orjson.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected JSON object, got {type(parsed).__name__}")
        return StructuredResponse(data=parsed, raw=raw)
