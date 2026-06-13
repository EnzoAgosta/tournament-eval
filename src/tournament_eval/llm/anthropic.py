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
from anthropic.types import Message, MessageParam, TextBlock

from tournament_eval.llm.base import (
    GenerationConfig,
    StructuredResponse,
    concurrency_guard,
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
        self._temperature = kwargs.get("temperature", 1.0)
        self._max_tokens = kwargs.get("max_tokens", _DEFAULT_MAX_TOKENS)
        self._system_prompt = kwargs.get("system_prompt")
        self._max_concurrency = kwargs.get("max_concurrency")
        self._sem: asyncio.Semaphore | None = semaphore or (
            asyncio.Semaphore(self._max_concurrency) if self._max_concurrency is not None else None
        )

    @property
    def name(self) -> str:
        return self._name or self._model_id

    @property
    def _system(self) -> str | Omit:
        return self._system_prompt if self._system_prompt is not None else omit

    def _text(self, message: Message) -> str:
        if message.stop_reason == "refusal":
            raise ValueError("Model refused to respond")
        for block in message.content:
            if isinstance(block, TextBlock):
                return block.text
        raise ValueError("Anthropic response had no text block")

    async def generate(self, prompt: str) -> str:
        async with concurrency_guard(self._sem):
            message = await self._client.messages.create(
                model=self._model_id,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                system=self._system,
                messages=[MessageParam(role="user", content=prompt)],
            )
        return self._text(message)

    async def generate_structured(self, prompt: str, schema: dict[str, object]) -> StructuredResponse:
        async with concurrency_guard(self._sem):
            message = await self._client.messages.create(
                model=self._model_id,
                max_tokens=self._max_tokens,
                temperature=self._temperature,
                system=self._system,
                messages=[MessageParam(role="user", content=prompt)],
                output_config={"format": {"type": "json_schema", "schema": schema}},
            )
        raw = self._text(message)
        parsed = orjson.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected JSON object, got {type(parsed).__name__}")
        return StructuredResponse(data=parsed, raw=raw)
