"""OpenAI client over the official SDK's Responses API.  Requires the ``openai`` extra.

Wraps an :class:`openai.AsyncOpenAI` you construct and own — auth, base URL,
timeouts and retries are configured the SDK's way, so this adapter is just the
shared :class:`~tournament_eval.llm.base.GenerationConfig` knobs mapped onto two
``responses.create`` calls.  Nothing to open or close: the SDK client's lifecycle
is yours (it's itself an async context manager).
"""

import asyncio
from typing import Unpack

import orjson
from openai import AsyncOpenAI
from openai.types.responses import ResponseOutputRefusal
from openai.types.shared.reasoning_effort import ReasoningEffort as OpenAIReasoningEffort
from openai.types.shared_params import Reasoning

from tournament_eval.llm.base import (
    GenerationConfig,
    GenerationResponse,
    ReasoningEffort,
    StructuredResponse,
    concurrency_guard,
    resolve_semaphore,
)

# OpenAI's Responses effort tops out at "xhigh", so "max" maps down to it.
_OPENAI_EFFORT: dict[ReasoningEffort, OpenAIReasoningEffort] = {
    "low": "low",
    "medium": "medium",
    "high": "high",
    "xhigh": "xhigh",
    "max": "xhigh",
}


class OpenAIClient:
    """Adapter over a caller-provided :class:`openai.AsyncOpenAI` (Responses API)."""

    def __init__(
        self,
        client: AsyncOpenAI,
        *,
        semaphore: asyncio.Semaphore | None = None,
        **kwargs: Unpack[GenerationConfig],
    ) -> None:
        self._client = client
        self._model_id = kwargs["model_id"]
        self._name = kwargs.get("name")
        self._temperature = kwargs.get("temperature", 1.0)
        self._max_tokens = kwargs.get("max_tokens")
        self._system_prompt = kwargs.get("system_prompt")
        self._reasoning_effort: ReasoningEffort | None = kwargs.get("reasoning_effort")
        self._sem = resolve_semaphore(semaphore, kwargs.get("max_concurrency"))

    @property
    def name(self) -> str:
        return self._name or self._model_id

    @property
    def _reasoning(self) -> Reasoning | None:
        if self._reasoning_effort is None:
            return None
        return {"effort": _OPENAI_EFFORT[self._reasoning_effort], "summary": "auto"}

    async def generate(self, prompt: str) -> GenerationResponse:
        async with concurrency_guard(self._sem):
            response = await self._client.responses.create(
                model=self._model_id,
                input=prompt,
                instructions=self._system_prompt,
                temperature=self._temperature,
                max_output_tokens=self._max_tokens,
                reasoning=self._reasoning,
                stream=False,
            )
        for output in response.output:
            if isinstance(output, ResponseOutputRefusal):
                raise ValueError(f"model refused to reply: {response.output_text}")
        return GenerationResponse(
            text=response.output_text, reasoning=response.reasoning.summary if response.reasoning else None
        )

    async def generate_structured(self, prompt: str, schema: dict[str, object]) -> StructuredResponse:
        async with concurrency_guard(self._sem):
            response = await self._client.responses.create(
                model=self._model_id,
                input=prompt,
                instructions=self._system_prompt,
                temperature=self._temperature,
                max_output_tokens=self._max_tokens,
                reasoning=self._reasoning,
                stream=False,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "structured_response",
                        "schema": schema,
                        "strict": True,
                    }
                },
            )
        raw = response.output_text
        return StructuredResponse(data=orjson.loads(raw), raw=raw)
