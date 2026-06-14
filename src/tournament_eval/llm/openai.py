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
from openai.types.responses import Response, ResponseOutputMessage, ResponseOutputRefusal

from tournament_eval.llm.base import (
    GenerationConfig,
    GenerationResponse,
    StructuredResponse,
    concurrency_guard,
    resolve_semaphore,
)


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
        self._sem = resolve_semaphore(semaphore, kwargs.get("max_concurrency"))

    @property
    def name(self) -> str:
        return self._name or self._model_id

    def _text(self, response: Response) -> str:
        """Return the response text, raising if the model refused.

        ``output_text`` silently drops refusal parts (a refusal yields ``""``),
        so scan the output for one first and surface it as an error instead.
        """
        for item in response.output:
            if isinstance(item, ResponseOutputMessage):
                for part in item.content:
                    if isinstance(part, ResponseOutputRefusal):
                        raise ValueError(f"Model refused to respond: {part.refusal}")
        return response.output_text

    async def generate(self, prompt: str) -> GenerationResponse:
        async with concurrency_guard(self._sem):
            response = await self._client.responses.create(
                model=self._model_id,
                input=prompt,
                instructions=self._system_prompt,
                temperature=self._temperature,
                max_output_tokens=self._max_tokens,
            )
        # Reasoning models expose only a summary, and only when requested via a
        # reasoning config we don't set yet — so no trace is captured here.
        return GenerationResponse(text=self._text(response), reasoning=None)

    async def generate_structured(self, prompt: str, schema: dict[str, object]) -> StructuredResponse:
        async with concurrency_guard(self._sem):
            response = await self._client.responses.create(
                model=self._model_id,
                input=prompt,
                instructions=self._system_prompt,
                temperature=self._temperature,
                max_output_tokens=self._max_tokens,
                text={
                    "format": {
                        "type": "json_schema",
                        "name": "structured_response",
                        "schema": schema,
                        "strict": True,
                    }
                },
            )
        raw = self._text(response)
        parsed = orjson.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected JSON object, got {type(parsed).__name__}")
        return StructuredResponse(data=parsed, raw=raw)
