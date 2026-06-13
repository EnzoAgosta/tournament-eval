"""Ollama client over the official SDK.  Requires the ``ollama`` extra.

Wraps an :class:`ollama.AsyncClient` you construct and own — host and transport
are configured the SDK's way, so this adapter just maps the shared
:class:`~tournament_eval.llm.base.GenerationConfig` knobs (plus ``seed``) onto two
``chat`` calls.  Nothing to open or close: the SDK client's lifecycle is yours.

Structured output uses Ollama's native ``format`` field, which takes a JSON
schema and constrains decoding to it.
"""

import asyncio
from typing import NotRequired, Unpack

import orjson
from ollama import AsyncClient

from tournament_eval.llm.base import (
    GenerationConfig,
    StructuredResponse,
    concurrency_guard,
)


class OllamaConfig(GenerationConfig):
    """Shared generation knobs plus Ollama's ``seed``."""

    seed: NotRequired[int]
    """Optional seed for deterministic sampling."""


class OllamaClient:
    """Adapter over a caller-provided :class:`ollama.AsyncClient`."""

    def __init__(
        self,
        client: AsyncClient,
        *,
        semaphore: asyncio.Semaphore | None = None,
        **kwargs: Unpack[OllamaConfig],
    ) -> None:
        self._client = client
        self._model_id = kwargs["model_id"]
        self._name = kwargs.get("name")
        self._temperature = kwargs.get("temperature", 1.0)
        self._max_tokens = kwargs.get("max_tokens")
        self._system_prompt = kwargs.get("system_prompt")
        self._seed = kwargs.get("seed")
        self._max_concurrency = kwargs.get("max_concurrency")
        self._sem: asyncio.Semaphore | None = semaphore or (
            asyncio.Semaphore(self._max_concurrency) if self._max_concurrency is not None else None
        )

    @property
    def name(self) -> str:
        return self._name or self._model_id

    def _messages(self, prompt: str) -> list[dict[str, str]]:
        messages: list[dict[str, str]] = []
        if self._system_prompt is not None:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": prompt})
        return messages

    def _options(self) -> dict[str, object]:
        options: dict[str, object] = {"temperature": self._temperature}
        if self._max_tokens is not None:
            options["num_predict"] = self._max_tokens
        if self._seed is not None:
            options["seed"] = self._seed
        return options

    async def generate(self, prompt: str) -> str:
        async with concurrency_guard(self._sem):
            response = await self._client.chat(
                model=self._model_id,
                messages=self._messages(prompt),
                options=self._options(),
            )
        return response.message.content or ""

    async def generate_structured(self, prompt: str, schema: dict[str, object]) -> StructuredResponse:
        async with concurrency_guard(self._sem):
            response = await self._client.chat(
                model=self._model_id,
                messages=self._messages(prompt),
                options=self._options(),
                format=schema,
            )
        raw = response.message.content or ""
        parsed = orjson.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected JSON object, got {type(parsed).__name__}")
        return StructuredResponse(data=parsed, raw=raw)
