"""OpenAI-compatible ``/chat/completions`` client and config."""

import dataclasses
import os

from tournament_eval.llm.base import HTTPLLMClient, HTTPModelConfig


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


class OpenAICompatibleLLMClient[C: OpenAICompatibleModelConfig](HTTPLLMClient[C]):
    """Client for any OpenAI-compatible ``/chat/completions`` endpoint.

    Talks to the official OpenAI API or any server that speaks the same protocol
    (``mlx_lm.server``, vLLM, ...): point ``base_url`` at the server's ``/v1``
    root.  Structured output uses ``response_format`` with a strict
    ``json_schema``.

    Generic over its config so a subclass (e.g. ``BedrockLLMClient``) can
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
