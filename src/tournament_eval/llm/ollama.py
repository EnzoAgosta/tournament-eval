"""Ollama native ``/api/chat`` client and config."""

import dataclasses

from tournament_eval.llm.base import HTTPLLMClient, HTTPModelConfig


@dataclasses.dataclass(frozen=True)
class OllamaModelConfig(HTTPModelConfig):
    """Ollama configuration, using its native ``/api/chat`` endpoint.

    The native API accepts a full JSON schema in its ``format`` field, giving
    grammar-constrained structured output.
    """

    base_url: str | None = "http://localhost:11434"
    """Ollama server base URL.  Defaults to the local default port."""
    seed: int | None = None
    """Optional seed for deterministic sampling."""


class OllamaLLMClient(HTTPLLMClient[OllamaModelConfig]):
    """Ollama, via its native ``/api/chat`` endpoint.

    A sibling of :class:`OpenAICompatibleLLMClient`, not a subclass: Ollama
    speaks its own wire protocol.  The payoff is structured output via the
    native ``format`` field, which accepts a full JSON schema and so constrains
    decoding to it — stronger than the OpenAI-compatible endpoint, which ignores
    ``json_schema``.
    """

    _DEFAULT_BASE_URL = "http://localhost:11434"

    @property
    def _endpoint_url(self) -> str:
        base = self._config.base_url or self._DEFAULT_BASE_URL
        return f"{base.rstrip('/')}/api/chat"

    @property
    def _headers(self) -> dict[str, str]:
        return {"Content-Type": "application/json"}

    def _build_payload(
        self,
        prompt: str,
        *,
        schema: dict[str, object] | None = None,
    ) -> dict[str, object]:
        options: dict[str, object] = {}
        if self._config.temperature is not None:
            options["temperature"] = self._config.temperature
        if self._config.max_tokens is not None:
            options["num_predict"] = self._config.max_tokens
        if self._config.seed is not None:
            options["seed"] = self._config.seed

        payload: dict[str, object] = {
            "model": self._config.model_name,
            "messages": self._messages(prompt),
            "stream": False,
            "options": options,
        }
        # Ollama's native `format` takes a JSON schema directly and constrains
        # decoding to it; the orchestration layer also restates the shape in the
        # prompt as a belt-and-braces measure.
        if schema is not None:
            payload["format"] = schema
        return payload

    def _extract_text(self, body: dict[str, object]) -> str:
        """Extract the assistant text from an ``/api/chat`` response body."""
        message = body.get("message")
        if not isinstance(message, dict):
            raise ValueError("Ollama response missing 'message'")
        content = message.get("content")
        if not isinstance(content, str):
            raise ValueError("Ollama response 'content' is not a string")
        return content
