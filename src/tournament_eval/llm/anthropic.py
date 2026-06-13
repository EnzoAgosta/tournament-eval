"""Anthropic native ``/v1/messages`` client and config."""

import dataclasses
import json
import os
from typing import Literal

from tournament_eval.llm.base import HTTPLLMClient, HTTPModelConfig


@dataclasses.dataclass(frozen=True)
class AnthropicModelConfig(HTTPModelConfig):
    """Configuration for Anthropic's native Messages API.

    See :class:`AnthropicLLMClient`.
    """

    base_url: str | None = "https://api.anthropic.com"
    """Anthropic API base URL.  Defaults to the hosted API."""
    api_key: str | None = None
    """API key.  Falls back to the ``ANTHROPIC_API_KEY`` environment variable if
    ``None``.  Sent in the ``x-api-key`` header (Anthropic does not use a Bearer
    token)."""
    effort: str | None = None
    """Optional reasoning effort, sent as ``output_config={"effort": ...}``.

    One of ``"low"``, ``"medium"``, ``"high"``, ``"xhigh"`` (Opus 4.7+) or
    ``"max"`` (Opus 4.5+).  A GA control on the current Opus / Sonnet 4.6 / Fable
    generation that tunes reasoning depth and overall token spend; **older models
    (Sonnet 4.5, Haiku 4.5, 3.x) reject it**, so leave it ``None`` for those.

    This is the recommended quality knob for this client: unlike :attr:`thinking`
    (below) it is compatible with the forced-tool-use structured-output path, so it
    applies to both ``generate`` and ``generate_structured``."""
    thinking: Literal["adaptive"] | int | None = None
    """The model's reasoning mode.

    * ``None`` (default) — no ``thinking`` parameter; the model answers directly.
    * ``"adaptive"`` — adaptive thinking (``thinking={"type": "adaptive"}``); the
      model decides when and how much to think.  The modern path (Opus 4.6+, Sonnet
      4.6, Fable 5) and the recommended setting for reasoning-heavy work.
    * an ``int`` — a *manual* extended-thinking budget in tokens
      (``thinking={"type": "enabled", "budget_tokens": N}``), for older models that
      still support it (Sonnet 4.5, 3.x).  The current Opus/Fable generation rejects
      a fixed budget — use ``"adaptive"`` there.

    Either thinking mode is **incompatible with forced tool use**, so it cannot be
    combined with :attr:`structured_output` = ``"tool_use"`` (which pins
    ``tool_choice``).  It *is* compatible with the ``"json_schema"`` path — so on the
    current models, adaptive thinking and schema-enforced output work together."""
    structured_output: Literal["json_schema", "tool_use"] = "json_schema"
    """How :meth:`AnthropicLLMClient.generate_structured` constrains output.

    * ``"json_schema"`` (default) — hand the schema to the API via
      ``output_config={"format": {"type": "json_schema", "schema": ...}}`` and let
      it enforce the shape, the same API-enforced path the OpenAI and Ollama
      clients use.  Supported on Opus 4.1/4.5/4.8, Sonnet 4.6, Haiku 4.5, and Fable
      5, and — unlike forced tool use — compatible with :attr:`thinking` (adaptive
      or manual).
    * ``"tool_use"`` — express the schema as a *forced tool call* (``tool_choice``
      pinned to a tool whose ``input_schema`` is the schema).  Works on every
      Claude model, so it's the fallback for older models that don't support
      ``output_config.format``.
    """


class AnthropicLLMClient(HTTPLLMClient[AnthropicModelConfig]):
    """Anthropic, via its native ``/v1/messages`` endpoint.

    A sibling of :class:`OpenAICompatibleLLMClient`, not a subclass: the Messages
    API diverges from the OpenAI protocol in ways the shared template can't paper
    over — so this client builds its own payload rather than reusing
    :meth:`HTTPLLMClient._messages`:

    * The **system prompt is a top-level ``system`` parameter**, not a message
      turn, so the messages array carries user/assistant turns only.
    * **``max_tokens`` is required** by the API; when the config leaves it unset
      the client falls back to ``_DEFAULT_MAX_TOKENS`` (4096).
    * Structured output has **two strategies**, chosen by
      :attr:`AnthropicModelConfig.structured_output`.  ``"json_schema"`` (default)
      hands the schema to the API via ``output_config.format`` — the API-enforced
      path, returned as a normal text block — while ``"tool_use"`` expresses it as
      a forced tool call for older models that lack ``output_config.format``.
      Either way :meth:`_extract_text` yields a JSON string the template parses.

    Authentication is the ``x-api-key`` header plus ``anthropic-version`` — **not**
    a Bearer token.
    """

    _DEFAULT_BASE_URL = "https://api.anthropic.com"
    _ANTHROPIC_VERSION = "2023-06-01"
    _DEFAULT_MAX_TOKENS = 4096
    _STRUCTURED_TOOL_NAME = "structured_response"

    @property
    def _endpoint_url(self) -> str:
        base = self._config.base_url or self._DEFAULT_BASE_URL
        return f"{base.rstrip('/')}/v1/messages"

    @property
    def _headers(self) -> dict[str, str]:
        headers = {
            "Content-Type": "application/json",
            "anthropic-version": self._ANTHROPIC_VERSION,
        }
        key = self._config.api_key or os.environ.get("ANTHROPIC_API_KEY")
        if key:
            headers["x-api-key"] = key
        return headers

    def _build_payload(
        self,
        prompt: str,
        *,
        schema: dict[str, object] | None = None,
    ) -> dict[str, object]:
        # `system` is a top-level parameter, not a turn, so the messages array is
        # user/assistant only — don't reuse HTTPLLMClient._messages here.
        payload: dict[str, object] = {
            "model": self._config.model_name,
            "messages": [{"role": "user", "content": prompt}],
            "max_tokens": self._config.max_tokens or self._DEFAULT_MAX_TOKENS,
        }
        if self._config.system_prompt is not None:
            payload["system"] = self._config.system_prompt
        if self._config.temperature is not None:
            payload["temperature"] = self._config.temperature
        thinking = self._config.thinking
        if thinking == "adaptive":
            payload["thinking"] = {"type": "adaptive"}
        elif isinstance(thinking, int):
            payload["thinking"] = {"type": "enabled", "budget_tokens": thinking}

        json_schema_mode = (
            schema is not None and self._config.structured_output == "json_schema"
        )

        # `effort` and `format` both live under `output_config`; build it once so
        # they coexist rather than clobbering each other.
        output_config: dict[str, object] = {}
        if self._config.effort is not None:
            output_config["effort"] = self._config.effort
        if json_schema_mode:
            output_config["format"] = {"type": "json_schema", "schema": schema}
        if output_config:
            payload["output_config"] = output_config

        # The `tool_use` fallback: express the schema as a forced tool call — the
        # schema is the tool's input_schema and tool_choice pins the model to it.
        # Works on every Claude model, where `output_config.format` is newer-only.
        if schema is not None and not json_schema_mode:
            payload["tools"] = [
                {
                    "name": self._STRUCTURED_TOOL_NAME,
                    "description": "Return the structured response.",
                    "input_schema": schema,
                }
            ]
            payload["tool_choice"] = {
                "type": "tool",
                "name": self._STRUCTURED_TOOL_NAME,
            }
        return payload

    def _extract_text(self, body: dict[str, object]) -> str:
        """Extract the assistant text — or, for a forced tool call, its JSON input.

        The response ``content`` is a list of blocks; the first ``text`` block is
        returned verbatim (this also covers ``json_schema`` structured output,
        which the API returns as a text block of JSON), while a ``tool_use`` block
        (the forced-tool structured path) has its ``input`` re-serialised to JSON
        so the template's ``generate_structured`` can parse it back.  Any
        ``thinking`` blocks (e.g. from adaptive thinking) are skipped.
        """
        if body.get("stop_reason") == "refusal":
            raise ValueError("Model refused to respond")
        content = body.get("content")
        if not isinstance(content, list) or not content:
            raise ValueError("Anthropic response missing a non-empty 'content' list")
        for block in content:
            if not isinstance(block, dict):
                continue
            if block.get("type") == "text":
                text = block.get("text")
                if not isinstance(text, str):
                    raise ValueError("Anthropic 'text' block content is not a string")
                return text
            if block.get("type") == "tool_use":
                tool_input = block.get("input")
                if not isinstance(tool_input, dict):
                    raise ValueError("Anthropic 'tool_use' 'input' is not an object")
                return json.dumps(tool_input)
        raise ValueError("Anthropic response had no 'text' or 'tool_use' block")
