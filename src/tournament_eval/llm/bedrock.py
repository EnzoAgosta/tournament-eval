"""Amazon Bedrock clients over boto3's Invoke API.  Requires the ``bedrock`` extra.

Bedrock = boto.  Every model family is reached through a boto3
``bedrock-runtime`` client you construct and own (``boto3.client("bedrock-runtime",
region_name=...)``) via ``invoke_model``, so region, credentials and SigV4 signing
are handled entirely at the boto layer.  boto3 is synchronous, so each call runs
in a worker thread via :func:`asyncio.to_thread` (the optional concurrency cap
bounds the threads).

Invoke request/response bodies are **model-specific**, so the abstract
:class:`BedrockClient` owns the transport once and each family is a thin subclass
implementing just two formatters — :meth:`~BedrockClient._build_body` and
:meth:`~BedrockClient._extract_text` (plus :meth:`~BedrockClient._extract_reasoning`
for the rare family that surfaces a trace, e.g. gpt-oss).  Built-in families: Anthropic (Claude),
Nova, Titan, Llama, Mistral, Cohere (Command R/R+), AI21 Jamba, Writer Palmyra,
and OpenAI gpt-oss (the last three share an OpenAI chat-completions body shape).

Structured output: only :class:`AnthropicBedrockClient` constrains it natively
(json_schema ``output_config``).  Every other family has no native mechanism over
Invoke, so :meth:`~BedrockClient.generate_structured` falls back to injecting the
schema into the prompt and parsing the reply — **best-effort, documented flaky**.
Because generation and ranking are separate axes, run such a model as a *contestant*
(generation only) and let a model with reliable structured output do the *ranking*.

The injected client is typed structurally, so this module never imports boto3 —
the ``bedrock`` extra exists to put boto3 on your path to *build* that client.
"""

import abc
import asyncio
from typing import Any, Protocol, Unpack

import orjson

from tournament_eval.llm.base import (
    GenerationConfig,
    GenerationResponse,
    StructuredResponse,
    concurrency_guard,
    resolve_semaphore,
)

# The Anthropic Messages API requires max_tokens; GenerationConfig leaves it optional.
_DEFAULT_MAX_TOKENS = 4096


def _json_instruction(schema: dict[str, object]) -> str:
    """Best-effort structured-output instruction for families without a native one."""
    return (
        "\n\nRespond ONLY with a JSON object that conforms to this JSON Schema, "
        "with no surrounding text or markdown:\n" + orjson.dumps(schema).decode()
    )


def _str_field(obj: object, key: str) -> str:
    """Return ``obj[key]`` as a string, or raise if the shape is wrong."""
    if not isinstance(obj, dict):
        raise ValueError(f"Bedrock response: expected an object for {key!r}")
    value = obj.get(key)
    if not isinstance(value, str):
        raise ValueError(f"Bedrock response: {key!r} is not a string")
    return value


def _list_field(obj: object, key: str) -> list[Any]:
    """Return ``obj[key]`` as a non-empty list, or raise if the shape is wrong."""
    if not isinstance(obj, dict):
        raise ValueError(f"Bedrock response: expected an object for {key!r}")
    value = obj.get(key)
    if not isinstance(value, list) or not value:
        raise ValueError(f"Bedrock response: missing a non-empty {key!r}")
    return value


class BedrockRuntimeClient(Protocol):
    """The slice of a boto3 ``bedrock-runtime`` client this adapter uses."""

    def invoke_model(self, **kwargs: Any) -> dict[str, Any]: ...


class BedrockClient(abc.ABC):
    """Transport base for Bedrock ``invoke_model``; subclass per model family.

    Owns the whole boto round-trip — sign/send via ``invoke_model`` in a worker
    thread, read the streamed body, parse it — so a family only has to say how to
    build its request body and pull text from its response.
    """

    def __init__(
        self,
        client: BedrockRuntimeClient,
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

    @abc.abstractmethod
    def _build_body(self, prompt: str, schema: dict[str, object] | None) -> dict[str, object]:
        """Build the family-specific ``invoke_model`` body.

        ``schema`` is ``None`` for plain generation and the JSON schema for a
        structured call.  Families without native structured output should fold it
        into the prompt via :func:`_json_instruction` (best-effort).
        """
        ...

    @abc.abstractmethod
    def _extract_text(self, response: dict[str, Any]) -> str:
        """Pull the assistant text out of the family-specific response body."""
        ...

    def _extract_reasoning(self, _response: dict[str, Any]) -> str | None:
        """Pull the model's reasoning trace from the response, if it has one.

        Defaults to ``None`` — over the Invoke API most families surface no
        reasoning; a family that does (e.g. gpt-oss) overrides this.
        """
        return None

    async def _invoke(self, body: dict[str, object]) -> dict[str, Any]:
        def _call() -> Any:
            response = self._client.invoke_model(
                modelId=self._model_id,
                body=orjson.dumps(body),
                contentType="application/json",
                accept="application/json",
            )
            return orjson.loads(response["body"].read())

        async with concurrency_guard(self._sem):
            parsed = await asyncio.to_thread(_call)
        if not isinstance(parsed, dict):
            raise ValueError("Bedrock response body is not a JSON object")
        return parsed

    async def generate(self, prompt: str) -> GenerationResponse:
        response = await self._invoke(self._build_body(prompt, None))
        return GenerationResponse(
            text=self._extract_text(response),
            reasoning=self._extract_reasoning(response),
        )

    async def generate_structured(self, prompt: str, schema: dict[str, object]) -> StructuredResponse:
        response = await self._invoke(self._build_body(prompt, schema))
        raw = self._extract_text(response)
        parsed = orjson.loads(raw)
        if not isinstance(parsed, dict):
            raise ValueError(f"Expected JSON object, got {type(parsed).__name__}")
        return StructuredResponse(data=parsed, raw=raw)


class AnthropicBedrockClient(BedrockClient):
    """Claude on Bedrock — the Anthropic Messages body, with native json_schema.

    Structured output uses ``output_config`` (json_schema), so it's reliable on the
    models that support it (the current generation: Opus 4.1 / 4.5 / 4.8, Sonnet
    4.6, Haiku 4.5, Fable); older Claude models on Bedrock aren't supported yet.
    """

    _ANTHROPIC_VERSION = "bedrock-2023-05-31"

    def _build_body(self, prompt: str, schema: dict[str, object] | None) -> dict[str, object]:
        body: dict[str, object] = {
            "anthropic_version": self._ANTHROPIC_VERSION,
            "max_tokens": self._max_tokens if self._max_tokens is not None else _DEFAULT_MAX_TOKENS,
            "temperature": self._temperature,
            "messages": [{"role": "user", "content": prompt}],
        }
        if self._system_prompt is not None:
            body["system"] = self._system_prompt
        if schema is not None:
            body["output_config"] = {"format": {"type": "json_schema", "schema": schema}}
        return body

    def _extract_text(self, response: dict[str, Any]) -> str:
        if response.get("stop_reason") == "refusal":
            raise ValueError("Model refused to respond")
        for block in _list_field(response, "content"):
            if isinstance(block, dict) and block.get("type") == "text":
                return _str_field(block, "text")
        raise ValueError("Bedrock/Claude response had no text block")


class NovaBedrockClient(BedrockClient):
    """Amazon Nova — messages body; structured output is best-effort (prompt-injected)."""

    def _build_body(self, prompt: str, schema: dict[str, object] | None) -> dict[str, object]:
        text = prompt if schema is None else prompt + _json_instruction(schema)
        inference: dict[str, object] = {"temperature": self._temperature}
        if self._max_tokens is not None:
            inference["maxTokens"] = self._max_tokens
        body: dict[str, object] = {
            "messages": [{"role": "user", "content": [{"text": text}]}],
            "inferenceConfig": inference,
        }
        if self._system_prompt is not None:
            body["system"] = [{"text": self._system_prompt}]
        return body

    def _extract_text(self, response: dict[str, Any]) -> str:
        output = response.get("output")
        if not isinstance(output, dict):
            raise ValueError("Nova response missing 'output'")
        for block in _list_field(output.get("message"), "content"):
            if isinstance(block, dict) and "text" in block:
                return _str_field(block, "text")
        raise ValueError("Nova response had no text block")


class TitanBedrockClient(BedrockClient):
    """Amazon Titan Text — inputText body; structured output is best-effort."""

    def _build_body(self, prompt: str, schema: dict[str, object] | None) -> dict[str, object]:
        text = prompt if schema is None else prompt + _json_instruction(schema)
        if self._system_prompt is not None:
            text = f"{self._system_prompt}\n\n{text}"
        config: dict[str, object] = {"temperature": self._temperature}
        if self._max_tokens is not None:
            config["maxTokenCount"] = self._max_tokens
        return {"inputText": text, "textGenerationConfig": config}

    def _extract_text(self, response: dict[str, Any]) -> str:
        return _str_field(_list_field(response, "results")[0], "outputText")


class LlamaBedrockClient(BedrockClient):
    """Meta Llama 3+ Instruct — prompt body with the Llama chat template.

    Structured output is best-effort (prompt-injected).
    """

    def _build_body(self, prompt: str, schema: dict[str, object] | None) -> dict[str, object]:
        text = prompt if schema is None else prompt + _json_instruction(schema)
        body: dict[str, object] = {
            "prompt": self._format(text),
            "temperature": self._temperature,
        }
        if self._max_tokens is not None:
            body["max_gen_len"] = self._max_tokens
        return body

    def _format(self, text: str) -> str:
        parts = ["<|begin_of_text|>"]
        if self._system_prompt is not None:
            parts.append(f"<|start_header_id|>system<|end_header_id|>\n\n{self._system_prompt}<|eot_id|>")
        parts.append(f"<|start_header_id|>user<|end_header_id|>\n\n{text}<|eot_id|>")
        parts.append("<|start_header_id|>assistant<|end_header_id|>\n\n")
        return "".join(parts)

    def _extract_text(self, response: dict[str, Any]) -> str:
        return _str_field(response, "generation")


class MistralBedrockClient(BedrockClient):
    """Mistral AI — text-completion body with ``[INST]`` wrapping (covers 7B/Mixtral).

    Structured output is best-effort (prompt-injected).  Chat-only models
    (e.g. the newest Mistral Large) want a messages body — add a subclass for those.
    """

    def _build_body(self, prompt: str, schema: dict[str, object] | None) -> dict[str, object]:
        text = prompt if schema is None else prompt + _json_instruction(schema)
        if self._system_prompt is not None:
            text = f"{self._system_prompt}\n\n{text}"
        body: dict[str, object] = {
            "prompt": f"<s>[INST] {text} [/INST]",
            "temperature": self._temperature,
        }
        if self._max_tokens is not None:
            body["max_tokens"] = self._max_tokens
        return body

    def _extract_text(self, response: dict[str, Any]) -> str:
        return _str_field(_list_field(response, "outputs")[0], "text")


class CohereBedrockClient(BedrockClient):
    """Cohere Command R / R+ — chat body (``message`` + ``preamble``).

    Structured output is best-effort (prompt-injected).  The older Command
    (``generate``) family uses a different body — add a subclass if you need it.
    """

    def _build_body(self, prompt: str, schema: dict[str, object] | None) -> dict[str, object]:
        message = prompt if schema is None else prompt + _json_instruction(schema)
        body: dict[str, object] = {
            "message": message,
            "temperature": self._temperature,
        }
        if self._system_prompt is not None:
            body["preamble"] = self._system_prompt
        if self._max_tokens is not None:
            body["max_tokens"] = self._max_tokens
        return body

    def _extract_text(self, response: dict[str, Any]) -> str:
        return _str_field(response, "text")


class _OpenAIChatBedrockClient(BedrockClient):
    """Shared base for families using the OpenAI chat-completions body shape.

    ``messages: [{role, content}]`` in, ``choices[0].message.content`` out — used by
    AI21 Jamba, Writer Palmyra, and OpenAI gpt-oss.  Subclasses tweak the max-tokens
    field name (:attr:`_MAX_TOKENS_KEY`) or post-process the text (:meth:`_clean`).
    Structured output is best-effort (prompt-injected) for all of them.
    """

    _MAX_TOKENS_KEY = "max_tokens"

    def _build_body(self, prompt: str, schema: dict[str, object] | None) -> dict[str, object]:
        text = prompt if schema is None else prompt + _json_instruction(schema)
        messages: list[dict[str, str]] = []
        if self._system_prompt is not None:
            messages.append({"role": "system", "content": self._system_prompt})
        messages.append({"role": "user", "content": text})
        body: dict[str, object] = {
            "messages": messages,
            "temperature": self._temperature,
        }
        if self._max_tokens is not None:
            body[self._MAX_TOKENS_KEY] = self._max_tokens
        return body

    def _content(self, response: dict[str, Any]) -> str:
        """The raw ``choices[0].message.content`` string, before any cleaning."""
        choice = _list_field(response, "choices")[0]
        message = choice.get("message") if isinstance(choice, dict) else None
        return _str_field(message, "content")

    def _extract_text(self, response: dict[str, Any]) -> str:
        return self._clean(self._content(response))

    def _clean(self, text: str) -> str:
        """Post-process the raw content; identity by default."""
        return text


class JambaBedrockClient(_OpenAIChatBedrockClient):
    """AI21 Jamba — OpenAI-style messages body."""


class PalmyraBedrockClient(_OpenAIChatBedrockClient):
    """Writer Palmyra (X4/X5) — OpenAI-style messages body."""


class GptOssBedrockClient(_OpenAIChatBedrockClient):
    """OpenAI gpt-oss (20b/120b) — OpenAI-style messages body.

    Two quirks vs the plain chat shape: the token cap is ``max_completion_tokens``,
    and over Invoke the model inlines its reasoning in ``<reasoning>…</reasoning>``
    before the answer.  :meth:`_clean` strips that prefix off the answer, while
    :meth:`_extract_reasoning` keeps the trace from inside the tags.
    """

    _MAX_TOKENS_KEY = "max_completion_tokens"

    def _clean(self, text: str) -> str:
        _, separator, answer = text.partition("</reasoning>")
        return answer.lstrip() if separator else text

    def _extract_reasoning(self, response: dict[str, Any]) -> str | None:
        head, separator, _ = self._content(response).partition("</reasoning>")
        if not separator:
            return None
        reasoning = head.partition("<reasoning>")[2].strip()
        return reasoning or None
