"""Amazon Bedrock client and config — a thin OpenAI-compatible subclass."""

import dataclasses
import os

from tournament_eval.llm.openai import (
    OpenAICompatibleLLMClient,
    OpenAICompatibleModelConfig,
)


@dataclasses.dataclass(frozen=True)
class BedrockModelConfig(OpenAICompatibleModelConfig):
    """Configuration for Amazon Bedrock via its OpenAI-compatible endpoint.

    Bedrock fronts *every* model it hosts — Claude, GPT, and open-weight — behind
    the OpenAI ``/chat/completions`` protocol, so :class:`BedrockLLMClient` is just
    a thin :class:`OpenAICompatibleLLMClient` pointed at the ``bedrock-mantle``
    endpoint with a Bedrock API key as the bearer token.  This is why a Claude
    model on Bedrock uses *this* config, not ``AnthropicModelConfig`` — the
    wire protocol is OpenAI's, not Anthropic's native Messages API.
    """

    region: str = "us-east-1"
    """AWS region, used to build the default ``bedrock-mantle`` base URL
    (``https://bedrock-mantle.{region}.api.aws/v1``).  Ignored when ``base_url`` is
    set explicitly — e.g. to target ``bedrock-runtime`` or a self-hosted gateway."""
    api_key: str | None = None
    """Amazon Bedrock API key, sent as an ``Authorization: Bearer`` token.  Falls
    back to the ``AWS_BEARER_TOKEN_BEDROCK`` environment variable if ``None``.

    SigV4 (raw AWS credentials) is **not** supported yet — only the long- or
    short-lived bearer API key.  SigV4 can be added later as a config-selected auth
    mode on this client without disturbing the shared HTTP template."""


class BedrockLLMClient(OpenAICompatibleLLMClient[BedrockModelConfig]):
    """Amazon Bedrock, via its OpenAI-compatible ``/chat/completions`` endpoint.

    A *thin* subclass of :class:`OpenAICompatibleLLMClient`: Bedrock speaks the
    OpenAI protocol for every model it hosts, so all the payload/extraction logic
    is inherited.  Only two things differ, both small:

    * **Endpoint** — defaults to the ``bedrock-mantle`` URL built from
      :attr:`BedrockModelConfig.region`; an explicit ``base_url`` still wins (to
      target ``bedrock-runtime`` or a gateway).
    * **Auth** — the bearer token falls back to the ``AWS_BEARER_TOKEN_BEDROCK``
      environment variable rather than ``OPENAI_API_KEY``.

    Because the protocol is OpenAI's, **Claude on Bedrock uses this client, not**
    ``AnthropicLLMClient`` — the latter targets the first-party Messages API.

    Caveat: structured output is inherited as ``response_format`` with a strict
    ``json_schema``.  Whether Bedrock *enforces* that for every model (notably
    Claude) is not yet verified end-to-end — if a model ignores it, output falls
    back to the prompt-restated schema, same as any OpenAI-compatible endpoint.

    Only the bearer-token API key is supported today; SigV4 can be added later as a
    config-selected auth mode (via an ``httpx`` auth object, which sees the request
    body it needs to sign) without changing the shared template.
    """

    @property
    def _endpoint_url(self) -> str:
        base = self._config.base_url or (
            f"https://bedrock-mantle.{self._config.region}.api.aws/v1"
        )
        return f"{base.rstrip('/')}/chat/completions"

    @property
    def _headers(self) -> dict[str, str]:
        headers = {"Content-Type": "application/json"}
        key = self._config.api_key or os.environ.get("AWS_BEARER_TOKEN_BEDROCK")
        if key:
            headers["Authorization"] = f"Bearer {key}"
        return headers
