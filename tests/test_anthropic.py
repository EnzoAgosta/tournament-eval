import asyncio
from unittest import mock

import pytest
from anthropic import AsyncAnthropic, omit
from anthropic.types import TextBlock, ThinkingBlock

from tournament_eval.llm.anthropic import AnthropicClient


def test_minimal_anthropic_client() -> None:
    client = AnthropicClient(client=AsyncAnthropic(api_key="k"), model_id="m")
    assert client.name == "m"
    assert client._model_id == "m"
    assert client._temperature is omit
    assert client._max_tokens == 4096
    assert client._system_prompt is None
    assert client._name is None
    assert client._reasoning_effort is None
    assert client._sem is None


def test_create_anthropic_client_with_everything() -> None:
    semaphore = asyncio.Semaphore(2)
    client = AnthropicClient(
        client=AsyncAnthropic(api_key="k"),
        model_id="m",
        temperature=0.5,
        max_tokens=100,
        system_prompt="system",
        name="name",
        reasoning_effort="xhigh",
        max_concurrency=2,
        semaphore=semaphore,
    )
    assert client.name == "name"
    assert client._model_id == "m"
    assert client._temperature == 0.5
    assert client._max_tokens == 100
    assert client._system_prompt == "system"
    assert client._name == "name"
    assert client._reasoning_effort == "xhigh"
    assert client._sem is semaphore


def test_anthropic_client_with_max_concurrency_and_no_semaphore() -> None:
    client = AnthropicClient(client=AsyncAnthropic(api_key="k"), model_id="m", max_concurrency=2)
    assert isinstance(client._sem, asyncio.Semaphore)
    assert client._sem._value == 2


def test_name_defaults_to_model_id() -> None:
    assert AnthropicClient(client=AsyncAnthropic(api_key="k"), model_id="m").name == "m"


def test_name_can_be_overridden() -> None:
    assert AnthropicClient(client=AsyncAnthropic(api_key="k"), model_id="m", name="prod").name == "prod"


def test_system_omitted_when_unset() -> None:
    client = AnthropicClient(client=AsyncAnthropic(api_key="k"), model_id="m")
    assert client._system is omit


def test_system_passed_when_set() -> None:
    client = AnthropicClient(client=AsyncAnthropic(api_key="k"), model_id="m", system_prompt="sys")
    assert client._system == "sys"


def test_thinking_omitted_without_reasoning_effort() -> None:
    client = AnthropicClient(client=AsyncAnthropic(api_key="k"), model_id="m")
    assert client._thinking is omit


def test_thinking_enabled_with_reasoning_effort() -> None:
    client = AnthropicClient(client=AsyncAnthropic(api_key="k"), model_id="m", reasoning_effort="high")
    assert client._thinking == {"type": "adaptive", "display": "summarized"}


def test_output_config_omitted_without_schema_or_effort() -> None:
    client = AnthropicClient(client=AsyncAnthropic(api_key="k"), model_id="m")
    assert client._output_config(None) is omit


def test_output_config_with_schema_only() -> None:
    client = AnthropicClient(client=AsyncAnthropic(api_key="k"), model_id="m")
    config = client._output_config({"type": "object"})
    assert config == {"format": {"type": "json_schema", "schema": {"type": "object"}}}


def test_output_config_with_effort_only() -> None:
    client = AnthropicClient(client=AsyncAnthropic(api_key="k"), model_id="m", reasoning_effort="high")
    assert client._output_config(None) == {"effort": "high"}


def test_output_config_with_schema_and_effort() -> None:
    client = AnthropicClient(client=AsyncAnthropic(api_key="k"), model_id="m", reasoning_effort="max")
    assert client._output_config({"type": "object"}) == {
        "format": {"type": "json_schema", "schema": {"type": "object"}},
        "effort": "max",
    }


def test_extract_text_returns_text_block() -> None:
    client = AnthropicClient(client=AsyncAnthropic(api_key="k"), model_id="m")
    message = mock.Mock(stop_reason="end_turn", content=[TextBlock(type="text", text="bonjour")])
    assert client._extract_text(message) == "bonjour"


def test_extract_text_raises_on_refusal() -> None:
    client = AnthropicClient(client=AsyncAnthropic(api_key="k"), model_id="m")
    message = mock.Mock(stop_reason="refusal", content=[])
    with pytest.raises(ValueError, match="refused"):
        client._extract_text(message)


def test_extract_text_raises_without_text_block() -> None:
    client = AnthropicClient(client=AsyncAnthropic(api_key="k"), model_id="m")
    message = mock.Mock(stop_reason="end_turn", content=[ThinkingBlock(type="thinking", thinking="hmm", signature="s")])
    with pytest.raises(ValueError, match="no text block"):
        client._extract_text(message)


def test_extract_reasoning_returns_thinking_block() -> None:
    client = AnthropicClient(client=AsyncAnthropic(api_key="k"), model_id="m")
    message = mock.Mock(
        content=[ThinkingBlock(type="thinking", thinking="hmm", signature="s"), TextBlock(type="text", text="bonjour")]
    )
    assert client._extract_reasoning(message) == "hmm"


def test_extract_reasoning_none_without_thinking_block() -> None:
    client = AnthropicClient(client=AsyncAnthropic(api_key="k"), model_id="m")
    message = mock.Mock(content=[TextBlock(type="text", text="bonjour")])
    assert client._extract_reasoning(message) is None


async def test_generate_returns_text_and_reasoning() -> None:
    client = AnthropicClient(client=AsyncAnthropic(api_key="k"), model_id="m")
    message = mock.Mock(
        stop_reason="end_turn",
        content=[ThinkingBlock(type="thinking", thinking="hmm", signature="s"), TextBlock(type="text", text="bonjour")],
    )
    with mock.patch.object(client._client.messages, "create", mock.AsyncMock(return_value=message)) as create:
        out = await client.generate("hi")
    assert (out.text, out.reasoning) == ("bonjour", "hmm")
    kwargs = create.call_args.kwargs
    assert kwargs["model"] == "m"
    assert kwargs["max_tokens"] == 4096
    assert kwargs["messages"] == [{"role": "user", "content": "hi"}]
    assert kwargs["temperature"] is omit
    assert kwargs["system"] is omit
    assert kwargs["thinking"] is omit
    assert kwargs["output_config"] is omit


async def test_generate_passes_reasoning_and_system_config() -> None:
    client = AnthropicClient(
        client=AsyncAnthropic(api_key="k"),
        model_id="m",
        system_prompt="sys",
        temperature=0.5,
        reasoning_effort="high",
    )
    message = mock.Mock(stop_reason="end_turn", content=[TextBlock(type="text", text="bonjour")])
    with mock.patch.object(client._client.messages, "create", mock.AsyncMock(return_value=message)) as create:
        await client.generate("hi")
    kwargs = create.call_args.kwargs
    assert kwargs["temperature"] == 0.5
    assert kwargs["system"] == "sys"
    assert kwargs["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert kwargs["output_config"] == {"effort": "high"}


async def test_generate_structured_sends_schema_and_parses() -> None:
    client = AnthropicClient(client=AsyncAnthropic(api_key="k"), model_id="m")
    message = mock.Mock(stop_reason="end_turn", content=[TextBlock(type="text", text='{"a": 1}')])
    with mock.patch.object(client._client.messages, "create", mock.AsyncMock(return_value=message)) as create:
        result = await client.generate_structured("hi", {"type": "object"})
    assert result.data == {"a": 1}
    assert result.raw == '{"a": 1}'
    assert create.call_args.kwargs["output_config"] == {"format": {"type": "json_schema", "schema": {"type": "object"}}}
