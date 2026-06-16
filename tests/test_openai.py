import asyncio
from unittest import mock

import pytest
from openai import AsyncOpenAI
from openai.types.responses import ResponseOutputRefusal

from tournament_eval.llm.openai import OpenAIClient


def test_minimal_openai_client() -> None:
    client = OpenAIClient(client=AsyncOpenAI(api_key="k"), model_id="m")
    assert client.name == "m"
    assert client._model_id == "m"
    assert client._temperature == 1.0
    assert client._max_tokens is None
    assert client._system_prompt is None
    assert client._name is None
    assert client._reasoning_effort is None
    assert client._sem is None


def test_create_open_client_with_everything() -> None:
    semaphore = asyncio.Semaphore(2)
    client = OpenAIClient(
        client=AsyncOpenAI(api_key="k"),
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
    assert client._sem is not None
    assert client._sem is semaphore


def test_create_open_client_with_max_concurrency_and_no_semaphore() -> None:
    client = OpenAIClient(client=AsyncOpenAI(api_key="k"), model_id="m", max_concurrency=2)
    assert isinstance(client._sem, asyncio.Semaphore)
    assert client._sem._value == 2


def test_open_client_name_defaults_to_model_id() -> None:
    client = OpenAIClient(client=AsyncOpenAI(api_key="k"), model_id="m")
    assert client.name == "m"


def test_open_client_name_can_be_overridden() -> None:
    client = OpenAIClient(client=AsyncOpenAI(api_key="k"), model_id="m", name="name")
    assert client.name == "name"


def test_openai_client_returns_none_for_no_reasoning() -> None:
    client = OpenAIClient(client=AsyncOpenAI(api_key="k"), model_id="m")
    assert client._reasoning is None


def test_openai_client_returns_reasoning() -> None:
    client = OpenAIClient(client=AsyncOpenAI(api_key="k"), model_id="m", reasoning_effort="xhigh")
    assert client._reasoning == {"effort": "xhigh", "summary": "auto"}


def test_openai_client_maps_reasoning_effort() -> None:
    client = OpenAIClient(client=AsyncOpenAI(api_key="k"), model_id="m", reasoning_effort="max")
    assert client._reasoning == {"effort": "xhigh", "summary": "auto"}


async def test_openai_client_generate_returns_full_generation_response() -> None:
    client = OpenAIClient(client=AsyncOpenAI(api_key="k"), model_id="m")

    class FakeResponse:
        summary = "because"

    fake_sdk_response = mock.AsyncMock(output_text="ok", reasoning=FakeResponse())
    with mock.patch.object(client._client.responses, "create", return_value=fake_sdk_response):
        result = await client.generate("test")
    assert result.text == "ok"
    assert result.reasoning == "because"


async def test_openai_client_generate_raises_on_refusal() -> None:
    client = OpenAIClient(client=AsyncOpenAI(api_key="k"), model_id="m")
    refusal = ResponseOutputRefusal(type="refusal", refusal="nope")
    fake_sdk_response = mock.AsyncMock(output=[refusal], output_text="nope", reasoning=None)
    with (
        mock.patch.object(client._client.responses, "create", return_value=fake_sdk_response),
        pytest.raises(ValueError, match="model refused to reply"),
    ):
        await client.generate("test")


async def test_openai_client_generate_structured_sends_schema_and_parses() -> None:
    client = OpenAIClient(client=AsyncOpenAI(api_key="k"), model_id="m")
    fake_sdk_response = mock.AsyncMock(output_text='{"a": 1}', reasoning=None)
    with mock.patch.object(client._client.responses, "create", return_value=fake_sdk_response) as create:
        result = await client.generate_structured("test", {"type": "object"})
    sent = create.call_args.kwargs["text"]
    assert sent["format"]["type"] == "json_schema"
    assert sent["format"]["schema"] == {"type": "object"}
    assert sent["format"]["strict"] is True
    assert (result.data, result.raw) == ({"a": 1}, '{"a": 1}')
