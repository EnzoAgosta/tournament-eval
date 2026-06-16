import asyncio
from unittest import mock

from ollama import AsyncClient

from tournament_eval.llm.ollama import OllamaClient


def test_minimal_ollama_client() -> None:
    client = OllamaClient(client=AsyncClient(), model_id="m")
    assert client.name == "m"
    assert client._model_id == "m"
    assert client._temperature == 1.0
    assert client._max_tokens is None
    assert client._system_prompt is None
    assert client._seed is None
    assert client._name is None
    assert client._reasoning_effort is None
    assert client._sem is None


def test_create_ollama_client_with_everything() -> None:
    semaphore = asyncio.Semaphore(2)
    client = OllamaClient(
        client=AsyncClient(),
        model_id="m",
        temperature=0.5,
        max_tokens=100,
        system_prompt="system",
        seed=7,
        name="name",
        reasoning_effort="high",
        max_concurrency=2,
        semaphore=semaphore,
    )
    assert client.name == "name"
    assert client._model_id == "m"
    assert client._temperature == 0.5
    assert client._max_tokens == 100
    assert client._system_prompt == "system"
    assert client._seed == 7
    assert client._name == "name"
    assert client._reasoning_effort == "high"
    assert client._sem is semaphore


def test_ollama_client_with_max_concurrency_and_no_semaphore() -> None:
    client = OllamaClient(client=AsyncClient(), model_id="m", max_concurrency=2)
    assert isinstance(client._sem, asyncio.Semaphore)
    assert client._sem._value == 2


def test_name_defaults_to_model_id() -> None:
    assert OllamaClient(client=AsyncClient(), model_id="m").name == "m"


def test_name_can_be_overridden() -> None:
    assert OllamaClient(client=AsyncClient(), model_id="m", name="prod").name == "prod"


def test_think_false_without_reasoning_effort() -> None:
    client = OllamaClient(client=AsyncClient(), model_id="m")
    assert client._think is False


def test_think_true_with_reasoning_effort() -> None:
    client = OllamaClient(client=AsyncClient(), model_id="m", reasoning_effort="low")
    assert client._think is True


def test_messages_without_system_prompt() -> None:
    client = OllamaClient(client=AsyncClient(), model_id="m")
    assert client._messages("hi") == [{"role": "user", "content": "hi"}]


def test_messages_with_system_prompt() -> None:
    client = OllamaClient(client=AsyncClient(), model_id="m", system_prompt="sys")
    assert client._messages("hi") == [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]


def test_options_minimal() -> None:
    client = OllamaClient(client=AsyncClient(), model_id="m")
    assert client._options() == {"temperature": 1.0}


def test_options_with_max_tokens_and_seed() -> None:
    client = OllamaClient(client=AsyncClient(), model_id="m", temperature=0.2, max_tokens=100, seed=7)
    assert client._options() == {"temperature": 0.2, "num_predict": 100, "seed": 7}


async def test_generate_returns_text_and_reasoning() -> None:
    client = OllamaClient(client=AsyncClient(), model_id="m")
    response = mock.Mock(message=mock.Mock(content="bonjour", thinking="because"))
    with mock.patch.object(client._client, "chat", mock.AsyncMock(return_value=response)) as chat:
        out = await client.generate("hi")
    assert (out.text, out.reasoning) == ("bonjour", "because")
    kwargs = chat.call_args.kwargs
    assert kwargs["model"] == "m"
    assert kwargs["messages"] == [{"role": "user", "content": "hi"}]
    assert kwargs["options"] == {"temperature": 1.0}
    assert kwargs["think"] is False


async def test_generate_empty_content_becomes_empty_string() -> None:
    client = OllamaClient(client=AsyncClient(), model_id="m")
    response = mock.Mock(message=mock.Mock(content=None, thinking=None))
    with mock.patch.object(client._client, "chat", mock.AsyncMock(return_value=response)):
        out = await client.generate("hi")
    assert out.text == ""
    assert out.reasoning is None


async def test_generate_passes_think_when_reasoning_effort_set() -> None:
    client = OllamaClient(client=AsyncClient(), model_id="m", reasoning_effort="high")
    response = mock.Mock(message=mock.Mock(content="bonjour", thinking="because"))
    with mock.patch.object(client._client, "chat", mock.AsyncMock(return_value=response)) as chat:
        await client.generate("hi")
    assert chat.call_args.kwargs["think"] is True


async def test_generate_structured_sends_schema_and_parses() -> None:
    client = OllamaClient(client=AsyncClient(), model_id="m")
    response = mock.Mock(message=mock.Mock(content='{"a": 1}'))
    with mock.patch.object(client._client, "chat", mock.AsyncMock(return_value=response)) as chat:
        result = await client.generate_structured("hi", {"type": "object"})
    assert result.data == {"a": 1}
    assert result.raw == '{"a": 1}'
    assert chat.call_args.kwargs["format"] == {"type": "object"}
