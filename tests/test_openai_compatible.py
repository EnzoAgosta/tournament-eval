import asyncio

import httpx
import pytest

from tests.conftest import FakeTransport
from tournament_eval.llm.base import GenerationResponse, StructuredResponse
from tournament_eval.llm.openai_compatible import OpenAICompatibleClient


def test_create_minimal_openai_compatible_client() -> None:
    client = OpenAICompatibleClient(model_id="m", base_url="http://...", api_key="k")
    assert client.name == "m"
    assert client._endpoint_url == "http://.../chat/completions"
    assert client._model_id == "m"
    assert client._base_url == "http://..."
    assert client._api_key == "k"
    assert client._temperature == 1.0
    assert client._max_tokens is None
    assert client._system_prompt is None
    assert client._retry_count == 3
    assert client._name is None
    assert client._timeout == 300.0
    assert client._seed is None
    assert client._reasoning_effort is None
    assert client._transport is None
    assert client._http_client is None
    assert client._sem is None


def test_create_openai_compatible_client_with_everything() -> None:
    semaphore = asyncio.Semaphore(2)
    client = OpenAICompatibleClient(
        model_id="m",
        base_url="http://...",
        api_key="k",
        temperature=0.5,
        max_tokens=100,
        system_prompt="system",
        retry_count=5,
        name="name",
        timeout=60.0,
        seed=123,
        reasoning_effort="xhigh",
        max_concurrency=2,
        semaphore=semaphore,
    )
    assert client.name == "name"
    assert client._endpoint_url == "http://.../chat/completions"
    assert client._model_id == "m"
    assert client._base_url == "http://..."
    assert client._api_key == "k"
    assert client._temperature == 0.5
    assert client._max_tokens == 100
    assert client._system_prompt == "system"
    assert client._retry_count == 5
    assert client._name == "name"
    assert client._timeout == 60.0
    assert client._seed == 123
    assert client._reasoning_effort == "xhigh"
    assert client._transport is None
    assert client._http_client is None
    assert client._sem is not None
    assert client._sem is semaphore


def test_create_openai_compatible_client_with_max_concurrency_and_no_semaphore() -> None:
    client = OpenAICompatibleClient(model_id="m", base_url="http://...", api_key="k", max_concurrency=2)
    assert isinstance(client._sem, asyncio.Semaphore)
    assert client._sem._value == 2


def test_openai_compatible_client_name_defaults_to_model_id() -> None:
    client = OpenAICompatibleClient(model_id="m", base_url="http://...", api_key="k")
    assert client.name == "m"


def test_openai_compatible_client_name_can_be_overridden() -> None:
    client = OpenAICompatibleClient(model_id="m", base_url="http://...", api_key="k", name="name")
    assert client.name == "name"


def test_openai_compatible_client_appends_endpoint_to_base_url() -> None:
    client = OpenAICompatibleClient(model_id="m", base_url="http://...", api_key="k")
    assert client._endpoint_url == "http://.../chat/completions"


def test_openai_compatible_client_endpoint_url_strips_trailing_slash() -> None:
    client = OpenAICompatibleClient(model_id="m", base_url="http://.../", api_key="k")
    assert client._endpoint_url == "http://.../chat/completions"


def test_openai_compatible_client_build_headers_with_api_key() -> None:
    client = OpenAICompatibleClient(model_id="m", base_url="http://...", api_key="k")
    assert client._build_headers() == {"Content-Type": "application/json", "Authorization": "Bearer k"}


def test_openai_compatible_client_build_headers_without_api_key() -> None:
    client = OpenAICompatibleClient(model_id="m", base_url="http://...")
    assert client._build_headers() == {"Content-Type": "application/json"}


def test_openai_compatible_client_build_messages_does_not_add_system_prompt_when_none() -> None:
    client = OpenAICompatibleClient(model_id="m", base_url="http://...", api_key="k")
    assert client._build_messages("hello") == [
        {"role": "user", "content": "hello"},
    ]


def test_openai_compatible_client_build_messages_adds_system_prompt_when_present() -> None:
    client = OpenAICompatibleClient(model_id="m", base_url="http://...", api_key="k", system_prompt="system")
    assert client._build_messages("hello") == [
        {"role": "system", "content": "system"},
        {"role": "user", "content": "hello"},
    ]


def test_openai_compatible_client_build_payload_minimal() -> None:
    client = OpenAICompatibleClient(model_id="m", base_url="http://...", api_key="k")
    assert client._build_payload("hello") == {
        "model": "m",
        "messages": [
            {"role": "user", "content": "hello"},
        ],
        "stream": False,
        "temperature": 1.0,
    }


def test_openai_compatible_client_build_payload_with_max_tokens() -> None:
    client = OpenAICompatibleClient(model_id="m", base_url="http://...", api_key="k", max_tokens=100)
    assert client._build_payload("hello") == {
        "model": "m",
        "messages": [
            {"role": "user", "content": "hello"},
        ],
        "stream": False,
        "temperature": 1.0,
        "max_tokens": 100,
    }


def test_openai_compatible_client_build_payload_with_seed() -> None:
    client = OpenAICompatibleClient(model_id="m", base_url="http://...", api_key="k", seed=123)
    assert client._build_payload("hello") == {
        "model": "m",
        "messages": [
            {"role": "user", "content": "hello"},
        ],
        "stream": False,
        "temperature": 1.0,
        "seed": 123,
    }


def test_openai_compatible_client_build_payload_with_reasoning_effort() -> None:
    client = OpenAICompatibleClient(model_id="m", base_url="http://...", api_key="k", reasoning_effort="xhigh")
    assert client._build_payload("hello") == {
        "model": "m",
        "messages": [
            {"role": "user", "content": "hello"},
        ],
        "stream": False,
        "temperature": 1.0,
        "reasoning_effort": "high",
    }


def test_openai_compatible_client_build_payload_with_schema() -> None:
    client = OpenAICompatibleClient(model_id="m", base_url="http://...", api_key="k")
    assert client._build_payload("hello", schema={"a": int}) == {
        "model": "m",
        "messages": [
            {"role": "user", "content": "hello"},
        ],
        "stream": False,
        "temperature": 1.0,
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "structured_response",
                "schema": {"a": int},
                "strict": True,
            },
        },
    }


async def test_openai_compatible_client_make_request_fails_without_http_client() -> None:
    client = OpenAICompatibleClient(model_id="m", base_url="http://...", api_key="k")
    with pytest.raises(RuntimeError, match="must be used as an async context manager"):
        await client._make_request({}, {})


async def test_openai_compatible_client_make_request_sends_request_and_returns_response() -> None:
    client = OpenAICompatibleClient(model_id="m", base_url="http://...", api_key="k")
    client._transport = FakeTransport()
    async with client:
        response = await client._make_request({}, {})
    assert response == {"choices": [{"message": {"content": "ok"}}]}


async def test_openai_compatible_client_make_request_retries_on_429_until_success() -> None:
    client = OpenAICompatibleClient(model_id="m", base_url="http://...", api_key="k")
    client._transport = FakeTransport()
    client._transport.fail_amount = 1
    async with client:
        response = await client._make_request({}, {})
    assert response == {"choices": [{"message": {"content": "ok"}}]}
    assert client._transport.call_count == 2


async def test_openai_compatible_client_make_request_retries_on_429_until_failure_if_max_attempts_reached() -> None:
    client = OpenAICompatibleClient(model_id="m", base_url="http://...", api_key="k")
    client._transport = FakeTransport()
    client._transport.fail_amount = 5
    async with client:
        with pytest.raises(httpx.HTTPStatusError, match="429 Too Many Requests"):
            await client._make_request({}, {})
    assert client._transport.call_count == 3


async def test_openai_compatible_client_make_request_retries_on_5xx_until_success() -> None:
    client = OpenAICompatibleClient(model_id="m", base_url="http://...", api_key="k")
    client._transport = FakeTransport()
    client._transport.error_code = 521
    client._transport.fail_amount = 1
    async with client:
        response = await client._make_request({}, {})
    assert response == {"choices": [{"message": {"content": "ok"}}]}
    assert client._transport.call_count == 2


async def test_openai_compatible_client_make_request_retries_on_5xx_until_failure_if_max_attempts_reached() -> None:
    client = OpenAICompatibleClient(model_id="m", base_url="http://...", api_key="k")
    client._transport = FakeTransport()
    client._transport.error_code = 521
    client._transport.fail_amount = 5
    async with client:
        with pytest.raises(httpx.HTTPStatusError, match=str(client._transport.error_code)):
            await client._make_request({}, {})
    assert client._transport.call_count == 3


async def test_openai_compatible_client_make_request_retries_on_transport_errors_until_success() -> None:
    client = OpenAICompatibleClient(model_id="m", base_url="http://...", api_key="k")
    client._transport = FakeTransport()
    client._transport.fail_amount = 1
    client._transport.raise_error = True
    async with client:
        response = await client._make_request({}, {})
    assert response == {"choices": [{"message": {"content": "ok"}}]}
    assert client._transport.call_count == 2


async def test_openai_compatible_client_make_request_retries_on_transport_errors_until_failure_if_max_attempts_reached() -> (  # noqa: E501
    None
):
    client = OpenAICompatibleClient(model_id="m", base_url="http://...", api_key="k")
    client._transport = FakeTransport()
    client._transport.fail_amount = 5
    client._transport.raise_error = True
    async with client:
        with pytest.raises(httpx.TransportError, match="boom"):
            await client._make_request({}, {})
    assert client._transport.call_count == 3


async def test_openai_compatible_client_make_request_even_on_impossible_retries() -> None:
    client = OpenAICompatibleClient(model_id="m", base_url="http://...", api_key="k")
    client._transport = FakeTransport()
    client._retry_count = -1
    async with client:
        await client._make_request({}, {})
    assert client._transport.call_count == 1


def test_openai_compatible_client_message_fails_on_missing_choices() -> None:
    client = OpenAICompatibleClient(model_id="m", base_url="http://...", api_key="k")
    with pytest.raises(ValueError, match="missing a 'choices' list"):
        client._message({})


def test_openai_compatible_client_message_fails_on_non_list_choices() -> None:
    client = OpenAICompatibleClient(model_id="m", base_url="http://...", api_key="k")
    with pytest.raises(ValueError, match="is not a list"):
        client._message({"choices": 1})


def test_openai_compatible_client_message_fails_on_empty_choices() -> None:
    client = OpenAICompatibleClient(model_id="m", base_url="http://...", api_key="k")
    with pytest.raises(ValueError, match="is empty"):
        client._message({"choices": []})


def test_openai_compatible_client_message_fails_on_first_choice_not_dict() -> None:
    client = OpenAICompatibleClient(model_id="m", base_url="http://...", api_key="k")
    with pytest.raises(ValueError, match="not an object"):
        client._message({"choices": [1]})


def test_openai_compatible_client_message_fails_on_missing_message() -> None:
    client = OpenAICompatibleClient(model_id="m", base_url="http://...", api_key="k")
    with pytest.raises(ValueError, match="missing 'message'"):
        client._message({"choices": [{"foo": "bar"}]})


def test_openai_compatible_client_message_fails_on_non_dict_message() -> None:
    client = OpenAICompatibleClient(model_id="m", base_url="http://...", api_key="k")
    with pytest.raises(ValueError, match="is not an object"):
        client._message({"choices": [{"message": 1}]})


def test_openai_compatible_client_message_returns_message() -> None:
    client = OpenAICompatibleClient(model_id="m", base_url="http://...", api_key="k")
    assert client._message({"choices": [{"message": {"content": "ok"}}]}) == {"content": "ok"}


def test_openai_compatible_client_extract_response_fails_on_refusal() -> None:
    client = OpenAICompatibleClient(model_id="m", base_url="http://...", api_key="k")
    with pytest.raises(ValueError, match="refused to respond:"):
        client._extract_response({"refusal": "nah"})


def test_openai_compatible_client_extract_response_fails_on_missing_content() -> None:
    client = OpenAICompatibleClient(model_id="m", base_url="http://...", api_key="k")
    with pytest.raises(ValueError, match="content"):
        client._extract_response({"foo": "bar"})


def test_openai_compatible_client_extract_response_fails_on_non_string_content() -> None:
    client = OpenAICompatibleClient(model_id="m", base_url="http://...", api_key="k")
    with pytest.raises(ValueError, match="content"):
        client._extract_response({"content": 1})


def test_openai_compatible_client_extract_response_returns_content() -> None:
    client = OpenAICompatibleClient(model_id="m", base_url="http://...", api_key="k")
    assert client._extract_response({"content": "ok"}) == "ok"


def test_openai_compatible_client_extract_reasoning_returns_none_on_missing_reasoning() -> None:
    client = OpenAICompatibleClient(model_id="m", base_url="http://...", api_key="k")
    assert client._extract_reasoning({"foo": "bar"}) is None


def test_openai_compatible_client_extract_reasoning_returns_reasoning() -> None:
    client = OpenAICompatibleClient(model_id="m", base_url="http://...", api_key="k")
    assert client._extract_reasoning({"reasoning_content": "ok"}) == "ok"


async def test_generate_returns_generation_response() -> None:
    client = OpenAICompatibleClient(model_id="m", base_url="http://...", api_key="k")
    client._transport = FakeTransport()
    client._transport.return_body = {"choices": [{"message": {"content": "ok", "reasoning_content": "because"}}]}
    async with client:
        response = await client.generate("hello")
    assert isinstance(response, GenerationResponse)
    assert response.text == "ok"
    assert response.reasoning == "because"


async def test_generate_structured_returns_structured_response() -> None:
    client = OpenAICompatibleClient(model_id="m", base_url="http://...", api_key="k")
    client._transport = FakeTransport()
    client._transport.return_body = {"choices": [{"message": {"content": '{"a": 1}', "reasoning_content": "because"}}]}
    async with client:
        response = await client.generate_structured("hello", {"type": "object"})
    assert isinstance(response, StructuredResponse)
    assert response.data == {"a": 1}
    assert response.raw == '{"a": 1}'


async def test_generate_structured_fails_on_invalid_json() -> None:
    client = OpenAICompatibleClient(model_id="m", base_url="http://...", api_key="k")
    client._transport = FakeTransport()
    client._transport.return_body = {"choices": [{"message": {"content": "not json", "reasoning_content": "because"}}]}
    async with client:
        with pytest.raises(ValueError, match="invalid literal"):
            await client.generate_structured("hello", {"type": "object"})
