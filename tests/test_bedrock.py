"""Tests for the Bedrock families.

Two layers, matching the two things each family owns:

* the **formatters** (``_build_body`` / ``_extract_text`` / ``_extract_reasoning``)
  are pure, so they're called directly on a family built with a dummy client — no AWS;
* the **transport** (``invoke_model`` + to_thread + StreamingBody + guard) lives on
  the base, so it's tested once end-to-end through a real ``bedrock-runtime`` client
  stubbed with :class:`botocore.stub.Stubber` (the ``bedrock_stub`` fixture).
"""

from collections.abc import Callable
from typing import cast

import pytest
from botocore.stub import Stubber

from tournament_eval.llm.bedrock import (
    AnthropicBedrockClient,
    BedrockRuntimeClient,
    CohereBedrockClient,
    GptOssBedrockClient,
    JambaBedrockClient,
    LlamaBedrockClient,
    MistralBedrockClient,
    NovaBedrockClient,
    PalmyraBedrockClient,
    TitanBedrockClient,
)

# The formatters never touch the client, so a dummy stands in for one, cast to the
# structural BedrockRuntimeClient Protocol to satisfy the type checker.
_NO_CLIENT = cast(BedrockRuntimeClient, object())


def test_anthropic_build_body() -> None:
    client = AnthropicBedrockClient(_NO_CLIENT, model_id="m", system_prompt="sys", max_tokens=10, temperature=0.5)
    body = client._build_body("hi", None)
    assert body["anthropic_version"] == "bedrock-2023-05-31"
    assert body["max_tokens"] == 10
    assert body["temperature"] == 0.5
    assert body["messages"] == [{"role": "user", "content": "hi"}]
    assert body["system"] == "sys"
    assert "thinking" not in body
    assert "output_config" not in body


def test_anthropic_build_body_defaults_max_tokens() -> None:
    client = AnthropicBedrockClient(_NO_CLIENT, model_id="m")
    body = client._build_body("hi", None)
    assert body["max_tokens"] == 4096
    assert "system" not in body


def test_anthropic_omits_temperature_unless_set() -> None:
    client = AnthropicBedrockClient(_NO_CLIENT, model_id="m")
    body = client._build_body("hi", None)
    assert "temperature" not in body


def test_anthropic_native_schema() -> None:
    client = AnthropicBedrockClient(_NO_CLIENT, model_id="m")
    body = client._build_body("hi", {"type": "object"})
    assert body["output_config"] == {"format": {"type": "json_schema", "schema": {"type": "object"}}}


def test_anthropic_reasoning_effort_enables_thinking_and_effort() -> None:
    client = AnthropicBedrockClient(_NO_CLIENT, model_id="m", reasoning_effort="high")
    body = client._build_body("hi", None)
    assert body["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert body["output_config"] == {"effort": "high"}


def test_anthropic_reasoning_effort_with_schema_carries_both() -> None:
    client = AnthropicBedrockClient(_NO_CLIENT, model_id="m", reasoning_effort="max")
    body = client._build_body("hi", {"type": "object"})
    assert body["output_config"] == {
        "format": {"type": "json_schema", "schema": {"type": "object"}},
        "effort": "max",
    }


def test_anthropic_extract_text() -> None:
    client = AnthropicBedrockClient(_NO_CLIENT, model_id="m")
    assert client._extract_text({"content": [{"type": "text", "text": "bonjour"}]}) == "bonjour"


def test_anthropic_extract_refusal() -> None:
    client = AnthropicBedrockClient(_NO_CLIENT, model_id="m")
    with pytest.raises(ValueError, match="refused"):
        client._extract_text({"stop_reason": "refusal", "content": []})


def test_anthropic_extract_no_text_block() -> None:
    client = AnthropicBedrockClient(_NO_CLIENT, model_id="m")
    with pytest.raises(ValueError, match="no text block"):
        client._extract_text({"content": [{"type": "tool_use"}]})


def test_anthropic_extract_reasoning_from_thinking_block() -> None:
    client = AnthropicBedrockClient(_NO_CLIENT, model_id="m")
    response = {"content": [{"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": "bonjour"}]}
    assert client._extract_reasoning(response) == "hmm"


def test_anthropic_extract_reasoning_none_without_thinking_block() -> None:
    client = AnthropicBedrockClient(_NO_CLIENT, model_id="m")
    assert client._extract_reasoning({"content": [{"type": "text", "text": "bonjour"}]}) is None


def test_nova_build_body() -> None:
    client = NovaBedrockClient(_NO_CLIENT, model_id="m", system_prompt="sys", max_tokens=10)
    body = client._build_body("hi", None)
    assert body["messages"] == [{"role": "user", "content": [{"text": "hi"}]}]
    assert body["inferenceConfig"] == {"temperature": 1.0, "maxTokens": 10}
    assert body["system"] == [{"text": "sys"}]


def test_nova_schema_injected_into_prompt() -> None:
    client = NovaBedrockClient(_NO_CLIENT, model_id="m")
    body = client._build_body("hi", {"type": "object"})
    text = body["messages"][0]["content"][0]["text"]  # type: ignore[index]
    assert text.startswith("hi")
    assert "JSON Schema" in text
    assert "system" not in body


def test_nova_extract_text() -> None:
    client = NovaBedrockClient(_NO_CLIENT, model_id="m")
    assert client._extract_text({"output": {"message": {"content": [{"text": "bonjour"}]}}}) == "bonjour"


def test_nova_extract_missing_output() -> None:
    client = NovaBedrockClient(_NO_CLIENT, model_id="m")
    with pytest.raises(ValueError, match="missing 'output'"):
        client._extract_text({})


def test_nova_extract_no_text_block() -> None:
    client = NovaBedrockClient(_NO_CLIENT, model_id="m")
    with pytest.raises(ValueError, match="no text block"):
        client._extract_text({"output": {"message": {"content": [{"foo": 1}]}}})


def test_nova_extract_missing_message() -> None:
    client = NovaBedrockClient(_NO_CLIENT, model_id="m")
    with pytest.raises(ValueError, match=r"missing 'output\.message'"):
        client._extract_text({"output": {}})


def test_titan_build_body() -> None:
    client = TitanBedrockClient(_NO_CLIENT, model_id="m", system_prompt="sys", max_tokens=10)
    body = client._build_body("hi", None)
    assert body["inputText"] == "sys\n\nhi"
    assert body["textGenerationConfig"] == {"temperature": 1.0, "maxTokenCount": 10}


def test_titan_extract_text() -> None:
    client = TitanBedrockClient(_NO_CLIENT, model_id="m")
    assert client._extract_text({"results": [{"outputText": "bonjour"}]}) == "bonjour"


def test_titan_extract_empty_results() -> None:
    client = TitanBedrockClient(_NO_CLIENT, model_id="m")
    with pytest.raises(ValueError, match="non-empty 'results'"):
        client._extract_text({"results": []})


def test_llama_build_body_templates_prompt() -> None:
    client = LlamaBedrockClient(_NO_CLIENT, model_id="m", system_prompt="sys", max_tokens=10)
    body = client._build_body("hi", None)
    prompt = body["prompt"]
    assert isinstance(prompt, str)
    assert prompt.startswith("<|begin_of_text|>")
    assert "system<|end_header_id|>\n\nsys" in prompt
    assert "user<|end_header_id|>\n\nhi" in prompt
    assert body["max_gen_len"] == 10


def test_llama_extract_text() -> None:
    client = LlamaBedrockClient(_NO_CLIENT, model_id="m")
    assert client._extract_text({"generation": "bonjour"}) == "bonjour"


def test_mistral_build_body_inst_wrapping() -> None:
    client = MistralBedrockClient(_NO_CLIENT, model_id="m", system_prompt="sys", max_tokens=10)
    body = client._build_body("hi", None)
    assert body["prompt"] == "<s>[INST] sys\n\nhi [/INST]"
    assert body["max_tokens"] == 10


def test_mistral_extract_text() -> None:
    client = MistralBedrockClient(_NO_CLIENT, model_id="m")
    assert client._extract_text({"outputs": [{"text": "bonjour"}]}) == "bonjour"


def test_cohere_build_body() -> None:
    client = CohereBedrockClient(_NO_CLIENT, model_id="m", system_prompt="sys", max_tokens=10)
    body = client._build_body("hi", None)
    assert body["message"] == "hi"
    assert body["preamble"] == "sys"
    assert body["max_tokens"] == 10


def test_cohere_extract_text() -> None:
    client = CohereBedrockClient(_NO_CLIENT, model_id="m")
    assert client._extract_text({"text": "bonjour"}) == "bonjour"


def test_jamba_build_body() -> None:
    client = JambaBedrockClient(_NO_CLIENT, model_id="m", system_prompt="sys", max_tokens=10)
    body = client._build_body("hi", None)
    assert body["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
    assert body["max_tokens"] == 10


def test_palmyra_uses_max_tokens() -> None:
    client = PalmyraBedrockClient(_NO_CLIENT, model_id="m", max_tokens=10)
    assert client._build_body("hi", None)["max_tokens"] == 10


def test_gptoss_uses_max_completion_tokens() -> None:
    client = GptOssBedrockClient(_NO_CLIENT, model_id="m", max_tokens=10)
    body = client._build_body("hi", None)
    assert "max_tokens" not in body
    assert body["max_completion_tokens"] == 10


def test_openai_chat_schema_injected_into_prompt() -> None:
    client = JambaBedrockClient(_NO_CLIENT, model_id="m")
    body = client._build_body("hi", {"type": "object"})
    assert "JSON Schema" in body["messages"][-1]["content"]  # type: ignore[index]


def test_jamba_extract_text() -> None:
    client = JambaBedrockClient(_NO_CLIENT, model_id="m")
    assert client._extract_text({"choices": [{"message": {"content": "bonjour"}}]}) == "bonjour"


def test_gptoss_strips_reasoning_from_text() -> None:
    client = GptOssBedrockClient(_NO_CLIENT, model_id="m")
    response = {"choices": [{"message": {"content": "<reasoning>hmm</reasoning>\nBonjour"}}]}
    assert client._extract_text(response) == "Bonjour"


def test_gptoss_passes_text_through_without_reasoning() -> None:
    client = GptOssBedrockClient(_NO_CLIENT, model_id="m")
    assert client._extract_text({"choices": [{"message": {"content": "Bonjour"}}]}) == "Bonjour"


def test_gptoss_captures_reasoning() -> None:
    client = GptOssBedrockClient(_NO_CLIENT, model_id="m")
    response = {"choices": [{"message": {"content": "<reasoning>hmm</reasoning>\nBonjour"}}]}
    assert client._extract_reasoning(response) == "hmm"


def test_gptoss_reasoning_none_without_tags() -> None:
    client = GptOssBedrockClient(_NO_CLIENT, model_id="m")
    assert client._extract_reasoning({"choices": [{"message": {"content": "Bonjour"}}]}) is None


def test_name_defaults_to_model_id() -> None:
    assert AnthropicBedrockClient(_NO_CLIENT, model_id="m").name == "m"


def test_name_uses_explicit_name() -> None:
    assert AnthropicBedrockClient(_NO_CLIENT, model_id="m", name="claude-prod").name == "claude-prod"


def test_extract_response_not_an_object() -> None:
    client = JambaBedrockClient(_NO_CLIENT, model_id="m")
    with pytest.raises(ValueError, match="expected an object"):
        client._extract_text("not-a-dict")  # type: ignore[arg-type]


def test_extract_field_holder_not_an_object() -> None:
    client = JambaBedrockClient(_NO_CLIENT, model_id="m")
    with pytest.raises(ValueError, match="expected an object"):
        client._extract_text({"choices": [123]})


def test_extract_field_value_not_a_string() -> None:
    client = JambaBedrockClient(_NO_CLIENT, model_id="m")
    with pytest.raises(ValueError, match="not a string"):
        client._extract_text({"choices": [{"message": {"content": 5}}]})


async def test_transport_generate_round_trip(
    bedrock_stub: Callable[[object], tuple[BedrockRuntimeClient, Stubber]],
) -> None:
    client, stubber = bedrock_stub({"content": [{"type": "text", "text": "bonjour"}]})
    with stubber:
        out = await AnthropicBedrockClient(client, model_id="anthropic.claude").generate("hi")
    assert (out.text, out.reasoning) == ("bonjour", None)
    stubber.assert_no_pending_responses()


async def test_transport_generate_structured_round_trip(
    bedrock_stub: Callable[[object], tuple[BedrockRuntimeClient, Stubber]],
) -> None:
    client, stubber = bedrock_stub({"content": [{"type": "text", "text": '{"x": 1}'}]})
    with stubber:
        response = await AnthropicBedrockClient(client, model_id="m").generate_structured("hi", {"type": "object"})
    assert response.data == {"x": 1}
    stubber.assert_no_pending_responses()


async def test_transport_best_effort_family_round_trip(
    bedrock_stub: Callable[[object], tuple[BedrockRuntimeClient, Stubber]],
) -> None:
    client, stubber = bedrock_stub({"choices": [{"message": {"content": "bonjour"}}]})
    with stubber:
        out = await JambaBedrockClient(client, model_id="ai21.jamba").generate("hi")
    assert out.text == "bonjour"
    stubber.assert_no_pending_responses()


async def test_transport_captures_thinking_trace(
    bedrock_stub: Callable[[object], tuple[BedrockRuntimeClient, Stubber]],
) -> None:
    client, stubber = bedrock_stub(
        {"content": [{"type": "thinking", "thinking": "hmm"}, {"type": "text", "text": "bonjour"}]}
    )
    with stubber:
        out = await AnthropicBedrockClient(client, model_id="m", reasoning_effort="high").generate("hi")
    assert (out.text, out.reasoning) == ("bonjour", "hmm")
    stubber.assert_no_pending_responses()


async def test_transport_non_object_response_body_raises(
    bedrock_stub: Callable[[object], tuple[BedrockRuntimeClient, Stubber]],
) -> None:
    client, stubber = bedrock_stub([1, 2, 3])
    with stubber, pytest.raises(ValueError, match="not a JSON object"):
        await AnthropicBedrockClient(client, model_id="m").generate("hi")
