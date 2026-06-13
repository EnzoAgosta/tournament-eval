"""Tests for the Bedrock families.

Two layers, matching the two things each family owns:

* the **formatters** (``_build_body`` / ``_extract_text``) are pure, so they're
  tested directly with a placeholder client — no AWS;
* the **transport** (``invoke_model`` + to_thread + StreamingBody + guard) lives on
  the base, so it's tested once end-to-end through a real ``bedrock-runtime`` client
  stubbed with :class:`botocore.stub.Stubber`.
"""

from typing import Any

import pytest

from tournament_eval.llm.bedrock import (
    AnthropicBedrockClient,
    BedrockClient,
    CohereBedrockClient,
    GptOssBedrockClient,
    JambaBedrockClient,
    LlamaBedrockClient,
    MistralBedrockClient,
    NovaBedrockClient,
    PalmyraBedrockClient,
    TitanBedrockClient,
)

# Placeholder boto client: the formatters never touch it.
_SENTINEL = object()


def _family(cls: type[BedrockClient], **cfg: Any) -> BedrockClient:
    cfg.setdefault("model_id", "m")
    return cls(_SENTINEL, **cfg)  # type: ignore[arg-type]


class TestAnthropicBedrock:
    def test_build_body(self) -> None:
        body = _family(AnthropicBedrockClient, system_prompt="sys", max_tokens=10, temperature=0.5)._build_body(
            "hi", None
        )
        assert body["anthropic_version"] == "bedrock-2023-05-31"
        assert body["max_tokens"] == 10
        assert body["temperature"] == 0.5
        assert body["messages"] == [{"role": "user", "content": "hi"}]
        assert body["system"] == "sys"
        assert "output_config" not in body

    def test_build_body_defaults_max_tokens_and_native_schema(self) -> None:
        body = _family(AnthropicBedrockClient)._build_body("hi", {"type": "object"})
        assert body["max_tokens"] == 4096
        assert body["output_config"] == {"format": {"type": "json_schema", "schema": {"type": "object"}}}
        assert "system" not in body

    def test_extract_text(self) -> None:
        out = _family(AnthropicBedrockClient)._extract_text({"content": [{"type": "text", "text": "bonjour"}]})
        assert out == "bonjour"

    def test_extract_refusal(self) -> None:
        with pytest.raises(ValueError, match="refused"):
            _family(AnthropicBedrockClient)._extract_text({"stop_reason": "refusal", "content": []})

    def test_extract_no_text_block(self) -> None:
        with pytest.raises(ValueError, match="no text block"):
            _family(AnthropicBedrockClient)._extract_text({"content": [{"type": "tool_use"}]})


class TestNovaBedrock:
    def test_build_body(self) -> None:
        body = _family(NovaBedrockClient, system_prompt="sys", max_tokens=10)._build_body("hi", None)
        assert body["messages"] == [{"role": "user", "content": [{"text": "hi"}]}]
        assert body["inferenceConfig"] == {"temperature": 1.0, "maxTokens": 10}
        assert body["system"] == [{"text": "sys"}]

    def test_build_body_schema_injected_into_prompt(self) -> None:
        body = _family(NovaBedrockClient)._build_body("hi", {"type": "object"})
        text = body["messages"][0]["content"][0]["text"]
        assert text.startswith("hi")
        assert "JSON Schema" in text
        assert "system" not in body

    def test_extract_text(self) -> None:
        out = _family(NovaBedrockClient)._extract_text({"output": {"message": {"content": [{"text": "bonjour"}]}}})
        assert out == "bonjour"

    def test_extract_missing_output(self) -> None:
        with pytest.raises(ValueError, match="missing 'output'"):
            _family(NovaBedrockClient)._extract_text({})

    def test_extract_no_text_block(self) -> None:
        with pytest.raises(ValueError, match="no text block"):
            _family(NovaBedrockClient)._extract_text({"output": {"message": {"content": [{"foo": 1}]}}})


class TestTitanBedrock:
    def test_build_body(self) -> None:
        body = _family(TitanBedrockClient, system_prompt="sys", max_tokens=10)._build_body("hi", None)
        assert body["inputText"] == "sys\n\nhi"
        assert body["textGenerationConfig"] == {"temperature": 1.0, "maxTokenCount": 10}

    def test_extract_text(self) -> None:
        assert _family(TitanBedrockClient)._extract_text({"results": [{"outputText": "bonjour"}]}) == "bonjour"

    def test_extract_empty_results(self) -> None:
        with pytest.raises(ValueError, match="non-empty 'results'"):
            _family(TitanBedrockClient)._extract_text({"results": []})


class TestLlamaBedrock:
    def test_build_body_templates_prompt(self) -> None:
        body = _family(LlamaBedrockClient, system_prompt="sys", max_tokens=10)._build_body("hi", None)
        prompt = body["prompt"]
        assert isinstance(prompt, str)
        assert prompt.startswith("<|begin_of_text|>")
        assert "system<|end_header_id|>\n\nsys" in prompt
        assert "user<|end_header_id|>\n\nhi" in prompt
        assert body["max_gen_len"] == 10

    def test_extract_text(self) -> None:
        assert _family(LlamaBedrockClient)._extract_text({"generation": "bonjour"}) == "bonjour"


class TestMistralBedrock:
    def test_build_body_inst_wrapping(self) -> None:
        body = _family(MistralBedrockClient, system_prompt="sys", max_tokens=10)._build_body("hi", None)
        assert body["prompt"] == "<s>[INST] sys\n\nhi [/INST]"
        assert body["max_tokens"] == 10

    def test_extract_text(self) -> None:
        assert _family(MistralBedrockClient)._extract_text({"outputs": [{"text": "bonjour"}]}) == "bonjour"


class TestNameAndResponseShape:
    def test_name_defaults_to_model_id(self) -> None:
        assert _family(AnthropicBedrockClient).name == "m"

    def test_name_uses_explicit_name(self) -> None:
        assert _family(AnthropicBedrockClient, name="claude-prod").name == "claude-prod"

    def test_response_not_an_object(self) -> None:
        with pytest.raises(ValueError, match="expected an object"):
            _family(JambaBedrockClient)._extract_text("not-a-dict")  # type: ignore[arg-type]

    def test_field_holder_not_an_object(self) -> None:  # choice is not a dict -> _str_field(None, ...)
        with pytest.raises(ValueError, match="expected an object"):
            _family(JambaBedrockClient)._extract_text({"choices": [123]})

    def test_field_value_not_a_string(self) -> None:
        with pytest.raises(ValueError, match="not a string"):
            _family(JambaBedrockClient)._extract_text({"choices": [{"message": {"content": 5}}]})


class TestCohereBedrock:
    def test_build_body(self) -> None:
        body = _family(CohereBedrockClient, system_prompt="sys", max_tokens=10)._build_body("hi", None)
        assert body["message"] == "hi"
        assert body["preamble"] == "sys"
        assert body["max_tokens"] == 10

    def test_extract_text(self) -> None:
        assert _family(CohereBedrockClient)._extract_text({"text": "bonjour"}) == "bonjour"


class TestOpenAIChatFamilies:
    def test_jamba_build_body(self) -> None:
        body = _family(JambaBedrockClient, system_prompt="sys", max_tokens=10)._build_body("hi", None)
        assert body["messages"] == [{"role": "system", "content": "sys"}, {"role": "user", "content": "hi"}]
        assert body["max_tokens"] == 10

    def test_palmyra_uses_max_tokens(self) -> None:
        assert _family(PalmyraBedrockClient, max_tokens=10)._build_body("hi", None)["max_tokens"] == 10

    def test_gptoss_uses_max_completion_tokens(self) -> None:
        body = _family(GptOssBedrockClient, max_tokens=10)._build_body("hi", None)
        assert "max_tokens" not in body
        assert body["max_completion_tokens"] == 10

    def test_schema_injected_into_prompt(self) -> None:
        body = _family(JambaBedrockClient)._build_body("hi", {"type": "object"})
        assert "JSON Schema" in body["messages"][-1]["content"]

    def test_extract_text(self) -> None:
        out = _family(JambaBedrockClient)._extract_text({"choices": [{"message": {"content": "bonjour"}}]})
        assert out == "bonjour"

    def test_gptoss_strips_reasoning(self) -> None:
        body = {"choices": [{"message": {"content": "<reasoning>hmm</reasoning>\nBonjour"}}]}
        assert _family(GptOssBedrockClient)._extract_text(body) == "Bonjour"

    def test_gptoss_passthrough_without_reasoning(self) -> None:
        body = {"choices": [{"message": {"content": "Bonjour"}}]}
        assert _family(GptOssBedrockClient)._extract_text(body) == "Bonjour"


class TestBedrockTransport:
    """End-to-end through a real bedrock-runtime client + Stubber (covers _invoke)."""

    async def test_generate_round_trip(self, bedrock_stub: Any) -> None:
        client, stubber = bedrock_stub({"content": [{"type": "text", "text": "bonjour"}]})
        with stubber:
            out = await AnthropicBedrockClient(client, model_id="anthropic.claude").generate("hi")
        assert out == "bonjour"
        stubber.assert_no_pending_responses()

    async def test_generate_structured_round_trip(self, bedrock_stub: Any) -> None:
        client, stubber = bedrock_stub({"content": [{"type": "text", "text": '{"x": 1}'}]})
        with stubber:
            response = await AnthropicBedrockClient(client, model_id="m").generate_structured("hi", {"type": "object"})
        assert response.data == {"x": 1}

    async def test_best_effort_family_round_trip(self, bedrock_stub: Any) -> None:
        client, stubber = bedrock_stub({"choices": [{"message": {"content": "bonjour"}}]})
        with stubber:
            out = await JambaBedrockClient(client, model_id="ai21.jamba").generate("hi")
        assert out == "bonjour"

    async def test_non_object_response_body_raises(self, bedrock_stub: Any) -> None:
        client, stubber = bedrock_stub([1, 2, 3])  # top-level JSON array
        with stubber, pytest.raises(ValueError, match="not a JSON object"):
            await AnthropicBedrockClient(client, model_id="m").generate("hi")

    async def test_structured_non_object_text_raises(self, bedrock_stub: Any) -> None:
        client, stubber = bedrock_stub({"content": [{"type": "text", "text": "[1, 2]"}]})  # valid text, not an object
        with stubber, pytest.raises(ValueError, match="Expected JSON object"):
            await AnthropicBedrockClient(client, model_id="m").generate_structured("hi", {"type": "object"})
