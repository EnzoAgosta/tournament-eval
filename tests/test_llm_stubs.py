"""Tests for intentionally unimplemented LLM client stubs."""

import pytest

from tournament_eval.llm import (
    AnthropicLLMClient,
    AnthropicModelConfig,
    OpenAILLMClient,
    OpenAIModelConfig,
)


class TestOpenAILLMClient:
    async def test_generate_raises_not_implemented(self) -> None:
        client = OpenAILLMClient(OpenAIModelConfig(model_name="gpt-4o"))
        with pytest.raises(NotImplementedError):
            await client.generate("test")

    async def test_generate_structured_raises_not_implemented(self) -> None:
        client = OpenAILLMClient(OpenAIModelConfig(model_name="gpt-4o"))
        with pytest.raises(NotImplementedError):
            await client.generate_structured("test", {})


class TestAnthropicLLMClient:
    async def test_generate_raises_not_implemented(self) -> None:
        client = AnthropicLLMClient(AnthropicModelConfig(model_name="claude"))
        with pytest.raises(NotImplementedError):
            await client.generate("test")

    async def test_generate_structured_raises_not_implemented(self) -> None:
        client = AnthropicLLMClient(AnthropicModelConfig(model_name="claude"))
        with pytest.raises(NotImplementedError):
            await client.generate_structured("test", {})
