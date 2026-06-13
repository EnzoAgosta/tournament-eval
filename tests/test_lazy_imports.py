"""Tests for lazy optional-client resolution at the package top level."""

import importlib

import pytest

import tournament_eval

_OPTIONAL_NAMES = [
    "OpenAIClient",
    "AnthropicClient",
    "OllamaClient",
    "BedrockClient",
    "AnthropicBedrockClient",
    "NovaBedrockClient",
    "TitanBedrockClient",
    "LlamaBedrockClient",
    "MistralBedrockClient",
    "CohereBedrockClient",
    "JambaBedrockClient",
    "PalmyraBedrockClient",
    "GptOssBedrockClient",
]


class TestLazyOptionalClients:
    def test_resolves_to_the_submodule_class(self) -> None:
        from tournament_eval import AnthropicBedrockClient
        from tournament_eval.llm.bedrock import AnthropicBedrockClient as Direct

        assert AnthropicBedrockClient is Direct

    @pytest.mark.parametrize("name", _OPTIONAL_NAMES)
    def test_all_optional_names_resolve_to_classes(self, name: str) -> None:
        assert isinstance(getattr(tournament_eval, name), type)

    def test_unknown_attribute_raises(self) -> None:
        missing = "Nope"  # via a variable so it isn't flagged as constant getattr
        with pytest.raises(AttributeError, match="no attribute 'Nope'"):
            getattr(tournament_eval, missing)

    def test_missing_extra_gives_install_hint(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def boom(_name: str) -> object:
            raise ImportError("simulated missing dependency")

        monkeypatch.setattr(importlib, "import_module", boom)
        with pytest.raises(ImportError, match=r"requires the 'bedrock' extra"):
            tournament_eval.__getattr__("BedrockClient")
