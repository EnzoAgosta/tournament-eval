"""Circular ranking for language models, plus clean primitives for wiring LLMs together.

Models generate outputs, then a pool of judges rank them — contestants and judges are
independent lists you pass in.  Bias becomes measurable signal instead of a single
grader's hidden assumption.
"""

import importlib
from typing import TYPE_CHECKING

from .llm import GenerationResponse, LLMClient, OpenAICompatibleClient, StructuredResponse
from .models import (
    GenerationFailure,
    GenerationFailureDict,
    GenerationResult,
    GenerationResultDict,
    GenerationTask,
    GenerationTaskDict,
    RankingFailure,
    RankingFailureDict,
    RankingResult,
    RankingResultDict,
    RankingTask,
    RankingTaskDict,
)
from .orchestration import (
    build_ranking_task,
    build_ranking_tasks,
    generate_all,
    generate_one,
    rank_all,
    rank_one,
)
from .persistence import (
    append_generation_failure,
    append_generation_result,
    append_generation_task,
    append_ranking_failure,
    append_ranking_result,
    append_ranking_task,
    read_generation_failure_file,
    read_generation_result_file,
    read_generation_task_file,
    read_ranking_failure_file,
    read_ranking_result_file,
    read_ranking_task_file,
)
from .ranking import (
    DefaultRankingTemplate,
    ParsedRanking,
    RankingTemplate,
)

# Optional, SDK-backed clients — resolved lazily so importing this package never
# requires an extra. `from tournament_eval import BedrockClient` works once the
# matching extra is installed, and otherwise raises a clear install hint.
_OPTIONAL_CLIENTS = {
    "OpenAIClient": ("tournament_eval.llm.openai", "openai"),
    "AnthropicClient": ("tournament_eval.llm.anthropic", "anthropic"),
    "OllamaClient": ("tournament_eval.llm.ollama", "ollama"),
    # Bedrock: the abstract base (for your own families) + the built-in families.
    "BedrockClient": ("tournament_eval.llm.bedrock", "bedrock"),
    "AnthropicBedrockClient": ("tournament_eval.llm.bedrock", "bedrock"),
    "NovaBedrockClient": ("tournament_eval.llm.bedrock", "bedrock"),
    "TitanBedrockClient": ("tournament_eval.llm.bedrock", "bedrock"),
    "LlamaBedrockClient": ("tournament_eval.llm.bedrock", "bedrock"),
    "MistralBedrockClient": ("tournament_eval.llm.bedrock", "bedrock"),
    "CohereBedrockClient": ("tournament_eval.llm.bedrock", "bedrock"),
    "JambaBedrockClient": ("tournament_eval.llm.bedrock", "bedrock"),
    "PalmyraBedrockClient": ("tournament_eval.llm.bedrock", "bedrock"),
    "GptOssBedrockClient": ("tournament_eval.llm.bedrock", "bedrock"),
}

if TYPE_CHECKING:
    # Make the lazy names visible to type checkers and IDEs (re-export aliases).
    from .llm.anthropic import AnthropicClient as AnthropicClient
    from .llm.bedrock import AnthropicBedrockClient as AnthropicBedrockClient
    from .llm.bedrock import BedrockClient as BedrockClient
    from .llm.bedrock import CohereBedrockClient as CohereBedrockClient
    from .llm.bedrock import GptOssBedrockClient as GptOssBedrockClient
    from .llm.bedrock import JambaBedrockClient as JambaBedrockClient
    from .llm.bedrock import LlamaBedrockClient as LlamaBedrockClient
    from .llm.bedrock import MistralBedrockClient as MistralBedrockClient
    from .llm.bedrock import NovaBedrockClient as NovaBedrockClient
    from .llm.bedrock import PalmyraBedrockClient as PalmyraBedrockClient
    from .llm.bedrock import TitanBedrockClient as TitanBedrockClient
    from .llm.ollama import OllamaClient as OllamaClient
    from .llm.openai import OpenAIClient as OpenAIClient


def __getattr__(name: str) -> object:
    target = _OPTIONAL_CLIENTS.get(name)
    if target is None:
        raise AttributeError(f"module {__name__!r} has no attribute {name!r}")
    module_name, extra = target
    try:
        module = importlib.import_module(module_name)
    except ImportError as exc:
        raise ImportError(
            f"{name} requires the '{extra}' extra — install it with: pip install 'tournament-eval[{extra}]'"
        ) from exc
    return getattr(module, name)


# Optional clients are deliberately omitted from __all__: listing them would make
# `from tournament_eval import *` try to import every SDK. They are still importable
# by name (via __getattr__ above) when their extra is installed.
__all__ = [
    "DefaultRankingTemplate",
    "GenerationFailure",
    "GenerationFailureDict",
    "GenerationResponse",
    "GenerationResult",
    "GenerationResultDict",
    "GenerationTask",
    "GenerationTaskDict",
    "LLMClient",
    "OpenAICompatibleClient",
    "ParsedRanking",
    "RankingFailure",
    "RankingFailureDict",
    "RankingResult",
    "RankingResultDict",
    "RankingTask",
    "RankingTaskDict",
    "RankingTemplate",
    "StructuredResponse",
    "append_generation_failure",
    "append_generation_result",
    "append_generation_task",
    "append_ranking_failure",
    "append_ranking_result",
    "append_ranking_task",
    "build_ranking_task",
    "build_ranking_tasks",
    "generate_all",
    "generate_one",
    "rank_all",
    "rank_one",
    "read_generation_failure_file",
    "read_generation_result_file",
    "read_generation_task_file",
    "read_ranking_failure_file",
    "read_ranking_result_file",
    "read_ranking_task_file",
]
