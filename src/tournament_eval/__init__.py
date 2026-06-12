"""Circular Tournament Evaluation Framework.

A framework for evaluating language models via circular ranking:
every model generates outputs, then every model judges all outputs.
"""

from .llm import (
    HTTPLLMClient,
    HTTPModelConfig,
    LLMClient,
    ModelConfig,
    OllamaLLMClient,
    OllamaModelConfig,
    OpenAICompatibleLLMClient,
    OpenAICompatibleModelConfig,
    StructuredResponse,
)
from .models import (
    GenerationResult,
    GenerationTask,
    LetterGenerator,
    RankingResult,
    RankingTask,
)
from .orchestration import (
    build_ranking_tasks,
    generate_all,
    generate_one,
    rank_all,
    rank_one,
)

__all__ = [
    "GenerationResult",
    "GenerationTask",
    "HTTPLLMClient",
    "HTTPModelConfig",
    "LLMClient",
    "LetterGenerator",
    "ModelConfig",
    "OllamaLLMClient",
    "OllamaModelConfig",
    "OpenAICompatibleLLMClient",
    "OpenAICompatibleModelConfig",
    "RankingResult",
    "RankingTask",
    "StructuredResponse",
    "build_ranking_tasks",
    "generate_all",
    "generate_one",
    "rank_all",
    "rank_one",
]
