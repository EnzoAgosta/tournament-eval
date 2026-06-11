"""Circular Tournament Evaluation Framework.

A framework for evaluating language models via circular ranking:
every model generates outputs, then every model judges all outputs.
"""

from .llm import (
    LLMClient,
    ModelConfig,
    OllamaLLMClient,
    OllamaModelConfig,
    OpenAILLMClient,
    OpenAIModelConfig,
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
    rank_all,
)

__all__ = [
    "GenerationResult",
    "GenerationTask",
    "LLMClient",
    "LetterGenerator",
    "ModelConfig",
    "OllamaLLMClient",
    "OllamaModelConfig",
    "OpenAILLMClient",
    "OpenAIModelConfig",
    "RankingResult",
    "RankingTask",
    "StructuredResponse",
    "build_ranking_tasks",
    "generate_all",
    "rank_all",
]
