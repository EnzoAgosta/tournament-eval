"""Circular Tournament Evaluation Framework.

A framework for evaluating language models via circular ranking:
every model generates outputs, then every model judges all outputs.
"""

from tournament_eval.llm import (
    LLMClient,
    ModelConfig,
    OllamaLLMClient,
    OllamaModelConfig,
    OpenAIModelConfig,
    AnthropicModelConfig,
    StructuredResponse,
)
from tournament_eval.models import (
    GenerationResult,
    GenerationTask,
    LetterGenerator,
    RankingResult,
    RankingTask,
)
from tournament_eval.orchestration import (
    build_ranking_tasks,
    generate_all,
    rank_all,
)

__all__ = [
    # Data model
    "GenerationTask",
    "GenerationResult",
    "RankingTask",
    "RankingResult",
    "LetterGenerator",
    # Transport
    "LLMClient",
    "ModelConfig",
    "OllamaModelConfig",
    "OpenAIModelConfig",
    "AnthropicModelConfig",
    "OllamaLLMClient",
    "StructuredResponse",
    # Orchestration
    "generate_all",
    "build_ranking_tasks",
    "rank_all",
]
