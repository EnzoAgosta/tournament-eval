"""Circular ranking for language models, plus clean primitives for wiring LLMs together.

Models generate outputs, then a pool of rankers rank them — contestants and rankers are
independent lists you pass in.  Bias becomes measurable signal instead of a single
ranker's hidden assumption.

The pipeline is a thin wrapper over `pydantic-ai <https://ai.pydantic.dev>`_:
you construct and own the :class:`pydantic_ai.Agent` instances (model, settings,
thinking, retries, concurrency all configured the pydantic-ai way) and hand them
to :func:`~tournament_eval.orchestration.generate_all` /
:func:`~tournament_eval.orchestration.rank_all`.  This package owns the
methodology — the tasks, the aliasing, the resume, the persistence — and leans on
pydantic-ai for everything LLM.

The top level is the happy path: build tasks, generate, build ranking tasks,
rank, and the records + ranking contract you touch along the way.  Everything
else keeps its module home — the on-disk JSON shapes (``*Dict``) and the typed
readers/writer live in :mod:`tournament_eval.persistence` and
:mod:`tournament_eval.models`; the template-authoring internals
(:class:`~tournament_eval.ranking.RankingResponse`, :class:`~tournament_eval.ranking.ParsedRanking`,
:func:`~tournament_eval.ranking.alias_for_index`) live in
:mod:`tournament_eval.ranking`; and aggregation lives in
:mod:`tournament_eval.aggregation`.
"""

from .models import (
    GenerationFailure,
    GenerationResult,
    GenerationTask,
    RankingFailure,
    RankingResult,
    RankingTask,
)
from .orchestration import (
    build_generation_task,
    build_generation_tasks,
    build_ranking_task,
    build_ranking_tasks,
    generate_all,
    generate_one,
    rank_all,
    rank_one,
)
from .ranking import DefaultRankingTemplate, RankingTemplate

__all__ = [
    "DefaultRankingTemplate",
    "GenerationFailure",
    "GenerationResult",
    "GenerationTask",
    "RankingFailure",
    "RankingResult",
    "RankingTask",
    "RankingTemplate",
    "build_generation_task",
    "build_generation_tasks",
    "build_ranking_task",
    "build_ranking_tasks",
    "generate_all",
    "generate_one",
    "rank_all",
    "rank_one",
]
