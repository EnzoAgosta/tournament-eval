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
"""

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
    build_generation_task,
    build_generation_tasks,
    build_ranking_task,
    build_ranking_tasks,
    deanonymize_ranking,
    generate_all,
    generate_one,
    rank_all,
    rank_one,
)
from .persistence import (
    append_record,
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
    RankingResponse,
    RankingTemplate,
    alias_for_index,
)

__all__ = [
    "DefaultRankingTemplate",
    "GenerationFailure",
    "GenerationFailureDict",
    "GenerationResult",
    "GenerationResultDict",
    "GenerationTask",
    "GenerationTaskDict",
    "ParsedRanking",
    "RankingFailure",
    "RankingFailureDict",
    "RankingResponse",
    "RankingResult",
    "RankingResultDict",
    "RankingTask",
    "RankingTaskDict",
    "RankingTemplate",
    "alias_for_index",
    "append_record",
    "build_generation_task",
    "build_generation_tasks",
    "build_ranking_task",
    "build_ranking_tasks",
    "deanonymize_ranking",
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
