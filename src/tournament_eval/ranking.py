"""Ranking strategy: how a ranking model is prompted, constrained, and parsed.

The :class:`RankingTemplate` contract bundles the three pieces that must agree
with each other — the response model (the *shape* of the verdict, validated by
pydantic-ai at the provider), the prompt, and the dynamic alias check — plus the
built-in :class:`DefaultRankingTemplate` (a strict total order, no ties) and the
:func:`alias_for_index` helper used to anonymise candidates behind aliases.

Because structured output is now pydantic-ai's concern, the *static* shape of the
response (every field's type) is validated by pydantic before
:meth:`RankingTemplate.parse` ever sees it — the model instance handed to
:meth:`parse` is already a validated :class:`pydantic.BaseModel`.  What stays
here is the *dynamic* rule that can't be expressed in a static schema: every
alias must appear exactly once, drawn from the task's candidate set (which isn't
known at schema-definition time).  A ranking that violates that rule raises
:class:`ValueError` so the orchestration records it as a
:class:`~tournament_eval.models.RankingFailure`.

This module depends only on :mod:`tournament_eval.models` for the data it ranks
(:class:`~tournament_eval.models.RankingTask` /
:class:`~tournament_eval.models.GenerationResult`); the dependency never runs the
other way.
"""

import abc
import dataclasses
from collections.abc import Mapping

from pydantic import BaseModel

from tournament_eval.models import GenerationResult, RankingTask


def alias_for_index(index: int) -> str:
    """Map a 0-based index to an Excel-style alias: 0->A, 25->Z, 26->AA, 27->AB, ...

    Used to anonymise candidates behind position-independent labels; the caller
    enumerates its candidates and asks for one alias per index.
    """
    result = ""
    n = index + 1
    while n > 0:
        n, remainder = divmod(n - 1, 26)
        result = chr(ord("A") + remainder) + result
    return result


@dataclasses.dataclass(frozen=True, slots=True)
class ParsedRanking:
    """A ranking model's validated verdict, as aliases.

    The output of :meth:`RankingTemplate.parse`, still in alias space (``"A"``,
    ``"B"``); the orchestration maps these back to :class:`GenerationResult` IDs
    when building the :class:`~tournament_eval.models.RankingResult`.
    """

    ranking: list[str]
    """Aliases in ranked order, best first."""
    reasoning: str | None
    """The ranking model's explanation, if it provided one."""


class RankingTemplate(abc.ABC):
    """The ranking contract: how a ranking model is prompted, constrained, and parsed.

    Bundles the three pieces that must agree with each other — they can't be
    varied independently:

    * :attr:`response_model` — the pydantic model the ranking agent's
      ``output_type`` is set to.  pydantic-ai constrains the model's reply to this
      shape and validates it, so the *static* type of every field is enforced
      before :meth:`parse` runs.
    * :meth:`render` — the full prompt the ranking model sees.
    * :meth:`parse` — the *dynamic* alias check (every alias exactly once, drawn
      from the task's candidates) that can't live in a static schema.

    Wire a ranking agent with ``Agent(model, output_type=template.response_model)``
    and hand it to :func:`~tournament_eval.orchestration.rank_all` together with the
    template.  Subclass to customise ranking.  Adding context (a ranking rubric,
    domain notes) is a one-method override of :meth:`render`; each candidate is
    passed as its full :class:`GenerationResult`, so ``.output``, ``.reasoning``,
    ``.generation_prompt`` and ``.metadata`` are all in reach.  Changing the
    *shape* of the verdict (e.g. allowing ties) means overriding all three — the
    response model, the prompt, and the parser.

    See :class:`DefaultRankingTemplate` for the built-in strict-total-order ranker.
    """

    @property
    @abc.abstractmethod
    def response_model(self) -> type[BaseModel]:
        """The pydantic model a ranking agent's ``output_type`` is set to.

        pydantic-ai constrains the model's reply to this schema and validates it,
        so the static shape of the verdict (field types) is enforced at the
        provider.  Set the ranking agent's ``output_type`` to this.
        """
        ...

    @abc.abstractmethod
    def render(
        self,
        ranking_task: RankingTask,
        candidates: Mapping[str, GenerationResult],
    ) -> str:
        """Build the full prompt sent to a ranking model.

        ``candidates`` maps each alias to its :class:`GenerationResult` (resolved
        from ``ranking_task.generations``), so derive any context you need from it.
        Note it exposes ``.author``: rendering that into the prompt would defeat the
        anonymisation, so a custom template should read only ``.output`` and
        task-level fields like ``.generation_prompt``.
        """
        ...

    @abc.abstractmethod
    def parse(self, instance: BaseModel, valid_aliases: set[str]) -> ParsedRanking:
        """Apply the *dynamic* alias check to an already-validated response.

        ``instance`` is the pydantic model returned by the ranking agent — its
        static shape is already validated, so this only enforces the rule that
        can't be expressed statically: every alias appears exactly once, drawn from
        ``valid_aliases``.  Raise :class:`ValueError` on anything malformed so the
        orchestration records it as a :class:`RankingFailure`.
        """
        ...


class RankingResponse(BaseModel):
    """The response model for :class:`DefaultRankingTemplate` — a strict total order."""

    ranking: list[str]
    """Aliases in ranked order, best first."""
    reasoning: str | None = None
    """The ranking model's explanation, if it provided one."""


class DefaultRankingTemplate(RankingTemplate):
    """The built-in ranking model: a strict total order over the candidates, no ties.

    The prompt shows the original task the models were given (shared across
    candidates, so it leaks no authorship) followed by the anonymised candidate
    outputs and the ranking instruction; the reply is constrained to a
    :class:`RankingResponse` (``ranking`` + optional ``reasoning``) and the
    aliases are validated so every one appears exactly once.

    Set ``include_reasoning=True`` to also surface each candidate's reasoning trace
    (when it has one) to the ranker.  It's off by default: traces are long, and a
    verbose reasoner can look more thorough than it is, biasing the panel.
    """

    def __init__(self, *, include_reasoning: bool = False) -> None:
        self.include_reasoning = include_reasoning

    @property
    def response_model(self) -> type[BaseModel]:
        return RankingResponse

    def render(
        self,
        ranking_task: RankingTask,
        candidates: Mapping[str, GenerationResult],
    ) -> str:
        lines: list[str] = []
        if not candidates:
            raise ValueError("render needs at least one candidate")
        generation_prompts = {result.generation_prompt for result in candidates.values()}
        if len(generation_prompts) != 1:
            raise ValueError("All candidates must have the same generation_prompt")
        lines.extend(
            [
                f"The models were given this task:\n{generation_prompts.pop()}",
                "",
                ranking_task.ranking_prompt,
                "",
            ]
        )

        lines.extend(
            [
                "",
                "Respond ONLY with a JSON object in this exact format:",
                '{"ranking": ["A", "B", "C"], "reasoning": "..."}',
                "",
                'The "ranking" field must list every candidate alias exactly once, from best to worst. '
                'Explain the reasoning behind your ranking in the "reasoning" field.',
                "",
                "NO TIES ARE ALLOWED. If two outputs appear equal, break the tie as you see fit "
                'and explain why in the "reasoning" field.',
                "",
                "Candidates:",
                "",
            ]
        )

        for alias, result in candidates.items():
            lines.append(f"{alias}: {result.output}")
            if self.include_reasoning and result.reasoning:
                lines.append(f"{alias} reasoning: {result.reasoning}")

        return "\n".join(lines)

    def parse(self, instance: BaseModel, valid_aliases: set[str]) -> ParsedRanking:
        if not isinstance(instance, RankingResponse):
            raise ValueError(f"Expected a RankingResponse, got {type(instance).__name__}")

        ranking: list[str] = []
        seen: set[str] = set()
        for alias in instance.ranking:
            if alias not in valid_aliases:
                raise ValueError(f"Unknown alias in ranking: {alias!r}")
            if alias in seen:
                raise ValueError(f"Duplicate alias in ranking: {alias!r}")
            seen.add(alias)
            ranking.append(alias)

        missing = valid_aliases - seen
        if missing:
            raise ValueError(f"Missing aliases in ranking: {sorted(missing)}")

        return ParsedRanking(ranking=ranking, reasoning=instance.reasoning)
