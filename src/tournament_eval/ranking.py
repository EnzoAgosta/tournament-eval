"""Ranking strategy: how a ranking model is prompted, constrained, and parsed.

The :class:`RankingTemplate` contract bundles the three pieces that must agree:
the response model (the verdict's *shape*, validated by pydantic-ai at the
provider), the prompt, and the dynamic alias check.  Plus the built-in
:class:`DefaultRankingTemplate` (a strict total order, no ties) and the
:func:`alias_for_index` helper used to anonymise candidates behind aliases.

The *static* shape of the response is validated by pydantic before
:meth:`RankingTemplate.parse` sees it; what stays here is the *dynamic* rule a
static schema can't express: every alias must appear exactly once, drawn from the
task's candidate set (unknown at schema-definition time).  A violation raises
:class:`ValueError`, which the orchestration records as a
:class:`~tournament_eval.models.RankingFailure`.

Depends only on :mod:`tournament_eval.models` for the data it ranks; the
dependency never runs the other way.
"""

import abc
import dataclasses
from collections.abc import Mapping

from pydantic import BaseModel, Field

from tournament_eval.models import GenerationResult, RankingTask


def alias_for_index(index: int) -> str:
    """Map a 0-based index to an Excel-style alias: 0->A, 25->Z, 26->AA, 27->AB, ...

    Used to anonymise candidates behind position-independent labels.
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

    Output of :meth:`RankingTemplate.parse`, still in alias space; the
    orchestration maps these back to :class:`GenerationResult` IDs when building
    the :class:`~tournament_eval.models.RankingResult`.
    """

    ranking: list[str]
    """Aliases in ranked order, best first."""
    reasoning: str | None
    """The ranking model's explanation, if it provided one."""


class RankingTemplate(abc.ABC):
    """The ranking contract: how a ranking model is prompted, constrained, and parsed.

    Three pieces that must agree — they can't be varied independently:

    * :attr:`response_model` — the pydantic model the ranking agent's
      ``output_type`` is set to.  pydantic-ai constrains the reply to this shape
      and validates it, so the static field types are enforced before :meth:`parse`
      runs.
    * :meth:`render` — the full prompt the ranking model sees.
    * :meth:`parse` — the dynamic alias check (every alias exactly once, drawn from
      the task's candidates) that can't live in a static schema.

    Wire a ranking agent with ``Agent(model, output_type=template.response_model)``
    and hand it to :func:`~tournament_eval.orchestration.rank_all` with the template.
    Subclass to customise.  Adding context (a rubric, domain notes) is a one-method
    override of :meth:`render`; each candidate is its full :class:`GenerationResult`,
    so ``.output``, ``.reasoning``, ``.generation_prompt`` and ``.metadata`` are in
    reach.  Changing the verdict's *shape* (e.g. allowing ties) means overriding all
    three.  See :class:`DefaultRankingTemplate` for the built-in strict-total-order
    ranker.
    """

    @property
    @abc.abstractmethod
    def response_model(self) -> type[BaseModel]:
        """The pydantic model a ranking agent's ``output_type`` is set to.

        pydantic-ai constrains the reply to this schema and validates it, so the
        static shape is enforced at the provider.
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
        from ``ranking_task.generations``).  It exposes ``.author`` — rendering
        that into the prompt would defeat the anonymisation, so a custom template
        reads only ``.output`` and task-level fields like ``.generation_prompt``.
        """
        ...

    @abc.abstractmethod
    def parse(self, instance: BaseModel, valid_aliases: set[str]) -> ParsedRanking:
        """Apply the dynamic alias check to an already-validated response.

        ``instance``'s static shape is already validated, so this only enforces
        what a static schema can't: every alias appears exactly once, drawn from
        ``valid_aliases``.  Raise :class:`ValueError` on anything malformed so the
        orchestration records it as a :class:`RankingFailure`.
        """
        ...


class RankingResponse(BaseModel):
    """A strict total order over every candidate alias — no ties, no omissions, no duplicates.

    ``ranking`` lists every alias best-first; ``reasoning`` briefly justifies the order.
    The dynamic check that every alias from the task's candidate set appears exactly
    once is applied by :meth:`RankingTemplate.parse` (it can't be expressed in a
    static schema, since the candidate set isn't known at schema-definition time).
    """

    ranking: list[str] = Field(
        description="Every candidate alias, best to worst. Each alias must appear exactly once "
        "— no omissions, no duplicates, no extras, no ties.",
    )
    reasoning: str | None = Field(
        default=None,
        description="A short justification of the ranking.",
    )


class DefaultRankingTemplate(RankingTemplate):
    """The built-in ranker: a strict total order over the candidates, no ties.

    The prompt shows the original task (shared across candidates, so it leaks no
    authorship) then the anonymised outputs and the ranking instruction; the reply
    is constrained to a :class:`RankingResponse` and every alias is validated to
    appear exactly once.

    Set ``include_reasoning=True`` to also surface each candidate's reasoning trace
    to the ranker.  Off by default: traces are long, and a verbose reasoner can
    look more thorough than it is, biasing the panel.
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

        aliases = sorted(candidates.keys())
        n = len(aliases)
        example = ", ".join(f'"{a}"' for a in aliases)
        lines.extend(
            [
                "",
                f"There are {n} candidates, labelled: {', '.join(aliases)}.",
                f"Your `ranking` must list every one of these {n} aliases exactly once — "
                "no omissions, no duplicates, no extras — from best to worst.",
                "",
                "Return two fields:",
                f"  - `ranking`: the {n} aliases in order, best to worst (e.g. [{example}]).",
                "  - `reasoning`: a short justification of the order.",
                "",
                "NO TIES. If two outputs appear equal, break the tie as you see fit and explain why in `reasoning`.",
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
