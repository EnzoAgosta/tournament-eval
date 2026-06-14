"""Ranking strategy: how a ranking model is prompted, constrained, and parsed.

The :class:`RankingTemplate` contract bundles the three pieces that must agree
with each other — the prompt, the response schema, and the validation — plus the
built-in :class:`DefaultRankingTemplate` (a strict total order, no ties) and the
:func:`alias_for_index` helper used to anonymise candidates behind aliases.

This module depends only on :mod:`tournament_eval.models` for the data it ranks
(:class:`~tournament_eval.models.RankingTask` /
:class:`~tournament_eval.models.GenerationResult`); the dependency never runs the
other way.
"""

import abc
import dataclasses
from collections.abc import Mapping
from typing import ClassVar

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
    when building the :class:`RankingResult`.
    """

    ranking: list[str]
    """Aliases in ranked order, best first."""
    reasoning: str | None
    """The ranking model's explanation, if it provided one."""


class RankingTemplate(abc.ABC):
    """The ranking contract: how a ranking model is prompted, constrained, and parsed.

    Bundles the three pieces that must agree with each other — they can't be
    varied independently:

    * :meth:`render` — the full prompt the ranking model sees.
    * :meth:`schema` — the JSON schema its response is constrained to.
    * :meth:`parse` — validation of that response into a :class:`ParsedRanking`.

    Subclass to customise ranking.  Adding context (a grading rubric, domain
    notes) is a one-method override of :meth:`render`; each candidate is passed as
    its full :class:`GenerationResult`, so ``.output``, ``.reasoning``,
    ``.generation_prompt`` and ``.metadata`` are all in reach.  Changing the
    *shape* of the verdict (e.g. allowing ties) means overriding all three.

    See :class:`DefaultRankingTemplate` for the built-in strict-total-order ranker.
    """

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

    @property
    @abc.abstractmethod
    def schema(self) -> dict[str, object]:
        """The JSON schema ``generate_structured`` constrains the model's reply to."""
        ...

    @abc.abstractmethod
    def parse(
        self,
        data: dict[str, object],
        valid_aliases: set[str],
    ) -> ParsedRanking:
        """Validate a ranking model's parsed JSON reply and extract the ranking.

        ``valid_aliases`` are the aliases that must appear; raise
        :class:`ValueError` on anything malformed so the orchestration records it
        as a :class:`RankingFailure`.
        """
        ...


class DefaultRankingTemplate(RankingTemplate):
    """The built-in ranking model: a strict total order over the candidates, no ties.

    The prompt shows the original task the models were given (shared across
    candidates, so it leaks no authorship) followed by the anonymised candidate
    outputs and the ranking instruction; the reply is constrained to
    ``{"ranking": [...], "reasoning": ...}`` and validated so every alias appears
    exactly once.

    Set ``include_reasoning=True`` to also surface each candidate's reasoning trace
    (when it has one) to the ranker.  It's off by default: traces are long, and a
    verbose reasoner can look more thorough than it is, biasing the panel.
    """

    def __init__(self, *, include_reasoning: bool = False) -> None:
        self.include_reasoning = include_reasoning

    _SCHEMA: ClassVar[dict[str, object]] = {
        "type": "object",
        "properties": {
            "ranking": {
                "type": "array",
                "items": {"type": "string"},
                "description": "Aliases in ranked order, best first.",
            },
            "reasoning": {
                "type": "string",
                "description": (
                    "Optional step-by-step analysis and explanation behind overall ranking and tie breakers."
                ),
            },
        },
        "required": ["ranking"],
        "additionalProperties": False,
    }

    def render(
        self,
        ranking_task: RankingTask,
        candidates: Mapping[str, GenerationResult],
    ) -> str:
        lines: list[str] = []
        first = next(iter(candidates.values()), None)
        if first is not None:
            lines += [f"The models were given this task:\n{first.generation_prompt}", ""]

        lines += [ranking_task.ranking_prompt, "", "Candidates:"]
        for alias, result in candidates.items():
            lines.append(f"{alias}. {result.output}")
            if self.include_reasoning and result.reasoning:
                lines.append(f"{alias} reasoning: {result.reasoning}")

        lines.extend(
            [
                "",
                "Respond ONLY with a JSON object in this exact format:",
                '{"ranking": ["A", "B", "C"], "reasoning": "..."}',
                "",
                'The "ranking" field must list every candidate alias exactly once,',
                'from best to worst. The "reasoning" field is optional.',
                (
                    "NO TIES ARE ALLOWED. If two outputs appear equal, break "
                    'the tie as you see fit and explain why in the "reasoning" '
                    "field."
                ),
            ]
        )
        return "\n".join(lines)

    @property
    def schema(self) -> dict[str, object]:
        return self._SCHEMA

    def parse(
        self,
        data: dict[str, object],
        valid_aliases: set[str],
    ) -> ParsedRanking:
        if not isinstance(data, dict):
            raise ValueError(f"Expected JSON object, got {type(data).__name__}")

        ranking_raw = data.get("ranking")
        if not isinstance(ranking_raw, list):
            raise ValueError(f'"ranking" must be a list, got {type(ranking_raw).__name__}')

        ranking: list[str] = []
        seen: set[str] = set()
        for alias in ranking_raw:
            if not isinstance(alias, str):
                raise ValueError(f"Ranking entry must be a string, got {type(alias).__name__}")
            if alias not in valid_aliases:
                raise ValueError(f"Unknown alias in ranking: {alias!r}")
            if alias in seen:
                raise ValueError(f"Duplicate alias in ranking: {alias!r}")
            seen.add(alias)
            ranking.append(alias)

        missing = valid_aliases - seen
        if missing:
            raise ValueError(f"Missing aliases in ranking: {sorted(missing)}")

        reasoning = data.get("reasoning")
        if reasoning is not None and not isinstance(reasoning, str):
            raise ValueError(f'"reasoning" must be a string, got {type(reasoning).__name__}')

        return ParsedRanking(ranking=ranking, reasoning=reasoning)
