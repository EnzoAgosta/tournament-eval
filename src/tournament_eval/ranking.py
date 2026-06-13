"""Ranking strategy: how a ranking model is prompted, constrained, and parsed.

The :class:`RankingTemplate` contract bundles the three pieces that must agree
with each other — the prompt, the response schema, and the validation — plus the
built-in :class:`DefaultRankingTemplate` (a strict total order, no ties) and the
:class:`LetterGenerator` used to anonymise candidates behind aliases.

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


class LetterGenerator:
    """Simple generator for letters, Excel-style: A, B, C, ..., AA, AB, AC, ..."""

    def __init__(self) -> None:
        self.n = 0

    def get_next_letter(self) -> str:
        self.n += 1
        result = ""
        temp = self.n
        while temp > 0:
            temp, remainder = divmod(temp - 1, 26)
            result = chr(65 + remainder) + result
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

    Subclass to customise ranking.  The common case — injecting context such as
    the original generation prompt — is a one-method override of :meth:`render`;
    each candidate is passed as its full :class:`GenerationResult`, so ``.output``,
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

    Prompts with the ranking instruction plus the anonymised candidate outputs,
    constrains the reply to ``{"ranking": [...], "reasoning": ...}``, and validates
    that every alias appears exactly once.
    """

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
        lines: list[str] = [ranking_task.ranking_prompt, "", "Candidates:"]
        for alias, result in candidates.items():
            lines.append(f"{alias}. {result.output}")

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
