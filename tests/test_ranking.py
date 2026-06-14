"""Tests for the ranking strategy: alias_for_index, DefaultRankingTemplate, custom templates."""

from collections.abc import Mapping

import pytest

from tests.conftest import _MakeGeneration, _MakeRankingTask
from tournament_eval.models import GenerationResult, RankingTask
from tournament_eval.ranking import DefaultRankingTemplate, alias_for_index


def test_alias_for_index_sequence() -> None:
    assert [alias_for_index(i) for i in range(3)] == ["A", "B", "C"]


def test_alias_for_index_wraps_past_z() -> None:
    assert alias_for_index(25) == "Z"
    assert alias_for_index(26) == "AA"
    assert alias_for_index(27) == "AB"


def test_schema_constrains_ranking() -> None:
    schema = DefaultRankingTemplate().schema
    assert schema["type"] == "object"
    assert "ranking" in schema["properties"]
    assert schema["required"] == ["ranking"]


def test_render_includes_source_prompt_candidates_and_format(
    make_generation: _MakeGeneration, make_ranking_task: _MakeRankingTask
) -> None:
    gen_a = make_generation(output="hello")  # generation_prompt defaults to "test prompt"
    gen_b = make_generation(output="world")
    task = make_ranking_task(ranking_prompt="Rank by charm.", generations={"A": gen_a.id, "B": gen_b.id})
    prompt = DefaultRankingTemplate().render(task, {"A": gen_a, "B": gen_b})
    assert "test prompt" in prompt  # the source task, surfaced by default
    assert "Rank by charm." in prompt
    assert "A. hello" in prompt
    assert "B. world" in prompt
    assert '"ranking"' in prompt
    assert "NO TIES" in prompt


def test_render_omits_reasoning_by_default(
    make_generation: _MakeGeneration, make_ranking_task: _MakeRankingTask
) -> None:
    gen = make_generation(output="hello", reasoning="i thought hard")
    task = make_ranking_task(generations={"A": gen.id})
    assert "i thought hard" not in DefaultRankingTemplate().render(task, {"A": gen})


def test_render_includes_reasoning_when_enabled(
    make_generation: _MakeGeneration, make_ranking_task: _MakeRankingTask
) -> None:
    gen = make_generation(output="hello", reasoning="i thought hard")
    task = make_ranking_task(generations={"A": gen.id})
    prompt = DefaultRankingTemplate(include_reasoning=True).render(task, {"A": gen})
    assert "i thought hard" in prompt


class TestDefaultParse:
    def test_valid(self) -> None:
        parsed = DefaultRankingTemplate().parse({"ranking": ["B", "A"]}, {"A", "B"})
        assert parsed.ranking == ["B", "A"]
        assert parsed.reasoning is None

    def test_with_reasoning(self) -> None:
        parsed = DefaultRankingTemplate().parse({"ranking": ["A"], "reasoning": "best"}, {"A"})
        assert parsed.reasoning == "best"

    def test_not_a_dict(self) -> None:
        with pytest.raises(ValueError, match="Expected JSON object"):
            DefaultRankingTemplate().parse(["A"], {"A"})  # type: ignore[arg-type]

    def test_ranking_not_a_list(self) -> None:
        with pytest.raises(ValueError, match='"ranking" must be a list'):
            DefaultRankingTemplate().parse({"ranking": "A"}, {"A"})

    def test_non_string_entry(self) -> None:
        with pytest.raises(ValueError, match="must be a string"):
            DefaultRankingTemplate().parse({"ranking": [1]}, {"A"})

    def test_unknown_alias(self) -> None:
        with pytest.raises(ValueError, match="Unknown alias"):
            DefaultRankingTemplate().parse({"ranking": ["Z"]}, {"A"})

    def test_duplicate_alias(self) -> None:
        with pytest.raises(ValueError, match="Duplicate alias"):
            DefaultRankingTemplate().parse({"ranking": ["A", "A"]}, {"A"})

    def test_missing_alias(self) -> None:
        with pytest.raises(ValueError, match="Missing aliases"):
            DefaultRankingTemplate().parse({"ranking": ["A"]}, {"A", "B"})

    def test_non_string_reasoning(self) -> None:
        with pytest.raises(ValueError, match='"reasoning" must be a string'):
            DefaultRankingTemplate().parse({"ranking": ["A"], "reasoning": 1}, {"A"})


def test_custom_template_can_add_context(
    make_generation: _MakeGeneration, make_ranking_task: _MakeRankingTask
) -> None:
    """Extensibility: a template that prepends extra context onto the default prompt."""

    class RubricAware(DefaultRankingTemplate):
        def render(self, ranking_task: RankingTask, candidates: Mapping[str, GenerationResult]) -> str:
            return f"Grading rubric: prioritise factual accuracy.\n\n{super().render(ranking_task, candidates)}"

    gen = make_generation(output="bonjour")
    task = make_ranking_task(generations={"A": gen.id})
    prompt = RubricAware().render(task, {"A": gen})
    assert prompt.startswith("Grading rubric: prioritise factual accuracy.")
    assert "A. bonjour" in prompt
