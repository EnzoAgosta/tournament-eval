"""Tests for the ranking strategy: LetterGenerator, DefaultRankingTemplate, custom templates."""

from collections.abc import Mapping

import pytest

from tests.conftest import _MakeGeneration, _MakeRankingTask
from tournament_eval.models import GenerationResult, RankingTask
from tournament_eval.ranking import DefaultRankingTemplate, LetterGenerator


def test_letter_generator_sequence() -> None:
    gen = LetterGenerator()
    assert [gen.get_next_letter() for _ in range(3)] == ["A", "B", "C"]


def test_letter_generator_wraps_past_z() -> None:
    gen = LetterGenerator()
    for _ in range(25):
        gen.get_next_letter()
    assert gen.get_next_letter() == "Z"
    assert gen.get_next_letter() == "AA"
    assert gen.get_next_letter() == "AB"


def test_schema_constrains_ranking() -> None:
    schema = DefaultRankingTemplate().schema
    assert schema["type"] == "object"
    assert "ranking" in schema["properties"]
    assert schema["required"] == ["ranking"]


def test_render_includes_prompt_candidates_and_format(
    make_generation: _MakeGeneration, make_ranking_task: _MakeRankingTask
) -> None:
    gen_a = make_generation(output="hello")
    gen_b = make_generation(output="world")
    task = make_ranking_task(ranking_prompt="Rank by charm.", generations={"A": gen_a.id, "B": gen_b.id})
    prompt = DefaultRankingTemplate().render(task, {"A": gen_a, "B": gen_b})
    assert "Rank by charm." in prompt
    assert "A. hello" in prompt
    assert "B. world" in prompt
    assert '"ranking"' in prompt
    assert "NO TIES" in prompt


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


def test_custom_template_can_inject_source(
    make_generation: _MakeGeneration, make_ranking_task: _MakeRankingTask
) -> None:
    """The headline extensibility case: a template that surfaces the source prompt."""

    class SourceAware(DefaultRankingTemplate):
        def render(self, ranking_task: RankingTask, candidates: Mapping[str, GenerationResult]) -> str:
            source = next(iter(candidates.values())).generation_prompt
            return f"SOURCE: {source}\n\n{super().render(ranking_task, candidates)}"

    gen = make_generation(output="bonjour")
    task = make_ranking_task(generations={"A": gen.id})
    prompt = SourceAware().render(task, {"A": gen})
    assert prompt.startswith("SOURCE: test prompt")
    assert "A. bonjour" in prompt
