"""Unit tests for pure helper functions (no LLM calls, no async, no I/O)."""

import pytest

from tests.conftest import _MakeGeneration, _MakeRankingTask, _MakeTask
from tournament_eval.orchestration import (
    _build_generation_lookup,
    build_ranking_task,
    build_ranking_tasks,
)
from tournament_eval.ranking import DefaultRankingTemplate, LetterGenerator


def test_letter_generator_sequence() -> None:
    gen = LetterGenerator()
    assert gen.get_next_letter() == "A"
    assert gen.get_next_letter() == "B"
    assert gen.get_next_letter() == "C"


def test_letter_generator_wraps_after_z() -> None:
    gen = LetterGenerator()
    for _ in range(25):
        gen.get_next_letter()
    assert gen.get_next_letter() == "Z"
    assert gen.get_next_letter() == "AA"
    assert gen.get_next_letter() == "AB"


class TestDefaultRankingTemplateParse:
    def test_valid(self) -> None:
        parsed = DefaultRankingTemplate().parse({"ranking": ["B", "A"]}, {"A", "B"})
        assert parsed.ranking == ["B", "A"]
        assert parsed.reasoning is None

    def test_with_reasoning(self) -> None:
        parsed = DefaultRankingTemplate().parse(
            {"ranking": ["A"], "reasoning": "great"}, {"A"}
        )
        assert parsed.reasoning == "great"

    def test_missing_alias(self) -> None:
        with pytest.raises(ValueError, match="Missing aliases"):
            DefaultRankingTemplate().parse({"ranking": ["A"]}, {"A", "B"})

    def test_duplicate_alias(self) -> None:
        with pytest.raises(ValueError, match="Duplicate alias"):
            DefaultRankingTemplate().parse({"ranking": ["A", "A"]}, {"A"})

    def test_unknown_alias(self) -> None:
        with pytest.raises(ValueError, match="Unknown alias"):
            DefaultRankingTemplate().parse({"ranking": ["Z"]}, {"A"})

    def test_not_a_dict(self) -> None:
        with pytest.raises(ValueError, match="Expected JSON object"):
            DefaultRankingTemplate().parse(["A"], {"A"})  # type: ignore[arg-type]

    def test_ranking_not_a_list(self) -> None:
        with pytest.raises(ValueError, match='"ranking" must be a list'):
            DefaultRankingTemplate().parse({"ranking": "A"}, {"A"})

    def test_non_string_entry(self) -> None:
        with pytest.raises(ValueError, match="Ranking entry must be a string"):
            DefaultRankingTemplate().parse({"ranking": [42]}, {"A"})

    def test_non_string_reasoning(self) -> None:
        with pytest.raises(ValueError, match='"reasoning" must be a string'):
            DefaultRankingTemplate().parse({"ranking": ["A"], "reasoning": 42}, {"A"})


def test_lookup_indexes_by_id(make_generation: _MakeGeneration) -> None:
    r1 = make_generation(author="a")
    r2 = make_generation(author="b")
    lookup = _build_generation_lookup([r1, r2])
    assert lookup[r1.id] == r1
    assert lookup[r2.id] == r2


def test_lookup_empty() -> None:
    assert _build_generation_lookup([]) == {}


def test_default_template_renders_rubric_and_candidates(
    make_generation: _MakeGeneration,
    make_ranking_task: _MakeRankingTask,
) -> None:
    gen_a = make_generation(output="hello")
    gen_b = make_generation(output="world")
    rt = make_ranking_task(
        ranking_prompt="Rank by charm.",
        generations={"A": gen_a.id, "B": gen_b.id},
    )
    prompt = DefaultRankingTemplate().render(rt, {"A": gen_a, "B": gen_b})
    assert "Rank by charm." in prompt
    assert "A. hello" in prompt
    assert "B. world" in prompt
    assert '"ranking"' in prompt


def test_custom_template_can_inject_generation_prompt(
    make_generation: _MakeGeneration,
    make_ranking_task: _MakeRankingTask,
) -> None:
    # The headline use case: a custom render that shows the judge what the models
    # were originally asked, derived from the candidates' GenerationResults.
    from collections.abc import Mapping

    from tournament_eval.models import GenerationResult, RankingTask

    class SourceAwareTemplate(DefaultRankingTemplate):
        def render(
            self,
            ranking_task: RankingTask,
            candidates: Mapping[str, GenerationResult],
        ) -> str:
            source = next(iter(candidates.values())).generation_prompt
            base = super().render(ranking_task, candidates)
            return f"The models were asked:\n{source}\n\n{base}"

    gen_a = make_generation(output="bonjour")
    gen_b = make_generation(output="salut")
    rt = make_ranking_task(generations={"A": gen_a.id, "B": gen_b.id})
    prompt = SourceAwareTemplate().render(rt, {"A": gen_a, "B": gen_b})
    assert prompt.startswith("The models were asked:\ntest prompt")
    assert "A. bonjour" in prompt


class TestBuildRankingTask:
    def test_assigns_an_alias_per_result(
        self, make_generation: _MakeGeneration
    ) -> None:
        results = [make_generation(), make_generation(), make_generation()]
        rt = build_ranking_task(results, "Rank.")
        assert list(rt.generations.keys()) == ["A", "B", "C"]
        assert set(rt.generations.values()) == {r.id for r in results}
        assert rt.ranking_prompt == "Rank."

    def test_seed_makes_the_shuffle_deterministic(
        self, make_generation: _MakeGeneration
    ) -> None:
        results = [make_generation() for _ in range(5)]
        a = build_ranking_task(results, "Rank.", random_seed=7)
        b = build_ranking_task(results, "Rank.", random_seed=7)
        assert list(a.generations.values()) == list(b.generations.values())

    def test_does_not_mutate_input(self, make_generation: _MakeGeneration) -> None:
        results = [make_generation(), make_generation()]
        before = list(results)
        build_ranking_task(results, "Rank.", random_seed=1)
        assert results == before  # shuffled a copy, not the caller's list


class TestBuildRankingTasks:
    def test_groups_by_task(
        self,
        make_task: _MakeTask,
        make_generation: _MakeGeneration,
    ) -> None:
        t1 = make_task()
        t2 = make_task()
        r1 = make_generation(task_id=t1.id)
        r2 = make_generation(task_id=t1.id)
        r3 = make_generation(task_id=t2.id)

        rts = build_ranking_tasks([t1, t2], [r1, r2, r3], "Rank.")

        assert len(rts) == 2
        assert set(rts[0].generations.values()) == {r1.id, r2.id}
        assert set(rts[1].generations.values()) == {r3.id}

    def test_skips_empty_tasks(
        self,
        make_task: _MakeTask,
        make_generation: _MakeGeneration,
    ) -> None:
        t1 = make_task()
        t2 = make_task()
        rts = build_ranking_tasks([t1, t2], [make_generation(task_id=t1.id)], "Rank.")
        assert len(rts) == 1

    def test_assigns_aliases(
        self,
        make_task: _MakeTask,
        make_generation: _MakeGeneration,
    ) -> None:
        t = make_task()
        rts = build_ranking_tasks(
            [t],
            [make_generation(task_id=t.id), make_generation(task_id=t.id)],
            "Rank.",
        )
        assert list(rts[0].generations.keys()) == ["A", "B"]

    def test_preserves_task_order(
        self,
        make_task: _MakeTask,
        make_generation: _MakeGeneration,
    ) -> None:
        t1 = make_task()
        t2 = make_task()
        r1 = make_generation(task_id=t1.id)
        r2 = make_generation(task_id=t2.id)
        rts = build_ranking_tasks([t2, t1], [r1, r2], "Rank.")
        assert list(rts[0].generations.values()) == [r2.id]
        assert list(rts[1].generations.values()) == [r1.id]
