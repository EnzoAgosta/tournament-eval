"""Unit tests for pure helper functions (no LLM calls, no async, no I/O)."""


import pytest

from tests.conftest import _MakeGeneration, _MakeRankingTask, _MakeTask
from tournament_eval.models import LetterGenerator
from tournament_eval.orchestration import (
    _build_generation_lookup,
    _build_judge_prompt,
    _parse_ranking_response,
    build_ranking_tasks,
)


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


class TestParseRankingResponse:
    def test_valid(self) -> None:
        ranking, reasoning = _parse_ranking_response(
            {"ranking": ["B", "A"]}, {"A", "B"}
        )
        assert ranking == ["B", "A"]
        assert reasoning is None

    def test_with_reasoning(self) -> None:
        _, reasoning = _parse_ranking_response(
            {"ranking": ["A"], "reasoning": "great"}, {"A"}
        )
        assert reasoning == "great"

    def test_missing_alias(self) -> None:
        with pytest.raises(ValueError, match="Missing aliases"):
            _parse_ranking_response({"ranking": ["A"]}, {"A", "B"})

    def test_duplicate_alias(self) -> None:
        with pytest.raises(ValueError, match="Duplicate alias"):
            _parse_ranking_response({"ranking": ["A", "A"]}, {"A"})

    def test_unknown_alias(self) -> None:
        with pytest.raises(ValueError, match="Unknown alias"):
            _parse_ranking_response({"ranking": ["Z"]}, {"A"})

    def test_not_a_dict(self) -> None:
        with pytest.raises(ValueError, match="Expected JSON object"):
            _parse_ranking_response(["A"], {"A"})

    def test_ranking_not_a_list(self) -> None:
        with pytest.raises(ValueError, match='"ranking" must be a list'):
            _parse_ranking_response({"ranking": "A"}, {"A"})

    def test_non_string_entry(self) -> None:
        with pytest.raises(ValueError, match="Ranking entry must be a string"):
            _parse_ranking_response({"ranking": [42]}, {"A"})

    def test_non_string_reasoning(self) -> None:
        with pytest.raises(ValueError, match='"reasoning" must be a string'):
            _parse_ranking_response({"ranking": ["A"], "reasoning": 42}, {"A"})


def test_lookup_indexes_by_id(make_generation: _MakeGeneration) -> None:
    r1 = make_generation(author="a")
    r2 = make_generation(author="b")
    lookup = _build_generation_lookup([r1, r2])
    assert lookup[r1.id] == r1
    assert lookup[r2.id] == r2


def test_lookup_empty() -> None:
    assert _build_generation_lookup([]) == {}


def test_prompt_includes_rubric_and_candidates(
    make_generation: _MakeGeneration,
    make_ranking_task: _MakeRankingTask,
) -> None:
    gen_a = make_generation(output="hello")
    gen_b = make_generation(output="world")
    rt = make_ranking_task(
        ranking_prompt="Rank by charm.",
        generations={"A": gen_a.id, "B": gen_b.id},
    )
    prompt = _build_judge_prompt(rt, _build_generation_lookup([gen_a, gen_b]))
    assert "Rank by charm." in prompt
    assert "A. hello" in prompt
    assert "B. world" in prompt
    assert '"ranking"' in prompt


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
