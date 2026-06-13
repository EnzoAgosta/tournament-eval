"""Edge-case tests for the orchestration layer."""

import uuid

import pytest

from tournament_eval.models import DefaultRankingTemplate, RankingTask
from tournament_eval.orchestration import (
    build_ranking_tasks,
    generate_all,
    rank_all,
)


class TestGenerateAllEdgeCases:
    async def test_empty_tasks(self) -> None:
        results, failures = await generate_all([], [])
        assert results == []
        assert failures == []

    async def test_empty_clients(self, make_task) -> None:
        tasks = [make_task("p1")]
        results, failures = await generate_all(tasks, [])
        assert results == []
        assert failures == []

    async def test_empty_everything(self) -> None:
        results, failures = await generate_all([], [])
        assert results == []
        assert failures == []


class TestRankAllEdgeCases:
    async def test_empty_ranking_tasks(self, make_generation) -> None:
        gen = make_generation()
        results, failures = await rank_all([], [gen], [])
        assert results == []
        assert failures == []

    async def test_empty_clients(self, make_generation) -> None:
        gen = make_generation()
        ranking_task = RankingTask(
            id=uuid.uuid4(),
            ranking_prompt="Rank.",
            generations={"A": gen.id},
        )
        results, failures = await rank_all([ranking_task], [gen], [])
        assert results == []
        assert failures == []

    async def test_empty_generations(self) -> None:
        ranking_task = RankingTask(
            id=uuid.uuid4(),
            ranking_prompt="Rank.",
            generations={},
        )
        results, failures = await rank_all([ranking_task], [], [])
        assert results == []
        assert failures == []

    async def test_empty_everything(self) -> None:
        results, failures = await rank_all([], [], [])
        assert results == []
        assert failures == []


class TestBuildRankingTasksEdgeCases:
    def test_empty_tasks(self) -> None:
        rts = build_ranking_tasks([], [], "Rank.")
        assert rts == []

    def test_empty_generation_results(self, make_task) -> None:
        t = make_task("p")
        rts = build_ranking_tasks([t], [], "Rank.")
        assert rts == []

    def test_empty_both(self) -> None:
        rts = build_ranking_tasks([], [], "Rank.")
        assert rts == []


class TestParseRankingResponseEdgeCases:
    def test_single_alias(self) -> None:
        parsed = DefaultRankingTemplate().parse({"ranking": ["A"]}, {"A"})
        assert parsed.ranking == ["A"]
        assert parsed.reasoning is None

    def test_reasoning_is_none_explicitly(self) -> None:
        parsed = DefaultRankingTemplate().parse(
            {"ranking": ["A"], "reasoning": None}, {"A"}
        )
        assert parsed.ranking == ["A"]
        assert parsed.reasoning is None

    def test_reasoning_is_empty_string(self) -> None:
        parsed = DefaultRankingTemplate().parse(
            {"ranking": ["A"], "reasoning": ""}, {"A"}
        )
        assert parsed.ranking == ["A"]
        assert parsed.reasoning == ""

    def test_extra_keys_ignored(self) -> None:
        parsed = DefaultRankingTemplate().parse(
            {"ranking": ["A"], "confidence": 0.95}, {"A"}
        )
        assert parsed.ranking == ["A"]
        assert parsed.reasoning is None

    def test_empty_ranking_list(self) -> None:
        with pytest.raises(ValueError, match="Missing aliases"):
            DefaultRankingTemplate().parse({"ranking": []}, {"A"})
