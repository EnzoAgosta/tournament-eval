"""Integration tests for the orchestration layer using a mock LLM client."""

import uuid
from collections.abc import Callable

from tests.conftest import MockLLMClient
from tournament_eval.models import (
    GenerationResult,
    GenerationTask,
    RankingTask,
)
from tournament_eval.orchestration import (
    _build_generation_lookup,
    _build_judge_prompt,
    generate_all,
    rank_all,
)


class TestGenerateAll:
    async def test_generates_for_all_tasks_and_clients(
        self,
        make_client: Callable[..., MockLLMClient],
        make_task: Callable[[str], GenerationTask],
    ) -> None:
        tasks = [make_task("p1"), make_task("p2")]
        clients = [
            make_client("a", generate_responses={"p1": "r1a", "p2": "r2a"}),
            make_client("b", generate_responses={"p1": "r1b", "p2": "r2b"}),
        ]

        results, failures = await generate_all(tasks, clients)

        assert len(results) == 4
        assert len(failures) == 0

        authors = {r.author for r in results}
        assert authors == {"a", "b"}

        by_task = {r.task_id: r.output for r in results}
        for t in tasks:
            assert t.id in by_task

    async def test_returns_failures(
        self,
        make_client: Callable[..., MockLLMClient],
        make_task: Callable[[str], GenerationTask],
    ) -> None:
        tasks = [make_task("p1"), make_task("p2")]
        clients = [
            make_client(
                "a",
                generate_responses={"p1": "ok", "p2": "ok"},
                fail_on={"p2"},
            ),
        ]

        results, failures = await generate_all(tasks, clients)

        assert len(results) == 1
        assert len(failures) == 1

    async def test_all_failures(
        self,
        make_client: Callable[..., MockLLMClient],
        make_task: Callable[[str], GenerationTask],
    ) -> None:
        tasks = [make_task("p1")]
        clients = [
            make_client("a", generate_responses={}, fail_on={"p1"}),
        ]

        results, failures = await generate_all(tasks, clients)

        assert len(results) == 0
        assert len(failures) == 1


class TestRankAll:
    async def test_ranks_for_all_tasks_and_clients(
        self,
        make_client: Callable[..., MockLLMClient],
    ) -> None:
        gen1 = GenerationResult(
            id=uuid.uuid4(),
            task_id=uuid.uuid4(),
            generation_prompt="p",
            raw_response="r1",
            output="hello",
            author="a",
        )
        gen2 = GenerationResult(
            id=uuid.uuid4(),
            task_id=gen1.task_id,
            generation_prompt="p",
            raw_response="r2",
            output="world",
            author="b",
        )

        ranking_task = RankingTask(
            id=uuid.uuid4(),
            ranking_prompt="Rank.",
            generations={"A": gen1.id, "B": gen2.id},
        )

        prompt = _build_judge_prompt(
            ranking_task,
            _build_generation_lookup([gen1, gen2]),
        )

        clients = [
            make_client(
                "judge1",
                structured_responses={
                    prompt: {"ranking": ["B", "A"], "reasoning": "B wins"},
                },
            ),
            make_client(
                "judge2",
                structured_responses={
                    prompt: {"ranking": ["A", "B"]},
                },
            ),
        ]

        results, failures = await rank_all(
            [ranking_task],
            [gen1, gen2],
            clients,
        )

        assert len(results) == 2
        assert len(failures) == 0

        judge_names = {r.author for r in results}
        assert judge_names == {"judge1", "judge2"}

        for r in results:
            assert r.ranking_task_id == ranking_task.id
            assert set(r.raw_model_ranking) == {"A", "B"}
            assert len(r.ranking) == 2

    async def test_returns_parsing_failures(
        self,
        make_client: Callable[..., MockLLMClient],
    ) -> None:
        gen1 = GenerationResult(
            id=uuid.uuid4(),
            task_id=uuid.uuid4(),
            generation_prompt="p",
            raw_response="r1",
            output="hello",
            author="a",
        )

        ranking_task = RankingTask(
            id=uuid.uuid4(),
            ranking_prompt="Rank.",
            generations={"A": gen1.id},
        )

        prompt = _build_judge_prompt(
            ranking_task,
            _build_generation_lookup([gen1]),
        )

        client = make_client(
            "bad-judge",
            structured_responses={
                # Missing alias "A" → validation failure
                prompt: {"ranking": []},
            },
        )

        results, failures = await rank_all(
            [ranking_task],
            [gen1],
            [client],
        )

        assert len(results) == 0
        assert len(failures) == 1

    async def test_returns_client_failure(
        self,
        make_client: Callable[..., MockLLMClient],
    ) -> None:
        gen1 = GenerationResult(
            id=uuid.uuid4(),
            task_id=uuid.uuid4(),
            generation_prompt="p",
            raw_response="r1",
            output="hello",
            author="a",
        )

        ranking_task = RankingTask(
            id=uuid.uuid4(),
            ranking_prompt="Rank.",
            generations={"A": gen1.id},
        )

        prompt = _build_judge_prompt(
            ranking_task,
            _build_generation_lookup([gen1]),
        )

        client = make_client(
            "failing-judge",
            structured_responses={},
            fail_on={prompt},
        )

        results, failures = await rank_all(
            [ranking_task],
            [gen1],
            [client],
        )

        assert len(results) == 0
        assert len(failures) == 1
