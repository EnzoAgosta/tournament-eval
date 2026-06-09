"""End-to-end smoke test of the tournament evaluation framework."""

import asyncio
import uuid

from tournament_eval import (
    GenerationTask,
    OllamaLLMClient,
    OllamaModelConfig,
    build_ranking_tasks,
    generate_all,
    rank_all,
)


async def main() -> None:
    clients = [
        OllamaLLMClient(OllamaModelConfig(model_name="gemma4:e2b")),
        OllamaLLMClient(OllamaModelConfig(model_name="gemma4:e4b")),
    ]

    sentences = [
        "The quick brown fox jumps over the lazy dog.",
        "Hello world!",
    ]

    tasks = [
        GenerationTask(
            id=uuid.uuid4(),
            generation_prompt=f"Translate this sentence to French:\n{sentence}",
        )
        for sentence in sentences
    ]

    print("=== Phase 1: Generation ===")
    generation_results, gen_failures = await generate_all(tasks, clients)
    print(f"  Generated: {len(generation_results)}")
    if gen_failures:
        print(f"  Failures:  {len(gen_failures)}")
        for (task_id, client_name), err in gen_failures.items():
            print(f"    - {client_name} on task {task_id}: {err}")

    print()
    print("=== Phase 2: Build Ranking Tasks ===")
    ranking_tasks = build_ranking_tasks(
        tasks=tasks,
        generation_results=generation_results,
        ranking_prompt="Rank these French translations by fluency and accuracy, best first.",
    )
    print(f"  Built: {len(ranking_tasks)} ranking tasks")

    print()
    print("=== Phase 3: Ranking ===")
    ranking_results, rank_failures = await rank_all(
        ranking_tasks=ranking_tasks,
        generation_results=generation_results,
        clients=clients,
    )
    print(f"  Rankings:  {len(ranking_results)}")
    if rank_failures:
        print(f"  Failures:  {len(rank_failures)}")
        for (task_id, client_name), err in rank_failures.items():
            print(f"    - {client_name} on ranking task {task_id}: {err}")

    print()
    print("=== Results ===")
    for r in ranking_results:
        print(f"\n  Judge:    {r.author}")
        print(f"  Aliases:  {r.raw_model_ranking}")
        print(f"  UUIDs:    {[str(x)[:8] + '...' for x in r.ranking]}")
        if r.reasoning:
            snippet = r.reasoning[:200].replace("\n", " ")
            print(f"  Reasoning: {snippet}...")


if __name__ == "__main__":
    asyncio.run(main())
