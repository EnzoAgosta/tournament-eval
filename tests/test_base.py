"""Tests for the base contract: StructuredResponse and concurrency_guard."""

import asyncio
import contextlib

from tournament_eval.llm.base import StructuredResponse, concurrency_guard


def test_structured_response_holds_data_and_raw() -> None:
    response = StructuredResponse(data={"a": 1}, raw='{"a": 1}')
    assert response.data == {"a": 1}
    assert response.raw == '{"a": 1}'


async def test_guard_without_semaphore_is_a_noop_nullcontext() -> None:
    guard = concurrency_guard(None)
    assert isinstance(guard, contextlib.nullcontext)
    async with guard:
        pass  # must not raise


async def test_guard_with_semaphore_returns_that_semaphore() -> None:
    semaphore = asyncio.Semaphore(1)
    assert concurrency_guard(semaphore) is semaphore


async def test_guard_with_semaphore_acquires_and_releases() -> None:
    semaphore = asyncio.Semaphore(1)
    async with concurrency_guard(semaphore):
        assert semaphore.locked()  # the lone permit is held inside the guard
    assert not semaphore.locked()  # and released on exit
