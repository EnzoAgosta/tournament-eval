"""Tests for the base contract: StructuredResponse and concurrency_guard."""

import asyncio
import contextlib

from tournament_eval.llm.base import concurrency_guard, resolve_semaphore


def test_resolve_semaphore_returns_given_semaphore() -> None:
    semaphore = asyncio.Semaphore(1)
    assert resolve_semaphore(semaphore, 1) == semaphore


def test_resolve_semaphore_returns_semaphore_with_max_concurrency() -> None:
    semaphore = resolve_semaphore(None, 2)
    assert isinstance(semaphore, asyncio.Semaphore)
    assert semaphore._value == 2


def test_resolve_semaphore_returns_none_for_unlimited_concurrency() -> None:
    assert resolve_semaphore(None, None) is None


def test_concurrency_guard_returns_semaphore() -> None:
    semaphore = asyncio.Semaphore(1)
    assert concurrency_guard(semaphore) == semaphore


def test_concurrency_guard_returns_null_context_manager() -> None:
    assert isinstance(concurrency_guard(None), contextlib.nullcontext)
