"""Testes do supervisor de tasks asyncio do motor."""

from __future__ import annotations

import asyncio

import pytest

from src.application.services.orchestrator.engine_supervisor import (
    EngineTaskRegistry,
    cancel_background_tasks,
    cancel_task_quiet,
    registry_of,
    run_task_group,
    shield_critical,
    spawn_background,
)


@pytest.mark.asyncio
async def test_registry_spawn_and_cancel_all():
    orch = type("O", (), {})()
    reg = registry_of(orch)
    started = asyncio.Event()
    hang = asyncio.Event()

    async def worker() -> None:
        started.set()
        await hang.wait()

    task = reg.spawn(worker(), name="t1")
    await started.wait()
    assert not task.done()
    await reg.cancel_all()
    assert task.done()


@pytest.mark.asyncio
async def test_spawn_background_tracks_on_orch():
    orch = type("O", (), {})()

    async def noop() -> None:
        return None

    task = spawn_background(orch, noop(), name="noop")
    await task
    await cancel_background_tasks(orch)


@pytest.mark.asyncio
async def test_run_task_group_propagates_exception_group():
    async def ok() -> None:
        await asyncio.sleep(0.01)

    async def boom() -> None:
        raise RuntimeError("fail-worker")

    with pytest.raises(ExceptionGroup) as excinfo:
        await run_task_group(ok, boom)
    assert any(isinstance(e, RuntimeError) for e in excinfo.value.exceptions)


@pytest.mark.asyncio
async def test_shield_critical_and_cancel_quiet():
    async def work() -> int:
        await asyncio.sleep(0.01)
        return 7

    assert await shield_critical(work()) == 7
    await cancel_task_quiet(None)

    async def noop() -> None:
        return None

    done = asyncio.create_task(noop())
    await done
    await cancel_task_quiet(done)

    reg = EngineTaskRegistry()

    async def hang() -> None:
        await asyncio.Event().wait()

    task = reg.spawn(hang(), name="hang")
    await cancel_task_quiet(task)
    assert task.done()
    assert registry_of(type("O2", (), {"_engine_task_registry": reg})()) is reg
