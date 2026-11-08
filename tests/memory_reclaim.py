"""Politicas de liberacao de memoria para execucao fragmentada da suite de testes."""

from __future__ import annotations

import asyncio
import ctypes
import gc
import logging
import os
import sys
from pathlib import Path
from typing import TYPE_CHECKING
from unittest.mock import patch

from tests.torch_test_support import release_torch_heap


if TYPE_CHECKING:
    import pytest


_GC_BATCH_SIZE = max(1, int(os.environ.get("AETHER_GC_BATCH_SIZE", "10")))
_XDIST_GB_PER_WORKER = float(os.environ.get("AETHER_XDIST_GB_PER_WORKER", "1.5"))
_tests_since_trim = {"count": 0}


def available_ram_gb() -> float | None:
    if sys.platform != "linux":
        return None
    try:
        with Path("/proc/meminfo").open(encoding="utf-8") as meminfo:
            for line in meminfo:
                if line.startswith("MemAvailable:"):
                    kb = int(line.split()[1])
                    return kb / (1024 * 1024)
    except OSError:
        return None
    return None


def ram_based_xdist_workers() -> int:
    avail = available_ram_gb()
    cpu_cap = os.cpu_count() or 2
    if avail is None:
        return max(1, min(2, cpu_cap))
    workers = int(avail // _XDIST_GB_PER_WORKER)
    return max(1, min(cpu_cap, workers))


def release_os_memory() -> None:
    if sys.platform == "linux":
        try:
            ctypes.CDLL("libc.so.6").malloc_trim(0)
        except (OSError, AttributeError):
            return
    elif sys.platform == "win32":
        try:
            ctypes.windll.kernel32.SetProcessWorkingSetSize(-1, -1, -1)
        except (OSError, AttributeError):
            return


def _drain_asyncio_loop() -> None:
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        return
    if loop.is_closed():
        return
    current = asyncio.current_task(loop)
    pending = [task for task in asyncio.all_tasks(loop) if task is not current and not task.done()]
    for task in pending:
        task.cancel()


def _clear_item_fixture_defs(item: pytest.Item) -> None:
    fixturedefs = getattr(item, "_fixturedefs", None)
    if fixturedefs is not None:
        fixturedefs.clear()
    request = getattr(item, "_request", None)
    if request is not None:
        request_fixturedefs = getattr(request, "_fixturedefs", None)
        if request_fixturedefs is not None:
            request_fixturedefs.clear()


def _clear_item_references(item: pytest.Item) -> None:
    for attr in ("user_properties", "_report_sections"):
        container = getattr(item, attr, None)
        if container is not None and hasattr(container, "clear"):
            container.clear()
    stash = getattr(item, "stash", None)
    if stash is not None and hasattr(stash, "clear"):
        stash.clear()


def _stop_active_patches() -> None:
    patch.stopall()


def _clear_logging_records() -> None:
    for logger in (logging.root, logging.getLogger("pytest")):
        for handler in logger.handlers:
            if hasattr(handler, "records"):
                handler.records.clear()
            buffer = getattr(handler, "buffer", None)
            if buffer is not None and hasattr(buffer, "clear"):
                buffer.clear()


def run_per_test_reclaim(item: pytest.Item) -> None:
    _clear_item_fixture_defs(item)
    _clear_item_references(item)
    _stop_active_patches()
    _drain_asyncio_loop()
    _clear_logging_records()
    release_torch_heap()

    gc.collect()
    gc.garbage.clear()

    _tests_since_trim["count"] += 1
    if _tests_since_trim["count"] < _GC_BATCH_SIZE:
        return

    _tests_since_trim["count"] = 0
    gc.collect(2)
    gc.garbage.clear()
    release_os_memory()


def run_session_finish_reclaim() -> None:
    gc.collect(2)
    gc.garbage.clear()
    release_os_memory()
