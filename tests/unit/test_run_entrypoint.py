"""Testes unitarios do ponto de entrada run.py."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

import run as run_entry


@pytest.mark.asyncio
async def test_main_returns_zero_on_graceful_stop_win(monkeypatch: pytest.MonkeyPatch) -> None:
    """Encerramento por stop_win nao deve logar erro nem retornar codigo 1."""
    logger = MagicMock()
    orch = SimpleNamespace(
        running=False,
        shutdown_reason="stop_win",
        run=AsyncMock(return_value=None),
        stop=AsyncMock(return_value=None),
    )
    monkeypatch.setattr(run_entry, "ensure_polars_max_threads", lambda: None)
    monkeypatch.setattr(run_entry, "load_engine_config", lambda: ({}, logger))
    monkeypatch.setattr(run_entry, "create_authenticated_auth", lambda _cfg, _log: object())
    monkeypatch.setattr(run_entry, "Orchestrator", lambda _cfg, _auth: orch)

    code = await run_entry.main()
    assert code == 0
    logger.error.assert_not_called()
    orch.stop.assert_awaited_once()


@pytest.mark.asyncio
async def test_main_returns_one_when_stopped_before_main_loop(monkeypatch: pytest.MonkeyPatch) -> None:
    """Encerramento antes do loop principal sem shutdown_reason deve logar erro e retornar 1."""
    logger = MagicMock()
    orch = SimpleNamespace(
        running=False,
        shutdown_reason=None,
        run=AsyncMock(return_value=None),
        stop=AsyncMock(return_value=None),
    )
    monkeypatch.setattr(run_entry, "ensure_polars_max_threads", lambda: None)
    monkeypatch.setattr(run_entry, "load_engine_config", lambda: ({}, logger))
    monkeypatch.setattr(run_entry, "create_authenticated_auth", lambda _cfg, _log: object())
    monkeypatch.setattr(run_entry, "Orchestrator", lambda _cfg, _auth: orch)

    code = await run_entry.main()
    assert code == 1
    logger.error.assert_called_once()
    orch.stop.assert_awaited_once()
