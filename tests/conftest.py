"""Configuracoes e fixtures compartilhadas para suite de testes Laya."""

import asyncio
from unittest.mock import patch

import numpy as np
import pytest

from src.application.services.orchestrator.execution_manager import ExecutionManager
from src.infrastructure.state.trading_state import TradingState


@pytest.fixture(autouse=True)
def fixed_numpy_random_seed():
    """Semente fixa para reproducibilidade."""
    np.random.seed(42)


@pytest.fixture(autouse=True)
def reset_trading_state():
    """Redefine o singleton TradingState antes de cada teste."""
    TradingState.reset()


@pytest.fixture(autouse=True)
def block_os_hard_exit():
    """Impede os._exit de encerrar o runner durante a suite de testes."""
    with patch("os._exit"):
        yield


@pytest.fixture(autouse=True)
def noop_background_settlement_watch(request):
    """Evita settlement watch em background travar testes de execucao."""
    if request.node.get_closest_marker("real_settlement_watch"):
        yield
        return

    async def _noop(_self) -> None:
        return

    with patch.object(ExecutionManager, "_run_settlement_watch", _noop):
        yield


@pytest.fixture
async def cancel_leftover_async_tasks():
    """Cancela tasks asyncio orfas ao final de cada teste."""
    yield
    current = asyncio.current_task()
    pending = [task for task in asyncio.all_tasks() if task is not current and not task.done()]
    for task in pending:
        task.cancel()
