"""Configurações e fixtures compartilhadas para suíte de testes."""

import asyncio
import gc
from unittest.mock import patch

import pytest
import torch

from src.application.services.deep_learning import dl_device
from src.application.services.orchestrator.execution_manager import ExecutionManager
from src.infrastructure.state.trading_state import TradingState


_TORCH_CPU = torch.device("cpu")
_TORCH_DEVICE_PATCHES = (
    "src.application.services.deep_learning.dl_training.resolve_torch_device",
    "src.application.services.deep_learning.dl_symbol_runtime.resolve_torch_device",
    "src.application.services.deep_learning.dl_symbol_train.resolve_torch_device",
)


@pytest.fixture(autouse=True)
def reset_trading_state():
    """Redefine o singleton TradingState antes de cada teste."""
    TradingState.reset()


@pytest.fixture(scope="session", autouse=True)
def force_dl_training_on_cpu():
    """Forca treino DL em CPU na sessao inteira sem limpar CUDA a cada teste."""
    dl_device._DEVICE_LOGGED.clear()
    active = [patch(target, return_value=_TORCH_CPU) for target in _TORCH_DEVICE_PATCHES]
    for item in active:
        item.start()
    yield
    for item in active:
        item.stop()
    dl_device._DEVICE_LOGGED.clear()
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


@pytest.fixture(autouse=True)
def noop_background_settlement_watch(request):
    """Evita settlement watch em background travar a suíte durante execute_cluster."""
    if request.node.get_closest_marker("real_settlement_watch"):
        yield
        return

    async def _noop(_self) -> None:
        return

    with patch.object(ExecutionManager, "_run_settlement_watch", _noop):
        yield


@pytest.fixture(autouse=True)
async def cancel_leftover_async_tasks():
    """Cancela tasks asyncio orfas (ex.: settlement watch) ao final de cada teste."""
    yield
    current = asyncio.current_task()
    pending = [task for task in asyncio.all_tasks() if task is not current and not task.done()]
    for task in pending:
        task.cancel()
    if pending:
        await asyncio.gather(*pending, return_exceptions=True)
