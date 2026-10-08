import asyncio
from types import SimpleNamespace
from unittest.mock import patch

import pytest

from src.application.services.orchestrator.orchestrator_atomic_state import (
    _caller_function_name,
    orchestrator_atomic_state_context,
    orchestrator_balance_snapshot,
)
from src.application.services.orchestrator.settlement_logic import _sync_state_manager_session
from src.infrastructure.state.state_manager import StateManager


@pytest.mark.asyncio
async def test_atomic_state_context_serializes_concurrent_mutations(tmp_path):
    mgr = StateManager(file_path=tmp_path / "atomic.json")
    order: list[str] = []

    async def writer(label: str, balance: float) -> None:
        async with mgr.atomic_state_context():
            order.append(f"{label}-start")
            await asyncio.sleep(0.01)
            mgr.mirror_balance(balance)
            order.append(f"{label}-end")

    await asyncio.gather(writer("a", 1000.0), writer("b", 1100.0))
    assert order.index("a-start") < order.index("a-end")
    assert order.index("b-start") < order.index("b-end")
    assert (order.index("a-end") < order.index("b-start")) or (order.index("b-end") < order.index("a-start"))


@pytest.mark.asyncio
async def test_orchestrator_atomic_state_context_uses_state_manager_lock(orch_ready):
    orch = orch_ready
    orch.state.balance = 1234.5
    orch.state_mgr.mirror_balance(1234.5)
    async with orchestrator_atomic_state_context(orch):
        assert orch.state_mgr._state_lock.locked()
    assert not orch.state_mgr._state_lock.locked()


@pytest.mark.asyncio
async def test_orchestrator_atomic_state_context_times_out_on_deadlock(orch_ready, caplog):
    orch = orch_ready
    await orch.state_mgr._state_lock.acquire()
    with (
        patch(
            "src.application.services.orchestrator.orchestrator_atomic_state._state_lock_timeout",
            return_value=0.05,
        ),
        caplog.at_level("DEBUG", logger="AETH"),
        pytest.raises(RuntimeError, match="STATE_LOCK_TIMEOUT") as captured,
    ):
        async with orchestrator_atomic_state_context(orch):
            pass
    assert any("LOCK_TRACE" in record.message for record in caplog.records)
    assert "\n" in str(captured.value)
    orch.state_mgr._state_lock.release()


def test_caller_function_name_fallbacks():
    with patch(
        "src.application.services.orchestrator.orchestrator_atomic_state.inspect.currentframe",
        return_value=None,
    ):
        assert _caller_function_name() == "<unknown>"

    class _Frame:
        def __init__(self, name: str, back=None):
            self.f_back = back
            self.f_globals = {"__name__": name}
            self.f_code = SimpleNamespace(co_name="inner")

    root = _Frame("contextlib")
    with patch(
        "src.application.services.orchestrator.orchestrator_atomic_state.inspect.currentframe",
        return_value=root,
    ):
        assert _caller_function_name() == "<unknown>"


def test_orchestrator_balance_snapshot_prefers_cached_balance(orch_ready):
    orch = orch_ready
    orch.state_mgr.mirror_balance(1500.0)
    orch.state.balance = 900.0
    assert orchestrator_balance_snapshot(orch) == pytest.approx(1500.0)


def test_orchestrator_balance_snapshot_falls_back_to_trading_state(orch_ready):
    orch = orch_ready
    orch.state_mgr._balance_snapshot = 0.0
    orch.state.balance = 875.0
    assert orchestrator_balance_snapshot(orch) == pytest.approx(875.0)


def test_sync_state_manager_session_legacy_state_manager_branch(orch_ready):
    class StateManagerLegacy:
        def __init__(self):
            self.state = SimpleNamespace(
                current_balance=0.0,
                initial_balance=0.0,
                daily_stop_win_target=0.0,
                stop_win_triggered=False,
                total_trades_today=0,
            )

        def check_session_limits(self):
            return None

    StateManagerLegacy.__name__ = "StateManager"
    orch = orch_ready
    legacy_mgr = StateManagerLegacy()
    orch.state_mgr = legacy_mgr
    orch.state.balance = 1010.0
    orch.risk_manager.initial_bankroll = 1000.0
    triggered = _sync_state_manager_session(orch, 50.0, increment_trades=True)
    assert legacy_mgr.state.current_balance == pytest.approx(1010.0)
    assert legacy_mgr.state.total_trades_today == 1
    assert triggered is False
