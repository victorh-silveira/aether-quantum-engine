from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator.execution_collect_helpers import (
    mandatory_fallback_candidates,
    resolve_mandatory_ultimate_candidate,
)
from src.application.services.orchestrator.execution_manager import ExecutionManager
from src.application.services.orchestrator.execution_manager_execute import execute_cluster_orders
from src.domain.models.trade import TradeDirection


def test_mandatory_fallback_candidates_returns_build_fallback_when_present():
    orch = SimpleNamespace(
        config={"orchestrator": {"execution": {"force_trade_every_cycle": False}}},
        risk_manager=SimpleNamespace(consecutive_losses_linear=0),
        _quality_skipped_cycles_counter=0,
    )
    exec_mgr = SimpleNamespace(_trade_symbols=lambda: ["R_10"], orch=orch)
    fallback = ("R_10", TradeDirection.CALL, {"trade_score": 0.55})
    with (
        patch(
            "src.application.services.orchestrator.execution_collect_helpers.pick_entropy_fallback_candidate",
            return_value=None,
        ),
        patch(
            "src.application.services.orchestrator.execution_collect_helpers.build_mandatory_fallback_candidate",
            return_value=fallback,
        ),
    ):
        picks = mandatory_fallback_candidates(
            exec_mgr,
            {},
            recovery_active=False,
            last_loss_symbol=None,
            last_loss_direction=None,
            skip_symbols=frozenset(),
            min_signal=0.5,
            min_val=0.5,
        )
    assert picks == [fallback]


def test_mandatory_fallback_candidates_uses_force_trade_when_empty():
    orch = SimpleNamespace(
        config={"orchestrator": {"execution": {"force_trade_every_cycle": True}}},
        risk_manager=SimpleNamespace(consecutive_losses_linear=0),
        _quality_skipped_cycles_counter=0,
    )
    exec_mgr = SimpleNamespace(_trade_symbols=lambda: ["R_10"], orch=orch)
    decisions = {"R_10": {"direction": None, "metrics": {"raw_prob": 0.61, "calibrated_prob": 0.61}}}
    with (
        patch(
            "src.application.services.orchestrator.execution_collect_helpers.pick_entropy_fallback_candidate",
            return_value=None,
        ),
        patch(
            "src.application.services.orchestrator.execution_collect_helpers.build_mandatory_fallback_candidate",
            return_value=None,
        ),
    ):
        picks = mandatory_fallback_candidates(
            exec_mgr,
            decisions,
            recovery_active=False,
            last_loss_symbol=None,
            last_loss_direction=None,
            skip_symbols=frozenset(),
            min_signal=0.5,
            min_val=0.5,
        )
    assert len(picks) == 1
    assert picks[0][0] == "R_10"


def test_resolve_ultimate_mandatory_candidate_force_path():
    orch = SimpleNamespace(
        config={"orchestrator": {"execution": {"force_trade_every_cycle": True}}},
        risk_manager=SimpleNamespace(consecutive_losses_linear=0),
    )
    exec_mgr = SimpleNamespace(_trade_symbols=lambda: ["R_10"], orch=orch)
    decisions = {"R_10": {"direction": None, "metrics": {"raw_prob": 0.58}}}
    with (
        patch(
            "src.application.services.orchestrator.execution_collect_helpers.build_mandatory_fallback_candidate",
            return_value=None,
        ),
        patch(
            "src.application.services.orchestrator.execution_collect_helpers.pick_absolute_mandatory_candidate",
            return_value=None,
        ),
    ):
        ultimate, wrapped = resolve_mandatory_ultimate_candidate(
            exec_mgr,
            decisions,
            mandatory=True,
            recovery_active=False,
            last_loss=None,
            last_loss_dir=None,
            skip_symbols=frozenset(),
            min_signal=0.5,
            min_val=0.5,
            mean_reversion=True,
            low_accuracy=True,
        )
    assert ultimate is not None
    assert wrapped == [ultimate]


@pytest.mark.asyncio
async def test_execute_cluster_orders_force_min_stake_when_kelly_zero():
    risk = SimpleNamespace(
        kelly_config={},
        pending_loss={},
        calculate_stake=MagicMock(return_value=0.0),
        register_entry_conviction=MagicMock(),
        record_contract_stake=MagicMock(),
        active_contract_ids=[],
    )
    orch = SimpleNamespace(
        config={
            "orchestrator": {"execution": {"force_trade_every_cycle": True}},
            "deep_learning": {},
            "risk_management": {"params": {"stake_min": 0.5, "duration": 60}},
        },
        risk_manager=risk,
        _active_cycle_id=1,
        _contract_cycle={},
        state=SimpleNamespace(add_contract=AsyncMock()),
    )
    executor = SimpleNamespace(
        orch=orch,
        _mandatory_trade_each_cycle=lambda: False,
        _place_order=AsyncMock(return_value=SimpleNamespace(contract_id=99, buy_price=0.5)),
        _log_exec=MagicMock(),
        logger=MagicMock(),
    )
    count = await execute_cluster_orders(
        executor,
        [("R_10", TradeDirection.CALL, {"execute": True, "raw_prob": 0.55})],
        0.0,
        100.0,
    )
    assert count == 1
    assert risk.calculate_stake.called


def test_cluster_stake_block_bypassed_by_force_trade():
    orch = SimpleNamespace(
        config={"orchestrator": {"execution": {"force_trade_every_cycle": True}}},
        risk_manager=SimpleNamespace(kelly_config={}, stake_block_reason=MagicMock(return_value="blocked")),
        _active_cycle_id=1,
    )
    executor = ExecutionManager.__new__(ExecutionManager)
    executor.orch = orch
    executor.logger = MagicMock()
    orders = [("R_10", TradeDirection.CALL, {"conviction": 0.5})]
    assert executor._cluster_stake_block(orders, 100.0) is None
    orch.risk_manager.stake_block_reason.assert_not_called()
