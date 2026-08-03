from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.execution_price_zone_gate import align_or_keep_meta_side
from src.domain.models.trade import TradeDirection


def test_align_or_keep_meta_side_soft_drift_keeps_strong_tcn():
    metrics = {"calib_drift_soft": True, "direction_margin": 0.35, "calibrated_prob": 0.15}
    with patch(
        "src.application.services.execution_price_zone_meta.align_direction_to_price_zone",
        return_value=TradeDirection.CALL,
    ):
        assert (
            align_or_keep_meta_side(
                TradeDirection.PUT,
                metrics,
                dl_dir=TradeDirection.PUT,
                predicted_edge=0.2,
                meta_applied=True,
            )
            == TradeDirection.PUT
        )
    assert metrics.get("tcn_direction_lock") is True


def test_log_execution_blockers_forced_trade_mode():
    executor_mock = SimpleNamespace(
        orch=SimpleNamespace(
            _active_cycle_id=1,
            config={"orchestrator": {"execution": {"force_trade_every_cycle": True}}},
        ),
        _trade_symbols=lambda: ["R_100"],
        logger=MagicMock(),
    )
    from src.application.services.orchestrator.execution_blockers import log_execution_blockers

    assert log_execution_blockers(executor_mock, {}) is None


def test_log_execution_blockers_standard_flow():
    executor_mock = SimpleNamespace(
        orch=SimpleNamespace(_active_cycle_id=1, config={}, risk_manager=SimpleNamespace(consecutive_losses_linear=1)),
        _trade_symbols=lambda: ["R_100"],
        logger=MagicMock(),
    )
    from src.application.services.orchestrator.execution_blockers import log_execution_blockers

    log_execution_blockers(executor_mock, {}, pending=10.0)
    assert executor_mock.logger.info.called


def test_execution_manager_reversal_block_reason_branches():
    from src.application.services.orchestrator.execution_manager import ExecutionManager

    orch_blocked = SimpleNamespace(
        config={},
        _active_cycle_id=1,
        risk_manager=SimpleNamespace(
            kelly_config={},
            stake_block_reason=MagicMock(return_value="kelly_no_edge"),
            consecutive_losses_linear=0,
            pending_loss_total=lambda: 0.0,
        ),
    )
    manager = ExecutionManager(orch_blocked)
    orders = [("R_100", TradeDirection.CALL, {"raw_prob": 0.4, "trade_score": 0.6, "conviction": 0.6})]
    result = manager._cluster_stake_block(orders, 1000.0)
    assert result == "kelly_no_edge"
    assert orders[0][1] == TradeDirection.CALL

    orch_other = SimpleNamespace(
        config={},
        _active_cycle_id=1,
        risk_manager=SimpleNamespace(
            kelly_config={},
            stake_block_reason=MagicMock(return_value="other_reason"),
            consecutive_losses_linear=0,
            pending_loss_total=lambda: 0.0,
        ),
    )
    manager2 = ExecutionManager(orch_other)
    orders2 = [("R_100", TradeDirection.PUT, {"raw_prob": 0.6, "trade_score": 0.6, "conviction": 0.6})]
    assert manager2._cluster_stake_block(orders2, 1000.0) == "other_reason"
    assert orders2[0][1] == TradeDirection.PUT


def test_cluster_stake_block_never_flips_direction():
    from src.application.services.orchestrator.execution_manager import ExecutionManager

    orch = SimpleNamespace(
        config={},
        _active_cycle_id=1,
        risk_manager=SimpleNamespace(
            kelly_config={},
            stake_block_reason=MagicMock(side_effect=["kelly_no_edge", None]),
            consecutive_losses_linear=0,
            pending_loss_total=lambda: 0.0,
        ),
    )
    manager = ExecutionManager(orch)
    orders = [("R_10", TradeDirection.PUT, {"raw_prob": 0.35, "trade_score": 0.65, "conviction": 0.65})]
    assert manager._cluster_stake_block(orders, 1000.0) == "kelly_no_edge"
    assert orders[0][1] == TradeDirection.PUT
    assert orch.risk_manager.stake_block_reason.call_count == 1
    assert "flipped_from" not in orders[0][2]


@pytest.mark.asyncio
async def test_execution_manager_reversal_stake_floor():
    from src.application.services.orchestrator.execution_manager_execute import execute_cluster_orders

    orch_recovery = SimpleNamespace(
        config={},
        _active_cycle_id=1,
        risk_manager=SimpleNamespace(
            kelly_config={"neutral_bankroll_pct": 0.0015},
            consecutive_losses_linear=1,
            pending_loss={},
            pending_loss_total=lambda: 5.0,
            calculate_stake=lambda *a, **k: 0.0,
            register_entry_conviction=MagicMock(),
        ),
    )
    mock_place_order = AsyncMock(return_value=None)
    executor_mock = SimpleNamespace(
        orch=orch_recovery,
        _mandatory_trade_each_cycle=lambda: False,
        logger=MagicMock(),
        _flush_result_buffer=MagicMock(),
        _place_order=mock_place_order,
    )
    orders = [("R_100", TradeDirection.CALL, {"conviction": 0.6, "reversal_stake_floor": True})]
    await execute_cluster_orders(executor_mock, orders, 0.0, 1000.0)
    mock_place_order.assert_called_once()
