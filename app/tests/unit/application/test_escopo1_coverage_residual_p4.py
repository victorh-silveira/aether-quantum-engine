"""Cobertura residual (parte 2) apos remocao dos vetos."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.orchestrator.execution_manager_execute import execute_cluster_orders
from src.domain.models.trade import TradeDirection


@pytest.mark.asyncio
async def test_execute_cluster_orders_force_and_reversal_stake():
    executor = MagicMock()
    executor.orch = SimpleNamespace(
        _active_cycle_id=1,
        config={
            "deep_learning": {"max_val_brier_execute": 0.28},
            "risk_management": {"params": {"duration": 120}, "kelly": {"neutral_bankroll_pct": 0.002}},
            "orchestrator": {"execution": {"force_trade_every_cycle": True}},
        },
        risk_manager=SimpleNamespace(
            kelly_config={"neutral_bankroll_pct": 0.002, "stop_win_kelly_enabled": True},
            pending_loss={},
            consecutive_losses_linear=2,
            pending_loss_total=lambda: 5.0,
            calculate_stake=MagicMock(return_value=0.0),
            register_entry_conviction=MagicMock(),
            record_contract_stake=MagicMock(),
            active_contract_ids=[],
        ),
    )
    executor._mandatory_trade_each_cycle = MagicMock(return_value=True)
    executor._place_order = AsyncMock(
        return_value=SimpleNamespace(contract_id=901, buy_price=2.5),
    )
    executor._log_exec = MagicMock()
    executor.orch.state = SimpleNamespace(add_contract=AsyncMock())
    executor.orch._contract_cycle = {}
    with (
        patch(
            "src.application.services.orchestrator.execution_manager_execute.force_trade_from_orch",
            return_value=True,
        ),
        patch(
            "src.application.services.orchestrator.execution_manager_execute.resolve_force_min_stake",
            return_value=2.5,
        ),
    ):
        count = await execute_cluster_orders(
            executor,
            [("R_10", TradeDirection.CALL, {"execute": True, "meta_feature_vector": [0.1] * 43})],
            0.0,
            1000.0,
        )
    assert count == 1
    assert "cid:901" in getattr(executor.orch, "_meta_clf_vectors", {})


@pytest.mark.asyncio
async def test_execute_cluster_orders_reversal_stake_floor():
    executor = MagicMock()
    executor.orch = SimpleNamespace(
        _active_cycle_id=1,
        config={
            "deep_learning": {"max_val_brier_execute": 0.28},
            "risk_management": {"params": {"duration": 120}, "kelly": {"neutral_bankroll_pct": 0.002}},
            "orchestrator": {"execution": {}},
        },
        risk_manager=SimpleNamespace(
            kelly_config={"neutral_bankroll_pct": 0.002, "stop_win_kelly_enabled": True},
            pending_loss={"R_10": 1.0},
            consecutive_losses_linear=2,
            pending_loss_total=lambda: 5.0,
            calculate_stake=MagicMock(return_value=0.0),
            register_entry_conviction=MagicMock(),
            record_contract_stake=MagicMock(),
            active_contract_ids=[],
        ),
    )
    executor._mandatory_trade_each_cycle = MagicMock(return_value=True)
    executor._place_order = AsyncMock(
        return_value=SimpleNamespace(contract_id=902, buy_price=3.0),
    )
    executor._log_exec = MagicMock()
    executor.orch.state = SimpleNamespace(add_contract=AsyncMock())
    executor.orch._contract_cycle = {}
    metrics = {"reversal_stake_floor": True, "execute": True}
    with patch(
        "src.application.services.orchestrator.execution_manager_execute.force_trade_from_orch",
        return_value=False,
    ):
        count = await execute_cluster_orders(
            executor,
            [("R_10", TradeDirection.CALL, metrics)],
            0.0,
            1000.0,
        )
    assert count == 1
