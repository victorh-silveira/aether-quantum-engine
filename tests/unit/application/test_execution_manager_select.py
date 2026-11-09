import pytest

from src.application.services.execution_symbols import select_best_execution_candidate
from src.domain.models.trade import TradeDirection
from tests.market_symbols import ANCHOR, PAIR


def test_select_best_candidate_keeps_top_when_clear_lead():
    candidates = [
        (ANCHOR, TradeDirection.PUT, {"trade_score": 0.81, "raw_prob": 0.19, "val_accuracy": 0.50}),
        (PAIR, TradeDirection.CALL, {"trade_score": 0.64, "raw_prob": 0.64, "val_accuracy": 0.45}),
    ]
    best = select_best_execution_candidate(
        candidates,
        last_loss_symbol=ANCHOR,
        recovery_active=False,
    )
    assert best[0] == ANCHOR


def test_select_best_candidate_diversifies_after_loss():
    candidates = [
        (ANCHOR, TradeDirection.PUT, {"conviction": 0.67, "raw_conviction": 0.67, "val_accuracy": 0.40}),
        (PAIR, TradeDirection.CALL, {"conviction": 0.62, "raw_conviction": 0.64, "val_accuracy": 0.50}),
    ]
    best = select_best_execution_candidate(
        candidates,
        last_loss_symbol=ANCHOR,
        recovery_active=True,
    )
    assert best[0] == PAIR


def test_select_best_candidate_prefers_high_val_in_recovery():
    candidates = [
        (
            PAIR,
            TradeDirection.PUT,
            {"trade_score": 0.57, "conviction": 0.57, "val_accuracy": 0.60, "edge": 0.07},
        ),
        (
            ANCHOR,
            TradeDirection.CALL,
            {"trade_score": 0.59, "conviction": 0.59, "val_accuracy": 0.42, "edge": 0.09},
        ),
    ]
    best = select_best_execution_candidate(
        candidates,
        last_loss_symbol=None,
        recovery_active=True,
    )
    assert best[0] == PAIR


def test_select_best_candidate_single_candidate():
    candidates = [(ANCHOR, TradeDirection.PUT, {"conviction": 0.70, "raw_conviction": 0.70})]
    assert (
        select_best_execution_candidate(
            candidates,
            last_loss_symbol=ANCHOR,
            recovery_active=False,
        )[0]
        == ANCHOR
    )


@pytest.mark.asyncio
async def test_execution_manager_pullback_never_skip_flag():
    from unittest.mock import AsyncMock, MagicMock, patch

    from src.application.services.orchestrator.execution_manager import ExecutionManager

    orch = MagicMock()
    orch.config = {
        "pullback_strategy": {"enabled": True, "never_skip": False},
        "indicator_consensus": {"multi_period_trading": False},
        "orchestrator": {"execution": {}},
        "risk_management": {},
    }
    orch.state = MagicMock(balance=1000.0)
    orch.risk_manager = MagicMock(pending_loss={})
    orch._active_cycle_id = 1
    orch.logger = MagicMock()

    mgr = ExecutionManager(orch)

    with (
        patch(
            "src.application.services.orchestrator.execution_manager.select_period_orders",
            new_callable=AsyncMock,
            return_value=[],
        ),
        patch(
            "src.application.services.orchestrator.execution_manager.resolve_mandatory_fallback_order",
            new_callable=AsyncMock,
        ) as mock_fallback,
        patch(
            "src.application.services.orchestrator.execution_manager.prepare_recovery_skip_counter",
            new_callable=AsyncMock,
        ),
    ):
        await mgr.execute_cluster({})
        mock_fallback.assert_not_called()

        orch.config["pullback_strategy"]["never_skip"] = True
        await mgr.execute_cluster({})
        mock_fallback.assert_called_once()
