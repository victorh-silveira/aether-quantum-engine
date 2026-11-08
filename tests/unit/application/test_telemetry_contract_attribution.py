"""Vinculo da probabilidade do lado executado ao contrato liquidado."""

from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock

import pytest

from src.application.services.orchestrator.execution_manager_execute import execute_cluster_orders
from src.domain.models.trade import TradeDirection
from src.infrastructure.telemetry.otel_business_instrumentor import BusinessMetricsInstrumentor


@pytest.mark.asyncio
async def test_put_execution_tracks_side_probability_by_contract():
    """O Brier posterior usa P(PUT) da compra confirmada."""
    risk = MagicMock()
    risk.kelly_config = {}
    risk.pending_loss = {}
    risk.calculate_stake.return_value = 10.0
    risk.active_contract_ids = []
    metrics = BusinessMetricsInstrumentor()
    orch = SimpleNamespace(
        _active_cycle_id=7,
        _contract_cycle={},
        business_metrics=metrics,
        config={"deep_learning": {}, "risk_management": {"params": {"duration": 5}}},
        risk_manager=risk,
        state=SimpleNamespace(add_contract=AsyncMock()),
    )
    executor = MagicMock(orch=orch)
    executor._mandatory_trade_each_cycle.return_value = False
    executor._place_order = AsyncMock(return_value=SimpleNamespace(contract_id=91, buy_price=10.0))

    count = await execute_cluster_orders(
        executor,
        [("1HZ75V", TradeDirection.PUT, {"calibrated_prob": 0.28, "raw_prob": 0.30, "trade_score": 0.72})],
        0.0,
        1000.0,
    )

    assert count == 1
    assert metrics._contract_predictions[91] == pytest.approx(0.72)
    metrics.record_trade("1HZ75V", "PUT", contract_id=91, won=False, profit=-10.0)
    assert metrics.compute_rolling_brier_score() == pytest.approx(0.72**2)
