from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.application.services.orchestrator.execution_quality_skip_yield import (
    await_quality_skip_yield,
    quality_skip_yield_seconds,
    sanitize_quality_skip_decisions,
)
from src.application.services.orchestrator.orchestrator_data_signature import resolve_signature_boundary_seconds
from src.application.services.orchestrator.trading_cycle_entry import run_trading_cycle_if_ready


def test_quality_skip_yield_seconds_always_zero():
    orch = SimpleNamespace(config={"orchestrator": {"cycle_interval_seconds": 180}}, _last_cluster_cycle_end=0.0)
    assert quality_skip_yield_seconds(orch) == 0.0


def test_sanitize_quality_skip_decisions_strips_reject_metadata():
    decisions = {
        "RDBULL": {
            "metrics": {
                "quality_guard_reject": True,
                "quality_gate_reason": "[Meta Z-Score]",
                "regime_skip_cycle": True,
                "trade_score": 0.55,
            }
        }
    }
    sanitize_quality_skip_decisions(decisions)
    metrics = decisions["RDBULL"]["metrics"]
    assert "quality_guard_reject" not in metrics
    assert "quality_gate_reason" not in metrics
    assert "regime_skip_cycle" not in metrics
    assert metrics["trade_score"] == 0.55


def test_resolve_signature_boundary_seconds_uses_custom_boundary(orch_ready):
    orch = orch_ready
    orch.config.setdefault("orchestrator", {})["signature_boundary_seconds"] = 300
    assert resolve_signature_boundary_seconds(orch) == 300


def test_resolve_signature_boundary_seconds_defaults_when_config_missing():
    assert resolve_signature_boundary_seconds(SimpleNamespace(config=None)) == 300


def test_resolve_signature_boundary_seconds_defaults_when_boundary_invalid(orch_ready):
    orch = orch_ready
    orch.config.setdefault("orchestrator", {})["signature_boundary_seconds"] = "invalid"
    orch.config["orchestrator"]["cycle_interval_seconds"] = "bad"
    assert resolve_signature_boundary_seconds(orch) == 300


def test_resolve_signature_boundary_seconds_defaults_when_config_invalid():
    assert resolve_signature_boundary_seconds(SimpleNamespace(config={"orchestrator": "invalid"})) == 300


def test_sanitize_quality_skip_decisions_ignores_invalid_payload():
    sanitize_quality_skip_decisions([])
    sanitize_quality_skip_decisions({"RDBULL": "invalid", "RDBEAR": {"metrics": "invalid"}})


@pytest.mark.asyncio
async def test_await_quality_skip_yield_is_noop():
    orch = SimpleNamespace(config={"orchestrator": {}}, _last_cluster_cycle_end=0.0)
    delay = await await_quality_skip_yield(orch)
    assert delay == 0.0


@pytest.mark.asyncio
async def test_trading_cycle_skips_execution_on_quality_gate_without_yield(orch_ready):
    orch = orch_ready
    orch._last_cluster_cycle_end = 0.0
    orch.config.setdefault("orchestrator", {})["cycle_interval_seconds"] = 0
    orch.config.setdefault("orchestrator", {}).setdefault("execution", {})["mandatory_trade_each_cycle"] = False
    weak_decisions = {
        "RDBULL": {
            "direction": "CALL",
            "metrics": {
                "calibrated_prob": 0.51,
                "predicted_payoff_edge": 0.01,
                "meta_payoff_edge_zscore": 0.10,
                "edge_zscore_samples": 15,
                "deploy_ok": True,
            },
        },
    }
    with (
        patch(
            "src.application.services.orchestrator.trading_cycle_entry.collect_deep_learning_decisions",
            new_callable=AsyncMock,
            return_value=weak_decisions,
        ),
        patch("src.application.services.orchestrator.trading_cycle_entry.mark_bar_processed", new_callable=AsyncMock),
        patch(
            "src.application.services.orchestrator.trading_cycle_entry.await_quality_skip_yield",
            new_callable=AsyncMock,
            return_value=0.0,
        ) as skip_yield_mock,
        patch(
            "src.application.services.orchestrator.trading_cycle_entry.await_regime_freeze_yield",
            new_callable=AsyncMock,
        ) as freeze_yield_mock,
    ):
        orch.executor.execute_cluster = AsyncMock()
        ran = await run_trading_cycle_if_ready(orch)
    assert ran is True
    orch.executor.execute_cluster.assert_awaited_once()
    skip_yield_mock.assert_not_awaited()
    freeze_yield_mock.assert_awaited_once()
