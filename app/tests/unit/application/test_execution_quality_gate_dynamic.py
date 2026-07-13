from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.application.services.execution_quality_gate_fallback import cluster_quality_gate_blocks_mandatory_fallback
from src.application.services.orchestrator.trading_cycle_entry import run_trading_cycle_if_ready


TRADING_CYCLE_MODULE = "src.application.services.orchestrator.trading_cycle_entry"


def test_cluster_quality_gate_blocks_mandatory_fallback_when_all_viable_rejected():
    decisions = {
        "RDBULL": {
            "direction": "CALL",
            "metrics": {"calibrated_prob": 0.55, "quality_guard_reject": True, "deploy_ok": True},
        }
    }
    risk_manager = SimpleNamespace(consecutive_losses_linear=2, pending_loss={}, pending_loss_total=lambda: 0.0)
    assert cluster_quality_gate_blocks_mandatory_fallback(
        decisions,
        exec_cfg={},
        risk_manager=risk_manager,
        trade_symbols=["RDBULL"],
    )


def test_cluster_quality_gate_allows_mandatory_fallback_when_one_symbol_passes():
    decisions = {
        "RDBULL": {
            "direction": "CALL",
            "metrics": {"calibrated_prob": 0.55, "quality_guard_reject": True, "deploy_ok": True},
        },
        "RDBEAR": {
            "direction": "PUT",
            "metrics": {"calibrated_prob": 0.70, "deploy_ok": True},
        },
    }
    risk_manager = SimpleNamespace(
        consecutive_losses_linear=0,
        pending_loss={"RDBULL": 1.0},
        pending_loss_total=lambda: 1.0,
    )
    assert not cluster_quality_gate_blocks_mandatory_fallback(
        decisions,
        exec_cfg={},
        risk_manager=risk_manager,
        trade_symbols=["RDBULL", "RDBEAR"],
    )


def test_cluster_quality_gate_skips_non_viable_entries():
    decisions = {
        "RDBULL": {"direction": "CALL", "metrics": {"deploy_ok": False, "quality_guard_reject": True}},
        "RDBEAR": {"direction": "PUT", "metrics": {"gate_reason": "data", "quality_guard_reject": True}},
        "ALT": {"metrics": "invalid"},
    }
    risk_manager = SimpleNamespace(consecutive_losses_linear=1, pending_loss={}, pending_loss_total=lambda: 0.0)
    assert not cluster_quality_gate_blocks_mandatory_fallback(
        decisions,
        exec_cfg={},
        risk_manager=risk_manager,
        trade_symbols=["RDBULL", "RDBEAR", "ALT", "MISSING"],
    )


def test_cluster_quality_gate_counts_raw_prob_entry():
    decisions = {"RDBULL": {"metrics": {"raw_prob": 0.55, "quality_guard_reject": True, "deploy_ok": True}}}
    risk_manager = SimpleNamespace(consecutive_losses_linear=1, pending_loss={}, pending_loss_total=lambda: 0.0)
    assert cluster_quality_gate_blocks_mandatory_fallback(
        decisions,
        exec_cfg={},
        risk_manager=risk_manager,
        trade_symbols=["RDBULL"],
    )


def test_cluster_quality_gate_blocks_mandatory_fallback_ignores_non_dict_decisions():
    risk_manager = SimpleNamespace(consecutive_losses_linear=1, pending_loss={}, pending_loss_total=lambda: 0.0)
    assert not cluster_quality_gate_blocks_mandatory_fallback(
        [],
        exec_cfg={},
        risk_manager=risk_manager,
        trade_symbols=["RDBULL"],
    )


@pytest.mark.asyncio
async def test_trading_cycle_executes_cluster_on_mandatory_quality_telemetry(orch_ready, caplog):
    orch = orch_ready
    orch.risk_manager.consecutive_losses_linear = 0
    orch.risk_manager.pending_loss_total = lambda: 0.0
    orch._last_cluster_cycle_end = 0.0
    orch.config.setdefault("orchestrator", {})["cycle_interval_seconds"] = 0
    orch.executor.execute_cluster = AsyncMock()
    weak_decisions = {
        "RDBULL": {
            "metrics": {
                "calibrated_prob": 0.52,
                "predicted_payoff_edge": 0.01,
                "meta_classifier_applied": True,
                "meta_payoff_edge_zscore": 0.10,
            }
        },
        "RDBEAR": {
            "metrics": {
                "calibrated_prob": 0.48,
                "predicted_payoff_edge": 0.01,
                "meta_classifier_applied": True,
                "meta_payoff_edge_zscore": 0.10,
            }
        },
    }
    with (
        patch(
            f"{TRADING_CYCLE_MODULE}.collect_deep_learning_decisions",
            new_callable=AsyncMock,
            return_value=weak_decisions,
        ),
        patch(f"{TRADING_CYCLE_MODULE}.mark_bar_processed", new_callable=AsyncMock),
        patch(f"{TRADING_CYCLE_MODULE}.await_quality_skip_yield", new_callable=AsyncMock),
        caplog.at_level("INFO", logger="AETH"),
    ):
        ran = await run_trading_cycle_if_ready(orch)
    assert ran is True
    orch.executor.execute_cluster.assert_awaited_once()
    assert orch._last_cycle_cluster_executed is True
