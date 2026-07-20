from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest

from src.application.services.execution_quality_gate_fallback import cluster_quality_gate_blocks_mandatory_fallback
from src.application.services.orchestrator.trading_cycle_entry import run_trading_cycle_if_ready


TRADING_CYCLE_MODULE = "src.application.services.orchestrator.trading_cycle_entry"


def test_cluster_quality_gate_blocks_mandatory_fallback_when_all_viable_rejected():
    decisions = {
        "R_10": {
            "direction": "CALL",
            "metrics": {
                "calibrated_prob": 0.55,
                "quality_guard_reject": True,
                "deploy_ok": True,
                "meta_payoff_edge_zscore": -0.85,
                "edge_zscore": -0.85,
            },
        }
    }
    risk_manager = SimpleNamespace(consecutive_losses_linear=2, pending_loss={}, pending_loss_total=lambda: 0.0)
    assert cluster_quality_gate_blocks_mandatory_fallback(
        decisions,
        exec_cfg={},
        risk_manager=risk_manager,
        trade_symbols=["R_10"],
    )


def test_cluster_quality_gate_allows_soft_meta_zscore_reject():
    decisions = {
        "R_10": {
            "direction": "CALL",
            "metrics": {
                "calibrated_prob": 0.64,
                "quality_guard_reject": True,
                "deploy_ok": True,
                "execution_gate_state": "meta_zscore_reject",
                "meta_payoff_edge_zscore": 0.10,
                "edge_zscore": 0.10,
            },
        }
    }
    risk_manager = SimpleNamespace(
        consecutive_losses_linear=4,
        pending_loss={"R_10": 540.0},
        pending_loss_total=lambda: 540.0,
    )
    assert not cluster_quality_gate_blocks_mandatory_fallback(
        decisions,
        exec_cfg={},
        risk_manager=risk_manager,
        trade_symbols=["R_10"],
    )


def test_cluster_quality_gate_allows_mandatory_fallback_when_one_symbol_passes():
    decisions = {
        "R_10": {
            "direction": "CALL",
            "metrics": {"calibrated_prob": 0.55, "quality_guard_reject": True, "deploy_ok": True},
        },
        "R_50": {
            "direction": "PUT",
            "metrics": {"calibrated_prob": 0.70, "deploy_ok": True},
        },
    }
    risk_manager = SimpleNamespace(
        consecutive_losses_linear=0,
        pending_loss={"R_10": 1.0},
        pending_loss_total=lambda: 1.0,
    )
    assert not cluster_quality_gate_blocks_mandatory_fallback(
        decisions,
        exec_cfg={},
        risk_manager=risk_manager,
        trade_symbols=["R_10", "R_50"],
    )


def test_cluster_quality_gate_skips_non_viable_entries():
    decisions = {
        "R_10": {"direction": "CALL", "metrics": {"deploy_ok": False, "quality_guard_reject": True}},
        "R_50": {"direction": "PUT", "metrics": {"gate_reason": "data", "quality_guard_reject": True}},
        "ALT": {"metrics": "invalid"},
    }
    risk_manager = SimpleNamespace(consecutive_losses_linear=1, pending_loss={}, pending_loss_total=lambda: 0.0)
    assert not cluster_quality_gate_blocks_mandatory_fallback(
        decisions,
        exec_cfg={},
        risk_manager=risk_manager,
        trade_symbols=["R_10", "R_50", "ALT", "MISSING"],
    )


def test_cluster_quality_gate_counts_raw_prob_entry():
    decisions = {
        "R_10": {
            "metrics": {
                "raw_prob": 0.55,
                "quality_guard_reject": True,
                "deploy_ok": True,
                "meta_payoff_edge_zscore": -0.55,
            }
        }
    }
    risk_manager = SimpleNamespace(consecutive_losses_linear=1, pending_loss={}, pending_loss_total=lambda: 0.0)
    assert cluster_quality_gate_blocks_mandatory_fallback(
        decisions,
        exec_cfg={},
        risk_manager=risk_manager,
        trade_symbols=["R_10"],
    )


def test_cluster_quality_gate_allows_fallback_on_soft_tcn_margin_reject():
    decisions = {
        "R_10": {
            "direction": "PUT",
            "metrics": {
                "calibrated_prob": 0.52,
                "deploy_ok": True,
                "quality_guard_reject": True,
                "quality_gate_reason": "[TCN Margin 0.02 < min 0.04]",
                "meta_payoff_edge_zscore": 1.75,
                "edge_zscore": 1.75,
                "predicted_payoff_edge": 1.33,
            },
        }
    }
    risk_manager = SimpleNamespace(
        consecutive_losses_linear=1,
        pending_loss={"R_10": 38.56},
        pending_loss_total=lambda: 38.56,
    )
    assert not cluster_quality_gate_blocks_mandatory_fallback(
        decisions,
        exec_cfg={},
        risk_manager=risk_manager,
        trade_symbols=["R_10"],
    )


def test_cluster_quality_gate_blocks_soft_tcn_when_zscore_strongly_negative():
    decisions = {
        "R_10": {
            "direction": "CALL",
            "metrics": {
                "calibrated_prob": 0.52,
                "deploy_ok": True,
                "quality_guard_reject": True,
                "quality_gate_reason": "[TCN Margin 0.02 < min 0.04]",
                "meta_payoff_edge_zscore": -0.85,
                "edge_zscore": -0.85,
            },
        }
    }
    risk_manager = SimpleNamespace(
        consecutive_losses_linear=1,
        pending_loss={"R_10": 20.0},
        pending_loss_total=lambda: 20.0,
    )
    assert cluster_quality_gate_blocks_mandatory_fallback(
        decisions,
        exec_cfg={},
        risk_manager=risk_manager,
        trade_symbols=["R_10"],
    )


def test_cluster_quality_gate_allows_soft_tcn_reject_without_zscore():
    decisions = {
        "R_10": {
            "direction": "CALL",
            "metrics": {
                "calibrated_prob": 0.52,
                "deploy_ok": True,
                "quality_guard_reject": True,
                "quality_gate_reason": "[TCN Margin 0.02 < min 0.04]",
            },
        }
    }
    risk_manager = SimpleNamespace(
        consecutive_losses_linear=1,
        pending_loss={"R_10": 20.0},
        pending_loss_total=lambda: 20.0,
    )
    assert not cluster_quality_gate_blocks_mandatory_fallback(
        decisions,
        exec_cfg={},
        risk_manager=risk_manager,
        trade_symbols=["R_10"],
    )


def test_cluster_quality_gate_blocks_mandatory_fallback_ignores_non_dict_decisions():
    risk_manager = SimpleNamespace(consecutive_losses_linear=1, pending_loss={}, pending_loss_total=lambda: 0.0)
    assert not cluster_quality_gate_blocks_mandatory_fallback(
        [],
        exec_cfg={},
        risk_manager=risk_manager,
        trade_symbols=["R_10"],
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
        "R_10": {
            "metrics": {
                "calibrated_prob": 0.52,
                "predicted_payoff_edge": 0.01,
                "meta_classifier_applied": True,
                "meta_payoff_edge_zscore": 0.10,
            }
        },
        "R_50": {
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
