from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.application.services.deep_learning.decision_bridge import _maybe_schedule_training
from src.application.services.execution_entropy_fallback import pick_entropy_fallback_candidate
from src.application.services.execution_mandatory_pick import _symbol_order
from src.application.services.meta_classifier_cross_symbol import (
    attach_cross_symbol_features_to_decisions,
    compute_cross_symbol_triplet,
)
from src.application.services.orchestrator.trading_cycle_entry import _execute_inference_cluster_cycle
from src.domain.models.trade import TradeDirection


TRADING_CYCLE_MODULE = "src.application.services.orchestrator.trading_cycle_entry"


def _metrics(*, prob: float, rsi: float, vol_ratio: float) -> dict:
    return {
        "calibrated_prob": prob,
        "micro_indicators": {"rsi": rsi, "vol_ratio": vol_ratio},
    }


def _peer_for_primary(symbol: str) -> str | None:
    return "R_50" if symbol == "R_10" else None


def test_maybe_schedule_training_skips_non_first_bootstrap_symbol():
    orch = MagicMock()
    orch.symbols = ["R_10", "R_50"]
    runtime = {"deploy_ok": False}
    prices = np.linspace(1.0, 2.0, 32)
    dl_config = {}
    params = {"training_history_bars": 32}
    with (
        patch(
            "src.application.services.deep_learning.decision_bridge.should_retrain_symbol",
            return_value=(True, "bootstrap"),
        ),
        patch(
            "src.application.services.deep_learning.decision_bridge.enqueue_deferred_symbol_training"
        ) as mock_enqueue,
    ):
        reason = _maybe_schedule_training(
            orch,
            "R_50",
            runtime,
            prices,
            dl_config,
            params,
            100,
            600,
            frozenset({"R_10", "R_50"}),
            None,
            None,
            None,
            None,
        )
    assert reason is None
    mock_enqueue.assert_not_called()


def test_symbol_order_deprioritizes_last_loss_symbol_in_core():
    with patch(
        "src.application.services.execution_mandatory_pick.TRADING_SYMBOLS",
        ("R_10", "R_50"),
    ):
        order = _symbol_order(["R_10", "R_50", "R_75"], "R_10", skip_symbols=frozenset())
    assert order[0] == "R_50"
    assert "R_10" in order


def test_entropy_fallback_uses_raw_prob_when_calibrated_missing():
    entry = {
        "direction": None,
        "metrics": {
            "raw_prob": 0.82,
            "deploy_ok": True,
            "dynamic_call_threshold": 0.53,
            "dynamic_put_threshold": 0.47,
        },
    }
    picked = pick_entropy_fallback_candidate(["R_10"], {"R_10": entry})
    assert picked is not None
    assert picked[0] == "R_10"


@patch("src.application.services.execution_entropy_fallback.ANCHOR_BULL", "PEER_BULL")
@patch("src.application.services.execution_entropy_fallback.ANCHOR_BEAR", "PEER_BEAR")
@patch("src.application.services.execution_entropy_fallback.infer_dl_direction", return_value=None)
def test_entropy_fallback_uses_anchor_direction_when_dl_missing(_mock_infer):
    entry = {
        "direction": None,
        "metrics": {
            "raw_prob": 0.82,
            "deploy_ok": True,
            "dynamic_call_threshold": 0.53,
            "dynamic_put_threshold": 0.47,
        },
    }
    bull = pick_entropy_fallback_candidate(["PEER_BULL"], {"PEER_BULL": entry})
    bear = pick_entropy_fallback_candidate(["PEER_BEAR"], {"PEER_BEAR": entry})
    assert bull is not None and bull[1] == TradeDirection.CALL
    assert bear is not None and bear[1] == TradeDirection.PUT


def test_compute_cross_symbol_triplet_zeros_when_same_metrics_object():
    metrics = _metrics(prob=0.62, rsi=58.0, vol_ratio=1.05)
    triplet = compute_cross_symbol_triplet(metrics, metrics)
    assert all(value == 0.0 for value in triplet.values())


def test_attach_cross_symbol_features_with_configured_peer():
    decisions = {
        "R_10": {"metrics": _metrics(prob=0.66, rsi=60.0, vol_ratio=1.1)},
        "R_50": {"metrics": _metrics(prob=0.41, rsi=44.0, vol_ratio=0.92)},
    }
    with patch(
        "src.application.services.meta_classifier_cross_symbol.hedge_peer",
        side_effect=_peer_for_primary,
    ):
        attach_cross_symbol_features_to_decisions(decisions)
    spread = decisions["R_10"]["metrics"]["cross_symbol_features"]["cross_symbol_rsi_spread"]
    assert spread == pytest.approx(16.0)


@pytest.mark.asyncio
async def test_execute_inference_cluster_runs_without_quality_suspend(orch_ready):
    orch = orch_ready
    decisions = {
        "R_10": {
            "metrics": {
                "calibrated_prob": 0.9,
                "deploy_ok": True,
            }
        }
    }
    orch.executor.execute_cluster = AsyncMock(return_value=1)
    with (
        patch(
            f"{TRADING_CYCLE_MODULE}.collect_deep_learning_decisions",
            new_callable=AsyncMock,
            return_value=decisions,
        ),
        patch(f"{TRADING_CYCLE_MODULE}.session_persistence_blocks_trading_cycle", return_value=False),
        patch(f"{TRADING_CYCLE_MODULE}.await_regime_freeze_yield", new_callable=AsyncMock),
        patch(f"{TRADING_CYCLE_MODULE}.refresh_correlation_cache", new_callable=AsyncMock),
    ):
        executed = await _execute_inference_cluster_cycle(orch)
    assert executed is True
    orch.executor.execute_cluster.assert_awaited_once()
