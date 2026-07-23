from types import SimpleNamespace
from unittest.mock import patch

from src.application.services.execution_market_rank import market_decision_score
from src.application.services.live_signal_metrics import (
    apply_live_calib_drift_soft,
    attach_live_signal_metrics,
    live_signal_snapshot,
    record_live_signal_outcome,
    reset_live_signal_metrics,
)
from src.application.services.meta_payoff_shadow import (
    meta_hard_veto_allowed,
    meta_inverted_shadow_active,
    record_meta_payoff_shadow_pair,
    reset_meta_payoff_shadow,
    shadow_ready,
)
from src.application.services.meta_payoff_veto_gate import should_veto_meta_payoff_negative_zscore
from src.domain.models.trade import TradeDirection


def _assert_export_mae_gap(train_mae: float, val_mae: float, *, max_gap: float = 1.25) -> None:
    train = max(float(train_mae), 1e-9)
    ratio = float(val_mae) / train
    if ratio > float(max_gap) + 1e-12:
        raise RuntimeError(f"gap={ratio}")


def test_live_signal_metrics_rolling_wr_brier():
    orch = SimpleNamespace()
    reset_live_signal_metrics(orch)
    for i in range(24):
        record_live_signal_outcome(
            orch,
            "R_10",
            won=True,
            raw_prob=0.70 if i % 2 == 0 else 0.30,
            direction="CALL" if i % 2 == 0 else "PUT",
        )
    snap = live_signal_snapshot(orch, "R_10")
    assert snap["live_n"] == 24
    assert snap["live_wr"] == 1.0
    assert snap["live_brier"] < 0.2
    metrics = {}
    attach_live_signal_metrics(orch, "R_10", metrics)
    assert metrics["live_n"] == 24
    reset_live_signal_metrics(orch)


def test_market_decision_score_prefers_live_wr():
    base = {
        "raw_prob": 0.80,
        "val_accuracy": 0.90,
        "edge": 0.30,
        "execute": True,
        "deploy_ok": True,
        "direction_margin": 0.12,
        "deploy_settlement_win_rate": 0.40,
        "live_n": 24,
        "live_wr": 0.80,
        "live_brier": 0.18,
        "live_ece": 0.04,
    }
    low_live = dict(base)
    low_live["live_wr"] = 0.35
    assert market_decision_score(base) > market_decision_score(low_live)


def test_calib_drift_soft_penalty():
    metrics = {
        "live_n": 48,
        "live_ece": 0.20,
        "live_wr": 0.40,
        "raw_prob": 0.85,
        "trade_score": 0.85,
    }
    orch = SimpleNamespace(_active_cycle_id=3, _log_dedupe={})
    with patch("src.application.services.live_signal_metrics.logger") as mock_logger:
        assert apply_live_calib_drift_soft(metrics, orch=orch, symbol="R_10") is True
        assert apply_live_calib_drift_soft(dict(metrics), orch=orch, symbol="R_10") is True
        assert apply_live_calib_drift_soft(dict(metrics), orch=None, symbol="R_10") is True
    assert mock_logger.info.call_count == 1
    assert metrics["calib_drift_soft"] is True
    assert metrics["calib_drift_soft_penalty"] > 0.0
    assert metrics["trade_score"] < 0.85


def test_meta_hard_veto_requires_shadow():
    reset_meta_payoff_shadow()
    orch = SimpleNamespace()
    metrics = {
        "predicted_payoff_edge": -0.05,
        "edge_expectancy": "LOSS_EXPECTED",
        "trade_score": 0.80,
        "meta_payoff_edge_zscore": -0.90,
        "edge_zscore": -0.90,
        "edge_zscore_samples": 20,
        "edge_zscore_window": 15,
        "raw_prob": 0.80,
    }
    assert should_veto_meta_payoff_negative_zscore(metrics, direction=TradeDirection.CALL, orch=orch) is False
    assert metrics.get("meta_veto_mode") == "soft"
    for i in range(64):
        record_meta_payoff_shadow_pair(z_score=float(i) * 0.1, profit=float(i), orch=orch)
    assert shadow_ready(orch) is True
    assert meta_hard_veto_allowed(orch) is True
    metrics2 = dict(metrics)
    hard = should_veto_meta_payoff_negative_zscore(metrics2, direction=TradeDirection.CALL, orch=orch)
    assert hard is True
    assert metrics2.get("gate_reason") == "meta_payoff_negative_zscore_veto"
    reset_meta_payoff_shadow()


def test_meta_inverted_shadow_hard_vetoes_high_z():
    reset_meta_payoff_shadow()
    orch = SimpleNamespace()
    for i in range(16):
        record_meta_payoff_shadow_pair(z_score=float(i), profit=-float(i), orch=orch)
    assert meta_inverted_shadow_active(orch) is True
    metrics = {
        "predicted_payoff_edge": 0.05,
        "edge_expectancy": "WIN_EXPECTED",
        "trade_score": 0.80,
        "meta_payoff_edge_zscore": 1.20,
        "edge_zscore": 1.20,
        "edge_zscore_samples": 20,
        "edge_zscore_window": 15,
        "raw_prob": 0.70,
    }
    hard = should_veto_meta_payoff_negative_zscore(metrics, direction=TradeDirection.PUT, orch=orch)
    assert hard is True
    assert metrics.get("gate_reason") == "meta_shadow_inverted_veto"
    assert metrics.get("meta_shadow_inverted") is True
    reset_meta_payoff_shadow()


def test_meta_inverted_shadow_soft_in_recovery_with_positive_edge():
    reset_meta_payoff_shadow()
    orch = SimpleNamespace()
    for i in range(16):
        record_meta_payoff_shadow_pair(z_score=float(i), profit=-float(i), orch=orch)
    metrics = {
        "predicted_payoff_edge": 0.08,
        "edge_expectancy": "WIN_EXPECTED",
        "trade_score": 0.80,
        "meta_payoff_edge_zscore": 1.20,
        "edge_zscore": 1.20,
        "edge_zscore_samples": 20,
        "edge_zscore_window": 15,
        "raw_prob": 0.48,
    }
    risk = SimpleNamespace(
        consecutive_losses_linear=2,
        pending_loss={"R_10": 40.0},
        pending_loss_total=lambda: 40.0,
    )
    hard = should_veto_meta_payoff_negative_zscore(
        metrics,
        direction=TradeDirection.PUT,
        orch=orch,
        risk_manager=risk,
    )
    assert hard is False
    assert metrics.get("meta_shadow_inverted_recovery_soft") is True
    assert metrics.get("meta_veto_mode") == "soft"
    assert metrics.get("gate_reason") != "meta_shadow_inverted_veto"
    reset_meta_payoff_shadow()


def test_meta_inverted_shadow_soft_via_recovery_flag_without_risk_manager():
    reset_meta_payoff_shadow()
    orch = SimpleNamespace()
    for i in range(16):
        record_meta_payoff_shadow_pair(z_score=float(i), profit=-float(i), orch=orch)
    metrics = {
        "predicted_payoff_edge": 0.09,
        "edge_expectancy": "WIN_EXPECTED",
        "trade_score": 0.80,
        "meta_payoff_edge_zscore": 1.20,
        "edge_zscore": 1.20,
        "edge_zscore_samples": 20,
        "edge_zscore_window": 15,
        "raw_prob": 0.48,
    }
    hard = should_veto_meta_payoff_negative_zscore(
        metrics,
        direction=TradeDirection.PUT,
        orch=orch,
        recovery_active=True,
    )
    assert hard is False
    assert metrics.get("meta_shadow_inverted_recovery_soft") is True
    assert metrics.get("meta_recovery_active") is True
    reset_meta_payoff_shadow()


def test_assert_export_mae_gap_blocks_overfit():
    _assert_export_mae_gap(1.0, 1.20, max_gap=1.25)
    try:
        _assert_export_mae_gap(1.0, 1.50, max_gap=1.25)
        raised = False
    except RuntimeError:
        raised = True
    assert raised is True


def test_live_signal_snapshot_empty_symbol():
    orch = SimpleNamespace(_live_signal_metrics={})
    snap = live_signal_snapshot(orch, "MISSING")
    assert snap["live_n"] == 0
    assert snap["live_brier"] == 1.0
    orch2 = SimpleNamespace()
    assert live_signal_snapshot(orch2, "X")["live_n"] == 0
