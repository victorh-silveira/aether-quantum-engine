from src.application.services.orchestrator.execution_recovery_gate import (
    cluster_entry_eligible,
    effective_signal,
    recovery_min_signal,
    recovery_min_val_accuracy,
)
from src.domain.models.trade import TradeDirection


def test_cluster_entry_accepts_technically_ok_with_raw_prob():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "execute": True,
            "gate_reason": None,
            "trade_score": 0.52,
            "val_accuracy": 0.55,
            "raw_prob": 0.52,
            "deploy_ok": True,
        },
    }
    assert cluster_entry_eligible(
        entry,
        mandatory=True,
        recovery_active=True,
        min_signal=0.53,
        min_val=0.52,
    )


def test_recovery_min_signal_uses_recovery_floor_when_active():
    cfg = {
        "mandatory_min_trade_score": 0.53,
        "recovery_min_trade_score": 0.50,
        "recovery_force_min_trade_score": 0.46,
        "recovery_force_pending_min": 1.0,
    }
    assert recovery_min_signal(cfg, recovery_active=True) == 0.50
    assert recovery_min_signal(cfg, recovery_active=True, pending_total=50.0) == 0.46
    assert recovery_min_signal(cfg, recovery_active=True, pending_total=200.0) == 0.46
    assert recovery_min_signal(cfg, recovery_active=False) == 0.53


def test_recovery_min_signal_uses_mandatory_floor_when_inactive():
    cfg = {"mandatory_min_trade_score": 0.45}
    assert recovery_min_signal(cfg, recovery_active=True, pending_total=0.0) == 0.45
    assert recovery_min_signal(cfg, recovery_active=False) == 0.45


def test_recovery_min_signal_recovery_floor_below_mandatory():
    cfg = {"mandatory_min_trade_score": 0.53, "recovery_min_trade_score": 0.50}
    assert recovery_min_signal(cfg, recovery_active=True) == 0.50
    assert recovery_min_signal(cfg, recovery_active=False) == 0.53


def test_recovery_min_val_accuracy_default():
    assert recovery_min_val_accuracy({}) == 0.50


def test_recovery_min_signal_scaling_consecutive_losses():
    cfg = {
        "mandatory_min_trade_score": 0.45,
        "recovery_min_trade_score": 0.45,
    }
    assert recovery_min_signal(cfg, recovery_active=True, consecutive_losses=0) == 0.45
    assert recovery_min_signal(cfg, recovery_active=True, consecutive_losses=1) == 0.52
    assert recovery_min_signal(cfg, recovery_active=True, consecutive_losses=2) == 0.54
    assert recovery_min_signal(cfg, recovery_active=True, consecutive_losses=3) == 0.56
    assert recovery_min_signal(cfg, recovery_active=True, consecutive_losses=4) == 0.58
    assert recovery_min_signal(cfg, recovery_active=True, consecutive_losses=5) == 0.58


def test_recovery_min_signal_hurst_log_adjustment():
    cfg = {
        "mandatory_min_trade_score": 0.45,
        "recovery_min_trade_score": 0.64,
        "recovery_hurst_persistence_min": 0.58,
        "recovery_hurst_log_scale": 0.08,
    }
    base = recovery_min_signal(cfg, recovery_active=True, consecutive_losses=2, hurst=0.60)
    raised = recovery_min_signal(cfg, recovery_active=True, consecutive_losses=2, hurst=0.45)
    assert base == 0.64
    assert raised > base


def test_recovery_min_val_accuracy_scaling_consecutive_losses():
    cfg = {
        "recovery_min_val_accuracy": 0.50,
    }
    assert recovery_min_val_accuracy(cfg, consecutive_losses=0) == 0.50
    assert recovery_min_val_accuracy(cfg, consecutive_losses=1) == 0.50
    assert recovery_min_val_accuracy(cfg, consecutive_losses=2) == 0.52
    assert recovery_min_val_accuracy(cfg, consecutive_losses=3) == 0.53
    assert recovery_min_val_accuracy(cfg, consecutive_losses=4) == 0.55


def test_cluster_entry_rejects_technical_blocks_only():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "execute": False,
            "gate_reason": "predict_error",
            "trade_score": 0.55,
            "val_accuracy": 0.53,
            "deploy_ok": True,
        },
    }
    assert not cluster_entry_eligible(
        entry,
        mandatory=True,
        recovery_active=True,
        min_signal=0.53,
        min_val=0.52,
    )


def test_cluster_entry_eligible_recovery_deploy_ok_false():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "execute": True,
            "deploy_ok": False,
            "trade_score": 0.55,
            "val_accuracy": 0.53,
            "raw_prob": 0.55,
        },
    }
    assert not cluster_entry_eligible(
        entry,
        mandatory=True,
        recovery_active=True,
        min_signal=0.53,
        min_val=0.52,
    )


def test_cluster_entry_accepts_without_execute_flag_when_raw_prob_present():
    entry = {
        "direction": TradeDirection.PUT,
        "metrics": {
            "execute": False,
            "gate_reason": "trend_conflict",
            "trade_score": 0.58,
            "val_accuracy": 0.62,
            "raw_prob": 0.42,
            "deploy_ok": True,
        },
    }
    assert cluster_entry_eligible(
        entry,
        mandatory=True,
        recovery_active=False,
        min_signal=0.50,
        min_val=0.50,
    )


def test_effective_signal_uses_raw_side():
    assert effective_signal({"trade_score": 0.40, "raw_prob": 0.62}) == 0.62
