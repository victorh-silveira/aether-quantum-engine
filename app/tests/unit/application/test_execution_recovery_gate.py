from src.application.services.orchestrator.execution_recovery_gate import (
    cluster_entry_eligible,
    recovery_min_signal,
    recovery_min_val_accuracy,
)
from src.domain.models.trade import TradeDirection


def test_cluster_entry_mandatory_recovery_uses_mandatory_floor():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "execute": False,
            "gate_reason": "confidence",
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
        recovery_cfg={"min_conviction_execute": 0.58},
        min_signal=0.52,
        min_val=0.48,
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
