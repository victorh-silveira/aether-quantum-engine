from src.application.services.orchestrator.execution_recovery_gate import (
    recovery_min_signal,
    recovery_min_val_accuracy,
)


def test_recovery_min_signal_uses_mandatory_floor():
    cfg = {"mandatory_min_trade_score": 0.45}
    assert recovery_min_signal(cfg, recovery_active=True) == 0.45
    assert recovery_min_signal(cfg, recovery_active=False) == 0.45


def test_recovery_min_val_accuracy_default():
    assert recovery_min_val_accuracy({}) == 0.50
