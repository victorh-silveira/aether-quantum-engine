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


def test_recovery_min_signal_scaling_consecutive_losses():
    cfg = {
        "mandatory_min_trade_score": 0.45,
        "recovery_min_trade_score": 0.45,
    }
    # consecutive_losses = 0 ou 1 -> deve manter o valor base (0.45)
    assert recovery_min_signal(cfg, recovery_active=True, consecutive_losses=0) == 0.45
    assert recovery_min_signal(cfg, recovery_active=True, consecutive_losses=1) == 0.45

    # consecutive_losses = 2 -> deve escalar para 0.53
    assert recovery_min_signal(cfg, recovery_active=True, consecutive_losses=2) == 0.53

    # consecutive_losses = 3 -> deve escalar para 0.55
    assert recovery_min_signal(cfg, recovery_active=True, consecutive_losses=3) == 0.55

    # consecutive_losses >= 4 -> deve escalar para 0.58
    assert recovery_min_signal(cfg, recovery_active=True, consecutive_losses=4) == 0.58
    assert recovery_min_signal(cfg, recovery_active=True, consecutive_losses=5) == 0.58


def test_recovery_min_val_accuracy_scaling_consecutive_losses():
    cfg = {
        "recovery_min_val_accuracy": 0.50,
    }
    # consecutive_losses = 0 ou 1 -> deve manter o valor base (0.50)
    assert recovery_min_val_accuracy(cfg, consecutive_losses=0) == 0.50
    assert recovery_min_val_accuracy(cfg, consecutive_losses=1) == 0.50

    # consecutive_losses = 2 -> deve escalar para 0.52
    assert recovery_min_val_accuracy(cfg, consecutive_losses=2) == 0.52

    # consecutive_losses = 3 -> deve escalar para 0.53
    assert recovery_min_val_accuracy(cfg, consecutive_losses=3) == 0.53

    # consecutive_losses >= 4 -> deve escalar para 0.55
    assert recovery_min_val_accuracy(cfg, consecutive_losses=4) == 0.55


def test_cluster_entry_eligible_forces_quality_in_recovery():
    # Sinal com execute=True do DL, mas com qualidade abaixo dos limites de recuperacao
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "execute": True,  # DL liberou
            "trade_score": 0.51,  # Abaixo do limite de recuperacao de 0.53
            "val_accuracy": 0.50,  # Abaixo do limite de recuperacao de 0.52
            "deploy_ok": True,
        },
    }
    # Em recuperacao, deve rejeitar mesmo com execute=True porque a qualidade esta abaixo dos limites
    assert not cluster_entry_eligible(
        entry,
        mandatory=True,
        recovery_active=True,
        recovery_cfg={},
        min_signal=0.53,
        min_val=0.52,
    )

    # Em recuperacao, deve aceitar se a qualidade estiver acima dos limites
    entry_good = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "execute": True,
            "trade_score": 0.55,
            "val_accuracy": 0.53,
            "deploy_ok": True,
        },
    }
    assert cluster_entry_eligible(
        entry_good,
        mandatory=True,
        recovery_active=True,
        recovery_cfg={},
        min_signal=0.53,
        min_val=0.52,
    )

    # Se NAO estiver em recuperacao, o execute=True do DL deve permitir entrada direta
    assert cluster_entry_eligible(
        entry,
        mandatory=True,
        recovery_active=False,
        recovery_cfg={},
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
        },
    }
    assert not cluster_entry_eligible(
        entry,
        mandatory=True,
        recovery_active=True,
        recovery_cfg={},
        min_signal=0.53,
        min_val=0.52,
    )


def test_cluster_entry_eligible_recovery_hard_block():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "execute": True,
            "deploy_ok": True,
            "gate_reason": "trend_conflict",
            "trade_score": 0.55,
            "val_accuracy": 0.53,
        },
    }
    assert not cluster_entry_eligible(
        entry,
        mandatory=True,
        recovery_active=True,
        recovery_cfg={},
        min_signal=0.53,
        min_val=0.52,
    )
