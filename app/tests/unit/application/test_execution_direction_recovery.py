from src.application.services.execution_direction import recovery_execution_eligible
from src.domain.models.trade import TradeDirection


def test_recovery_execution_eligible_rejects_deploy_not_ok():
    entry = {
        "direction": TradeDirection.PUT,
        "metrics": {"execute": False, "deploy_ok": False, "trade_score": 0.70, "val_accuracy": 0.60},
    }
    assert recovery_execution_eligible(entry) is False


def test_recovery_execution_eligible_rejects_predict_error():
    entry = {
        "direction": TradeDirection.PUT,
        "metrics": {"execute": False, "gate_reason": "predict_error", "trade_score": 0.65, "val_accuracy": 0.55},
    }
    assert recovery_execution_eligible(entry) is False


def test_recovery_execution_eligible_rejects_cooldown_gate():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "execute": False,
            "gate_reason": "cooldown",
            "trade_score": 0.58,
            "val_accuracy": 0.55,
            "raw_prob": 0.58,
        },
    }
    assert recovery_execution_eligible(entry) is True


def test_recovery_execution_eligible_accepts_execute_true():
    entry = {
        "direction": TradeDirection.PUT,
        "metrics": {"execute": True, "trade_score": 0.40, "val_accuracy": 0.40},
    }
    assert recovery_execution_eligible(entry) is True


def test_recovery_execution_eligible_rejects_missing_direction():
    entry = {"direction": None, "metrics": {"trade_score": 0.60, "val_accuracy": 0.55}}
    assert recovery_execution_eligible(entry) is False


def test_recovery_execution_eligible_accepts_quality_signal():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {"execute": False, "trade_score": 0.60, "val_accuracy": 0.55, "raw_prob": 0.58},
    }
    assert recovery_execution_eligible(entry) is True


def test_recovery_execution_eligible_accepts_inferable_direction_without_quality_gate():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "execute": False,
            "trade_score": 0.52,
            "val_accuracy": 0.47,
            "raw_prob": 0.51,
        },
    }
    assert recovery_execution_eligible(entry) is True


def test_recovery_execution_eligible_ignores_recovery_cfg_bypass():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "execute": False,
            "gate_reason": "val_acc",
            "trade_score": 0.67,
            "val_accuracy": 0.47,
            "raw_prob": 0.67,
        },
    }
    assert recovery_execution_eligible(entry, {"min_val_accuracy": 0.48}) is True
