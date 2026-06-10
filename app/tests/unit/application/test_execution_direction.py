from unittest.mock import patch

from src.application.services.execution_direction import (
    _entry_gate_blocked,
    build_execution_candidate,
    build_forced_direction_candidate,
    build_forced_recovery_candidate,
    infer_dl_direction,
    mandatory_execution_eligible,
    recovery_execution_eligible,
    recovery_hedge_target,
)
from src.application.services.execution_direction_fallback import _forced_recovery_pick
from src.domain.models.trade import TradeDirection


def test_infer_dl_direction_from_raw():
    entry = {"direction": None, "metrics": {"raw_prob": 0.62}}
    assert infer_dl_direction(entry) == TradeDirection.CALL


def test_mandatory_execution_eligible_rejects_hard_blocks():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "execute": False,
            "gate_reason": "data",
            "conviction": 0.7,
            "raw_prob": 0.6,
            "deploy_ok": True,
            "val_accuracy": 0.55,
        },
    }
    assert mandatory_execution_eligible(entry) is False
    deploy_entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "execute": False,
            "gate_reason": "deploy",
            "conviction": 0.7,
            "raw_prob": 0.6,
            "deploy_ok": False,
            "val_accuracy": 0.55,
        },
    }
    assert mandatory_execution_eligible(deploy_entry) is False


def test_mandatory_execution_eligible_rejects_training_cooldown_and_pause():
    for gate in ("training", "cooldown", "session_pause"):
        entry = {
            "direction": TradeDirection.CALL,
            "metrics": {
                "execute": False,
                "gate_reason": gate,
                "conviction": 0.7,
                "raw_prob": 0.62,
                "val_accuracy": 0.60,
            },
        }
        assert mandatory_execution_eligible(entry) is False


def test_mandatory_execution_eligible_accepts_weak_signal():
    entry = {
        "direction": TradeDirection.PUT,
        "metrics": {
            "execute": False,
            "gate_reason": "candle_reject",
            "conviction": 0.58,
            "raw_prob": 0.45,
            "deploy_ok": True,
            "val_accuracy": 0.52,
        },
    }
    assert mandatory_execution_eligible(entry) is True


def test_mandatory_execution_eligible_rejects_missing_direction():
    entry = {"direction": None, "metrics": {"deploy_ok": True, "val_accuracy": 0.55}}
    assert mandatory_execution_eligible(entry) is False


def test_mandatory_execution_eligible_accepts_low_val_accuracy():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {"deploy_ok": True, "val_accuracy": 0.40, "conviction": 0.60, "raw_prob": 0.6},
    }
    assert mandatory_execution_eligible(entry) is True


def test_mandatory_execution_eligible_rejects_deploy_not_ok():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {"deploy_ok": False, "val_accuracy": 0.55, "conviction": 0.60, "raw_prob": 0.6},
    }
    assert mandatory_execution_eligible(entry) is False


def test_recovery_execution_eligible_rejects_deploy_not_ok():
    entry = {
        "direction": TradeDirection.PUT,
        "metrics": {"execute": False, "deploy_ok": False, "trade_score": 0.70, "val_accuracy": 0.60},
    }
    assert recovery_execution_eligible(entry) is False


def test_entry_gate_blocked_rejects_deploy_not_ok():
    assert _entry_gate_blocked({"deploy_ok": False, "gate_reason": ""}) is True
    assert _entry_gate_blocked({"deploy_ok": True, "gate_reason": ""}) is False


def test_recovery_execution_eligible_rejects_hard_block():
    entry = {
        "direction": TradeDirection.PUT,
        "metrics": {"execute": False, "gate_reason": "deploy", "trade_score": 0.65, "val_accuracy": 0.55},
    }
    assert recovery_execution_eligible(entry) is False


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


def test_recovery_execution_eligible_requires_quality_when_not_execute():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "execute": False,
            "trade_score": 0.52,
            "val_accuracy": 0.47,
            "raw_prob": 0.51,
        },
    }
    assert recovery_execution_eligible(entry) is False


def test_build_execution_candidate_returns_none_without_direction():
    entry = {"direction": None, "metrics": {"execute": True}}
    assert build_execution_candidate("R_50", entry) is None


def test_build_candidate_uses_dl_direction():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {"execute": False, "conviction": 0.61, "raw_prob": 0.52},
    }
    sym, exec_dir, metrics = build_execution_candidate("R_50", entry)
    assert sym == "R_50"
    assert exec_dir == TradeDirection.CALL
    assert metrics["dl_direction"] == "CALL"
    assert metrics["exec_direction"] == "CALL"
    assert metrics["direction_inverted"] is False


def test_build_forced_direction_candidate_after_high_side_call_loss():
    entry = {"direction": TradeDirection.CALL, "metrics": {"raw_prob": 0.56, "conviction": 0.56}}
    target = recovery_hedge_target("R_100", "CALL")
    assert target == ("R_10", TradeDirection.PUT)
    sym, exec_dir, metrics = build_forced_direction_candidate("R_10", entry, TradeDirection.PUT)
    assert sym == "R_10"
    assert exec_dir == TradeDirection.PUT
    assert metrics["recovery_hedge_forced"] is True


def test_build_forced_direction_candidate_without_dl_direction():
    entry = {"direction": None, "metrics": {}}
    assert build_forced_direction_candidate("R_50", entry, TradeDirection.PUT) is None


def test_recovery_hedge_target_returns_none_without_inputs():
    assert recovery_hedge_target(None, "CALL") is None
    assert recovery_hedge_target("R_50", "CALL") is None
    assert recovery_hedge_target("R_10", None) is None


def test_recovery_hedge_target_low_side_call_loss():
    assert recovery_hedge_target("R_10", "CALL") == ("R_100", TradeDirection.PUT)
    assert recovery_hedge_target("R_25", "PUT") == ("R_75", TradeDirection.CALL)


def test_recovery_hedge_target_when_peer_lookup_empty():
    with patch(
        "src.application.services.execution_direction.hedge_peer",
        return_value=None,
    ):
        assert recovery_hedge_target("R_10", "CALL") is None


def test_mandatory_execution_eligible_accepts_direction_margin_with_raw():
    entry = {
        "direction": None,
        "metrics": {
            "execute": False,
            "gate_reason": "direction_margin",
            "deploy_ok": True,
            "val_accuracy": 0.52,
            "conviction": 0.58,
            "raw_prob": 0.47,
        },
    }
    assert mandatory_execution_eligible(entry) is True


def test_build_forced_recovery_candidate_without_dl_direction():
    entry = {"direction": None, "metrics": {"trade_score": 0.55}}
    sym, side, metrics = build_forced_recovery_candidate("R_75", entry, TradeDirection.PUT)
    assert sym == "R_75"
    assert side == TradeDirection.PUT
    assert metrics["recovery_forced"] is True
    assert metrics["trade_score"] == 0.58


def test_build_forced_recovery_candidate_uses_raw_side_floor():
    entry = {"direction": None, "metrics": {"raw_prob": 0.62, "trade_score": 0.0}}
    _, _, metrics = build_forced_recovery_candidate("R_75", entry, TradeDirection.CALL)
    assert metrics["trade_score"] == 0.62
    assert metrics["direction_inverted"] is False


def test_forced_recovery_pick_prefers_dl_aligned_symbol():
    decisions = {
        "R_50": {"direction": None, "metrics": {"raw_prob": 0.40, "trade_score": 0.58, "val_accuracy": 0.55}},
        "R_75": {"direction": None, "metrics": {"raw_prob": 0.62, "trade_score": 0.58, "val_accuracy": 0.55}},
    }
    picked = _forced_recovery_pick(["R_50", "R_75"], decisions, TradeDirection.CALL)
    assert picked is not None
    assert picked[0] == "R_75"
