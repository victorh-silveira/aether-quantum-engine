from src.application.services.execution_direction import (
    _entry_gate_blocked,
    build_execution_candidate,
    infer_dl_direction,
    mandatory_execution_eligible,
    recovery_execution_eligible,
)
from src.domain.models.trade import TradeDirection


def test_infer_dl_direction_from_raw():
    entry = {"direction": None, "metrics": {"raw_prob": 0.62}}
    assert infer_dl_direction(entry) == TradeDirection.CALL


def test_mandatory_execution_eligible_rejects_hard_blocks():
    for gate in ("predict_error", "data", "training"):
        entry = {
            "direction": TradeDirection.CALL,
            "metrics": {
                "execute": False,
                "gate_reason": gate,
                "conviction": 0.7,
                "raw_prob": 0.62,
                "val_accuracy": 0.60,
                "deploy_ok": True,
            },
        }
        assert mandatory_execution_eligible(entry) is False


def test_mandatory_execution_eligible_accepts_soft_gate_reasons():
    for gate in ("trend_conflict", "exhaustion_conflict"):
        entry = {
            "direction": TradeDirection.CALL,
            "metrics": {
                "execute": False,
                "gate_reason": gate,
                "conviction": 0.7,
                "raw_prob": 0.62,
                "val_accuracy": 0.60,
                "deploy_ok": True,
            },
        }
        assert mandatory_execution_eligible(entry) is True


def test_build_execution_candidate_returns_none_without_direction():
    entry = {"direction": None, "metrics": {"execute": True}}
    assert build_execution_candidate("R_10", entry) is None


def test_build_candidate_uses_dl_direction():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {"execute": False, "conviction": 0.61, "raw_prob": 0.52, "calibrated_prob": 0.70, "deploy_ok": True},
    }
    sym, exec_dir, metrics = build_execution_candidate("R_10", entry)
    assert sym in {"R_10", "R_50"}
    assert exec_dir == TradeDirection.CALL
    assert metrics["dl_direction"] == "CALL"
    assert metrics["exec_direction"] == "CALL"
    assert "direction_inverted" not in metrics


def test_entry_gate_blocked_rejects_deploy_not_ok():
    assert _entry_gate_blocked({"deploy_ok": False, "gate_reason": ""}) is True
    assert _entry_gate_blocked({"deploy_ok": True, "gate_reason": ""}) is False


def test_recovery_execution_eligible():
    entry = {"direction": TradeDirection.PUT, "metrics": {"deploy_ok": True}}
    assert recovery_execution_eligible(entry) is True
