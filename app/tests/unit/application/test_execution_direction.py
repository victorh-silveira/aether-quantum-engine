from src.application.services.execution_direction import (
    _entry_gate_blocked,
    build_execution_candidate,
    infer_dl_direction,
    mandatory_execution_eligible,
    recovery_execution_eligible,
)
from src.application.services.execution_direction_resolver import resolve_execution_direction
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


def test_build_execution_candidate_allows_thin_cal_margin():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "execute": True,
            "conviction": 0.61,
            "raw_prob": 0.52,
            "calibrated_prob": 0.51,
            "cal_margin": 0.01,
            "quality_min_direction_margin": 0.03,
            "deploy_ok": True,
        },
    }
    built = build_execution_candidate("R_10", entry)
    assert built is not None
    assert built[0] == "R_10"
    assert built[1] == TradeDirection.CALL


def test_resolve_execution_direction_already_resolved_in_cycle():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "_direction_resolved_cycle": 7,
            "execution_candidate_ready": True,
            "exec_direction": "CALL",
        },
    }
    result = resolve_execution_direction(entry, cycle_id=7)
    assert result is not None
    assert result[0] == TradeDirection.CALL
    assert result[1] is entry["metrics"]


def test_resolve_execution_direction_sync_pending_loss_error_handling():
    from types import SimpleNamespace

    # Case 1: pending_loss_total returns non-float string
    class BrokenRiskMgr1:
        def pending_loss_total(self):
            return "not_a_float"

    entry1 = {
        "direction": TradeDirection.CALL,
        "metrics": {"raw_prob": 0.60, "calibrated_prob": 0.60, "val_accuracy": 0.60, "deploy_ok": True},
    }
    orch1 = SimpleNamespace(
        risk_manager=BrokenRiskMgr1(),
        config={"infra": {"loss_classifier": {"enabled": False}}},
    )
    resolve_execution_direction(entry1, orch=orch1, exec_cfg={"skip_neg_edge": False})
    assert entry1["metrics"].get("pending_loss_total") == 0.0

    # Case 2: pending_loss dict sum throws error
    class BrokenRiskMgr2:
        pending_loss = {"1HZ75V": "invalid"}

    entry2 = {
        "direction": TradeDirection.CALL,
        "metrics": {"raw_prob": 0.60, "calibrated_prob": 0.60, "val_accuracy": 0.60, "deploy_ok": True},
    }
    orch2 = SimpleNamespace(
        risk_manager=BrokenRiskMgr2(),
        config={"infra": {"loss_classifier": {"enabled": False}}},
    )
    resolve_execution_direction(entry2, orch=orch2, exec_cfg={"skip_neg_edge": False})
    assert entry2["metrics"].get("pending_loss_total") == 0.0
