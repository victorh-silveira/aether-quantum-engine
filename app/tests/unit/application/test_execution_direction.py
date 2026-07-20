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


def test_mandatory_execution_eligible_rejects_low_val_accuracy():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {"deploy_ok": True, "val_accuracy": 0.40, "conviction": 0.60, "raw_prob": 0.6},
    }
    assert mandatory_execution_eligible(entry) is False


def test_mandatory_execution_eligible_accepts_low_val_when_floor_disabled():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {"deploy_ok": True, "val_accuracy": 0.40, "conviction": 0.60, "raw_prob": 0.6},
    }
    assert mandatory_execution_eligible(entry, min_val_accuracy=0.0) is True


def test_mandatory_execution_eligible_rejects_zero_signal():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {"deploy_ok": True, "val_accuracy": 0.55, "conviction": 0.0},
    }
    assert mandatory_execution_eligible(entry) is False


def test_mandatory_execution_eligible_rejects_deploy_not_ok():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {"deploy_ok": False, "val_accuracy": 0.55, "conviction": 0.60, "raw_prob": 0.6},
    }
    assert mandatory_execution_eligible(entry) is False


def test_mandatory_execution_eligible_accepts_low_trade_score():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {"deploy_ok": True, "val_accuracy": 0.40, "conviction": 0.40, "raw_prob": 0.6},
    }
    assert mandatory_execution_eligible(entry, min_signal=0.53, min_val_accuracy=0.0) is True


def test_entry_gate_blocked_rejects_deploy_not_ok():
    assert _entry_gate_blocked({"deploy_ok": False, "gate_reason": ""}) is True
    assert _entry_gate_blocked({"deploy_ok": True, "gate_reason": ""}) is False


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
    assert metrics["direction_inverted"] is False


def test_build_forced_direction_candidate_after_high_side_call_loss():
    entry = {"direction": TradeDirection.CALL, "metrics": {"raw_prob": 0.56, "conviction": 0.56}}
    target = recovery_hedge_target("R_10", "CALL")
    assert target is None
    sym, exec_dir, metrics = build_forced_direction_candidate("R_10", entry, TradeDirection.PUT)
    assert sym in {"R_10", "R_50"}
    assert exec_dir == TradeDirection.PUT
    assert metrics["recovery_hedge_forced"] is True


def test_build_forced_direction_candidate_without_dl_direction():
    entry = {"direction": None, "metrics": {}}
    assert build_forced_direction_candidate("R_10", entry, TradeDirection.PUT) is None


def test_recovery_hedge_target_returns_none_without_inputs():
    assert recovery_hedge_target(None, "CALL") is None
    assert recovery_hedge_target("R_10", None) is None


def test_recovery_hedge_target_low_side_call_loss():
    assert recovery_hedge_target("R_10", "CALL") is None
    assert recovery_hedge_target("R_10", "PUT") is None


def test_recovery_hedge_target_when_peer_lookup_empty():
    with patch(
        "src.application.services.execution_direction.hedge_peer",
        return_value=None,
    ):
        assert recovery_hedge_target("R_10", "CALL") is None


def test_recovery_hedge_target_returns_none_when_peer_lookup_empty_despite_map():
    with (
        patch("src.application.services.execution_direction.HEDGE_PEER", {"RDBULL": "RDBEAR"}),
        patch("src.application.services.execution_direction.hedge_peer", return_value=None),
    ):
        assert recovery_hedge_target("RDBULL", "CALL") is None


def _hedge_peer_map(symbol: str) -> str | None:
    return {"RDBULL": "RDBEAR", "RDBEAR": "RDBULL"}.get(symbol)


def _is_bull_high_side(symbol: str) -> bool:
    return symbol == "RDBULL"


def test_recovery_hedge_target_high_side_and_low_side_branches():
    peer_map = {"RDBULL": "RDBEAR", "RDBEAR": "RDBULL"}
    with (
        patch("src.application.services.execution_direction.HEDGE_PEER", peer_map),
        patch("src.application.services.execution_direction.hedge_peer", side_effect=_hedge_peer_map),
        patch("src.application.services.execution_direction.is_high_side", side_effect=_is_bull_high_side),
    ):
        assert recovery_hedge_target("RDBULL", "PUT") == ("RDBEAR", TradeDirection.CALL)
        assert recovery_hedge_target("RDBEAR", "CALL") == ("RDBULL", TradeDirection.PUT)


def test_recovery_hedge_target_returns_none_without_loss_direction():
    with patch("src.application.services.execution_direction.HEDGE_PEER", {"RDBULL": "RDBEAR"}):
        assert recovery_hedge_target("RDBULL", "") is None


def test_build_execution_candidate_reads_peer_entry_from_decisions():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {"calibrated_prob": 0.62, "deploy_ok": True, "conviction": 0.62, "raw_prob": 0.62},
    }
    peer_entry = {
        "direction": TradeDirection.PUT,
        "metrics": {"calibrated_prob": 0.38, "deploy_ok": True},
    }
    with patch("src.application.services.execution_direction.hedge_peer", return_value="R_50"):
        candidate = build_execution_candidate(
            "R_10",
            entry,
            decisions={"R_10": entry, "R_50": peer_entry},
        )
    assert candidate is not None
    assert candidate[0] == "R_10"


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
    sym, side, metrics = build_forced_recovery_candidate("R_10", entry, TradeDirection.PUT)
    assert sym in {"R_10", "R_50"}
    assert side == TradeDirection.PUT
    assert metrics["recovery_forced"] is True
    assert metrics["trade_score"] == 0.55


def test_build_forced_recovery_candidate_uses_raw_side_floor():
    entry = {"direction": None, "metrics": {"raw_prob": 0.62, "trade_score": 0.0}}
    _, _, metrics = build_forced_recovery_candidate("R_10", entry, TradeDirection.CALL)
    assert metrics["trade_score"] == 0.62
    assert metrics["direction_inverted"] is False


def test_forced_recovery_pick_prefers_dl_aligned_symbol():
    decisions = {
        "R_10": {"direction": None, "metrics": {"raw_prob": 0.40, "trade_score": 0.58, "val_accuracy": 0.55}},
        "R_50": {"direction": None, "metrics": {"raw_prob": 0.62, "trade_score": 0.58, "val_accuracy": 0.55}},
    }
    picked = _forced_recovery_pick(["R_10", "R_50"], decisions, TradeDirection.CALL)
    assert picked is not None
    assert picked[0] in {"R_10", "R_50"}


def test_recovery_execution_eligible_rejects_technical_blocks():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "execute": False,
            "gate_reason": "predict_error",
            "raw_prob": 0.62,
            "deploy_ok": True,
        },
    }
    assert recovery_execution_eligible(entry, {}) is False
