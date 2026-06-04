from unittest.mock import patch

from src.application.services.execution_direction import (
    build_execution_candidate,
    build_forced_direction_candidate,
    infer_dl_direction,
    invert_direction,
    recovery_hedge_target,
)
from src.domain.models.trade import TradeDirection


def test_infer_dl_direction_from_raw():
    entry = {"direction": None, "metrics": {"raw_prob": 0.62}}
    assert infer_dl_direction(entry) == TradeDirection.CALL


def test_invert_direction_put_to_call():
    assert invert_direction(TradeDirection.PUT) == TradeDirection.CALL


def test_invert_and_build_candidate():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {"execute": False, "conviction": 0.61, "raw_prob": 0.52},
    }
    sym, exec_dir, metrics = build_execution_candidate("R_50", entry, invert_dl_direction=True)
    assert sym == "R_50"
    assert exec_dir == TradeDirection.PUT
    assert metrics["dl_direction"] == "CALL"
    assert metrics["exec_direction"] == "PUT"
    assert metrics["direction_inverted"] is True


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
