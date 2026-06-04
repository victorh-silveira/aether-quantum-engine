from src.application.services.execution_direction import (
    build_execution_candidate,
    infer_dl_direction,
    invert_direction,
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
    sym, exec_dir, metrics = build_execution_candidate("RDBULL", entry, invert_dl_direction=True)
    assert sym == "RDBULL"
    assert exec_dir == TradeDirection.PUT
    assert metrics["dl_direction"] == "CALL"
    assert metrics["exec_direction"] == "PUT"
    assert metrics["direction_inverted"] is True
