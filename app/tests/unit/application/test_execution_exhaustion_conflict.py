"""Testes de conflito DL vs exaustao CMO/RSI."""

from src.application.services.execution_exhaustion_conflict import (
    exhaustion_conflict_penalty,
    exhaustion_conflict_side,
)
from src.domain.models.trade import TradeDirection


_CFG = {
    "exhaustion_gate": {
        "enabled": True,
        "rsi_overbought": 0.72,
        "rsi_oversold": 0.28,
        "cmo_bull": 0.55,
        "cmo_bear": -0.55,
        "min_penalty_skip": 0.12,
    }
}


def test_exhaustion_side_overbought():
    metrics = {"indicators": {"rsi": 0.80, "cmo": 0.60}}
    assert exhaustion_conflict_side(metrics, cfg=_CFG) == "put"


def test_exhaustion_side_oversold():
    metrics = {"indicators": {"rsi": 0.20, "cmo": -0.60}}
    assert exhaustion_conflict_side(metrics, cfg=_CFG) == "call"


def test_exhaustion_conflict_penalty_when_dl_disagrees():
    metrics = {
        "indicators": {"rsi": 0.80, "cmo": 0.60},
        "direction_margin": 0.10,
    }
    conflict, penalty = exhaustion_conflict_penalty(
        metrics,
        TradeDirection.CALL,
        cfg=_CFG,
    )
    assert conflict is True
    assert penalty >= 0.12


def test_no_conflict_when_dl_aligns_with_exhaustion():
    metrics = {"indicators": {"rsi": 0.80, "cmo": 0.60}}
    conflict, penalty = exhaustion_conflict_penalty(
        metrics,
        TradeDirection.PUT,
        cfg=_CFG,
    )
    assert conflict is False
    assert penalty == 0.0


def test_exhaustion_gate_disabled():
    metrics = {"indicators": {"rsi": 0.80, "cmo": 0.60}}
    assert exhaustion_conflict_side(metrics, cfg={"exhaustion_gate": {"enabled": False}}) is None
    conflict, penalty = exhaustion_conflict_penalty(
        metrics,
        TradeDirection.CALL,
        cfg={"exhaustion_gate": {"enabled": False}},
    )
    assert conflict is False
    assert penalty == 0.0
