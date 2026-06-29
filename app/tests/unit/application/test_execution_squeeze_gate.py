"""Testes do gate de squeeze extremo."""

from src.application.services.execution_squeeze_gate import passes_squeeze_gate, squeeze_consensus_side
from src.domain.models.trade import TradeDirection


def test_squeeze_gate_skips_when_not_extreme():
    metrics = {"squeeze_extreme": False, "direction_margin": 0.01}
    assert passes_squeeze_gate(metrics) is True


def test_squeeze_gate_blocks_low_margin():
    metrics = {"squeeze_extreme": True, "direction_margin": 0.05}
    cfg = {"squeeze_min_margin": 0.12}
    assert passes_squeeze_gate(metrics, cfg=cfg) is False


def test_squeeze_gate_requires_consensus():
    metrics = {
        "squeeze_extreme": True,
        "direction_margin": 0.2,
        "trend_direction": "CALL",
        "indicator_regime_side": "call",
        "calibrated_prob": 0.7,
        "resolved_direction": TradeDirection.CALL.name,
    }
    cfg = {"squeeze_min_margin": 0.12, "require_indicator_consensus": True}
    assert passes_squeeze_gate(metrics, cfg=cfg) is True
    assert squeeze_consensus_side(metrics) == "call"
