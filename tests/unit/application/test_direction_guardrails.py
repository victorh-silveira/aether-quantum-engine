from src.application.services.llm import llm_bridge_guards as g
from src.domain.models.trade import TradeDirection


def test_rsi_exhaustion_gate_passes_through_always():
    d, tag = g.rsi_exhaustion_execution_gate(
        TradeDirection.CALL,
        85.0,
        rsi_block_call_above=70.0,
        rsi_block_put_below=30.0,
        gate_enabled=True,
    )
    assert d == TradeDirection.CALL
    assert tag == g.TREND_FOLLOWING_ACTIVE


def test_is_sawtooth_pattern_detects_oscillation():
    assert g.is_sawtooth_pattern("P/M/P/M") is True
    assert g.is_sawtooth_pattern("M/P/M/P") is True
    assert g.is_sawtooth_pattern("P/P/M/M") is False


def test_is_highly_divergent_mtf_detects_conflict():
    assert (
        g.is_highly_divergent_mtf(
            "Momentum Alpha (Bull)", "Momentum Alpha (Bull)", "Momentum Alpha (Bear)", "Momentum Alpha (Bear)"
        )
        is True
    )
    assert (
        g.is_highly_divergent_mtf(
            "Momentum Alpha (Bull)", "Momentum Alpha (Bull)", "Momentum Alpha (Bull)", "Momentum Alpha (Bull)"
        )
        is False
    )


def test_invert_call_put():
    assert g.invert_call_put(TradeDirection.CALL) == TradeDirection.PUT
    assert g.invert_call_put(TradeDirection.PUT) == TradeDirection.CALL
    assert g.invert_call_put(None) is None


def test_is_overextended_detects_stretch():
    closes_ok = [100.0] * 30
    assert g.is_overextended(closes_ok) is False

    closes_stretched = [100.0] * 29 + [150.0]
    assert g.is_overextended(closes_stretched) is True

    assert g.is_overextended([100.0] * 5) is False
