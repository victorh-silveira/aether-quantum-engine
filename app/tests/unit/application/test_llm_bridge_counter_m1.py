from src.application.services.llm import llm_bridge as bridge
from src.domain.models.trade import TradeDirection


def test_build_metrics_for_decision_no_longer_blocks_m1_counter_trend_call():
    runtime = {"min_conviction_execute": 0.5, "model": "m"}
    direction, metrics = bridge._build_metrics_for_decision_core(
        runtime,
        TradeDirection.CALL,
        0.99,
        "entry note",
        None,
        "A/A/A/B",
        "H1 alta",
        "M15 alta",
        "M5 alta",
        "M1 gatilho: RSI=30; vela=bearish; pavio_superior",
        bridge.llm_metrics,
    )
    assert direction == TradeDirection.CALL
    assert metrics["execute"] is True
    assert metrics["llm_direction_adjusted"] is False


def test_build_metrics_for_decision_no_longer_blocks_m1_counter_trend_put():
    runtime = {"min_conviction_execute": 0.5, "model": "m"}
    direction, metrics = bridge._build_metrics_for_decision_core(
        runtime,
        TradeDirection.PUT,
        0.99,
        "entry note",
        None,
        "B/B/B/A",
        "H1 baixa",
        "M15 baixa",
        "M5 baixa",
        "M1 gatilho: RSI=70; vela=bullish; pavio_inferior",
        bridge.llm_metrics,
    )
    assert direction == TradeDirection.PUT
    assert metrics["execute"] is True
    assert metrics["llm_direction_adjusted"] is False
