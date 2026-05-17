from src.application.services.llm import llm_bridge as bridge
from src.domain.models.trade import TradeDirection


def test_build_metrics_for_decision_preserva_direcao_e_conviccao():
    direction, metrics = bridge._build_metrics_for_decision_core(
        {"min_conviction_execute": 0.66, "model": "m"},
        TradeDirection.CALL,
        0.85,
        "nota llm",
        None,
        "M30: lateral | M5: lateral | M1: lateral",
        "H1 lateral",
        "mapa RSI zona neutro sem tendencia clara",
        "filtro RSI 48 zona neutro",
        "gatilho RSI 50 zona neutro",
        bridge.llm_metrics,
    )
    assert direction == TradeDirection.CALL
    assert metrics["direction"] == "CALL"
    assert metrics["execute"] is True
    assert metrics["llm_direction_adjusted"] is False
    assert "neutral_weak_mtf_wait" not in metrics


def test_build_metrics_for_decision_wait_quando_conviccao_abaixo_do_limiar():
    direction, metrics = bridge._build_metrics_for_decision_core(
        {"min_conviction_execute": 0.66, "model": "m"},
        TradeDirection.CALL,
        0.50,
        "nota",
        None,
        "M15: baixa | M5: alta | M3: alta",
        "H1 neutro",
        "M15 tendencia_EMA=baixa",
        "M5 tendencia_EMA=alta",
        "M3 gatilho com tendencia_EMA=alta",
        bridge.llm_metrics,
    )
    assert direction == TradeDirection.CALL
    assert metrics["execute"] is False
