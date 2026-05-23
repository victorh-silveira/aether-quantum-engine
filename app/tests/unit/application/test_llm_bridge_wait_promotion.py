from src.application.services.llm import llm_bridge as bridge


def test_build_metrics_for_decision_keeps_wait_when_input_is_wait():
    direction, metrics = bridge._build_metrics_for_decision_core(
        {
            "min_conviction_execute": 0.66,
            "model": "m",
        },
        None,
        0.8,
        "aguardar",
        None,
        "M15: baixa | M5: baixa | M3: baixa",
        "H1 baixa",
        "M15 tendencia_EMA=baixa",
        "M5 tendencia_EMA=baixa",
        "M3 gatilho com tendencia_EMA=baixa",
        bridge.llm_metrics,
    )
    assert direction is None
    assert metrics["direction"] == "NONE"
    assert metrics["execute"] is False


def test_build_metrics_for_decision_wait_without_promotion_ignores_m3_confirmation():
    direction, metrics = bridge._build_metrics_for_decision_core(
        {
            "min_conviction_execute": 0.66,
            "model": "m",
        },
        None,
        0.8,
        "aguardar",
        None,
        "M15: baixa | M5: baixa | M3: alta",
        "H1 baixa",
        "M15 tendencia_EMA=baixa",
        "M5 tendencia_EMA=baixa",
        "M3 gatilho com tendencia_EMA=alta",
        bridge.llm_metrics,
    )
    assert direction is None
    assert metrics["direction"] == "NONE"
    assert metrics["execute"] is False
