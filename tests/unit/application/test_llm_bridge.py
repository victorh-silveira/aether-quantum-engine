from src.application.services.llm import llm_bridge as bridge
from src.application.services.llm.llm_bridge import llm_metrics
from src.application.services.llm.prompt_utils import (
    build_trading_prompt,
    iter_llm_prompt_audit_sections,
)
from src.domain.models.trade import TradeDirection


def test_build_trading_prompt_template_v15_shape():
    p = build_trading_prompt(
        "1HZ10V",
        "m15 mapa",
        "m5 filtro",
        "m3 gatilho",
        "M15: trend | M5: trend | M3: trend",
        "REGIME=trend_persistente",
        "SESSAO=ny",
        "MICRO=x",
        0.0123,
        [100.0, 100.1],
        0.88,
        0.80,
        1,
        "m",
    )
    assert "ATIVO: 1HZ10V" in p
    assert "REGIME" in p
    assert "ESTRUTURA: m15 mapa | FILTRO: m5 filtro" in p
    assert "ALINHAMENTO: M15: trend | M5: trend | M3: trend" in p
    assert "MEDALLION V15:" in p


def test_iter_llm_prompt_audit_sections_covers_blocks_sent_to_llm():
    rows = iter_llm_prompt_audit_sections(
        "1HZ10V",
        "mapa",
        "estrutura",
        "filtro",
        "gatilho H=0.6",
        {"hurst": "persist", "zscore": "normal", "entropy": "low", "velocity": "pos"},
        "T/T/T",
        "REGIME=trend",
        "SESSAO=ny",
        "MICRO=x",
        0.055,
        [100.0, 100.1],
        0.95,
        0.82,
        1,
        "m",
    )
    tags = [t for t, _ in rows]
    assert "ATIVO" in tags
    assert "SNIPER_INPUT" in tags


def test_llm_metrics_shape():
    m = llm_metrics(TradeDirection.CALL, 0.7, "ok")
    assert m["conviction"] == 0.7
    assert m["direction"] == "CALL"
    assert m["decision_source"] == "llm"

    m_fail = llm_metrics(None, 0.0, "api error")
    assert m_fail["decision_source"] == "llm_api_failure"

    m_skip = llm_metrics(None, 0.0, "SKIP: rWalk")
    assert m_skip["decision_source"] == "llm_skip"


def test_decision_from_payload_maps_values():
    d, c, n, us, eu = bridge._decision_from_payload(
        {"_direction_normalized": "CALL", "_conviction_normalized": 0.85, "note": "ok"}
    )
    assert d == TradeDirection.CALL
    assert c == 0.85
    assert n == "ok"


def test_build_metrics_for_decision_executes_even_on_low_conviction_sovereign():
    runtime = {
        "min_conviction_execute": 0.8,
        "model": "m",
    }
    direction, metrics = bridge._build_metrics_for_decision_core(
        runtime,
        TradeDirection.CALL,
        0.7,
        "fraco",
        123.4,
        "T/T/T",
        "trend_alta",
        "trend_alta",
        "trend_alta",
        "trend_alta",
        llm_metrics,
    )
    assert direction == TradeDirection.CALL
    assert metrics["execute"] is False


def test_build_metrics_for_decision_sovereignty_ignores_sawtooth():
    runtime = {"min_conviction_execute": 0.5, "model": "m"}
    direction, metrics = bridge._build_metrics_for_decision_core(
        runtime,
        TradeDirection.CALL,
        0.99,
        "confianca total",
        None,
        "T/R/T/R",
        "trend",
        "reversao",
        "trend",
        "reversao",
        llm_metrics,
    )
    assert direction == TradeDirection.CALL
    assert metrics["execute"] is True
