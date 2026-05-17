from src.application.services.llm.indicators import resolve_indicator_config
from src.application.services.llm.indicators_numeric import (
    _shrink_cf_tag,
    format_numeric_indicators_tight_line,
)


def test_format_numeric_indicators_tight_line_minifies_tokens():
    cfg = resolve_indicator_config({})
    data = [100.0] * 60
    al = "60: trend | 15: trend | 5: trend | 1: trend"
    cfline = "sinal_quant=FORTE_CONTINUIDADE_QUANT (High Conviction)"
    s = format_numeric_indicators_tight_line(
        data, data, data, data, cfg, 0.05, al, cfline, tf_tags=("60", "15", "5", "1")
    )
    assert " | 15:" in s and " | 5:" in s and " | 1:" in s
    assert "sigma=0.050" in s
    assert "mtf=P/P/P/P" in s
    assert "cf=fQuant" in s


def test_format_numeric_indicators_tight_line_sem_segmentos_na():
    cfg = resolve_indicator_config({})
    short = [100.0, 101.0, 102.0]
    al = "60: trend | 15: trend | 5: trend | 1: trend"
    cfline = "sinal_quant=dados_insuficientes"
    s = format_numeric_indicators_tight_line(
        short, short, short, short, cfg, None, al, cfline, tf_tags=("60", "15", "5", "1")
    )
    assert "60:na" in s and "15:na" in s and "5:na" in s and "1:na" in s
    assert "sigma=" not in s
    assert "mtf=P/P/P/P" in s
    assert "cf=" not in s


def test_shrink_cf_tag_branches():
    assert _shrink_cf_tag("") == "-"
    assert _shrink_cf_tag("FORTE_CONTINUIDADE_QUANT (High Conviction)") == "fQuant"
    assert _shrink_cf_tag("RANDOM_WALK_SEM_EDGE (Noisy)") == "rWalk"
    assert _shrink_cf_tag("DIVERGENCIA_ESTRUTURAL_DETECTADA (Risky)") == "divRisk"
