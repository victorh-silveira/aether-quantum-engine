import numpy as np

from src.application.services.llm import (
    abbrev_mtf_alignment_tokens,
    bundle_llm_indicators_for_log,
    extract_confluence_heuristic_tag,
    format_numeric_indicators_one_line,
    resolve_indicator_config,
    trend_token_from_label_word,
)
from src.application.services.llm.indicators_confluence import (
    mtf_confluence_line,
)
from src.application.services.llm.indicators_numeric import dual_confluence_shrunk_tags


def test_mtf_confluence_forte_continuidade():
    up = [100.0 * (1.01**i) for i in range(60)]
    line = mtf_confluence_line(up, up)
    assert "FORTE_CONTINUIDADE_QUANT (High Conviction)" in line
    assert "hurst=1.00" in line


def test_mtf_confluence_arbitragem_reversao():
    osc = [100.0 + (i % 2) * 5.0 for i in range(60)]
    up = [100.0 * (1.01**i) for i in range(60)]
    line = mtf_confluence_line(up, osc)
    assert "ARBITRAGEM_REVERSAO_ESTATISTICA (Counter-Trend)" in line


def test_mtf_confluence_ruido_excessivo():
    cfg = resolve_indicator_config({"entropy_window": 100, "entropy_bins": 50})
    np.random.seed(42)
    noise = (100.0 + np.random.uniform(-50, 50, 150)).tolist()
    up = [100.0 * (1.01**i) for i in range(150)]
    line = mtf_confluence_line(up, noise, cfg)
    assert "HIGH_ENTROPY_NOISE (Avoid High Stakes)" in line


def test_mtf_confluence_indefinido_short():
    short = [100.0, 101.0]
    line = mtf_confluence_line(short, short)
    assert "dados_insuficientes" in line


def test_format_numeric_indicators_one_line():
    cfg = resolve_indicator_config({})
    data = [100.0] * 60
    al = "M60: trend_alta | M15: trend_alta | M5: trend_alta | M1: trend_alta"
    cfline = "x | sinal_quant=FORTE_CONTINUIDADE_QUANT (High Conviction) | y"
    labs = ("M60", "M15", "M5", "M1")
    s = format_numeric_indicators_one_line(data, data, data, data, cfg, 0.05, al, cfline, tf_labels=labs)
    assert "M60 e" in s
    assert "mtf=P/P/P/P" in s
    assert "cf=FORTE_CONTINUIDADE_QUANT (High Conviction)" in s


def test_abbrev_mtf_alignment_tokens():
    assert abbrev_mtf_alignment_tokens("M30: trend | M5: reversao | M1: noise") == "P/M/N"


def test_trend_token_from_label_word():
    assert trend_token_from_label_word("trend alta") == "P"
    assert trend_token_from_label_word("reversao esperada") == "M"
    assert trend_token_from_label_word("noise") == "N"
    assert trend_token_from_label_word("indefinido") == "?"


def test_extract_confluence_heuristic_tag():
    line = "x | sinal_quant=ARBITRAGEM_REVERSAO_ESTATISTICA (Counter-Trend) | y"
    assert extract_confluence_heuristic_tag(line) == "ARBITRAGEM_REVERSAO_ESTATISTICA (Counter-Trend)"


def test_dual_confluence_shrunk_tags():
    a = "x | sinal_quant=FORTE_CONTINUIDADE_QUANT (High Conviction) | z"
    b = "y | sinal_quant=RUIDO_EXCESSIVO_STAT_SKIP | w"
    res = dual_confluence_shrunk_tags(a, b)
    assert "cf30_5=" in res
    assert "cf5_1=" in res


def test_bundle_llm_indicators_for_log():
    data = [100.0] * 60
    cfg = resolve_indicator_config({})
    s = bundle_llm_indicators_for_log(data, data, data, data, cfg, "H1", "M15", "M5", "M1")
    assert "H1 [QUANT]" in s
    assert "M15 [QUANT]" in s
