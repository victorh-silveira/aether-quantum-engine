import src.application.services.llm.prompt_utils as tpu


def test_iter_llm_prompt_audit_sections_handles_empty_regime():
    """Verifica se o auditor lida com regime/sessao vazios sem quebrar o prompt quant."""
    rows = tpu.iter_llm_prompt_audit_sections(
        symbol="frxEURUSD",
        macro_desc="h=0.5",
        structure_desc="h=0.6",
        swing_desc="h=0.7",
        trigger_desc="Hurst=0.55 | Z-Score=+1.2",
        sniper_tokens=None,
        mtf_align="P/P/P/P",
        regime_line="",
        session_line="",
        micro_line="",
        trigger_tail_closes=[1.0, 1.1],
        payout_estimate=0.95,
        min_payout_accept=0.85,
        duration=5,
        duration_unit="m",
        max_chars=800,
    )
    tags = [t for t, _ in rows]
    assert "REGIME" in tags
    regime_body = next(b for t, b in rows if t == "REGIME")
    assert regime_body == "-"


def test_iter_llm_prompt_audit_sections_short_body_no_ellipsis():
    rows = tpu.iter_llm_prompt_audit_sections(
        symbol="frxEURUSD",
        macro_desc="ab",
        structure_desc="cd",
        swing_desc="ef",
        trigger_desc="Hurst: 0.50",
        sniper_tokens=None,
        mtf_align="x",
        regime_line="rg",
        session_line="sess",
        micro_line="micro",
        trigger_tail_closes=[100.0],
        payout_estimate=0.9,
        min_payout_accept=0.8,
        duration=2,
        duration_unit="m",
        max_chars=500,
    )
    ab = dict(rows)["MAPA_TF_CASCADE"]
    assert ab == "macro: ab | estrutura: cd | swing: ef"


def test_iter_llm_prompt_audit_sections_long_body_truncates():
    long_txt = "Z" * 120
    rows = tpu.iter_llm_prompt_audit_sections(
        symbol="frxEURUSD",
        macro_desc=long_txt,
        structure_desc="m15",
        swing_desc="m5",
        trigger_desc="Hurst: 0.50",
        sniper_tokens=None,
        mtf_align="y",
        regime_line="rg",
        session_line="sess",
        micro_line="micro",
        trigger_tail_closes=[100.0],
        payout_estimate=0.9,
        min_payout_accept=0.8,
        duration=2,
        duration_unit="m",
        max_chars=40,
    )
    ab = dict(rows)["MAPA_TF_CASCADE"]
    assert len(ab) == 40
    assert ab.endswith("...")


def test_compact_m1_candles_csv_with_ohlc():
    closes = [100.0, 101.0, 102.0]
    ohlc = [(100.0, 101.0, 99.0, 100.5), (100.5, 102.0, 100.0, 101.5)]
    res = tpu.compact_m1_candles_csv(closes, ohlc_rows=ohlc)
    assert "(100.00000/101.00000/99.00000/100.50000)" in res
    assert "(100.50000/102.00000/100.00000/101.50000)" in res


def test_build_sniper_trading_prompt_with_ohlc():
    res = tpu.build_sniper_trading_prompt(
        symbol="frxEURUSD",
        macro_desc="M30",
        structure_desc="M15",
        swing_desc="M5",
        trigger_desc="M1 Hurst: 0.5",
        sniper_tokens={"mtf": "B/B/B/B"},
        mtf_align="B/B/B/B",
        regime_line="range",
        session_line="asia",
        micro_line="micro",
        trigger_tail_closes=[100.0, 101.0],
        payout_estimate=0.95,
        min_payout_accept=0.85,
        duration=1,
        duration_unit="m",
        trigger_ohlc=[(100, 105, 95, 102)],
        macro_confluence="MACRO_CONFLUENCIA: tag=risk_on",
        fx_reference_line="CONTEXTO_FX_REF: Risk-On",
        macro_sentiment="risk_on",
    )
    assert "CANDLES: (100.00000/105.00000/95.00000/102.00000)" in res
    assert "=== SINTESE FINAL ===" in res
    assert "MACRO_CONFLUENCIA" in res
    assert "CONTEXTO_FX_REF" in res
    assert "Risk-On" in res or "RISE" in res


def test_format_metrics_line():
    line = tpu.format_metrics_line(0.62, 1.5, 2.41)
    assert line == "H=0.62, Z=+1.50, E=2.41"
    line_na = tpu.format_metrics_line(None, None, None)
    assert line_na == "H=-, Z=-, E=-"
