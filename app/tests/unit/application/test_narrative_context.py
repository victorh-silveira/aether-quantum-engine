from datetime import UTC
from unittest.mock import MagicMock, patch

import numpy as np

import src.application.services.llm.narrative_context as nc
from src.application.services.llm.indicators import resolve_indicator_config
from src.application.services.llm.narrative_context import (
    describe_m3_trigger,
    describe_m5_filter,
    describe_m15_map,
    describe_micro_structure,
    describe_mtf_alignment,
    describe_session_context,
    describe_volatility_regime,
)


def test_describe_m15_map_insufficient_data():
    cfg = resolve_indicator_config({})
    assert "indisponivel" in describe_m15_map([], cfg).lower()


def test_describe_m15_map_quant():
    cfg = resolve_indicator_config({"hurst_window": 30})
    closes = [100.0] * 100
    txt = describe_m15_map(closes, cfg)
    assert "hurst=" in txt
    assert "distribuicao_normal" in txt


def test_describe_m5_filter_quant():
    cfg = resolve_indicator_config({})
    closes = [100.0] * 60
    txt = describe_m5_filter(closes, cfg)
    assert "entropia=" in txt
    assert "velocidade=" in txt


def test_describe_m3_trigger_quant():
    cfg = resolve_indicator_config({})
    closes = [100.0] * 60
    txt = describe_m3_trigger(closes, cfg)
    assert "aceleracao=" in txt
    assert "zscore=" in txt


def test_describe_mtf_alignment():
    cfg = resolve_indicator_config({})
    data = [100.0] * 60
    txt = describe_mtf_alignment(data, data, data, data, cfg, "M30", "M15", "M5", "M1")
    assert "M30:" in txt and "M15:" in txt and "M5:" in txt and "M1:" in txt


def test_describe_volatility_regime():
    cfg = resolve_indicator_config({})
    data = [100.0] * 60
    txt = describe_volatility_regime(data, data, cfg)
    assert "REGIME_quant=" in txt
    assert "sigma_swing=" in txt


def test_describe_session_context():
    txt = describe_session_context()
    assert "SESSAO_UTC=" in txt


def test_describe_micro_structure():
    cfg = resolve_indicator_config({})
    closes = [100.0] * 60
    txt = describe_micro_structure(closes, cfg)
    assert "MICRO_M1" in txt
    assert "swing=" in txt


def test_micro_swing_tag_branches():
    assert nc._micro_swing_tag(np.array([100.0, 100.1, 100.2, 100.3])) == "HH_recente"
    assert nc._micro_swing_tag(np.array([100.3, 100.2, 100.1, 100.0])) == "LL_recente"
    assert nc._micro_swing_tag(np.array([100.0, 100.2, 100.1, 100.2])) == "pressao_alta"


def test_describe_session_context_mock():
    with patch("src.application.services.llm.narrative_context.datetime") as mdt:
        mdt.UTC = UTC
        mock_now = MagicMock()
        mock_now.hour = 14
        mock_now.strftime.return_value = "14:00"
        mdt.now.return_value = mock_now
        txt = describe_session_context()
        assert "janela=ny" in txt
