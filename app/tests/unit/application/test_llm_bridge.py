from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.llm import llm_bridge as bridge
from src.application.services.llm.indicators import IndicatorConfig
from src.application.services.llm.llm_bridge import llm_metrics
from src.application.services.llm.prompt_extras import build_institutional_pa_bundle
from src.application.services.llm.prompt_utils import (
    iter_llm_prompt_audit_sections,
)
from src.application.services.llm.symbol_decision_utils import build_symbol_prompt
from src.domain.models.trade import TradeDirection


def test_build_institutional_pa_bundle_medallion_shape():
    p = build_institutional_pa_bundle(
        regime_label="range",
        entropy_swing=0.0123,
        vol_range_pct=0.25,
        indicators_numeric_line="H=0.55 Z=0.1",
        cf_dual="CF_DUAL=ok",
        line_macro_structure="macro",
        line_swing_trigger="swing",
    )
    assert "LLM_DADOS_NUM=" in p
    assert "reg=range" in p
    assert "CONFL_MACRO_ESTRUTURA=macro" in p


def test_iter_llm_prompt_audit_sections_covers_blocks_sent_to_llm():
    rows = iter_llm_prompt_audit_sections(
        "R_100",
        "mapa",
        "estrutura",
        "filtro",
        "gatilho H=0.6",
        {"hurst": "persist", "zscore": "normal", "entropy": "low", "velocity": "pos"},
        "T/T/T",
        "REGIME=trend",
        "SESSAO=ny",
        "MICRO=x",
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


def test_build_metrics_for_decision_high_conviction_executes_with_mtf_tokens():
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


@pytest.mark.asyncio
async def test_build_symbol_prompt_with_real_stream_coverage():
    orch = MagicMock()

    class DummyStream:
        async def fetch_candle_closes(self, s, _gran, _count):
            if "SPC" in s:
                return [100.0, 105.0]
            if "NDX" in s:
                return [100.0, 95.0]
            if "DJI" in s:
                return []
            return [100.0, 100.0]

        async def fetch_candle_ohlc(self, _s, _gran, _count):
            return []

    orch.stream = DummyStream()
    orch._active_cycle_id = 1

    runtime = {
        "tf_macro_gran": 3600,
        "tf_macro_bars": 10,
        "tf_structure_gran": 900,
        "tf_structure_bars": 10,
        "tf_swing_gran": 300,
        "tf_swing_bars": 10,
        "tf_trigger_gran": 60,
        "tf_trigger_bars": 10,
        "indicator_config": IndicatorConfig(),
        "payout_estimate": 0.85,
        "min_payout_accept": 0.82,
        "duration": 15,
        "du": "m",
        "strategy_payload": None,
    }

    with patch(
        "src.application.services.llm.symbol_decision_utils.fetch_context_blocks", new_callable=AsyncMock
    ) as mock_fetch:
        mock_fetch.return_value = (
            "macro",
            "struct",
            "swing",
            "trigger",
            "mtf",
            {
                "regime_label": "range",
                "llm_macro_closes": [100.0, 101.0],
                "llm_structure_closes": [100.0, 101.0],
                "llm_swing_closes": [100.0, 101.0],
                "llm_trigger_closes": [100.0, 101.0],
                "atr_m5_pct": 0.01,
                "sniper_tokens": {},
            },
        )

        (
            prompt,
            ctx,
            sniper_tok,
            baseline,
            ind_line,
            pa_bundle,
            m_d,
            s_d,
            sw_d,
            t_d,
            mtf_d,
            sw_c,
            _macro_snap,
        ) = await build_symbol_prompt(orch, "R_100", runtime)

        assert "US_CLUSTER" in prompt
        assert "MACRO_CONFLUENCIA" in prompt
        assert "EU_CLUSTER" in prompt


@pytest.mark.asyncio
async def test_build_symbol_prompt_stream_exception_fallback():
    orch = MagicMock()

    class BrokenStream:
        async def fetch_candle_closes(self, _s, _gran, _count):
            raise ValueError("fetch failed")

        async def fetch_candle_ohlc(self, _s, _gran, _count):
            return []

    orch.stream = BrokenStream()
    orch._active_cycle_id = 1

    runtime = {
        "tf_macro_gran": 3600,
        "tf_macro_bars": 10,
        "tf_structure_gran": 900,
        "tf_structure_bars": 10,
        "tf_swing_gran": 300,
        "tf_swing_bars": 10,
        "tf_trigger_gran": 60,
        "tf_trigger_bars": 10,
        "indicator_config": IndicatorConfig(),
        "payout_estimate": 0.85,
        "min_payout_accept": 0.82,
        "duration": 15,
        "du": "m",
        "strategy_payload": None,
    }

    with patch(
        "src.application.services.llm.symbol_decision_utils.fetch_context_blocks", new_callable=AsyncMock
    ) as mock_fetch:
        mock_fetch.return_value = (
            "macro",
            "struct",
            "swing",
            "trigger",
            "mtf",
            {
                "regime_label": "range",
                "llm_macro_closes": [100.0, 101.0],
                "llm_structure_closes": [100.0, 101.0],
                "llm_swing_closes": [100.0, 101.0],
                "llm_trigger_closes": [100.0, 101.0],
                "atr_m5_pct": 0.01,
                "sniper_tokens": {},
            },
        )

        (
            prompt,
            ctx,
            sniper_tok,
            baseline,
            ind_line,
            pa_bundle,
            m_d,
            s_d,
            sw_d,
            t_d,
            mtf_d,
            sw_c,
            _macro_snap,
        ) = await build_symbol_prompt(orch, "R_100", runtime)
        assert "N/A" not in prompt or "MACRO_CONFLUENCIA" in prompt
