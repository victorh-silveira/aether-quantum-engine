from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.llm.indicators import IndicatorConfig
from src.application.services.llm.llm_bridge import collect_llm_decisions
from src.application.services.llm.llm_bridge_utils import parse_llm_trade_response
from src.application.services.llm.macro_config import MacroSnapshot
from src.application.services.llm.symbol_decision_utils import build_symbol_prompt
from src.domain.models.trade import TradeDirection


def _risk_off_snapshot() -> MacroSnapshot:
    return MacroSnapshot(
        us_dir="down",
        eu_dir="down",
        us_strength=0.9,
        eu_strength=0.88,
        tag="risk_off",
        eurusd_bias="PUT",
        cluster_status="",
        macro_block="",
        fx_reference_line="",
        us_parts=("SPC: 105.00 (FALL -5.00%)",),
        eu_parts=("FCHI: 95.00 (FALL -5.00%)",),
    )


@pytest.mark.asyncio
async def test_collect_llm_decisions_with_dynamic_cluster_regime():
    """Verifica se o robô segue as direções específicas dos clusters enviadas pela Gemini."""
    orch = MagicMock()
    orch.anchor = "frxEURUSD"
    orch.symbols = ["frxEURUSD", "OTC_SPC", "OTC_FCHI"]
    orch._active_cycle_id = 1
    orch._last_llm_macro_tag = None
    orch._last_llm_decisions = None
    orch.config = {
        "strategy": {
            "clusters": {"us": ["OTC_SPC"], "eu": ["OTC_FCHI"]},
            "correlation": {
                "enabled": True,
                "exclusive_cluster_by_macro": False,
                "targets": {"OTC_SPC": 1.0, "OTC_FCHI": 1.0},
            },
        },
        "llm": {"max_decision_latency_seconds": 10},
    }

    metrics_anchor = {
        "conviction": 0.90,
        "direction": "PUT",
        "execute": True,
        "llm_note": "Morphological Regime",
        "us_cluster": "PUT",
        "eu_cluster": "CALL",
        "macro_sentiment": "risk_off",
    }

    with patch(
        "src.application.services.llm.llm_bridge.fetch_macro_snapshot",
        new_callable=AsyncMock,
        return_value=_risk_off_snapshot(),
    ):
        with patch(
            "src.application.services.llm.llm_bridge._collect_symbol_decision", new_callable=AsyncMock
        ) as mock_dec:
            mock_dec.return_value = (TradeDirection.PUT, metrics_anchor)

            decisions = await collect_llm_decisions(orch)

            assert decisions["frxEURUSD"]["direction"] == TradeDirection.PUT

            assert decisions["OTC_SPC"]["direction"] == TradeDirection.PUT
            assert decisions["OTC_SPC"]["metrics"]["decision_source"] == "cluster_regime"

            assert decisions["OTC_FCHI"]["direction"] == TradeDirection.CALL
            assert decisions["OTC_FCHI"]["metrics"]["decision_source"] == "cluster_regime"
        assert "CLUSTER_TAG" in decisions["OTC_FCHI"]["metrics"]["llm_note"]


def test_parse_llm_trade_response_with_clusters():
    """Valida o parsing das novas tags de cluster no texto da LLM."""
    raw_text = "EURUSD: CALL | US_CLUSTER: PUT | EU_CLUSTER: CALL | Probabilidade: 85%"
    out = parse_llm_trade_response(raw_text)

    assert out["direction"] == "CALL"
    assert out["us_cluster"] == "PUT"
    assert out["eu_cluster"] == "CALL"
    assert out["conviction"] == 0.85
    assert out["note"] == "EURUSD_CALL"


def test_parse_llm_trade_response_compact_cluster_tags():
    raw_text = "[PUT] EURUSD: PUT | US=PUT | EU=CALL | Probabilidade: 99%"
    out = parse_llm_trade_response(raw_text)
    assert out["direction"] == "PUT"
    assert out["us_cluster"] == "PUT"
    assert out["eu_cluster"] == "CALL"


@pytest.mark.asyncio
async def test_cluster_skipped_when_no_explicit_tag():
    """Clusters US/EU sao pulados quando a LLM nao retorna tag explicita."""
    orch = MagicMock()
    orch.anchor = "frxEURUSD"
    orch.symbols = ["frxEURUSD", "OTC_SPC", "OTC_FCHI"]
    orch._active_cycle_id = 1
    orch.config = {
        "strategy": {
            "correlation": {
                "enabled": True,
                "exclusive_cluster_by_macro": False,
                "targets": {"OTC_SPC": 1.0, "OTC_FCHI": 1.0},
            }
        },
        "llm": {"max_decision_latency_seconds": 10},
    }

    metrics_no_tags = {
        "conviction": 0.8,
        "direction": "PUT",
        "execute": True,
        "llm_note": "Bearish",
    }

    with patch(
        "src.application.services.llm.llm_bridge._collect_symbol_decision",
        new_callable=AsyncMock,
    ) as mock_dec:
        mock_dec.return_value = (TradeDirection.PUT, metrics_no_tags)
        decisions = await collect_llm_decisions(orch)

    assert "frxEURUSD" in decisions
    assert "OTC_SPC" not in decisions
    assert "OTC_FCHI" not in decisions


@pytest.mark.asyncio
async def test_build_symbol_prompt_clusters_dict_and_non_dict_coverage():
    # 1. Test dict clusters
    orch = MagicMock()
    orch.config = {"strategy": {"clusters": {"us": ["OTC_SPC"], "eu": ["OTC_FTSE"]}}}

    class DummyStream:
        async def fetch_candle_closes(self, _s, _gran, _count):
            return [100.0, 105.0]

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

        prompt, _, _, _, _, _, _, _, _, _, _, _, _ = await build_symbol_prompt(orch, "frxEURUSD", runtime)
        assert "SPC:" in prompt

    # 2. Test non-dict clusters fallback coverage
    orch_non_dict = MagicMock()
    orch_non_dict.config = {"strategy": {"clusters": None}}
    orch_non_dict.stream = DummyStream()
    orch_non_dict._active_cycle_id = 1

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

        prompt, _, _, _, _, _, _, _, _, _, _, _, _ = await build_symbol_prompt(orch_non_dict, "frxEURUSD", runtime)
        assert "NDX:" in prompt or "DJI:" in prompt or "R_10:" in prompt or "VOL10:" in prompt
