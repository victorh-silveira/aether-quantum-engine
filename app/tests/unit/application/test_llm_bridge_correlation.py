from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.llm import IndicatorConfig
from src.application.services.llm.global_macro_confluence import build_macro_snapshot
from src.application.services.llm.llm_bridge import collect_llm_decisions
from src.application.services.llm.symbol_decision import collect_symbol_llm_decision
from src.domain.models.trade import TradeDirection
from tests.unit.application.macro_guard_fixtures import RELAXED_MACRO_CFG


def _llm_metrics(direction, conviction, note):
    return {
        "direction": direction.name if direction else "NONE",
        "conviction": conviction,
        "llm_note": note,
        "execute": direction is not None,
    }


@pytest.mark.asyncio
async def test_collect_llm_decisions_propagates_cluster_tags_only():
    """Indices seguem US_CLUSTER e EU_CLUSTER da LLM, sem inversao por coeficiente."""
    orch = MagicMock()
    orch.anchor = "R_100"
    orch.symbols = ["R_100", "R_25", "R_50", "R_75"]
    orch._active_cycle_id = 1
    orch.config = {
        "strategy": {
            "correlation": {
                "enabled": True,
                "exclusive_cluster_by_macro": False,
                "statarb_index_select_enabled": False,
            },
            "clusters": {
                "us": ["R_25", "R_50"],
                "eu": ["R_75"],
            },
        },
        "llm": {"max_decision_latency_seconds": 10},
    }

    metrics_anchor = {
        "conviction": 0.90,
        "direction": "CALL",
        "execute": True,
        "llm_note": "Strong trend",
        "us_cluster": "PUT",
        "eu_cluster": "PUT",
    }

    with patch("src.application.services.llm.llm_bridge._collect_symbol_decision", new_callable=AsyncMock) as mock_dec:
        mock_dec.return_value = (TradeDirection.CALL, metrics_anchor)

        decisions = await collect_llm_decisions(orch)

        assert mock_dec.call_count == 1
        assert decisions["R_100"]["direction"] == TradeDirection.CALL
        assert decisions["R_25"]["direction"] == TradeDirection.PUT
        assert decisions["R_50"]["direction"] == TradeDirection.PUT
        assert decisions["R_75"]["direction"] == TradeDirection.PUT
        assert decisions["R_25"]["metrics"]["decision_source"] == "cluster_regime"
        assert "CLUSTER_TAG" in decisions["R_25"]["metrics"]["llm_note"]


@pytest.mark.asyncio
async def test_collect_llm_decisions_exclusive_risk_on_us_only():
    orch = MagicMock()
    orch.anchor = "R_100"
    orch.symbols = ["R_100", "R_25", "R_75"]
    orch._active_cycle_id = 4
    orch.logger = MagicMock()
    orch.config = {
        "strategy": {
            "correlation": {
                "enabled": True,
                "exclusive_cluster_by_macro": True,
                "cluster_invert_on_block": False,
            },
            "clusters": {"us": ["R_25"], "eu": ["R_75"]},
        },
        "llm": {"max_decision_latency_seconds": 10, "min_conviction_execute": 0.5},
    }

    with patch("src.application.services.llm.llm_bridge._collect_symbol_decision", new_callable=AsyncMock) as mock_dec:
        mock_dec.return_value = (
            TradeDirection.CALL,
            {
                "conviction": 0.68,
                "llm_note": "weak",
                "us_cluster": "CALL",
                "eu_cluster": "PUT",
                "macro_sentiment": "risk_on",
            },
        )
        decisions = await collect_llm_decisions(orch)

    assert decisions["R_25"]["direction"] == TradeDirection.CALL
    assert "R_75" not in decisions


@pytest.mark.asyncio
async def test_collect_llm_decisions_skips_when_indefinido_tie_exclusive():
    orch = MagicMock()
    orch.anchor = "R_100"
    orch.symbols = ["R_100", "R_25", "R_50"]
    orch._active_cycle_id = 6
    orch.logger = MagicMock()
    orch.config = {
        "strategy": {
            "correlation": {"enabled": True, "exclusive_cluster_by_macro": True},
            "clusters": {"us": ["R_25", "R_50"], "eu": []},
        },
        "llm": {"max_decision_latency_seconds": 10},
    }

    with patch("src.application.services.llm.llm_bridge._collect_symbol_decision", new_callable=AsyncMock) as mock_dec:
        mock_dec.return_value = (
            TradeDirection.CALL,
            {
                "conviction": 0.7,
                "us_cluster": "CALL",
                "eu_cluster": "CALL",
                "macro_sentiment": "indefinido",
                "macro_us_strength_quant": 0.5,
                "macro_eu_strength_quant": 0.5,
            },
        )
        decisions = await collect_llm_decisions(orch)

    assert list(decisions.keys()) == ["R_100"]
    orch.logger.debug.assert_called()


@pytest.mark.asyncio
async def test_collect_llm_decisions_statarb_picks_single_us_index():
    orch = MagicMock()
    orch.anchor = "R_100"
    orch.symbols = ["R_100", "R_25", "R_50"]
    orch._active_cycle_id = 5
    orch.logger = MagicMock()
    orch.config = {
        "strategy": {
            "correlation": {
                "enabled": True,
                "exclusive_cluster_by_macro": False,
                "statarb_index_select_enabled": True,
                "statarb_index_max_per_cluster": 1,
            },
            "clusters": {"us": ["R_25", "R_50"], "eu": []},
            "macro": {"statarb_z_threshold": 2.5},
        },
        "llm": {"max_decision_latency_seconds": 10},
    }

    with patch("src.application.services.llm.llm_bridge._collect_symbol_decision", new_callable=AsyncMock) as mock_dec:
        mock_dec.return_value = (
            TradeDirection.CALL,
            {
                "conviction": 0.9,
                "us_cluster": "CALL",
                "eu_cluster": "CALL",
                "statarb_spreads": {"R_25": -2.5, "R_50": 0.3},
                "hmm_state": 0,
            },
        )
        decisions = await collect_llm_decisions(orch)

    assert "R_25" in decisions
    assert "R_50" not in decisions
    assert "STATARB_INDEX" in decisions["R_25"]["metrics"]["llm_note"]


@pytest.mark.asyncio
async def test_collect_llm_decisions_correlation_no_direction():
    """Verifica se nao propaga nada se a ancora nao tiver direcao."""
    orch = MagicMock()
    orch.anchor = "R_100"
    orch.symbols = ["R_100", "R_50"]
    orch._active_cycle_id = 1
    orch.config = {
        "strategy": {"correlation": {"enabled": True}},
        "llm": {"max_decision_latency_seconds": 10},
    }

    with patch("src.application.services.llm.llm_bridge._collect_symbol_decision", new_callable=AsyncMock) as mock_dec:
        mock_dec.return_value = (None, {"execute": False})

        decisions = await collect_llm_decisions(orch)

        assert "R_100" in decisions
        assert "R_50" not in decisions


@pytest.mark.asyncio
async def test_collect_llm_decisions_skips_symbols_outside_cluster_lists():
    orch = MagicMock()
    orch.anchor = "R_100"
    orch.symbols = ["R_100", "1HZ50V", "R_25"]
    orch._active_cycle_id = 3
    orch.config = {
        "strategy": {
            "correlation": {
                "enabled": True,
                "exclusive_cluster_by_macro": False,
                "statarb_index_select_enabled": False,
            },
            "clusters": {"us": ["R_25"], "eu": []},
        },
        "llm": {"max_decision_latency_seconds": 10},
    }

    with patch("src.application.services.llm.llm_bridge._collect_symbol_decision", new_callable=AsyncMock) as mock_dec:
        mock_dec.return_value = (
            TradeDirection.PUT,
            {"conviction": 0.8, "us_cluster": "PUT", "eu_cluster": "PUT"},
        )
        decisions = await collect_llm_decisions(orch)

    assert "R_25" in decisions
    assert "1HZ50V" not in decisions


@pytest.mark.asyncio
async def test_collect_symbol_appends_macro_guard_note():
    snap = build_macro_snapshot(
        ["R_25"],
        ["R_75"],
        {"R_25": [100.0, 105.0], "R_75": [100.0, 95.0]},
        {"min_indices_for_vote": 1},
    )
    orch = MagicMock()
    orch._active_cycle_id = 6
    orch.symbols = ["R_100"]
    orch.logger = MagicMock()
    ctx = {"macro_cfg": {**RELAXED_MACRO_CFG, "macro_intelligence_only": True}}
    with (
        patch(
            "src.application.services.llm.symbol_decision.build_symbol_prompt",
            AsyncMock(return_value=("prompt", ctx, {}, 0.5, "line", "bundle", "m", "st", "sw", "tr", "mtf", [], snap)),
        ),
        patch(
            "src.application.services.llm.symbol_decision._request_payload",
            AsyncMock(
                return_value={
                    "_direction_normalized": "CALL",
                    "_conviction_normalized": 0.9,
                    "note": "base",
                }
            ),
        ),
    ):
        runtime = {
            "indicator_config": IndicatorConfig(),
            "payout_estimate": 0.85,
            "min_payout_accept": 0.80,
            "duration": 5,
            "du": "m",
            "model": "m",
            "base_url": "http://x",
            "timeout": 10,
            "num_predict": 64,
            "min_conviction_execute": 0.51,
        }
        _dir, metrics = await collect_symbol_llm_decision(
            orch, sym="R_100", runtime=runtime, llm_metrics=_llm_metrics
        )
        assert "MACRO_INTEL" in metrics["llm_note"]
