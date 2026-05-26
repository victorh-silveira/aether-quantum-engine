from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.llm.llm_bridge import collect_llm_decisions
from src.application.services.llm.llm_cluster_refresh import (
    cluster_refresh_due,
    refresh_cluster_decisions_from_cache,
    resolve_cluster_refresh_interval_seconds,
)
from src.application.services.llm.macro_config import MacroSnapshot
from src.domain.models.trade import TradeDirection


def _snapshot() -> MacroSnapshot:
    return MacroSnapshot(
        us_dir="PUT",
        eu_dir="PUT",
        us_strength=0.7,
        eu_strength=0.72,
        tag="risk_off",
        eurusd_bias="bear",
        cluster_status="ok",
        macro_block="",
        fx_reference_line="",
        us_parts=(),
        eu_parts=(),
        statarb_spreads={"OTC_FCHI": 1.1, "OTC_GDAXI": 0.6},
        hmm_state=0,
        hmm_prob=0.9,
    )


def test_resolve_cluster_refresh_interval_falls_back_to_cycle():
    cfg = {"orchestrator": {"cycle_interval_seconds": 60}}
    assert resolve_cluster_refresh_interval_seconds(cfg) == 60


def test_cluster_refresh_due_false_when_interval_zero():
    orch = MagicMock()
    assert cluster_refresh_due(orch, now_epoch=100.0, interval_seconds=0) is False


def test_resolve_cluster_refresh_interval_explicit():
    cfg = {"orchestrator": {"cluster_refresh_interval_seconds": 45, "cycle_interval_seconds": 60}}
    assert resolve_cluster_refresh_interval_seconds(cfg) == 45


def test_refresh_returns_none_when_cache_empty():
    orch = MagicMock()
    orch.anchor = "frxEURUSD"
    orch._last_llm_decisions = {}
    assert refresh_cluster_decisions_from_cache(orch, _snapshot(), "C1") is None


def test_refresh_returns_cached_when_anchor_entry_invalid():
    orch = MagicMock()
    orch.anchor = "frxEURUSD"
    orch._last_llm_decisions = {"frxEURUSD": "bad"}
    out = refresh_cluster_decisions_from_cache(orch, _snapshot(), "C1")
    assert out == {"frxEURUSD": "bad"}


def test_refresh_returns_cached_when_direction_missing():
    orch = MagicMock()
    orch.anchor = "frxEURUSD"
    orch._last_llm_decisions = {"frxEURUSD": {"direction": "PUT", "metrics": {}}}
    out = refresh_cluster_decisions_from_cache(orch, _snapshot(), "C1")
    assert "frxEURUSD" in out


def test_cluster_refresh_due_when_never_refreshed():
    orch = MagicMock()
    orch._last_cluster_refresh_epoch = None
    assert cluster_refresh_due(orch, now_epoch=100.0, interval_seconds=60) is True


def test_cluster_refresh_not_due_within_window():
    orch = MagicMock()
    orch._last_cluster_refresh_epoch = 90.0
    assert cluster_refresh_due(orch, now_epoch=120.0, interval_seconds=60) is False


@pytest.mark.asyncio
async def test_refresh_cluster_decisions_from_cache_repropagates():
    orch = MagicMock()
    orch.anchor = "frxEURUSD"
    orch.symbols = ["frxEURUSD", "OTC_FCHI", "OTC_GDAXI"]
    orch.config = {
        "strategy": {
            "clusters": {"us": [], "eu": ["OTC_FCHI", "OTC_GDAXI"]},
            "correlation": {"enabled": True, "best_symbol_only": True},
            "macro": {},
        },
        "llm": {"min_conviction_execute": 0.5},
        "risk_management": {},
    }
    orch.logger = MagicMock()
    metrics = {
        "conviction": 0.7,
        "us_cluster": "PUT",
        "eu_cluster": "PUT",
        "macro_sentiment": "risk_off",
        "statarb_spreads": {"OTC_FCHI": 0.2},
    }
    orch._last_llm_decisions = {
        "frxEURUSD": {"direction": TradeDirection.PUT, "metrics": dict(metrics)},
    }
    out = refresh_cluster_decisions_from_cache(orch, _snapshot(), "C0001")
    assert out is not None
    assert "OTC_FCHI" in out or "OTC_GDAXI" in out
    orch.logger.info.assert_called()


@pytest.mark.asyncio
async def test_collect_llm_decisions_refreshes_cluster_from_cache():
    orch = MagicMock()
    orch.anchor = "frxEURUSD"
    orch.symbols = ["frxEURUSD", "OTC_FCHI"]
    orch._active_cycle_id = 3
    orch._last_llm_macro_tag = "risk_off"
    orch._last_cluster_refresh_epoch = None
    orch.config = {
        "orchestrator": {"cluster_refresh_interval_seconds": 60},
        "strategy": {
            "correlation": {"enabled": True, "cluster_invert_on_block": False},
            "clusters": {"us": [], "eu": ["OTC_FCHI"]},
            "macro": {"allowed_execute_tags": ("risk_off",)},
        },
        "llm": {"max_decision_latency_seconds": 10, "min_conviction_execute": 0.5},
    }
    orch._last_llm_decisions = {
        "frxEURUSD": {
            "direction": TradeDirection.PUT,
            "metrics": {
                "conviction": 0.7,
                "us_cluster": "PUT",
                "eu_cluster": "PUT",
                "macro_sentiment": "risk_off",
            },
        }
    }
    snap = _snapshot()

    with (
        patch("src.application.services.llm.llm_bridge.resolve_llm_runtime") as mock_rt,
        patch("src.application.services.llm.llm_bridge.fetch_macro_snapshot", new_callable=AsyncMock) as mock_macro,
        patch("src.application.services.llm.llm_bridge.should_refresh_llm_decision", return_value=False),
        patch("src.application.services.llm.llm_bridge.macro_tag_allows_llm_call", return_value=(True, "")),
    ):
        mock_rt.return_value = {"max_decision_latency_seconds": 10}
        mock_macro.return_value = snap
        out = await collect_llm_decisions(orch)

    assert "OTC_FCHI" in out or "frxEURUSD" in out
