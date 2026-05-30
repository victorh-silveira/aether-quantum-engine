from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from src.application.services.llm.llm_bridge import collect_llm_decisions
from src.application.services.llm.llm_cluster_refresh import (
    cluster_refresh_due,
    merge_macro_snapshot_into_metrics,
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
        statarb_spreads={"R_75": 1.1, "1HZ100V": 0.6},
        hmm_state=0,
        hmm_prob=0.9,
    )


def test_resolve_cluster_refresh_interval_falls_back_to_cycle():
    cfg = {"orchestrator": {"cycle_interval_seconds": 60}}
    assert resolve_cluster_refresh_interval_seconds(cfg) == 60


def test_merge_macro_snapshot_into_metrics_includes_m5_map():
    snap = MacroSnapshot(
        us_dir="up",
        eu_dir="down",
        us_strength=0.7,
        eu_strength=0.6,
        tag="divergence_us_leads",
        eurusd_bias="CALL",
        cluster_status="",
        macro_block="",
        fx_reference_line="",
        us_parts=(),
        eu_parts=(),
        index_m5_dir_by_symbol={"R_50": "down"},
    )
    out = merge_macro_snapshot_into_metrics({"us_cluster": "CALL"}, snap)
    assert out["index_m5_dir_by_symbol"] == {"R_50": "down"}


def test_cluster_refresh_due_false_when_interval_zero():
    orch = MagicMock()
    assert cluster_refresh_due(orch, now_epoch=100.0, interval_seconds=0) is False


def test_resolve_cluster_refresh_interval_explicit():
    cfg = {"orchestrator": {"cluster_refresh_interval_seconds": 45, "cycle_interval_seconds": 60}}
    assert resolve_cluster_refresh_interval_seconds(cfg) == 45


def test_refresh_returns_none_when_cache_empty():
    orch = MagicMock()
    orch.anchor = "R_100"
    orch._last_llm_decisions = {}
    assert refresh_cluster_decisions_from_cache(orch, _snapshot(), "C1") is None


def test_refresh_returns_cached_when_anchor_entry_invalid():
    orch = MagicMock()
    orch.anchor = "R_100"
    orch._last_llm_decisions = {"R_100": "bad"}
    out = refresh_cluster_decisions_from_cache(orch, _snapshot(), "C1")
    assert out == {"R_100": "bad"}


def test_refresh_returns_cached_when_direction_missing():
    orch = MagicMock()
    orch.anchor = "R_100"
    orch._last_llm_decisions = {"R_100": {"direction": "PUT", "metrics": {}}}
    out = refresh_cluster_decisions_from_cache(orch, _snapshot(), "C1")
    assert "R_100" in out


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
    orch.anchor = "R_100"
    orch.symbols = ["R_100", "R_75", "1HZ100V"]
    orch.config = {
        "strategy": {
            "clusters": {"us": [], "eu": ["R_75", "1HZ100V"]},
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
        "statarb_spreads": {"R_75": 0.2},
    }
    orch._last_llm_decisions = {
        "R_100": {"direction": TradeDirection.PUT, "metrics": dict(metrics)},
    }
    out = refresh_cluster_decisions_from_cache(orch, _snapshot(), "C0001")
    assert out is not None
    assert "R_75" in out or "1HZ100V" in out
    orch.logger.info.assert_called()


@pytest.mark.asyncio
async def test_collect_llm_decisions_refreshes_cluster_from_cache():
    orch = MagicMock()
    orch.anchor = "R_100"
    orch.symbols = ["R_100", "R_75"]
    orch._active_cycle_id = 3
    orch._last_llm_macro_tag = "risk_off"
    orch._last_cluster_refresh_epoch = None
    orch.config = {
        "orchestrator": {"cluster_refresh_interval_seconds": 60},
        "strategy": {
            "correlation": {"enabled": True, "cluster_invert_on_block": False},
            "clusters": {"us": [], "eu": ["R_75"]},
            "macro": {"allowed_execute_tags": ("risk_off",)},
        },
        "llm": {"max_decision_latency_seconds": 10, "min_conviction_execute": 0.5},
    }
    orch._last_llm_decisions = {
        "R_100": {
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

    assert "R_75" in out or "R_100" in out
