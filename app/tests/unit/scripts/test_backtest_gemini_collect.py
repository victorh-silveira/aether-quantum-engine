"""Testes do coletor Gemini no backtest (sem rede)."""

from unittest.mock import patch

import pytest

from scripts.backtest.gemini_collect import collect_hft_orders_gemini


def _config():
    return {
        "anchor": "R_100",
        "orchestrator": {"cycle_interval_seconds": 15},
        "llm": {
            "min_conviction_execute": 0.55,
            "timeout_seconds": 5,
            "model": "gemini-2.5-flash",
        },
        "risk_management": {"params": {"entry_cooldown_ticks": 0, "payout_estimate": 0.95}},
        "strategy": {
            "clusters": {
                "us": ["R_25"],
                "eu": ["1HZ100V"],
            },
            "correlation": {
                "enabled": True,
                "exclusive_cluster_by_macro": False,
                "statarb_index_select_enabled": True,
                "statarb_index_min_abs_z": 0.0,
            },
            "macro": {
                "min_indices_for_vote": 1,
                "cluster_return_threshold_pct": 0.02,
                "confluence_conviction_floor": 0.0,
                "divergence_min_leader_strength": 0.0,
                "divergence_min_strength_gap": 0.0,
                "assert_min_hmm_prob": 0.0,
                "divergence_max_conviction": 0.99,
                "statarb_z_threshold": 2.5,
            },
        },
    }


def _market():
    closes = [100.0 * (1.012**i) for i in range(40)]
    syms = {"R_100": closes, "R_25": closes, "1HZ100V": closes}
    return {"m15": syms, "m5": syms}


@pytest.mark.asyncio
async def test_collect_hft_orders_gemini_uses_llm_clusters(tmp_path):
    config = _config()
    market = _market()
    payload = {
        "_direction_normalized": "CALL",
        "_conviction_normalized": 0.88,
        "us_cluster": "CALL",
        "eu_cluster": "PUT",
    }
    runtime = {
        "gemini_api_key": "ci-test-key",
        "llm_system": "",
        "max_decision_latency_seconds": 30.0,
    }

    async def _payload_into_cache(_orch, *, anchor, runtime, cache, bar_index, cache_file):
        cache[str(bar_index)] = payload
        return payload

    with (
        patch(
            "scripts.backtest.gemini_collect.resolve_llm_runtime",
            return_value=runtime,
        ),
        patch(
            "scripts.backtest.gemini_collect_api.gemini_payload_for_bar",
            side_effect=_payload_into_cache,
        ),
    ):
        orders, stats = await collect_hft_orders_gemini(
            config=config,
            m15=market["m15"],
            m5=market["m5"],
            us_syms=["R_25"],
            eu_syms=["1HZ100V"],
            all_syms=["R_100", "R_25", "1HZ100V"],
            anchor="R_100",
            macro_cfg=config["strategy"]["macro"],
            start=10,
            end=12,
            cache_path=str(tmp_path / "cache.jsonl"),
            max_llm_bars=3,
        )
    assert stats["gemini_llm_calls"] >= 1
    assert orders
    assert orders[0].direction.name in ("CALL", "PUT")
