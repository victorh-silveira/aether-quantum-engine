from unittest.mock import MagicMock

import pytest

from src.application.services.llm.global_macro_confluence import ClusterVote, build_macro_snapshot, empty_macro_snapshot
from src.application.services.llm.macro_snapshot_build import apply_m5_fallback_to_snapshot, macro_snapshot_from_votes
from src.application.services.llm.macro_snapshot_fetch import fetch_macro_snapshot


def test_apply_m5_fallback_upgrades_flat_us_to_down():
    m15 = {
        "R_25": [100.0, 100.03],
        "R_50": [100.0, 100.02],
        "R_75": [100.0, 95.0],
        "1HZ100V": [100.0, 94.5],
    }
    snap = build_macro_snapshot(
        ["R_25", "R_50"],
        ["R_75", "1HZ100V"],
        m15,
        {"min_indices_for_vote": 2, "cluster_min_move_pct": 0.10},
    )
    assert snap.us_dir == "flat"
    m5 = {
        "R_25": [100.0, 99.0, 98.5, 98.0],
        "R_50": [100.0, 99.2, 98.8, 98.3],
    }
    out = apply_m5_fallback_to_snapshot(
        snap,
        us_symbols=["R_25", "R_50"],
        eu_symbols=["R_75", "1HZ100V"],
        fallback_closes=m5,
        macro_cfg={"cluster_fallback_min_move_pct": 0.05, "min_indices_for_vote": 2},
    )
    assert out.us_dir == "down"
    assert out.eu_dir == snap.eu_dir
    assert "CLUSTER_M5_FALLBACK_US" in out.macro_block


def test_apply_m5_fallback_no_change_when_disabled():
    snap = empty_macro_snapshot()
    out = apply_m5_fallback_to_snapshot(
        snap,
        us_symbols=["R_25"],
        eu_symbols=["R_75"],
        fallback_closes={"R_25": [100.0, 95.0]},
        macro_cfg={"cluster_use_m5_fallback_when_flat": False},
    )
    assert out is snap


def test_apply_m5_fallback_upgrades_flat_eu_to_down():
    m15 = {
        "R_25": [100.0, 105.0],
        "R_50": [100.0, 104.0],
        "R_75": [100.0, 100.02],
        "1HZ100V": [100.0, 100.01],
    }
    snap = build_macro_snapshot(
        ["R_25", "R_50"],
        ["R_75", "1HZ100V"],
        m15,
        {"min_indices_for_vote": 2, "cluster_min_move_pct": 0.10},
    )
    assert snap.eu_dir == "flat"
    m5 = {
        "R_75": [100.0, 99.0, 98.5, 98.0],
        "1HZ100V": [100.0, 99.2, 98.8, 98.3],
    }
    out = apply_m5_fallback_to_snapshot(
        snap,
        us_symbols=["R_25", "R_50"],
        eu_symbols=["R_75", "1HZ100V"],
        fallback_closes=m5,
        macro_cfg={"cluster_fallback_min_move_pct": 0.05, "min_indices_for_vote": 2},
    )
    assert out.eu_dir == "down"
    assert "CLUSTER_M5_FALLBACK_EU" in out.macro_block


@pytest.mark.asyncio
async def test_fetch_macro_snapshot_applies_m5_when_m15_flat():
    class DummyStream:
        @staticmethod
        async def fetch_candle_closes(sym, gran, _bars):
            if gran == 900:
                return [100.0, 100.03]
            return [100.0, 99.0, 98.5, 98.0, 97.5] if sym == "R_25" else [100.0, 98.8, 98.2, 97.8, 97.2]

    DummyStream.fetch_candle_closes._is_coroutine = True

    class DummyOrch:
        stream = DummyStream()
        logger = MagicMock()
        config = {
            "strategy": {
                "clusters": {"us": ["R_25", "R_50"], "eu": ["R_75"]},
                "macro": {
                    "min_indices_for_vote": 2,
                    "cluster_min_move_pct": 0.10,
                    "cluster_use_m5_fallback_when_flat": True,
                    "cluster_fallback_min_move_pct": 0.05,
                },
            }
        }

    orch = DummyOrch()
    snap = await fetch_macro_snapshot(orch, {})
    assert snap.us_dir == "down"
    assert "CLUSTER_M5_FALLBACK_US" in snap.macro_block


@pytest.mark.asyncio
async def test_fetch_macro_snapshot_empty_when_stream_missing():
    class NoStream:
        pass

    snap = await fetch_macro_snapshot(NoStream(), {})
    assert snap.tag == "indefinido"


def test_macro_snapshot_from_votes_suffix():
    us = ClusterVote("up", 1.0, ("S&P: RISE",))
    eu = ClusterVote("down", 1.0, ("DAX: FALL",))
    snap = macro_snapshot_from_votes(us, eu, cluster_suffix="CLUSTER_M5_FALLBACK_EU")
    assert snap.tag == "divergence_us_leads"
    assert "CLUSTER_M5_FALLBACK_EU" in snap.macro_block
