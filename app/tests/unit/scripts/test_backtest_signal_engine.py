"""Testes do motor de sinais do backtest Medallion (sem rede)."""

from scripts.backtest.backtest_cluster_runtime import BacktestClusterRuntime
from scripts.backtest.signal_engine import (
    apply_backtest_refresh_execute_gate,
    derive_quant_cluster_tags,
    resolve_orders_at_bar,
)
from scripts.backtest.snapshot_engine import build_snapshot_at_bar
from src.application.services.llm.macro_config import MacroSnapshot
from src.domain.models.trade import TradeDirection


def test_derive_quant_cluster_tags_risk_on():
    snap = MacroSnapshot(
        us_dir="up",
        eu_dir="up",
        us_strength=0.9,
        eu_strength=0.85,
        tag="risk_on",
        eurusd_bias="CALL",
        cluster_status="",
        macro_block="",
        fx_reference_line="",
        us_parts=(),
        eu_parts=(),
    )
    us, eu = derive_quant_cluster_tags(snap, {"confluence_conviction_floor": 0.6})
    assert us == "CALL"
    assert eu == "CALL"


def test_resolve_orders_risk_on_selects_us_index():
    snap = MacroSnapshot(
        us_dir="up",
        eu_dir="up",
        us_strength=0.9,
        eu_strength=0.85,
        tag="risk_on",
        eurusd_bias="CALL",
        cluster_status="",
        macro_block="",
        fx_reference_line="",
        us_parts=(),
        eu_parts=(),
        statarb_spreads={"OTC_SPC": -2.5, "OTC_NDX": -1.0, "OTC_DJI": -0.5},
        hmm_state=0,
        hmm_prob=0.9,
    )
    config = {
        "anchor": "frxEURUSD",
        "llm": {"min_conviction_execute": 0.55},
        "strategy": {
            "clusters": {
                "us": ["OTC_SPC", "OTC_NDX", "OTC_DJI"],
                "eu": ["OTC_FCHI", "OTC_GDAXI", "OTC_FTSE", "OTC_SSMI"],
            },
            "correlation": {
                "enabled": True,
                "exclusive_cluster_by_macro": True,
                "statarb_index_select_enabled": True,
                "statarb_index_max_per_cluster": 1,
                "statarb_index_min_abs_z": 0.85,
                "statarb_require_z_align": True,
                "statarb_weak_leader_on_no_align": False,
            },
            "macro": {
                "allowed_execute_tags": ("risk_on", "risk_off"),
                "confluence_conviction_floor": 0.55,
                "assert_min_hmm_prob": 0.0,
                "statarb_z_threshold": 2.5,
                "statarb_min_abs_z_by_tag": {"risk_on": 0.50},
            },
        },
    }
    all_syms = [
        "frxEURUSD",
        "OTC_SPC",
        "OTC_NDX",
        "OTC_DJI",
        "OTC_FCHI",
        "OTC_GDAXI",
        "OTC_FTSE",
        "OTC_SSMI",
    ]
    runtime = BacktestClusterRuntime(config, symbols=all_syms, anchor="frxEURUSD")
    orders = resolve_orders_at_bar(
        bar_index=10,
        snapshot=snap,
        config=config,
        us_symbols=["OTC_SPC", "OTC_NDX", "OTC_DJI"],
        eu_symbols=["OTC_FCHI", "OTC_GDAXI", "OTC_FTSE", "OTC_SSMI"],
        all_symbols=all_syms,
        anchor="frxEURUSD",
        runtime=runtime,
    )
    assert orders
    symbols = {o.symbol for o in orders}
    assert symbols.issubset({"OTC_SPC", "OTC_NDX", "OTC_DJI"})
    assert len(symbols) <= 1


def test_build_snapshot_propagates_index_m5_dir():
    us = ["R_50"]
    eu: list[str] = []
    m15 = {
        "R_100": [100.0 + i * 0.1 for i in range(40)],
        "R_50": [50.0 + i * 0.2 for i in range(40)],
    }
    m5 = {"R_50": [50.0 + i * 0.05 for i in range(120)]}
    snap = build_snapshot_at_bar(
        bar_index=10,
        m15_closes=m15,
        m5_closes=m5,
        us_symbols=us,
        eu_symbols=eu,
        macro_cfg={"cluster_bars": 8, "statarb_lookback": 8},
        anchor="R_100",
    )
    assert isinstance(snap.index_m5_dir_by_symbol, dict)
    assert "R_50" in snap.index_m5_dir_by_symbol


def test_backtest_refresh_gate_blocks_risk_off_without_llm():
    config = {
        "anchor": "frxEURUSD",
        "orchestrator": {
            "cluster_refresh_execute_enabled": False,
            "cluster_refresh_execute_on_quant_validate": True,
        },
        "strategy": {"clusters": {"us": [], "eu": []}, "correlation": {}, "macro": {}},
    }
    runtime = BacktestClusterRuntime(config, symbols=["frxEURUSD", "OTC_DJI"], anchor="frxEURUSD")
    runtime._cluster_refresh_without_llm = True
    decisions = {
        "frxEURUSD": {"direction": TradeDirection.PUT, "metrics": {"macro_sentiment": "risk_off"}},
        "OTC_DJI": {"direction": TradeDirection.PUT, "metrics": {"execute": False, "macro_sentiment": "risk_off"}},
    }
    apply_backtest_refresh_execute_gate(runtime, decisions)
    assert decisions["OTC_DJI"]["metrics"]["execute"] is False
