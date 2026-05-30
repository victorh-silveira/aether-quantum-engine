from scripts.backtest.data_loader import backtest_symbols
from src.application.services.llm.strategy_clusters import resolve_cluster_lists
from src.application.services.orchestrator.config_symbols import normalize_symbols_and_anchor


def test_resolve_cluster_lists_excludes_symbols():
    strategy = {
        "excluded_symbols": ["OTC_SPC"],
        "clusters": {"us": ["OTC_SPC", "OTC_NDX"], "eu": ["OTC_FCHI"]},
    }
    us, eu = resolve_cluster_lists(strategy)
    assert us == ["OTC_NDX"]
    assert eu == ["OTC_FCHI"]


def test_backtest_symbols_honors_excluded():
    cfg = {
        "anchor": "frxEURUSD",
        "strategy": {
            "excluded_symbols": ["OTC_SPC", "OTC_NDX"],
            "clusters": {"us": ["OTC_SPC", "OTC_NDX", "OTC_DJI"], "eu": ["OTC_FCHI"]},
        },
    }
    us, eu, all_syms, anchor = backtest_symbols(cfg)
    assert "OTC_SPC" not in us
    assert "OTC_NDX" not in us
    assert "OTC_DJI" in us
    assert anchor == "frxEURUSD"


def test_normalize_symbols_from_clusters():
    cfg = {
        "anchor": "R_100",
        "strategy": {
            "excluded_symbols": ["R_25"],
            "clusters": {"us": ["R_10", "R_25", "R_50"], "eu": ["R_75"]},
        },
    }
    anchor, symbols = normalize_symbols_and_anchor(cfg)
    assert anchor == "R_100"
    assert "R_25" not in symbols
    assert "R_50" in symbols


def test_backtest_symbols_synthetic():
    cfg = {
        "anchor": "R_100",
        "strategy": {
            "excluded_symbols": ["R_25"],
            "clusters": {"us": ["R_10", "R_25", "R_50"], "eu": ["R_75", "1HZ50V"]},
        },
    }
    us, eu, all_syms, anchor = backtest_symbols(cfg)
    assert anchor == "R_100"
    assert "R_25" not in us
    assert "R_50" in us
    assert "1HZ50V" in eu
    assert "R_100" in all_syms
