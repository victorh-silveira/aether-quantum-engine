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
        "anchor": "frxEURUSD",
        "strategy": {
            "excluded_symbols": ["OTC_SPC"],
            "clusters": {"us": ["OTC_SPC", "OTC_DJI"], "eu": ["OTC_FCHI"]},
        },
    }
    anchor, symbols = normalize_symbols_and_anchor(cfg)
    assert anchor == "frxEURUSD"
    assert "OTC_SPC" not in symbols
    assert "OTC_DJI" in symbols


def test_normalize_symbols_from_symbols_list_without_clusters():
    cfg = {
        "anchor": "frxEURUSD",
        "symbols": ["OTC_DJI", "OTC_SPC", "frxEURUSD"],
        "strategy": {"excluded_symbols": ["OTC_SPC"]},
    }
    anchor, symbols = normalize_symbols_and_anchor(cfg)
    assert anchor == "frxEURUSD"
    assert "frxEURUSD" in symbols
    assert "OTC_SPC" not in symbols
    assert "OTC_DJI" in symbols
