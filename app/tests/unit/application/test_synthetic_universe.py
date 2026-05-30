from src.application.services.llm.synthetic_universe import (
    DEFAULT_ANCHOR,
    DEFAULT_EU_CLUSTER,
    DEFAULT_US_CLUSTER,
    resolve_anchor,
)
from src.application.services.orchestrator.config_symbols import normalize_symbols_and_anchor


def test_resolve_anchor_from_correlation():
    cfg = {"anchor": "R_50", "strategy": {"correlation": {"anchor": "R_100"}}}
    assert resolve_anchor(cfg) == "R_100"


def test_normalize_symbols_synthetic_clusters():
    cfg = {
        "anchor": "R_100",
        "strategy": {
            "excluded_symbols": ["R_25"],
            "clusters": {"us": list(DEFAULT_US_CLUSTER), "eu": list(DEFAULT_EU_CLUSTER)},
        },
    }
    anchor, symbols = normalize_symbols_and_anchor(cfg)
    assert anchor == "R_100"
    assert "R_25" not in symbols
    assert DEFAULT_ANCHOR in symbols
    assert "R_10" in symbols
    assert "1HZ100V" in symbols
