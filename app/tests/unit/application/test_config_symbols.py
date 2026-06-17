from src.application.services.orchestrator.config_symbols import normalize_symbols_and_anchor, resolve_dl_train_symbols
from tests.market_symbols import ANCHOR, PAIR


def test_normalize_symbols_and_anchor_inserts_anchor_when_absent():
    anchor, symbols = normalize_symbols_and_anchor({"anchor": ANCHOR, "symbols": [PAIR]})
    assert anchor == ANCHOR
    assert symbols[0] == ANCHOR
    assert PAIR in symbols


def test_resolve_dl_train_symbols_defaults_to_active_symbols():
    config = {"anchor": ANCHOR, "symbols": [ANCHOR, PAIR]}
    assert resolve_dl_train_symbols(config) == [ANCHOR, PAIR]


def test_resolve_dl_train_symbols_honors_explicit_list():
    config = {
        "anchor": ANCHOR,
        "symbols": [ANCHOR, PAIR],
        "deep_learning": {"train_symbols": [PAIR]},
    }
    assert resolve_dl_train_symbols(config) == [PAIR]


def test_resolve_dl_train_symbols_empty_list_skips_training():
    config = {"anchor": ANCHOR, "symbols": [ANCHOR, PAIR], "deep_learning": {"train_symbols": []}}
    assert resolve_dl_train_symbols(config) == []
