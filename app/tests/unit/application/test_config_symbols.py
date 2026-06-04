from src.application.services.orchestrator.config_symbols import normalize_symbols_and_anchor
from tests.market_symbols import ANCHOR, PAIR


def test_normalize_symbols_and_anchor_inserts_anchor_when_absent():
    anchor, symbols = normalize_symbols_and_anchor({"anchor": ANCHOR, "symbols": [PAIR]})
    assert anchor == ANCHOR
    assert symbols[0] == ANCHOR
    assert PAIR in symbols
