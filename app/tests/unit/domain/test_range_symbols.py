from src.domain.symbols.range_symbols import (
    DEFAULT_ANCHOR,
    HEDGE_PEER,
    RANGE_SYMBOLS,
    hedge_peer,
    is_high_side,
    sym_is_low_barrier,
)


def test_range_symbols_constants():
    assert DEFAULT_ANCHOR == "R_50"
    assert len(RANGE_SYMBOLS) == 5
    assert HEDGE_PEER["R_10"] == "R_100"


def test_hedge_peer_known_and_unknown():
    assert hedge_peer("R_25") == "R_75"
    assert hedge_peer("R_50") is None
    assert hedge_peer("UNKNOWN") is None


def test_is_high_side():
    assert is_high_side("R_100")
    assert is_high_side("R_75")
    assert not is_high_side("R_10")
    assert not is_high_side("R_50")


def test_sym_is_low_barrier_with_and_without_peer():
    assert sym_is_low_barrier("R_10", "R_100")
    assert not sym_is_low_barrier("R_100", "R_10")
    assert sym_is_low_barrier("R_25")
    assert not sym_is_low_barrier("R_50")
    assert not sym_is_low_barrier("R_50", "R_25")
