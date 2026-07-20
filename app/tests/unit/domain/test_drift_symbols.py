from src.domain.symbols.drift_symbols import (
    DEFAULT_ANCHOR,
    DRIFT_SYMBOLS,
    HEDGE_PEER,
    TRADING_SYMBOLS,
    hedge_peer,
    is_high_side,
    sym_is_low_barrier,
)


def test_drift_symbols_constants():
    assert DEFAULT_ANCHOR == "R_10"
    assert TRADING_SYMBOLS == ("R_10",)
    assert DRIFT_SYMBOLS == ("R_10",)
    assert HEDGE_PEER == {}


def test_hedge_peer_known_and_unknown():
    assert hedge_peer("R_10") is None
    assert hedge_peer("UNKNOWN") is None


def test_is_high_side():
    assert not is_high_side("R_10")
    assert not is_high_side("UNKNOWN")


def test_sym_is_low_barrier_with_and_without_peer():
    assert not sym_is_low_barrier("R_10", "R_10")
    assert not sym_is_low_barrier("R_10")
    assert not sym_is_low_barrier("UNKNOWN")
