from src.domain.symbols.drift_symbols import (
    DEFAULT_ANCHOR,
    DRIFT_SYMBOLS,
    TRADING_SYMBOLS,
    hedge_peer,
    is_high_side,
    sym_is_low_barrier,
)


def test_drift_symbols_constants():
    assert DEFAULT_ANCHOR == "1HZ75V"
    assert TRADING_SYMBOLS == ("1HZ75V",)
    assert DRIFT_SYMBOLS == ("1HZ75V",)
    assert DEFAULT_ANCHOR == "1HZ75V"


def test_hedge_peer_known_and_unknown():
    assert hedge_peer("1HZ75V") is None
    assert hedge_peer("UNKNOWN") is None


def test_is_high_side():
    assert not is_high_side("1HZ75V")
    assert not is_high_side("UNKNOWN")


def test_sym_is_low_barrier_with_and_without_peer():
    assert not sym_is_low_barrier("1HZ75V", "1HZ75V")
    assert not sym_is_low_barrier("1HZ75V")
    assert not sym_is_low_barrier("UNKNOWN")
