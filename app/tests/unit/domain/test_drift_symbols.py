from src.domain.symbols.drift_symbols import (
    DEFAULT_ANCHOR,
    DRIFT_SYMBOLS,
    HEDGE_PEER,
    hedge_peer,
    is_high_side,
    sym_is_low_barrier,
)


def test_drift_symbols_constants():
    assert DEFAULT_ANCHOR == "RDBULL"
    assert DRIFT_SYMBOLS == ("RDBEAR", "RDBULL")
    assert HEDGE_PEER["RDBEAR"] == "RDBULL"
    assert HEDGE_PEER["RDBULL"] == "RDBEAR"


def test_hedge_peer_known_and_unknown():
    assert hedge_peer("RDBEAR") == "RDBULL"
    assert hedge_peer("RDBULL") == "RDBEAR"
    assert hedge_peer("UNKNOWN") is None


def test_is_high_side():
    assert is_high_side("RDBULL")
    assert not is_high_side("RDBEAR")


def test_sym_is_low_barrier_with_and_without_peer():
    assert sym_is_low_barrier("RDBEAR", "RDBULL")
    assert not sym_is_low_barrier("RDBULL", "RDBEAR")
    assert sym_is_low_barrier("RDBEAR")
    assert not sym_is_low_barrier("RDBULL")
    assert not sym_is_low_barrier("UNKNOWN")
