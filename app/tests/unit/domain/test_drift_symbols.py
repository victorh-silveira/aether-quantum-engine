from src.domain.symbols.drift_symbols import (
    DEFAULT_ANCHOR,
    DRIFT_SYMBOLS,
    TRADING_SYMBOLS,
    hedge_peer,
    is_high_side,
    sym_is_low_barrier,
)


def test_drift_symbols_constants():
    assert DEFAULT_ANCHOR == "OTC_SPC"
    assert TRADING_SYMBOLS == ("OTC_SPC",)
    assert DRIFT_SYMBOLS == ("OTC_SPC",)
    assert DEFAULT_ANCHOR == "OTC_SPC"


def test_hedge_peer_known_and_unknown():
    assert hedge_peer("OTC_SPC") is None
    assert hedge_peer("UNKNOWN") is None


def test_is_high_side():
    assert not is_high_side("OTC_SPC")
    assert not is_high_side("UNKNOWN")


def test_sym_is_low_barrier_with_and_without_peer():
    assert not sym_is_low_barrier("OTC_SPC", "OTC_SPC")
    assert not sym_is_low_barrier("OTC_SPC")
    assert not sym_is_low_barrier("UNKNOWN")
