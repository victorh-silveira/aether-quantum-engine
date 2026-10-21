from src.domain.symbols.drift_symbols import (
    DEFAULT_ANCHOR,
    DRIFT_SYMBOLS,
    TRADING_SYMBOLS,
    hedge_peer,
    is_high_side,
    sym_is_low_barrier,
)


def test_drift_symbols_constants():
    assert DEFAULT_ANCHOR in (
        "R_10",
        "R_25",
        "R_50",
        "R_75",
        "R_100",
        "1HZ10V",
        "1HZ15V",
        "1HZ25V",
        "1HZ30V",
        "1HZ50V",
        "1HZ75V",
        "1HZ90V",
        "1HZ100V",
        "OTC_SPC",
        "OTC_DJI",
        "OTC_NDX",
        "OTC_GDAXI",
        "OTC_FTSE",
        "OTC_N225",
        "OTC_AS51",
        "OTC_SX5E",
        "OTC_AEX",
        "OTC_FCHI",
        "OTC_SSMI",
    )
    assert DEFAULT_ANCHOR in TRADING_SYMBOLS
    assert DRIFT_SYMBOLS == TRADING_SYMBOLS


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
