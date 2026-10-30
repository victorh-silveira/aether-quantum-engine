from datetime import datetime, timedelta, timezone

from src.domain.symbols.drift_symbols import (
    DEFAULT_ANCHOR,
    DRIFT_SYMBOLS,
    SYMBOL_DISPLAY_NAMES,
    TRADING_SYMBOLS,
    hedge_peer,
    is_high_side,
    is_otc_exchange_open_brt,
    sym_is_low_barrier,
    symbol_display_name,
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


def test_is_otc_exchange_open_brt_all_regions():
    brt = timezone(timedelta(hours=-3))
    wed_morning = datetime(2026, 10, 21, 7, 0, tzinfo=brt).timestamp()
    wed_afternoon = datetime(2026, 10, 21, 15, 0, tzinfo=brt).timestamp()
    wed_night = datetime(2026, 10, 21, 22, 0, tzinfo=brt).timestamp()
    sat_noon = datetime(2026, 10, 24, 12, 0, tzinfo=brt).timestamp()

    assert is_otc_exchange_open_brt("R_25") is True
    assert is_otc_exchange_open_brt("OTC_SSMI", sat_noon) is False
    assert is_otc_exchange_open_brt("OTC_SSMI", wed_morning) is True
    assert is_otc_exchange_open_brt("OTC_SSMI", wed_afternoon) is False
    assert is_otc_exchange_open_brt("OTC_SPC", wed_morning) is False
    assert is_otc_exchange_open_brt("OTC_SPC", wed_afternoon) is True
    assert is_otc_exchange_open_brt("OTC_N225", wed_night) is True
    assert is_otc_exchange_open_brt("OTC_N225", wed_afternoon) is False
    assert is_otc_exchange_open_brt("OTC_CUSTOM", wed_afternoon) is True


def test_symbol_display_name():
    assert symbol_display_name("OTC_AEX") == "Netherlands 25"
    assert symbol_display_name("OTC_DJI") == "Wall Street 30"
    assert symbol_display_name("otc_spc") == "US 500"
    assert symbol_display_name("UNKNOWN_SYM") == "UNKNOWN_SYM"
    assert len(SYMBOL_DISPLAY_NAMES) == 12
