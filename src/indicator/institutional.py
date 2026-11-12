"""Modulo de analise institucional (referencia OHLC, FVG e killzones BRT)."""

from __future__ import annotations

from datetime import datetime, timedelta, timezone
from math import isfinite, sqrt

from indicator.domain import Candle


BRT_TIMEZONE = timezone(timedelta(hours=-3))
# Killzones empiricas BRT acima do breakeven: London Mid (7), Pre-NY (9), NY Overlap (10, 12, 13) e London Fix/Pos-Fix (15, 16)
DEFAULT_KILLZONES_BRT = [7, 9, 10, 12, 13, 15, 16]


def session_price_reference(bars: list[Candle]) -> tuple[float, float, float]:
    """Calcula referencia temporal OHLC da sessao UTC, sem alegar volume inexistente."""
    if not bars:
        return 0.0, 0.0, 0.0
    last_day = bars[-1].epoch // 86400
    session_bars = [b for b in bars if b.epoch // 86400 == last_day]
    if len(session_bars) < 2:
        return 0.0, 0.0, 0.0

    sample_count = 0.0
    price_sum = 0.0
    typical_prices: list[float] = []
    weights: list[float] = []

    for b in session_bars:
        tp = (b.high + b.low + b.close) / 3.0
        w = 1.0
        typical_prices.append(tp)
        weights.append(w)
        price_sum += tp * w
        sample_count += w

    reference = price_sum / sample_count
    variance = sum(w * ((tp - reference) ** 2) for tp, w in zip(typical_prices, weights, strict=True)) / sample_count
    std_dev = sqrt(variance) if variance > 0.0 else 0.0
    upper_band = reference + 1.5 * std_dev
    lower_band = reference - 1.5 * std_dev
    return reference, upper_band, lower_band


def detect_fvg_mitigation(bars: list[Candle], atr: float = 1.0) -> tuple[bool, bool]:
    """Detecta Fair Value Gaps recentes e valida se o recuo mitigou a ineficiencia."""
    if len(bars) < 5 or atr <= 0 or not isfinite(atr):
        return False, False

    pullback = bars[-2]
    bull_mitigated = False
    bear_mitigated = False

    for i in range(max(2, len(bars) - 10), len(bars) - 2):
        bar_prev2 = bars[i - 2]
        bar_curr = bars[i]

        if bar_curr.low > bar_prev2.high:
            fvg_top = bar_curr.low
            fvg_bottom = bar_prev2.high
            if pullback.low <= fvg_top + 0.35 * atr and pullback.low >= fvg_bottom - 0.35 * atr:
                bull_mitigated = True

        if bar_curr.high < bar_prev2.low:
            fvg_bottom = bar_curr.high
            fvg_top = bar_prev2.low
            if pullback.high >= fvg_bottom - 0.35 * atr and pullback.high <= fvg_top + 0.35 * atr:
                bear_mitigated = True

    return bull_mitigated, bear_mitigated


def is_within_killzone(
    epoch: int,
    optimal_hours: list[int] | None = None,
    *,
    period_seconds: int = 0,
) -> bool:
    """Verifica se o timestamp da entrada (fechamento = epoch + period_seconds) pertence a Killzone BRT."""
    if optimal_hours is None:
        optimal_hours = DEFAULT_KILLZONES_BRT
    if not optimal_hours:
        return False
    entry_epoch = int(epoch) + max(0, int(period_seconds))
    hour_brt = datetime.fromtimestamp(entry_epoch, tz=BRT_TIMEZONE).hour
    return hour_brt in optimal_hours


_REAL_CASH_HOURS_BRT: dict[str, tuple[int, ...]] = {
    "OTC_AS51": (0, 1, 20, 21, 22, 23),
    "OTC_N225": (0, 1, 2, 3, 21, 22, 23),
    "OTC_HSI": (0, 1, 2, 3, 4, 22, 23),
    "OTC_GDAXI": tuple(range(5, 14)),
    "OTC_FCHI": tuple(range(5, 14)),
    "OTC_FTSE": tuple(range(5, 14)),
    "OTC_SX5E": tuple(range(5, 14)),
    "OTC_AEX": tuple(range(5, 14)),
    "OTC_SSMI": tuple(range(5, 14)),
    "OTC_SPC": tuple(range(10, 18)),
    "OTC_NDX": tuple(range(10, 18)),
    "OTC_DJI": tuple(range(10, 18)),
}
_DERIV_OTC_HOURS_BRT: dict[str, tuple[int, ...]] = {
    **dict.fromkeys(
        ("OTC_SPC", "OTC_NDX", "OTC_DJI", "OTC_GDAXI", "OTC_FCHI", "OTC_FTSE", "OTC_SX5E", "OTC_AEX", "OTC_SSMI"),
        tuple(range(4, 18)),
    ),
    "OTC_AS51": tuple(range(0, 16)) + (20, 21, 22, 23),
    "OTC_N225": tuple(range(0, 17)) + (20, 21, 22, 23),
    "OTC_HSI": (0, 1, 2, 3, 4, 22, 23),
}


def market_session_status_brt(symbol: str, epoch_ts: float | None = None) -> tuple[bool, bool]:
    """Retorna (bolsa_real_aberta, deriv_otc_ativo) em BRT sem acoplamento externo."""
    sym = str(symbol)
    if not sym.startswith("OTC_"):
        return True, True
    dt = (
        datetime.fromtimestamp(float(epoch_ts), tz=BRT_TIMEZONE)
        if epoch_ts is not None
        else datetime.now(tz=BRT_TIMEZONE)
    )
    wd = dt.weekday()
    if wd == 5:
        return False, False
    if wd == 6:
        asia = sym in ("OTC_AS51", "OTC_N225", "OTC_HSI") and dt.hour >= 20
        return asia, asia
    real_hours = _REAL_CASH_HOURS_BRT.get(sym)
    otc_hours = _DERIV_OTC_HOURS_BRT.get(sym)
    real_open = True if real_hours is None else dt.hour in real_hours
    otc_open = real_open if otc_hours is None else dt.hour in otc_hours
    return real_open, otc_open
