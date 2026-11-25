"""Estrategia Persistent Trend v3 (5 ramos Wilder DI + H1 EMA20/50) 100% causal em M15."""

from __future__ import annotations

from src.indicator.domain import Candle
from src.indicator.pullback_indicators import ema_direct


TOXIC_HOURS_UTC: frozenset[int] = frozenset({3, 6, 19})
MIN_H1_BARS = 51


def _true_range(curr: Candle, prev: Candle) -> float:
    """Calcula True Range entre a vela atual e o fechamento anterior."""
    return max(curr.high - curr.low, abs(curr.high - prev.close), abs(curr.low - prev.close))


def calc_wilder_di_adx(bars: list[Candle], period: int) -> tuple[float, float]:
    """Calcula (DI+ - DI-, ADX) causais pelo metodo Wilder EWM."""
    n = len(bars)
    if n <= period or period <= 0:
        return 0.0, 0.0
    alpha = 1.0 / period
    one_minus = 1.0 - alpha
    ewm_tr = bars[0].high - bars[0].low
    ewm_pdm = 0.0
    ewm_mdm = 0.0
    ewm_dx = 0.0
    for idx in range(1, n):
        cur, prev = bars[idx], bars[idx - 1]
        tr = _true_range(cur, prev)
        up = cur.high - prev.high
        dn = prev.low - cur.low
        pdm = up if (up > dn and up > 0.0) else 0.0
        mdm = dn if (dn > up and dn > 0.0) else 0.0
        ewm_tr = one_minus * ewm_tr + alpha * tr
        ewm_pdm = one_minus * ewm_pdm + alpha * pdm
        ewm_mdm = one_minus * ewm_mdm + alpha * mdm
        if ewm_tr > 1e-12:
            pdi = 100.0 * ewm_pdm / ewm_tr
            mdi = 100.0 * ewm_mdm / ewm_tr
            di_sum = pdi + mdi
            dx = (100.0 * abs(pdi - mdi) / di_sum) if di_sum > 1e-12 else 0.0
            ewm_dx = one_minus * ewm_dx + alpha * dx
    if ewm_tr <= 1e-12:
        return 0.0, 0.0
    return 100.0 * (ewm_pdm - ewm_mdm) / ewm_tr, ewm_dx


def _resample_h1_closes_causal(bars: list[Candle], cutoff_epoch: int) -> list[float]:
    """Agrega fechamentos H1 estritamente fechados ate cutoff_epoch a partir de M15."""
    closes: list[float] = []
    cur_hour = -1
    last_close = 0.0
    for bar in bars:
        hour_epoch = (bar.epoch // 3600) * 3600
        if hour_epoch != cur_hour:
            if cur_hour >= 0 and cur_hour + 3600 <= cutoff_epoch:
                closes.append(last_close)
            cur_hour = hour_epoch
        last_close = bar.close
    if cur_hour >= 0 and cur_hour + 3600 <= cutoff_epoch:
        closes.append(last_close)
    return closes


def _resolve_h1_trend(bars: list[Candle], h1_bars: list[Candle] | None) -> int:
    """Retorna tendencia H1 causal (+1 EMA20 > EMA50, -1 caso contrario, 0 sem historico)."""
    cutoff_epoch = bars[-1].epoch + 900
    if h1_bars:
        closes = [bar.close for bar in h1_bars if bar.epoch + 3600 <= cutoff_epoch]
    else:
        closes = _resample_h1_closes_causal(bars, cutoff_epoch)
    if len(closes) < MIN_H1_BARS:
        return 0
    ema20 = ema_direct(closes, 20)[-1]
    ema50 = ema_direct(closes, 50)[-1]
    return 1 if ema20 > ema50 else -1


def _candle_dir(bar: Candle) -> int:
    """Retorna direcao assinada da vela (+1 alta, -1 baixa, 0 empate)."""
    if bar.close > bar.open:
        return 1
    if bar.close < bar.open:
        return -1
    return 0


def _branch_h1_di(h1_trend: int, di_diff: float, threshold: float) -> str | None:
    """Ramos B1/B2: alinhamento entre tendencia H1 (EMA20/50) e Wilder DI em M15."""
    if h1_trend > 0 and di_diff > threshold:
        return "CALL"
    if h1_trend < 0 and di_diff < -threshold:
        return "PUT"
    return None


def _branch_di_pullback(di_diff: float, dir0: int, threshold: float) -> str | None:
    """Ramo B3: pullback de 1 vela contra pressao direcional Wilder DI(14)."""
    if di_diff > threshold and dir0 < 0:
        return "CALL"
    if di_diff < -threshold and dir0 > 0:
        return "PUT"
    return None


def _branch_di_continuation(di_diff: float, dir0: int, threshold: float) -> str | None:
    """Ramo B4: continuacao de 1 vela a favor da pressao direcional Wilder DI(28)."""
    if di_diff > threshold and dir0 > 0:
        return "CALL"
    if di_diff < -threshold and dir0 < 0:
        return "PUT"
    return None


def _branch_di_adx(di_diff: float, adx: float, adx_min: float, di_min: float) -> str | None:
    """Ramo B5: tendencia forte ADX(20) combinada com pressao direcional Wilder DI(20)."""
    if adx < adx_min:
        return None
    if di_diff > di_min:
        return "CALL"
    if di_diff < -di_min:
        return "PUT"
    return None


def evaluate_persistent_trend_v3(
    bars: list[Candle],
    h1_bars: list[Candle] | None = None,
) -> tuple[str, str]:
    """Avalia o portfolio persistent_trend_v3 (v3_balanced_5b_noToxHr) em M15 causal."""
    if len(bars) < 51 or any(bar.high == bar.low for bar in bars[-2:]):
        return "SEM SINAL", "mercado_sem_oscilacao"
    if (bars[-1].epoch // 3600) % 24 in TOXIC_HOURS_UTC:
        return "SEM SINAL", "hora_toxica_utc"
    di14, _ = calc_wilder_di_adx(bars, 14)
    di20, adx20 = calc_wilder_di_adx(bars, 20)
    di24, _ = calc_wilder_di_adx(bars, 24)
    di28, _ = calc_wilder_di_adx(bars, 28)
    h1_trend = _resolve_h1_trend(bars, h1_bars)
    dir0 = _candle_dir(bars[-1])
    candidates = (
        _branch_h1_di(h1_trend, di24, 40.0),
        _branch_h1_di(h1_trend, di14, 45.0),
        _branch_di_pullback(di14, dir0, 45.0),
        _branch_di_continuation(di28, dir0, 38.0),
        _branch_di_adx(di20, adx20, 35.0, 40.0),
    )
    hits = [side for side in candidates if side is not None]
    if not hits:
        return "SEM SINAL", "sem_setup_persistent_trend"
    if len(set(hits)) > 1:
        return "SEM SINAL", "conflito_persistent_trend"
    return hits[0], "setup_confirmado"
