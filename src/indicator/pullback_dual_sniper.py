"""Avaliador bimodal Sniper M15 (T18 Tendencia MTF + C08 Compressao Cross-Asset)."""

from __future__ import annotations

from src.indicator.domain import Candle
from src.indicator.pullback_indicators import calc_ema


EUROPE_BLOCK = frozenset({"OTC_GDAXI", "OTC_FCHI", "OTC_FTSE", "OTC_SX5E", "OTC_AEX", "OTC_SSMI"})
ASIA_PAC_BLOCK = frozenset({"OTC_N225", "OTC_AS51", "OTC_HSI"})
US_BLOCK = frozenset({"OTC_SPC", "OTC_NDX", "OTC_DJI"})

BLOCKS_BY_SYMBOL: dict[str, frozenset[str]] = {
    **dict.fromkeys(EUROPE_BLOCK, EUROPE_BLOCK),
    **dict.fromkeys(ASIA_PAC_BLOCK, ASIA_PAC_BLOCK),
    **dict.fromkeys(US_BLOCK, US_BLOCK),
}
C08_ELIGIBLE_SYMBOLS = EUROPE_BLOCK | ASIA_PAC_BLOCK | US_BLOCK


def _sma_tr(bars: list[Candle], period: int) -> float:
    """Calcula media movel simples do True Range nas ultimas velas fechadas."""
    if len(bars) < 2 or period <= 0:
        return 0.0
    trs: list[float] = []
    start = max(1, len(bars) - period)
    for i in range(start, len(bars)):
        b = bars[i]
        prev_c = bars[i - 1].close
        trs.append(max(b.high - b.low, abs(b.high - prev_c), abs(b.low - prev_c)))
    return sum(trs) / len(trs) if trs else 0.0


def _closed_h1_trend(bars: list[Candle], h1_bars: list[Candle] | None = None) -> int:
    """Agrega velas M15 ou usa H1 diretas estritamente fechadas e retorna +1, -1 ou 0."""
    if not bars:
        return 0
    curr_close_ts = bars[-1].epoch + 900
    h1_closes: list[float] = []
    if h1_bars:
        h1_closes = [b.close for b in h1_bars if b.epoch + 3600 <= curr_close_ts]
    else:
        cur_h1_epoch = -1
        cur_close = 0.0
        for bar in bars:
            h_ep = (bar.epoch // 3600) * 3600
            if h_ep != cur_h1_epoch:
                if cur_h1_epoch >= 0 and cur_h1_epoch + 3600 <= curr_close_ts:
                    h1_closes.append(cur_close)
                cur_h1_epoch = h_ep
                cur_close = bar.close
            else:
                cur_close = bar.close
        if cur_h1_epoch >= 0 and cur_h1_epoch + 3600 <= curr_close_ts:
            h1_closes.append(cur_close)

    if len(h1_closes) < 51:
        return 0
    ema20 = calc_ema(h1_closes, 20)[-1]
    ema50 = calc_ema(h1_closes, 50)[-1]
    last_close = h1_closes[-1]
    if ema20 > ema50 and last_close > ema20:
        return 1
    if ema20 < ema50 and last_close < ema20:
        return -1
    return 0


def _evaluate_pb09_pb22(
    bars: list[Candle],
    ema9: float,
    ema20: float,
    ema50: float,
    atr14: float,
) -> str | None:
    """Avalia micro-pullback pos-impulso acima/abaixo da EMA9 (PB09 ou PB22)."""
    prev, curr = bars[-2], bars[-1]
    if curr.epoch - prev.epoch != 900 or atr14 <= 0.0:
        return None
    prev_body = abs(prev.close - prev.open)
    curr_body = abs(curr.close - curr.open)
    curr_rng = max(1e-10, curr.high - curr.low)
    uwick_ratio = (curr.high - max(curr.open, curr.close)) / curr_rng
    lwick_ratio = (min(curr.open, curr.close) - curr.low) / curr_rng

    if ema9 > ema20 > ema50 and prev.close > prev.open and curr.close < curr.open and curr.close > ema9:
        pb09 = prev_body >= 1.1 * atr14 and curr_body <= 0.45 * prev_body
        pb22 = prev_body >= 1.0 * atr14 and curr_body <= 0.50 * prev_body and uwick_ratio >= 0.25
        if pb09 or pb22:
            return "CALL"

    if ema9 < ema20 < ema50 and prev.close < prev.open and curr.close > curr.open and curr.close < ema9:
        pb09 = prev_body >= 1.1 * atr14 and curr_body <= 0.45 * prev_body
        pb22 = prev_body >= 1.0 * atr14 and curr_body <= 0.50 * prev_body and lwick_ratio >= 0.25
        if pb09 or pb22:
            return "PUT"

    return None


def _wilder_atr_adx(bars: list[Candle], period: int = 14) -> tuple[float, float]:
    """Calcula ATR e ADX causais pelo metodo de Wilder."""
    n = len(bars)
    if n <= 2 * period:
        return 0.0, 0.0
    trs: list[float] = [bars[0].high - bars[0].low]
    pdms: list[float] = [0.0]
    mdms: list[float] = [0.0]
    for i in range(1, n):
        cur, prev = bars[i], bars[i - 1]
        trs.append(max(cur.high - cur.low, abs(cur.high - prev.close), abs(cur.low - prev.close)))
        up = cur.high - prev.high
        dn = prev.low - cur.low
        pdms.append(up if (up > dn and up > 0.0) else 0.0)
        mdms.append(dn if (dn > up and dn > 0.0) else 0.0)
    sm_tr = sum(trs[1 : period + 1])
    sm_pdm = sum(pdms[1 : period + 1])
    sm_mdm = sum(mdms[1 : period + 1])
    atr_val = sm_tr / period
    dxs: list[float] = []
    for i in range(period, n):
        if i > period:
            sm_tr = sm_tr - (sm_tr / period) + trs[i]
            sm_pdm = sm_pdm - (sm_pdm / period) + pdms[i]
            sm_mdm = sm_mdm - (sm_mdm / period) + mdms[i]
            atr_val = sm_tr / period
        pdi = 100.0 * (sm_pdm / sm_tr) if sm_tr > 1e-12 else 0.0
        mdi = 100.0 * (sm_mdm / sm_tr) if sm_tr > 1e-12 else 0.0
        denom = pdi + mdi
        dxs.append(100.0 * abs(pdi - mdi) / denom if denom > 1e-12 else 0.0)
    adx_val = sum(dxs[1 : period + 1]) / period
    for dx in dxs[period + 1 :]:
        adx_val = ((adx_val * (period - 1)) + dx) / period
    return atr_val, adx_val


def _evaluate_t18_trend(
    bars: list[Candle],
    closes: list[float],
    h1_bars: list[Candle] | None = None,
) -> str | None:
    """Ramo 1 (T18): PB09|PB22 filtrado por ADX14 Wilder >= 20 e tendencia H1 fechada."""
    atr14, adx14 = _wilder_atr_adx(bars, 14)
    if adx14 < 20.0 or atr14 <= 0.0:
        return None
    ema9 = calc_ema(closes, 9)[-1]
    ema20 = calc_ema(closes, 20)[-1]
    ema50 = calc_ema(closes, 50)[-1]
    side = _evaluate_pb09_pb22(bars, ema9, ema20, ema50, atr14)
    if side is None:
        return None
    h1_dir = _closed_h1_trend(bars, h1_bars=h1_bars)
    if (side == "CALL" and h1_dir == 1) or (side == "PUT" and h1_dir == -1):
        return side
    return None


def _peer_directions_at_epoch(
    symbol: str,
    target_epoch: int,
    peer_bars: dict[str, list[Candle]] | None,
) -> list[int]:
    """Extrai direcao (+1/-1/0) dos ativos do mesmo bloco regional no epoch exato."""
    block = BLOCKS_BY_SYMBOL.get(symbol)
    if not block or not peer_bars:
        return []
    dirs: list[int] = []
    for peer_sym in block:
        candles = peer_bars.get(peer_sym)
        if not candles:
            continue
        for bar in reversed(candles[-8:]):
            if bar.epoch == target_epoch:
                dirs.append(1 if bar.close > bar.open else (-1 if bar.close < bar.open else 0))
                break
    return dirs


def _evaluate_c08_compression(
    bars: list[Candle],
    symbol: str,
    peer_bars: dict[str, list[Candle]] | None,
) -> str | None:
    """Ramo 2 (C08): reversao de barra forte em compressao de volatilidade (Europa/Asia-Pac/US)."""
    if symbol not in C08_ELIGIBLE_SYMBOLS or len(bars) < 51:
        return None
    curr = bars[-1]
    atr14 = _sma_tr(bars, 14)
    atr50 = _sma_tr(bars, 50)
    if atr14 <= 0.0 or atr50 <= 0.0:
        return None
    if (atr14 / atr50) >= 0.90:
        return None
    body_signed = curr.close - curr.open
    rng = max(1e-10, curr.high - curr.low)
    if (curr.high - max(curr.open, curr.close)) / rng <= 0.023:
        return None
    if (abs(body_signed) / rng) <= 0.55:
        return None
    ret_atr = body_signed / atr14
    if abs(ret_atr) <= 1.20:
        return None
    dirs = _peer_directions_at_epoch(symbol, curr.epoch, peer_bars)
    if len(dirs) < 3:
        return None
    ups = sum(1 for d in dirs if d > 0) / len(dirs)
    dns = sum(1 for d in dirs if d < 0) / len(dirs)
    if ups >= 0.75 and ret_atr > 1.20:
        return "PUT"
    if dns >= 0.75 and ret_atr < -1.20:
        return "CALL"
    return None


def evaluate_dual_sniper(
    bars: list[Candle],
    closes: list[float],
    symbol: str = "",
    peer_bars: dict[str, list[Candle]] | None = None,
    h1_bars: list[Candle] | None = None,
) -> tuple[str, str]:
    """Combina Ramo 1 (T18 Tendencia MTF) e Ramo 2 (C08 Compressao Cross-Asset)."""
    if len(bars) < 51 or any(bar.high == bar.low for bar in bars[-3:]):
        return "SEM SINAL", "mercado_sem_oscilacao"
    t18 = _evaluate_t18_trend(bars, closes, h1_bars=h1_bars)
    c08 = _evaluate_c08_compression(bars, symbol, peer_bars)
    if t18 and c08 and t18 != c08:
        return "SEM SINAL", "conflito_bimodal"
    if t18:
        return t18, "setup_confirmado"
    if c08:
        return c08, "setup_confirmado"
    return "SEM SINAL", "sem_setup_sniper"
