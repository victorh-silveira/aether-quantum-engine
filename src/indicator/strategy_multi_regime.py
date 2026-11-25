"""Portfolio Multi-Regime de Alta Frequencia v2 (anti-cluster) 100% causal em M15."""

from __future__ import annotations

from src.indicator.domain import Candle
from src.indicator.pullback_dual_sniper import BLOCKS_BY_SYMBOL
from src.indicator.pullback_indicators import calc_cci


def _true_range(curr: Candle, prev: Candle) -> float:
    """Calcula True Range entre a vela atual e o fechamento anterior."""
    return max(curr.high - curr.low, abs(curr.high - prev.close), abs(curr.low - prev.close))


def _wilder_atr(bars: list[Candle], period: int) -> float:
    """Calcula ATR causal pelo metodo de suavizacao de Wilder."""
    n = len(bars)
    if n <= period or period <= 0:
        return 0.0
    total = 0.0
    for idx in range(1, period + 1):
        total += _true_range(bars[idx], bars[idx - 1])
    atr = total / period
    for idx in range(period + 1, n):
        atr = (atr * (period - 1) + _true_range(bars[idx], bars[idx - 1])) / period
    return atr


def calc_wilder_di_diff(bars: list[Candle], period: int = 5) -> float:
    """Calcula diferenca direcional causal Wilder EWM (DI+ menos DI-)."""
    n = len(bars)
    if n <= period or period <= 0:
        return 0.0
    alpha = 1.0 / period
    one_minus = 1.0 - alpha
    ewm_tr = bars[0].high - bars[0].low
    ewm_pdm = 0.0
    ewm_mdm = 0.0
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
    if ewm_tr <= 1e-12:
        return 0.0
    return 100.0 * (ewm_pdm - ewm_mdm) / ewm_tr


def _candle_dir(bar: Candle) -> int:
    """Retorna direcao assinada da vela (+1 alta, -1 baixa, 0 empate)."""
    if bar.close > bar.open:
        return 1
    if bar.close < bar.open:
        return -1
    return 0


def _is_ny_premarket_entry(sig_epoch: int) -> bool:
    """Identifica entradas na janela pre-mercado NY (10:45-12:15 UTC / 07:45-09:15 BRT)."""
    entry_mod = ((sig_epoch + 900) % 86400) // 60
    return 645 <= entry_mod <= 735


def _is_marubozu(bar: Candle) -> bool:
    """Identifica vela seca de corpo cheio (> 80% do range) caracteristica de breakout."""
    rng = bar.high - bar.low
    return rng <= 0.0 or (abs(bar.close - bar.open) / rng) > 0.80


def _peer_snapshot(
    symbol: str,
    target_epoch: int,
    peer_bars: dict[str, list[Candle]] | None,
) -> tuple[list[int], list[int], list[float]]:
    """Extrai direcoes globais, direcoes do bloco regional e razoes ATR14/ATR50 no epoch."""
    if not peer_bars:
        return [], [], []
    block = BLOCKS_BY_SYMBOL.get(symbol, frozenset())
    glb_dirs: list[int] = []
    blk_dirs: list[int] = []
    blk_ratios: list[float] = []
    for peer_sym, candles in peer_bars.items():
        if not candles:
            continue
        match_idx = -1
        low_bound = max(0, len(candles) - 8)
        for idx in range(len(candles) - 1, low_bound - 1, -1):
            if candles[idx].epoch == target_epoch:
                match_idx = idx
                break
        if match_idx < 0:
            continue
        direction = _candle_dir(candles[match_idx])
        glb_dirs.append(direction)
        if peer_sym in block:
            blk_dirs.append(direction)
            prefix = candles[: match_idx + 1]
            p_atr14 = _wilder_atr(prefix, 14)
            p_atr50 = _wilder_atr(prefix, 50)
            if p_atr14 > 0.0 and p_atr50 > 0.0:
                blk_ratios.append(p_atr14 / p_atr50)
    return glb_dirs, blk_dirs, blk_ratios


def _eval_climax_global(
    bars: list[Candle],
    atr14: float,
    atr50: float,
    glb_dirs: list[int],
    blk_ratios: list[float],
) -> str | None:
    """Ramo 1A: fade de expansao de True Range sob compressao regional e consenso global."""
    curr = bars[-1]
    if (curr.epoch // 3600) % 24 == 20 or _is_ny_premarket_entry(curr.epoch) or _is_marubozu(curr):
        return None
    dir0 = _candle_dir(curr)
    if dir0 == 0:
        return None
    if (_true_range(curr, bars[-2]) / atr14) <= 1.5:
        return None
    blk_mean_atrr = (sum(blk_ratios) / len(blk_ratios)) if blk_ratios else (atr14 / atr50)
    if blk_mean_atrr >= 0.95 or len(glb_dirs) < 3:
        return None
    glb_cons = sum(1 for d in glb_dirs if d == dir0) / len(glb_dirs)
    if glb_cons < 0.65:
        return None
    return "PUT" if dir0 > 0 else "CALL"


def _eval_q2_cci14(
    bars: list[Candle],
    atr14: float,
    atr50: float,
    blk_dirs: list[int],
) -> str | None:
    """Ramo 1B: exaustao CCI(14) na vela :30 sob compressao e consenso regional."""
    curr = bars[-1]
    if (
        (curr.epoch // 3600) % 24 == 20
        or (curr.epoch % 3600) // 60 != 30
        or _is_ny_premarket_entry(curr.epoch)
        or _is_marubozu(curr)
    ):
        return None
    if (atr14 / atr50) >= 1.0 or len(blk_dirs) < 2:
        return None
    cci14 = calc_cci(bars, 14)
    if cci14 <= -120.0:
        if (sum(1 for d in blk_dirs if d < 0) / len(blk_dirs)) >= 0.66:
            return "CALL"
    elif cci14 >= 120.0 and (sum(1 for d in blk_dirs if d > 0) / len(blk_dirs)) >= 0.66:
        return "PUT"
    return None


def _eval_di5_pullback(bars: list[Candle]) -> str | None:
    """Ramo 2: pullback de 1 vela contra pressao direcional Wilder DI(5) fora da virada :45."""
    curr = bars[-1]
    body = curr.close - curr.open
    if body == 0.0:
        return None
    di_diff = calc_wilder_di_diff(bars, 5)
    if di_diff > 50.0 and body < 0.0:
        return "CALL"
    if di_diff < -50.0 and body > 0.0:
        return "PUT"
    return None


def evaluate_multi_regime_hf(
    bars: list[Candle],
    symbol: str = "",
    peer_bars: dict[str, list[Candle]] | None = None,
) -> tuple[str, str]:
    """Avalia o portfolio multi-regime v2 com filtros anti-cluster e veto de conflito."""
    if len(bars) < 51 or any(bar.high == bar.low for bar in bars[-2:]):
        return "SEM SINAL", "mercado_sem_oscilacao"
    if (bars[-1].epoch % 3600) // 60 == 45:
        return "SEM SINAL", "virada_de_hora"
    atr14 = _wilder_atr(bars, 14)
    atr50 = _wilder_atr(bars, 50)
    if atr14 <= 0.0 or atr50 <= 0.0:
        return "SEM SINAL", "mercado_sem_oscilacao"
    glb_dirs, blk_dirs, blk_ratios = _peer_snapshot(symbol, bars[-1].epoch, peer_bars)
    candidates = (
        _eval_climax_global(bars, atr14, atr50, glb_dirs, blk_ratios),
        _eval_q2_cci14(bars, atr14, atr50, blk_dirs),
        _eval_di5_pullback(bars),
    )
    hits = [side for side in candidates if side is not None]
    if not hits:
        return "SEM SINAL", "sem_setup_multi_regime"
    if len(set(hits)) > 1:
        return "SEM SINAL", "conflito_multi_regime"
    return hits[0], "setup_confirmado"
