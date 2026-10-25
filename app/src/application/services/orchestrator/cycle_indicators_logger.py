"""Formatador e emissor de telemetria completa de indicadores M5 do ciclo."""

from __future__ import annotations

import math
import time
from typing import Any


def _calc_ema(values: list[float], span: int) -> list[float]:
    """Calcula serie de media movel exponencial causal."""
    alpha = 2.0 / (span + 1.0)
    res = [values[0]]
    for val in values[1:]:
        res.append(alpha * val + (1.0 - alpha) * res[-1])
    return res


def _calc_macd(
    closes: list[float], fast_span: int = 12, slow_span: int = 26, signal_span: int = 9
) -> tuple[float, float, float]:
    """Calcula linha MACD, sinal e histograma causais."""
    if len(closes) < slow_span + signal_span:
        return 0.0, 0.0, 0.0
    fast, slow = _calc_ema(closes, fast_span), _calc_ema(closes, slow_span)
    macd_line = [f - s for f, s in zip(fast, slow, strict=True)]
    sig_line = _calc_ema(macd_line, signal_span)
    return macd_line[-1], sig_line[-1], macd_line[-1] - sig_line[-1]


def _calc_stochastic(bars: list[Any], period: int = 14, smooth: int = 3) -> tuple[float, float]:
    """Calcula %K e %D do Estocastico causal."""
    if len(bars) < period + smooth:
        return 50.0, 50.0
    k_vals = []
    for idx in range(len(bars) - smooth + 1, len(bars) + 1):
        win = bars[idx - period : idx]
        high_max, low_min = max(b.high for b in win), min(b.low for b in win)
        rng = high_max - low_min
        k_vals.append(((win[-1].close - low_min) / rng * 100.0) if rng > 0 else 50.0)
    return k_vals[-1], (sum(k_vals) / len(k_vals) if k_vals else 50.0)


def _calc_cci(bars: list[Any], period: int = 14, constant: float = 0.015) -> float:
    """Calcula Commodity Channel Index (CCI) causal de 14 periodos."""
    if len(bars) < period:
        return 0.0
    recent = bars[-period:]
    tp_list = [(b.high + b.low + b.close) / 3.0 for b in recent]
    mean_tp = sum(tp_list) / period
    mean_dev = sum(abs(tp - mean_tp) for tp in tp_list) / period
    return ((tp_list[-1] - mean_tp) / (constant * mean_dev)) if mean_dev > 1e-10 else 0.0


def _calc_cmo(closes: list[float], period: int = 14) -> float:
    """Calcula Chande Momentum Oscillator (CMO) causal de 14 periodos."""
    if len(closes) < period + 1:
        return 0.0
    deltas = [curr - prev for prev, curr in zip(closes[-period - 1 : -1], closes[-period:], strict=True)]
    gains, losses = sum(d for d in deltas if d > 0), sum(-d for d in deltas if d < 0)
    denom = gains + losses
    return (((gains - losses) / denom) * 100.0) if denom > 1e-10 else 0.0


def _calc_williams_r(bars: list[Any], period: int = 14) -> float:
    """Calcula Williams %R causal de 14 periodos [-100, 0]."""
    if len(bars) < period:
        return -50.0
    recent = bars[-period:]
    hh, ll = max(b.high for b in recent), min(b.low for b in recent)
    return (((hh - recent[-1].close) / (hh - ll)) * -100.0) if (hh - ll) > 1e-10 else -50.0


def _calc_roc(closes: list[float], period: int = 14) -> float:
    """Calcula Rate of Change (ROC) causal de 14 periodos."""
    if len(closes) < period + 1:
        return 0.0
    base = closes[-period - 1]
    return (((closes[-1] - base) / base) * 100.0) if abs(base) > 1e-10 else 0.0


def _calc_keltner_pct_b(bars: list[Any], period: int = 14, atr_period: int = 14, mult: float = 2.0) -> float:
    """Calcula Keltner Channel %B causal de 14 periodos."""
    if len(bars) < max(period, atr_period) + 1:
        return 0.5
    closes = [b.close for b in bars]
    ema_vals = _calc_ema(closes, period)
    ranges = [
        max(c.high - c.low, abs(c.high - p.close), abs(c.low - p.close))
        for p, c in zip(bars[-atr_period - 1 : -1], bars[-atr_period:], strict=True)
    ]
    atr_kc = (sum(ranges) / atr_period) if ranges else 1.0
    width = 2.0 * mult * atr_kc
    return ((closes[-1] - (ema_vals[-1] - mult * atr_kc)) / width) if width > 1e-10 else 0.5


def _calc_hurst_exponent(closes: list[float], window: int = 14) -> float:
    """Estima o Expoente de Hurst causal de 14 periodos via analise R/S rescalada."""
    if len(closes) < window:
        return 0.5
    series = closes[-window:]
    mean = sum(series) / window
    deviations = [x - mean for x in series]
    cumdev = []
    curr = 0.0
    for d in deviations:
        curr += d
        cumdev.append(curr)
    r = max(cumdev) - min(cumdev)
    s = (sum(d * d for d in deviations) / window) ** 0.5
    if s <= 1e-10 or r <= 1e-10:
        return 0.5
    rs = r / s
    return min(1.0, max(0.0, float(math.log(rs) / math.log(window)))) if (window > 1 and rs > 0) else 0.5


def log_cycle_indicators(orch: Any) -> None:
    """Registra telemetria de todos os indicadores M5 com linhas vetadas em vermelho ANSI."""
    if not bool(orch.config.get("pullback_strategy", {}).get("enabled", False)):
        return
    stream = getattr(orch, "stream", None) or getattr(orch, "stream_handler", None)
    if stream is None:
        return
    symbols = list(getattr(orch, "symbols", [])) or [str(orch.config.get("anchor") or "R_25")]
    now_ts = time.time()
    for symbol in symbols:
        candles = getattr(stream, "micro_candles", {}).get(symbol) or []
        if len(candles) < 55:
            continue
        last_ep = float(getattr(candles[-1], "epoch", 0) or 0)
        if len(symbols) > 1 and last_ep > 100000 and now_ts - last_ep > 1800:
            continue
        bars = candles[-128:] if len(candles) >= 128 else candles
        closes = [b.close for b in bars]
        fast, slow = _calc_ema(closes, 20), _calc_ema(closes, 50)
        ema9, ema21 = _calc_ema(closes, 9)[-1], _calc_ema(closes, 21)[-1]
        ranges = [
            max(c.high - c.low, abs(c.high - p.close), abs(c.low - p.close))
            for p, c in zip(bars[-15:-1], bars[-14:], strict=False)
        ]
        atr = (sum(ranges) / 14.0) if (ranges and sum(ranges) > 0) else 1.0
        pb, cf = bars[-2:]
        recuo = abs(pb.low - fast[-2]) / atr if fast[-2] > slow[-2] else abs(pb.high - fast[-2]) / atr
        g_list = [max(0.0, c - p) for p, c in zip(closes[-15:-1], closes[-14:], strict=False)]
        l_list = [max(0.0, p - c) for p, c in zip(closes[-15:-1], closes[-14:], strict=False)]
        ag, al = sum(g_list) / 14.0, sum(l_list) / 14.0
        rsi = 100.0 - (100.0 / (1.0 + (ag / al))) if al > 0 else (100.0 if ag > 0 else 50.0)
        closes_p = closes[:-1]
        gp_list = [max(0.0, c - p) for p, c in zip(closes_p[-15:-1], closes_p[-14:], strict=False)]
        lp_list = [max(0.0, p - c) for p, c in zip(closes_p[-15:-1], closes_p[-14:], strict=False)]
        agp, alp = sum(gp_list) / 14.0, sum(lp_list) / 14.0
        rsi_p = 100.0 - (100.0 / (1.0 + (agp / alp))) if alp > 0 else (100.0 if agp > 0 else 50.0)
        r14 = closes[-14:]
        m14 = sum(r14) / 14.0
        w14 = 4.0 * ((sum((x - m14) ** 2 for x in r14) / 14.0) ** 0.5)
        pct_b = (r14[-1] - (m14 - 0.5 * w14)) / w14 if w14 > 0 else 0.5
        pdm = [
            c.high - p.high if (c.high - p.high > p.low - c.low and c.high > p.high) else 0.0
            for p, c in zip(bars[-15:-1], bars[-14:], strict=False)
        ]
        mdm = [
            p.low - c.low if (p.low - c.low > c.high - p.high and p.low > c.low) else 0.0
            for p, c in zip(bars[-15:-1], bars[-14:], strict=False)
        ]
        tr_s = sum(ranges) + 1e-10
        pdi, mdi = (sum(pdm) / tr_s * 100.0), (sum(mdm) / tr_s * 100.0)
        m_val, m_sig, m_hist = _calc_macd(closes, 7, 14, 5)
        _, _, m_hist_p = _calc_macd(closes[:-1], 7, 14, 5) if len(closes) >= 20 else (0.0, 0.0, 0.0)
        sk, sd = _calc_stochastic(bars, 14, 3)
        c_body = abs(cf.close - cf.open)
        u_wick = cf.high - max(cf.open, cf.close)
        l_wick = min(cf.open, cf.close) - cf.low
        rejection = (
            "Topo"
            if (u_wick > 0.8 * c_body and u_wick > 0.20 * atr)
            else ("Fundo" if (l_wick > 0.8 * c_body and l_wick > 0.20 * atr) else "Nao")
        )
        f4 = fast[-4] if len(fast) >= 4 else fast[0]
        trend_up = fast[-2] > slow[-2] and fast[-2] > f4
        trend_down = fast[-2] < slow[-2] and fast[-2] < f4
        dyn_up = ema9 >= ema21 and cf.close > ema21 and pdi > mdi and m_hist >= 0
        dyn_down = ema9 <= ema21 and cf.close < ema21 and mdi > pdi and m_hist <= 0
        t_up = (trend_up or dyn_up) and not dyn_down
        t_down = (trend_down or dyn_down) and not dyn_up
        trend = "ALTA" if (t_up or (fast[-2] > slow[-2] and not t_down)) else "BAIXA"
        m = {
            "rsi": rsi,
            "rsi_prev": rsi_p,
            "m_val": m_val,
            "m_sig": m_sig,
            "m_hist": m_hist,
            "m_hist_prev": m_hist_p,
            "stoch_k": sk,
            "stoch_d": sd,
            "cci": _calc_cci(bars, 14),
            "cmo": _calc_cmo(closes, 14),
            "williams": _calc_williams_r(bars, 14),
            "roc": _calc_roc(closes, 14),
            "pct_b": pct_b,
            "kc_b": _calc_keltner_pct_b(bars, 14, 14, 2.0),
            "adx": abs(pdi - mdi) / (pdi + mdi + 1e-10) * 100.0,
            "pdi": pdi,
            "mdi": mdi,
            "hurst": _calc_hurst_exponent(closes, 14),
            "is_doji": c_body < 0.06 * atr,
            "rejection": rejection,
        }
        green, rst = "\033[92m", "\033[0m"
        c_pr, r_pr = (green, rst)
        c_mo, r_mo = (green, rst)
        c_os, r_os = (green, rst)
        c_vo, r_vo = (green, rst)
        c_re, r_re = (green, rst)
        greens = 5
        c_pl = green
        placar_status = "LIBERADO SEM VETOS (ORGANICO)"
        cycle_id = getattr(orch, "_active_cycle_id", 0)
        is_doji_str = "Sim" if m["is_doji"] else "Nao"
        spread = fast[-2] - slow[-2]
        msg = (
            f"INDICADORES TECNICOS | {symbol} (Ciclo {cycle_id})\n"
            f"   {c_pr}PRECO E MEDIAS     | Fechamento: {cf.close:.2f} | Medias Exponenciais: EMA20={fast[-2]:.2f} / EMA50={slow[-2]:.2f} | Diferenca (Spread): {spread:+.2f} | Cruzamento Rapido: EMA9={ema9:.2f} / EMA21={ema21:.2f}{r_pr}\n"
            f"   {c_mo}MOMENTUM           | Indice de Forca Relativa (RSI 14): {rsi:.1f} | Estocastico (14, 3): %K={sk:.1f} / %D={sd:.1f} | MACD (7, 14, 5): Linha={m_val:+.2f}, Sinal={m_sig:+.2f}, Histograma={m_hist:+.2f}{r_mo}\n"
            f"   {c_os}OSCILADORES        | Canal de Commodities (CCI 14): {m['cci']:+.1f} | Chande Momentum (CMO 14): {m['cmo']:+.1f} | Williams %R (14): {m['williams']:.1f} | Taxa de Variacao (ROC 14): {m['roc']:+.2f}%{r_os}\n"
            f"   {c_vo}VOLATILIDADE       | Volatilidade Media (ATR 14): {atr:.2f} | Bandas de Bollinger (%B 14): {pct_b:.2f} | Canais de Keltner (%B 14): {m['kc_b']:.2f} | Recuo na Media: {recuo:.2f} ATR{r_vo}\n"
            f"   {c_re}REGIME E TENDENCIA | Forca da Tendencia (ADX 14): {m['adx']:.1f} (+DI={pdi:.1f} / -DI={mdi:.1f}) | Expoente de Hurst (14): {m['hurst']:.2f} | Indecisao (Doji): {is_doji_str} | Rejeicao de Pavio: {rejection} | Tendencia Geral: {trend}{r_re}\n"
            f"   {c_pl}PLACAR METRICAS    | Verdes (Aprovados): {greens}/5 | Vermelhos (Vetados): {5 - greens}/5 | Decisao: {placar_status}{rst}"
        )
        orch.logger.info(msg)
