"""Setup fixo de tendencia, recuo e confirmacao em velas fechadas."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import isfinite, log

from indicator.domain import Candle
from indicator.institutional import detect_fvg_mitigation, is_within_killzone, session_vwap

SPEC = "ema20_50:warmup128:atr14:incl3:recuo0.5:corpo0.03_1.5:confirmacao1:m5"
VERSION = "pullback-1-" + sha256(SPEC.encode()).hexdigest()[:12]
MIN_BARS = 64
VARIANTS = (
    "base",
    "ema50_slope",
    "countertrend_retrace",
    "entry_distance",
    "improved",
    "pool_rsi_bb",
    "momentum_burst",
    "senior_institutional",
)


def version_for_variant(variant: str = "base") -> str:
    """Calcula a versao imutavel do setup para a variante especificada."""
    if variant == "base":
        return VERSION
    if variant in ("improved", "pool_rsi_bb"):
        spec = "ema20_50:warmup128:atr14:recuo1.8:dist1.4:rsi_bb_di_v3:m5"
        return "pullback-1-" + sha256(spec.encode()).hexdigest()[:12]
    if variant == "momentum_burst":
        spec = "momentum_burst:score4_conviction:hurst0.45:adx18:v1"
        return "momentum-1-" + sha256(spec.encode()).hexdigest()[:12]
    if variant == "senior_institutional":
        spec = "senior_institutional:vwap_fvg_killzone:v1"
        return "pullback-1-" + sha256(spec.encode()).hexdigest()[:12]
    spec = f"{SPEC}:{variant}"
    return "pullback-1-" + sha256(spec.encode()).hexdigest()[:12]


@dataclass(frozen=True)
class SetupResult:
    """Direcao do setup e motivo verificavel."""

    side: str
    reason: str
    version: str = VERSION


def _ema(values: list[float], span: int) -> list[float]:
    alpha = 2.0 / (span + 1.0)
    result = [values[0]]
    for value in values[1:]:
        result.append(alpha * value + (1.0 - alpha) * result[-1])
    return result


def _atr(candles: list[Candle]) -> float:
    ranges = []
    for previous, current in zip(candles[-15:-1], candles[-14:], strict=True):
        ranges.append(
            max(current.high - current.low, abs(current.high - previous.close), abs(current.low - previous.close))
        )
    return sum(ranges) / 14.0


def _rsi(closes: list[float], period: int = 14) -> float:
    """Calcula o RSI causal de 14 periodos."""
    if len(closes) < period + 1:
        return 50.0
    gains = []
    losses = []
    for prev, curr in zip(closes[-period - 1 : -1], closes[-period:], strict=True):
        chg = curr - prev
        gains.append(max(0.0, chg))
        losses.append(max(0.0, -chg))
    avg_g = sum(gains) / period
    avg_l = sum(losses) / period
    if avg_l == 0:
        return 100.0 if avg_g > 0 else 50.0
    rs = avg_g / avg_l
    return 100.0 - (100.0 / (1.0 + rs))


def _bollinger_pct_b(closes: list[float], period: int = 14, mult: float = 2.0) -> float:
    """Calcula o %B das Bandas de Bollinger causal de 14 periodos."""
    if len(closes) < period:
        return 0.5
    recent = closes[-period:]
    mean = sum(recent) / period
    variance = sum((x - mean) ** 2 for x in recent) / period
    std = variance**0.5
    upper = mean + mult * std
    lower = mean - mult * std
    width = upper - lower
    return (recent[-1] - lower) / width if width > 0 else 0.5


def _adx(bars: list[Candle], period: int = 14) -> tuple[float, float, float]:
    """Calcula ADX e direcionais +DI / -DI causais de 14 periodos."""
    if len(bars) < period + 1:
        return 0.0, 0.0, 0.0
    tr_sum = 0.0
    pdm_sum = 0.0
    mdm_sum = 0.0
    for prev, curr in zip(bars[-period - 1 : -1], bars[-period:], strict=True):
        tr = max(curr.high - curr.low, abs(curr.high - prev.close), abs(curr.low - prev.close))
        tr_sum += tr
        up = curr.high - prev.high
        down = prev.low - curr.low
        if up > down and up > 0:
            pdm_sum += up
        if down > up and down > 0:
            mdm_sum += down
    if tr_sum <= 0:
        return 0.0, 0.0, 0.0
    plus_di = (pdm_sum / tr_sum) * 100.0
    minus_di = (mdm_sum / tr_sum) * 100.0
    di_sum = plus_di + minus_di
    dx = (abs(plus_di - minus_di) / di_sum * 100.0) if di_sum > 0 else 0.0
    return dx, plus_di, minus_di


def _macd(
    closes: list[float], fast_span: int = 7, slow_span: int = 14, signal_span: int = 5
) -> tuple[float, float, float]:
    """Calcula linha MACD, sinal e histograma causais harmonizados com 14 periodos."""
    if len(closes) < slow_span + signal_span:
        return 0.0, 0.0, 0.0
    fast = _ema(closes, fast_span)
    slow = _ema(closes, slow_span)
    macd_line = [f - s for f, s in zip(fast, slow, strict=True)]
    sig_line = _ema(macd_line, signal_span)
    m_val = macd_line[-1]
    s_val = sig_line[-1]
    hist = m_val - s_val
    return m_val, s_val, hist


def _stochastic(bars: list[Candle], period: int = 14, smooth: int = 3) -> tuple[float, float]:
    """Calcula %K e %D do Estocastico causal."""
    if len(bars) < period + smooth:
        return 50.0, 50.0
    k_vals = []
    for idx in range(len(bars) - smooth + 1, len(bars) + 1):
        win = bars[idx - period : idx]
        high_max = max(b.high for b in win)
        low_min = min(b.low for b in win)
        rng = high_max - low_min
        close = win[-1].close
        k = ((close - low_min) / rng * 100.0) if rng > 0 else 50.0
        k_vals.append(k)
    pct_k = k_vals[-1]
    pct_d = sum(k_vals) / len(k_vals) if k_vals else 50.0
    return pct_k, pct_d


def _cci(bars: list[Candle], period: int = 14, constant: float = 0.015) -> float:
    """Calcula Commodity Channel Index (CCI) causal de 14 periodos."""
    if len(bars) < period:
        return 0.0
    recent = bars[-period:]
    tp_list = [(b.high + b.low + b.close) / 3.0 for b in recent]
    mean_tp = sum(tp_list) / period
    mean_dev = sum(abs(tp - mean_tp) for tp in tp_list) / period
    return ((tp_list[-1] - mean_tp) / (constant * mean_dev)) if mean_dev > 1e-10 else 0.0


def _cmo(closes: list[float], period: int = 14) -> float:
    """Calcula Chande Momentum Oscillator (CMO) causal de 14 periodos."""
    if len(closes) < period + 1:
        return 0.0
    deltas = [curr - prev for prev, curr in zip(closes[-period - 1 : -1], closes[-period:], strict=True)]
    gains = sum(d for d in deltas if d > 0)
    losses = sum(-d for d in deltas if d < 0)
    denom = gains + losses
    return (((gains - losses) / denom) * 100.0) if denom > 1e-10 else 0.0


def _williams_r(bars: list[Candle], period: int = 14) -> float:
    """Calcula Williams %R causal de 14 periodos [-100, 0]."""
    if len(bars) < period:
        return -50.0
    recent = bars[-period:]
    hh = max(b.high for b in recent)
    ll = min(b.low for b in recent)
    rng = hh - ll
    return (((hh - recent[-1].close) / rng) * -100.0) if rng > 1e-10 else -50.0


def _roc(closes: list[float], period: int = 14) -> float:
    """Calcula Rate of Change (ROC) causal de 14 periodos."""
    if len(closes) < period + 1:
        return 0.0
    base = closes[-period - 1]
    return (((closes[-1] - base) / base) * 100.0) if abs(base) > 1e-10 else 0.0


def _keltner_pct_b(bars: list[Candle], period: int = 14, atr_period: int = 14, mult: float = 2.0) -> float:
    """Calcula Keltner Channel %B causal de 14 periodos."""
    if len(bars) < max(period, atr_period) + 1:
        return 0.5
    closes = [b.close for b in bars]
    ema_vals = _ema(closes, period)
    ranges = [
        max(curr.high - curr.low, abs(curr.high - prev.close), abs(curr.low - prev.close))
        for prev, curr in zip(bars[-atr_period - 1 : -1], bars[-atr_period:], strict=True)
    ]
    atr_kc = (sum(ranges) / atr_period) if ranges else 1.0
    mid = ema_vals[-1]
    upper = mid + mult * atr_kc
    lower = mid - mult * atr_kc
    width = upper - lower
    return ((closes[-1] - lower) / width) if width > 1e-10 else 0.5


def _hurst_exponent(closes: list[float], window: int = 14) -> float:
    """Estima o Expoente de Hurst causal de 14 periodos via variancia rescalada."""
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
    variance = sum(d * d for d in deviations) / window
    s = variance**0.5
    if s <= 1e-10 or r <= 1e-10:
        return 0.5
    h = (log(r / s) / log(window)) if window > 1 else 0.5
    return min(1.0, max(0.0, float(h)))


def evaluate(
    candles: list[Candle],
    period: int,
    now: int,
    variant: str = "base",
    optimal_hours_brt: list[int] | None = None,
) -> SetupResult:
    """Avalia o ultimo recuo e a ultima confirmacao sem usar vela corrente."""
    version = version_for_variant(variant) if variant in VARIANTS else VERSION
    if period not in (300, 900, 3600, 86400):
        return SetupResult("SEM SINAL", "periodo_invalido", version=version)
    if len(candles) < MIN_BARS:
        return SetupResult("SEM SINAL", "historico_insuficiente", version=version)
    bars = candles[-MIN_BARS:]
    if any(
        candle.epoch % period
        or candle.high < max(candle.open, candle.close)
        or candle.low > min(candle.open, candle.close)
        or not all(isfinite(v) for v in (candle.open, candle.high, candle.low, candle.close))
        for candle in bars
    ):
        return SetupResult("SEM SINAL", "vela_invalida", version=version)
    if bars[-1].epoch - bars[-2].epoch != period:
        return SetupResult("SEM SINAL", "lacuna", version=version)
    if any(period < right.epoch - left.epoch <= 4 * period for left, right in zip(bars, bars[1:], strict=False)):
        return SetupResult("SEM SINAL", "lacuna", version=version)
    if bars[-1].epoch + period > now:
        return SetupResult("SEM SINAL", "vela_em_formacao", version=version)
    if bars[-1].epoch + 2 * period <= now:
        return SetupResult("SEM SINAL", "dados_atrasados", version=version)
    closes = [bar.close for bar in bars]
    fast = _ema(closes, 20)
    slow = _ema(closes, 50)
    pullback, confirmation = bars[-2:]
    atr = _atr(bars[:-1])
    if not isfinite(atr) or atr <= 0:
        return SetupResult("SEM SINAL", "atr_invalido", version=version)
    body = abs(confirmation.close - confirmation.open)
    max_body = 2.5 * atr
    if variant == "base" and not (0.02 * atr <= body <= max_body):
        return SetupResult("SEM SINAL", "corpo_fora_faixa", version=version)
    trend_up = fast[-2] > slow[-2] and fast[-4] < fast[-2]
    trend_down = fast[-2] < slow[-2] and fast[-4] > fast[-2]
    call = (
        trend_up
        and abs(pullback.low - fast[-2]) <= 2.20 * atr
        and pullback.close > slow[-2]
        and confirmation.close > fast[-1]
        and confirmation.close > confirmation.open
        and confirmation.close > pullback.high
    )
    put = (
        trend_down
        and abs(pullback.high - fast[-2]) <= 2.20 * atr
        and pullback.close < slow[-2]
        and confirmation.close < fast[-1]
        and confirmation.close < confirmation.open
        and confirmation.close < pullback.low
    )
    if variant == "ema50_slope":
        call = call and slow[-2] > slow[-5]
        put = put and slow[-2] < slow[-5]
    elif variant == "countertrend_retrace":
        call = call and pullback.close < pullback.open
        put = put and pullback.close > pullback.open
    elif variant == "entry_distance":
        call = call and abs(confirmation.close - fast[-1]) <= 1.40 * atr
        put = put and abs(confirmation.close - fast[-1]) <= 1.40 * atr
    elif variant in ("improved", "pool_rsi_bb", "momentum_burst", "senior_institutional"):
        if variant == "senior_institutional" and not is_within_killzone(confirmation.epoch, optimal_hours_brt):
            return SetupResult("SEM SINAL", "fora_da_killzone_brt", version=version)
        rsi = _rsi(closes, 14)
        rsi_prev = _rsi(closes[:-1], 14)
        pct_b = _bollinger_pct_b(closes, 14)
        dx, pdi, mdi = _adx(bars, 14)
        m_val, s_val, hist = _macd(closes, 7, 14, 5)
        _, _, hist_prev = _macd(closes[:-1], 7, 14, 5)
        stoch_k, stoch_d = _stochastic(bars, 14, 3)
        cci_val = _cci(bars, 14)
        cmo_val = _cmo(closes, 14)
        williams_val = _williams_r(bars, 14)
        roc_val = _roc(closes, 14)
        kc_b = _keltner_pct_b(bars, 14, 14, 2.0)
        hurst_val = _hurst_exponent(closes, 14)
        ema9 = _ema(closes, 9)[-1]
        ema21 = _ema(closes, 21)[-1]
        conf_body = abs(confirmation.close - confirmation.open)
        is_doji = conf_body < 0.06 * atr
        upper_wick = confirmation.high - max(confirmation.open, confirmation.close)
        lower_wick = min(confirmation.open, confirmation.close) - confirmation.low
        call_rejection = upper_wick > 0.8 * conf_body and upper_wick > 0.20 * atr
        put_rejection = lower_wick > 0.8 * conf_body and lower_wick > 0.20 * atr
        ema_spread = abs(fast[-2] - slow[-2])
        dyn_up = ema9 >= ema21 and confirmation.close > ema21 and pdi > mdi and hist >= 0
        dyn_down = ema9 <= ema21 and confirmation.close < ema21 and mdi > pdi and hist <= 0
        t_up = (trend_up or dyn_up) and not dyn_down
        t_down = (trend_down or dyn_down) and not dyn_up

        ok_pr_call = (
            t_up
            and (len(slow) < 5 or slow[-2] > slow[-5] or ema9 >= ema21)
            and ema9 >= ema21
            and ema_spread >= 0.15 * atr
            and confirmation.close > fast[-1]
            and confirmation.close > confirmation.open
            and confirmation.close > max(pullback.open, pullback.close)
        )
        ok_mo_call = (
            42.0 <= rsi <= 72.0
            and (rsi >= rsi_prev - 1.5)
            and (hist >= hist_prev or m_val >= s_val or rsi >= 52.0)
            and (22.0 <= stoch_k <= 85.0)
        )
        ok_os_call = (
            -50.0 <= cci_val <= 180.0 and cmo_val >= -15.0 and -82.0 <= williams_val <= -10.0 and roc_val >= -0.15
        )
        ok_vo_call = (
            0.18 <= pct_b <= 0.88
            and 0.18 <= kc_b <= 0.88
            and abs(pullback.low - fast[-2]) <= 1.20 * atr
            and pullback.close > slow[-2]
            and abs(confirmation.close - fast[-1]) <= 1.20 * atr
        )
        ok_re_call = dx >= 15.0 and pdi >= mdi and hurst_val >= 0.40 and not is_doji and not call_rejection

        ok_pr_put = (
            t_down
            and (len(slow) < 5 or slow[-2] < slow[-5] or ema9 <= ema21)
            and ema9 <= ema21
            and ema_spread >= 0.15 * atr
            and confirmation.close < fast[-1]
            and confirmation.close < confirmation.open
            and confirmation.close < min(pullback.open, pullback.close)
        )
        ok_mo_put = (
            28.0 <= rsi <= 58.0
            and (rsi <= rsi_prev + 1.5)
            and (hist <= hist_prev or m_val <= s_val or rsi <= 48.0)
            and (15.0 <= stoch_k <= 78.0)
        )
        ok_os_put = -180.0 <= cci_val <= 50.0 and cmo_val <= 15.0 and -90.0 <= williams_val <= -18.0 and roc_val <= 0.15
        ok_vo_put = (
            0.12 <= pct_b <= 0.82
            and 0.12 <= kc_b <= 0.82
            and abs(pullback.high - fast[-2]) <= 1.20 * atr
            and pullback.close < slow[-2]
            and abs(confirmation.close - fast[-1]) <= 1.20 * atr
        )
        ok_re_put = dx >= 15.0 and mdi >= pdi and hurst_val >= 0.40 and not is_doji and not put_rejection

        score_call = sum([ok_pr_call, ok_mo_call, ok_os_call, ok_vo_call, ok_re_call])
        score_put = sum([ok_pr_put, ok_mo_put, ok_os_put, ok_vo_put, ok_re_put])
        if variant == "momentum_burst":
            min_momentum_score = 4
            call = (
                score_call >= min_momentum_score and score_call >= score_put + 2 and not call_rejection and not is_doji
            )
            put = score_put >= min_momentum_score and score_put >= score_call + 2 and not put_rejection and not is_doji
        elif variant == "senior_institutional":
            min_momentum_score = 4
            vwap, vwap_upper, vwap_lower = session_vwap(bars)
            bull_fvg, bear_fvg = detect_fvg_mitigation(bars, atr)
            call = (
                score_call >= min_momentum_score
                and score_call >= score_put + 2
                and not call_rejection
                and not is_doji
                and confirmation.close >= vwap
                and confirmation.close <= vwap_upper
                and bull_fvg
            )
            put = (
                score_put >= min_momentum_score
                and score_put >= score_call + 2
                and not put_rejection
                and not is_doji
                and confirmation.close <= vwap
                and confirmation.close >= vwap_lower
                and bear_fvg
            )
        elif score_call > score_put:
            call, put = True, False
        elif score_put > score_call:
            call, put = False, True
        elif t_up or confirmation.close > fast[-1]:
            call, put = True, False
        elif t_down or confirmation.close < fast[-1]:
            call, put = False, True
        else:
            call = confirmation.close >= confirmation.open
            put = not call
    elif variant != "base":
        return SetupResult("SEM SINAL", "variante_invalida", version=version)
    if call:
        return SetupResult("CALL", "setup_confirmado", version=version)
    if put:
        return SetupResult("PUT", "setup_confirmado", version=version)
    return SetupResult("SEM SINAL", "padrao_incompleto", version=version)
