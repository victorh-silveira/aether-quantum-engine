"""Setup fixo de tendencia, recuo e confirmacao em velas fechadas."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from hashlib import sha256
from math import isfinite

from indicator.domain import Candle
from indicator.institutional import (  # noqa: F401
    detect_fvg_mitigation,
    is_within_killzone,
    session_price_reference,
)
from indicator.pullback_indicators import (
    calc_adx,
    calc_atr,
    calc_bollinger_pct_b,
    calc_cci,
    calc_cmo,
    calc_ema,
    calc_hurst_exponent,
    calc_keltner_pct_b,
    calc_macd,
    calc_roc,
    calc_rsi,
    calc_stochastic,
    calc_williams_r,
    ema_direct,
)
from indicator.pullback_senior import evaluate_senior_institutional


SPEC = "ema20_50:warmup128:atr14:incl3:recuo0.5:corpo0.03_1.5:confirmacao1:m5"
VERSION = "pullback-1-" + sha256(SPEC.encode()).hexdigest()[:12]
MIN_BARS = 128
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

# Compatibilidade retroativa para testes e consumidores internos
_ema_direct, _ema, _atr, _rsi, _bollinger_pct_b = ema_direct, calc_ema, calc_atr, calc_rsi, calc_bollinger_pct_b
_adx, _macd, _stochastic, _cci, _cmo = calc_adx, calc_macd, calc_stochastic, calc_cci, calc_cmo
_williams_r, _roc, _keltner_pct_b, _hurst_exponent = calc_williams_r, calc_roc, calc_keltner_pct_b, calc_hurst_exponent


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
        spec = "senior_institutional:m15:strict_ema_retrace:session_proxy:fvg:killzone:v2"
        return "pullback-1-" + sha256(spec.encode()).hexdigest()[:12]
    spec = f"{SPEC}:{variant}"
    return "pullback-1-" + sha256(spec.encode()).hexdigest()[:12]


@dataclass(frozen=True)
class SetupResult:
    """Direcao do setup e motivo verificavel."""

    side: str
    reason: str
    version: str = VERSION


def _evaluate_standard_variants(
    variant: str,
    bars: list[Candle],
    closes: list[float],
    fast: list[float],
    slow: list[float],
    atr: float,
    pullback: Candle,
    confirmation: Candle,
) -> tuple[bool, bool]:
    """Avalia variantes padrao e experimentais do setup de pullback."""
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
        return (call and slow[-2] > slow[-5]), (put and slow[-2] < slow[-5])
    if variant == "countertrend_retrace":
        return (call and pullback.close < pullback.open), (put and pullback.close > pullback.open)
    if variant == "entry_distance":
        return (call and abs(confirmation.close - fast[-1]) <= 1.40 * atr), (
            put and abs(confirmation.close - fast[-1]) <= 1.40 * atr
        )
    if variant in ("improved", "pool_rsi_bb", "momentum_burst"):
        return _evaluate_pool_variants(variant, bars, closes, fast, slow, atr, pullback, confirmation)
    return call, put


def _evaluate_pool_variants(
    variant: str,
    bars: list[Candle],
    closes: list[float],
    fast: list[float],
    slow: list[float],
    atr: float,
    pullback: Candle,
    confirmation: Candle,
) -> tuple[bool, bool]:
    """Avalia variantes combinadas pool_rsi_bb, improved e momentum_burst."""
    rsi = _rsi(closes, 14)
    rsi_prev = _rsi(closes[:-1], 14)
    pct_b = _bollinger_pct_b(closes, 14)
    dx, pdi, mdi = _adx(bars, 14)
    m_val, s_val, hist = _macd(closes, 7, 14, 5)
    _, _, hist_prev = _macd(closes[:-1], 7, 14, 5)
    stoch_k, _ = _stochastic(bars, 14, 3)
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
    trend_up = fast[-2] > slow[-2] and fast[-4] < fast[-2]
    trend_down = fast[-2] < slow[-2] and fast[-4] > fast[-2]
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
    ok_os_call = -50.0 <= cci_val <= 180.0 and cmo_val >= -15.0 and -82.0 <= williams_val <= -10.0 and roc_val >= -0.15
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
        call = score_call >= min_momentum_score and score_call >= score_put + 2 and not call_rejection and not is_doji
        put = score_put >= min_momentum_score and score_put >= score_call + 2 and not put_rejection and not is_doji
        return call, put
    if score_call > score_put:
        return True, False
    if score_put > score_call:
        return False, True
    if t_up or confirmation.close > fast[-1]:
        return True, False
    if t_down or confirmation.close < fast[-1]:
        return False, True
    call = confirmation.close >= confirmation.open
    return call, not call


def evaluate(
    candles: list[Candle],
    period: int,
    now: int,
    variant: str = "base",
    optimal_hours_brt: list[int] | None = None,
    allow_session_gaps: bool = False,  # noqa: FBT001, FBT002
) -> SetupResult:
    """Avalia o ultimo recuo e a ultima confirmacao sem usar vela corrente."""
    version = version_for_variant(variant) if variant in VARIANTS else VERSION
    if period not in (300, 900, 3600, 86400):
        return SetupResult("SEM SINAL", "periodo_invalido", version=version)
    min_required = 64 if variant == "senior_institutional" else MIN_BARS
    if len(candles) < min_required:
        return SetupResult("SEM SINAL", "historico_insuficiente", version=version)
    bars = candles[-MIN_BARS:] if len(candles) >= MIN_BARS else candles[-min_required:]
    if any(
        candle.epoch % period
        or candle.high < max(candle.open, candle.close)
        or candle.low > min(candle.open, candle.close)
        or not all(isfinite(v) for v in (candle.open, candle.high, candle.low, candle.close))
        for candle in bars
    ):
        return SetupResult("SEM SINAL", "vela_invalida", version=version)

    def _is_gap(diff: int) -> bool:
        """Determina se a diferenca entre velas constitui lacuna nao autorizada."""
        return diff != period and not (allow_session_gaps and period < diff <= 120 * 3600)

    if _is_gap(bars[-1].epoch - bars[-2].epoch) or any(
        _is_gap(right.epoch - left.epoch) for left, right in zip(bars, bars[1:], strict=False)
    ):
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
    if variant == "senior_institutional":
        side, reason = evaluate_senior_institutional(
            bars, closes, fast, slow, atr, optimal_hours_brt, module=sys.modules[__name__]
        )
        return SetupResult(side, reason, version=version)
    body = abs(confirmation.close - confirmation.open)
    if variant == "base" and not (0.02 * atr <= body <= 2.5 * atr):
        return SetupResult("SEM SINAL", "corpo_fora_faixa", version=version)
    if variant not in VARIANTS:
        return SetupResult("SEM SINAL", "variante_invalida", version=version)

    call, put = _evaluate_standard_variants(variant, bars, closes, fast, slow, atr, pullback, confirmation)
    if call:
        return SetupResult("CALL", "setup_confirmado", version=version)
    if put:
        return SetupResult("PUT", "setup_confirmado", version=version)
    return SetupResult("SEM SINAL", "padrao_incompleto", version=version)
