"""Setup fixo de tendencia, recuo e confirmacao em velas fechadas."""

from __future__ import annotations

from dataclasses import dataclass
from hashlib import sha256
from math import isfinite

from indicator.domain import Candle

SPEC = "ema20_50:warmup128:atr14:incl3:recuo0.5:corpo0.03_1.5:confirmacao1:m5"
VERSION = "pullback-1-" + sha256(SPEC.encode()).hexdigest()[:12]
MIN_BARS = 128
VARIANTS = ("base", "ema50_slope", "countertrend_retrace", "entry_distance", "improved")


def version_for_variant(variant: str = "base") -> str:
    """Calcula a versao imutavel do setup para a variante especificada."""
    if variant == "base":
        return VERSION
    if variant == "improved":
        spec = "ema20_50:warmup128:atr14:incl3:recuo0.5:corpo0.03_1.1:confirmacao1:countertrend:ema50slope:m5"
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


def evaluate(candles: list[Candle], period: int, now: int, variant: str = "base") -> SetupResult:
    """Avalia o ultimo recuo e a ultima confirmacao sem usar vela corrente."""
    version = version_for_variant(variant) if variant in VARIANTS else VERSION
    if period not in (300, 900, 3600):
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
    if any(right.epoch - left.epoch != period for left, right in zip(bars, bars[1:], strict=False)):
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
    max_body = 1.2 * atr
    if not 0.03 * atr <= body <= max_body:
        return SetupResult("SEM SINAL", "corpo_fora_faixa", version=version)
    trend_up = fast[-2] > slow[-2] and fast[-4] < fast[-2] and (fast[-2] - slow[-2]) >= 0.25 * atr
    trend_down = fast[-2] < slow[-2] and fast[-4] > fast[-2] and (slow[-2] - fast[-2]) >= 0.25 * atr
    cf_range = confirmation.high - confirmation.low
    call = (
        trend_up
        and abs(pullback.low - fast[-2]) <= 0.80 * atr
        and pullback.close > slow[-2]
        and confirmation.close > fast[-1]
        and confirmation.close > confirmation.open
        and confirmation.close > pullback.high
        and cf_range > 0
        and (confirmation.high - confirmation.close) <= 0.35 * cf_range
    )
    put = (
        trend_down
        and abs(pullback.high - fast[-2]) <= 0.80 * atr
        and pullback.close < slow[-2]
        and confirmation.close < fast[-1]
        and confirmation.close < confirmation.open
        and confirmation.close < pullback.low
        and cf_range > 0
        and (confirmation.close - confirmation.low) <= 0.35 * cf_range
    )
    if variant == "ema50_slope":
        call = call and slow[-2] > slow[-5]
        put = put and slow[-2] < slow[-5]
    elif variant == "countertrend_retrace":
        call = call and pullback.close < pullback.open
        put = put and pullback.close > pullback.open
    elif variant == "entry_distance":
        call = call and abs(confirmation.close - fast[-1]) <= 0.75 * atr
        put = put and abs(confirmation.close - fast[-1]) <= 0.75 * atr
    elif variant == "improved":
        call = (
            call
            and slow[-2] >= slow[-5]
            and pullback.close <= pullback.open
            and abs(confirmation.close - fast[-1]) <= 1.20 * atr
        )
        put = (
            put
            and slow[-2] <= slow[-5]
            and pullback.close >= pullback.open
            and abs(confirmation.close - fast[-1]) <= 1.20 * atr
        )
    elif variant != "base":
        return SetupResult("SEM SINAL", "variante_invalida", version=version)
    if call:
        return SetupResult("CALL", "setup_confirmado", version=version)
    if put:
        return SetupResult("PUT", "setup_confirmado", version=version)
    return SetupResult("SEM SINAL", "padrao_incompleto", version=version)
