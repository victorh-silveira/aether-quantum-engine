"""Indicadores tecnicos compactos para o prompt da LLM e confluencia MTF."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import numpy as np


_TYPE_VALUE_ERRORS = (TypeError, ValueError)


@dataclass(frozen=True, slots=True)
class IndicatorConfig:
    """Parametros normalizados para modelos Quant (Simons/Medallion style)."""

    entropy_bins: int = 30
    entropy_window: int = 20
    zscore_window: int = 10
    hurst_window: int = 30
    velocity_window: int = 5
    acceleration_window: int = 5
    volatility_window: int = 14
    confluence_include_entropy: bool = True
    confluence_mandate: str = "medallion"
    entropy_threshold: float = 4.0
    zscore_extreme_threshold: float = 3.0
    tf_labels: tuple[str, str, str, str] = ("H4", "H1", "M15", "M5")
    tf_tags: tuple[str, str, str, str] = ("240", "60", "15", "5")


def resolve_indicator_config(raw: Mapping[str, object] | None) -> IndicatorConfig:
    """Instancia ``IndicatorConfig`` a partir do mapa ``llm.indicator_config`` ou defaults."""
    if not raw:
        return IndicatorConfig()
    eb = _clamp_int(raw.get("entropy_bins"), 2, 50, 10)
    ew = _clamp_int(raw.get("entropy_window"), 10, 100, 30)
    zw = _clamp_int(raw.get("zscore_window"), 5, 100, 20)
    hw = _clamp_int(raw.get("hurst_window"), 10, 200, 50)
    vw = _clamp_int(raw.get("velocity_window"), 2, 50, 5)
    aw = _clamp_int(raw.get("acceleration_window"), 2, 50, 5)
    vow = _clamp_int(raw.get("volatility_window"), 2, 200, 14)
    cie = _bool_cfg(raw.get("confluence_include_entropy"), default=True)
    man = str(raw.get("confluence_mandate", "medallion")).strip().lower()
    et = _clamp_float(raw.get("entropy_threshold"), 0.1, 10.0, 4.0)
    zet = _clamp_float(raw.get("zscore_extreme_threshold"), 1.0, 10.0, 3.0)

    raw_labels = raw.get("tf_labels")
    tf_labels = (
        tuple(map(str, raw_labels))
        if isinstance(raw_labels, (list, tuple)) and len(raw_labels) == 4
        else ("H4", "H1", "M15", "M5")
    )

    raw_tags = raw.get("tf_tags")
    tf_tags = (
        tuple(map(str, raw_tags))
        if isinstance(raw_tags, (list, tuple)) and len(raw_tags) == 4
        else ("240", "60", "15", "5")
    )

    return IndicatorConfig(eb, ew, zw, hw, vw, aw, vow, cie, man, et, zet, tf_labels, tf_tags)


def _bool_cfg(v: object, *, default: bool) -> bool:
    """Converte valores heterogeneos de configuracao em booleano estavel."""
    if v is None:
        return default
    if isinstance(v, bool):
        return v
    if isinstance(v, (int, float)):
        return bool(v)
    if isinstance(v, str):
        return v.strip().lower() in ("1", "true", "yes", "on")
    return default


def _clamp_int(v: object, lo: int, hi: int, default: int) -> int:
    """Limita inteiro ao intervalo ``[lo, hi]`` com fallback em erro de conversao."""
    try:
        x = int(v) if v is not None else default
    except _TYPE_VALUE_ERRORS:
        return default
    return max(lo, min(hi, x))


def _clamp_float(v: object, lo: float, hi: float, default: float) -> float:
    """Limita float ao intervalo ``[lo, hi]`` com fallback em erro de conversao."""
    try:
        x = float(v) if v is not None else default
    except _TYPE_VALUE_ERRORS:
        return default
    return max(lo, min(hi, x))


def min_bars_for_indicators(cfg: IndicatorConfig) -> int:
    """Menor numero de fechamentos necessario para calcular todos os indicadores quant."""
    return max(
        cfg.entropy_window + 5,
        cfg.zscore_window + 2,
        cfg.hurst_window + 5,
        cfg.volatility_window,
    )


def _shannon_entropy(closes: np.ndarray, bins: int = 10, window: int = 30) -> float | None:
    """Calcula a Entropia de Shannon dos retornos logaritmicos."""
    if closes.size < window + 2:
        return None
    c = np.asarray(closes[-window:], dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = np.diff(np.log(c[c > 0]))
    if (np.max(rets) - np.min(rets)) < 1e-7:
        return 0.0

    counts, _ = np.histogram(rets, bins=bins)
    probs = counts / (np.sum(counts) + 1e-12)
    probs = probs[probs > 0]
    if probs.size <= 1:
        return 0.0
    return float(-np.sum(probs * np.log2(probs)))


def _hurst_exponent(closes: np.ndarray, window: int = 50) -> float | None:
    """Estimativa simplificada do Expoente de Hurst (H > 0.5 = Persistencia)."""
    if closes.size < window:
        return None
    c = np.asarray(closes[-window:], dtype=np.float64)
    with np.errstate(divide="ignore", invalid="ignore"):
        rets = np.diff(np.log(c[c > 0]))
    if rets.size < 10:
        return None

    n = rets.size
    y = np.cumsum(rets - np.mean(rets))
    r = np.max(y) - np.min(y)
    s = np.std(rets)
    if s < 1e-12:
        return 1.0 if abs(np.mean(rets)) > 1e-12 else 0.5
    h = np.log(r / s) / np.log(n)
    return float(np.clip(h, 0.0, 1.0))


def _z_score_last(closes: np.ndarray, window: int = 20) -> float | None:
    """Calcula o Z-Score do ultimo preco em relacao a media movel (Reversao a Media)."""
    if closes.size < window:
        return None
    seg = closes[-window:]
    mean = np.mean(seg)
    std = np.std(seg)
    if std < 1e-12:
        return 0.0
    return float((closes[-1] - mean) / std)


def _price_derivatives(closes: np.ndarray, window: int = 5) -> tuple[float | None, float | None]:
    """Calcula Velocidade (1a deriv) e Aceleracao (2a deriv) do preco."""
    if closes.size < window + 2:
        return None, None

    v_now = (closes[-1] - closes[-window]) / window
    v_prev = (closes[-2] - closes[-(window + 1)]) / window

    accel = v_now - v_prev
    return float(v_now), float(accel)


def _vol_range_pct(closes: np.ndarray, period: int) -> float | None:
    """Volatilidade simplificada (High-Low range %) na janela; retorna ``None`` se curta."""
    if closes.size < period:
        return None
    seg = closes[-period:]
    last = float(closes[-1])
    if abs(last) < 1e-12:
        return 0.0
    return (float(np.max(seg)) - float(np.min(seg))) / last * 100.0


def compact_indicators_line(timeframe_label: str, closes: Sequence[float], cfg: IndicatorConfig | None = None) -> str:
    """Monta uma linha unica com Entropia, Hurst, Z-Score, Velocidade e Volatilidade."""
    c = np.asarray(list(closes), dtype=np.float64)
    ic = cfg or IndicatorConfig()
    need = min_bars_for_indicators(ic)
    n = int(c.size)
    if n < need:
        return f"{timeframe_label} quant: amostra_curta n={n} (min={need})"

    entropy = _shannon_entropy(c, ic.entropy_bins, ic.entropy_window)
    hurst = _hurst_exponent(c, ic.hurst_window)
    zscore = _z_score_last(c, ic.zscore_window)
    velocity, acceleration = _price_derivatives(c, ic.velocity_window)
    vol_v = _vol_range_pct(c, ic.volatility_window)

    ent_txt = f"{entropy:.2f}" if entropy is not None else "n/a"
    hst_txt = f"{hurst:.2f}" if hurst is not None else "n/a"
    zsc_txt = f"{zscore:+.2f}" if zscore is not None else "n/a"
    vel_txt = f"{velocity:+.6f}" if velocity is not None else "n/a"
    acc_txt = f"{acceleration:+.6f}" if acceleration is not None else "n/a"
    vol_txt = f"{vol_v:.2f}%" if vol_v is not None else "n/a"

    return (
        f"{timeframe_label} [QUANT]: "
        f"Entropy={ent_txt} | "
        f"Hurst={hst_txt} | "
        f"Z-Score={zsc_txt} | "
        f"V/A={vel_txt}/{acc_txt} | "
        f"Sigma={vol_txt}"
    )


def _market_regime_quant(hurst: float | None, zscore: float | None) -> str:
    """Classifica o regime de mercado via Hurst e Z-Score."""
    if hurst is None or zscore is None:
        return "indefinido"
    if hurst > 0.53:
        return "trend_persistente"
    if hurst < 0.47:
        return "mean_reverting"
    return "random_walk_high_noise"


def effective_indicator_config_log(cfg: IndicatorConfig) -> dict[str, Any]:
    """Serializa configuracao de indicadores para auditoria em log."""
    return {
        "entropy_window": cfg.entropy_window,
        "zscore_window": cfg.zscore_window,
        "hurst_window": cfg.hurst_window,
        "volatility_window": cfg.volatility_window,
    }


def bundle_llm_indicators_for_log(
    macro: Sequence[float],
    structure: Sequence[float],
    swing: Sequence[float],
    trigger: Sequence[float],
    cfg: IndicatorConfig,
    lm: str,
    ls: str,
    lw: str,
    lt: str,
) -> str:
    """Concatena linhas por timeframe usadas no prompt para auditoria em log."""
    return " | ".join(
        (
            compact_indicators_line(lm, macro, cfg),
            compact_indicators_line(ls, structure, cfg),
            compact_indicators_line(lw, swing, cfg),
            compact_indicators_line(lt, trigger, cfg),
        )
    )
