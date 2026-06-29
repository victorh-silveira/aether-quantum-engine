"""Motor unificado de direcao CALL/PUT por scoring de indicadores e DL."""

from __future__ import annotations

import contextlib

from src.application.services.deep_learning.dl_calibration_fit import entropy_weight_penalty
from src.application.services.execution_direction_scoring import (
    accumulate_direction_scores,
    finalize_direction_metrics,
)
from src.domain.models.trade import TradeDirection


def _direction_prob(entry: dict) -> float | None:
    """Retorna probabilidade calibrada de CALL quando disponivel."""
    metrics = entry.get("metrics") or {}
    calibrated = metrics.get("calibrated_prob")
    if calibrated is not None:
        return float(calibrated)
    raw = metrics.get("raw_prob")
    if raw is None:
        return None
    return float(raw)


def _direction_pivot(metrics: dict) -> float:
    """Pivot CALL/PUT: medio dos thresholds dinamicos ou 0.5."""
    call_th = metrics.get("dynamic_call_threshold")
    put_th = metrics.get("dynamic_put_threshold")
    if call_th is not None and put_th is not None:
        return (float(call_th) + float(put_th)) * 0.5
    return 0.5


def infer_dl_direction(entry: dict) -> TradeDirection | None:
    """Obtem direcao prevista pelo DL ou infere a partir da probabilidade calibrada."""
    direction = entry.get("direction")
    if direction is not None:
        return direction
    metrics = entry.get("metrics") or {}
    prob = _direction_prob(entry)
    if prob is None:
        return None
    pivot = _direction_pivot(metrics)
    return TradeDirection.CALL if float(prob) > pivot else TradeDirection.PUT


_TECHNICAL_BLOCKS = frozenset({"data", "predict_error", "training"})
_DEFAULT_WEIGHTS = {
    "dl_raw_weight": 0.45,
    "val_accuracy_weight": 0.18,
    "trend_weight": 0.15,
    "exhaustion_weight": 0.12,
    "indicator_regime_weight": 0.10,
}


def is_technically_blocked(entry: dict) -> bool:
    """Indica bloqueio absoluto por falha tecnica."""
    metrics = entry.get("metrics") or {}
    if metrics.get("deploy_ok") is False:
        return True
    gate = str(metrics.get("gate_reason") or "")
    return gate in _TECHNICAL_BLOCKS


def _scoring_weights(cfg: dict) -> dict[str, float]:
    """Mescla pesos de direction_scoring com defaults do resolver."""
    merged = dict(_DEFAULT_WEIGHTS)
    chunk = cfg.get("direction_scoring") if isinstance(cfg, dict) else {}
    if isinstance(chunk, dict):
        for key in _DEFAULT_WEIGHTS:
            if key in chunk:
                merged[key] = float(chunk[key])
    return merged


def _clamp01(value: float) -> float:
    """Limita valor ao intervalo [0, 1]."""
    return max(0.0, min(1.0, float(value)))


def _dl_call_put_scores(entry: dict, weights: dict) -> tuple[float, float]:
    """Calcula contribuicao lateralizada da probabilidade calibrada no score CALL/PUT."""
    metrics = entry.get("metrics") or {}
    prob = _direction_prob(entry)
    if prob is None:
        dl_dir = infer_dl_direction(entry)
        if dl_dir is None:
            return 0.5, 0.5
        prob = 0.55 if dl_dir == TradeDirection.CALL else 0.45
    prob_f = _clamp01(float(prob))
    pivot = _direction_pivot(metrics)
    w = float(weights["dl_raw_weight"])
    penalty = entropy_weight_penalty(prob_f)
    if metrics.get("entropy_violation"):
        penalty = max(penalty, entropy_weight_penalty(prob_f))
    w_eff = w * (1.0 - penalty * penalty)
    metrics["dl_weight_penalty"] = penalty
    metrics["direction_entropy"] = float(metrics.get("calibrated_entropy", 0.0))
    return 0.5 + (prob_f - pivot) * w_eff, 0.5 + (pivot - prob_f) * w_eff


def _val_accuracy_bias(metrics: dict, weights: dict) -> tuple[float, float]:
    """Aplica bias de val_accuracy ao score lateral."""
    val = _clamp01(float(metrics.get("val_accuracy", 0.5)))
    w = float(weights["val_accuracy_weight"])
    bias = (val - 0.5) * w
    return 0.5 + bias, 0.5 - bias


def _trend_bias(metrics: dict, weights: dict) -> tuple[float, float, str | None]:
    """Ajusta score conforme alinhamento com tendencia de mercado."""
    trend_str = metrics.get("trend_direction")
    if not trend_str:
        return 0.5, 0.5, None
    indicators = metrics.get("indicators") or {}
    vol_ratio = float(indicators.get("vol_ratio", 1.0))
    adx = float(indicators.get("adx", 0.5))
    if vol_ratio < 0.85 and adx < 0.25:
        return 0.5, 0.5, None
    w = float(weights["trend_weight"])
    with contextlib.suppress(KeyError, ValueError):
        trend_dir = TradeDirection[str(trend_str).upper()]
        if trend_dir == TradeDirection.CALL:
            return 0.5 + w, 0.5 - w, "trend_bias"
        return 0.5 - w, 0.5 + w, "trend_bias"
    return 0.5, 0.5, None


def _exhaustion_bias(metrics: dict, weights: dict) -> tuple[float, float, str | None]:
    """Empurra direcao oposta em extremos de RSI/Keltner."""
    indicators = metrics.get("indicators") or {}
    rsi = float(indicators.get("rsi", 0.5))
    keltner = float(indicators.get("keltner", 0.5))
    w = float(weights["exhaustion_weight"])
    if rsi < 0.45 or keltner < 0.30:
        return 0.5 + w, 0.5 - w, "exhaustion_flip"
    if rsi > 0.55 or keltner > 0.70:
        return 0.5 - w, 0.5 + w, "exhaustion_flip"
    return 0.5, 0.5, None


def _indicator_regime_bias(metrics: dict, weights: dict) -> tuple[float, float, str | None]:
    """Aplica pesos de regime (hurst, adx, vol_ratio, cmo) ao score."""
    indicators = metrics.get("indicators") or {}
    hurst = float(indicators.get("hurst", 0.5))
    adx = float(indicators.get("adx", 0.5))
    vol_ratio = float(indicators.get("vol_ratio", 1.0))
    rsi = float(indicators.get("rsi", 0.5))
    w = float(weights["indicator_regime_weight"])
    if hurst < 0.48 and adx < 0.25 and vol_ratio < 1.0:
        if rsi < 0.45:
            return 0.5 + w, 0.5 - w, "mean_reversion"
        if rsi > 0.55:
            return 0.5 - w, 0.5 + w, "mean_reversion"
    cmo = float(indicators.get("cmo", 0.0))
    if cmo > 0.08:
        return 0.5 + w * 0.5, 0.5 - w * 0.5, "indicator_regime"
    if cmo < -0.08:
        return 0.5 - w * 0.5, 0.5 + w * 0.5, "indicator_regime"
    return 0.5, 0.5, None


def _low_val_accuracy_bias(entry: dict, metrics: dict, weights: dict) -> tuple[float, float, str | None]:
    """Inverte bias lateral quando val_accuracy esta abaixo de 0.5."""
    val_acc = float(metrics.get("val_accuracy", 0.5))
    if val_acc >= 0.50:
        return 0.5, 0.5, None
    dl_dir = infer_dl_direction(entry)
    if dl_dir is None:
        return 0.5, 0.5, None
    w = float(weights["val_accuracy_weight"]) * 0.5
    inverted = TradeDirection.PUT if dl_dir == TradeDirection.CALL else TradeDirection.CALL
    if inverted == TradeDirection.CALL:
        return 0.5 + w, 0.5 - w, "low_val_flip"
    return 0.5 - w, 0.5 + w, "low_val_flip"


def resolve_execution_direction(
    entry: dict,
    *,
    exec_cfg: dict | None = None,
    recovery_active: bool = False,
) -> tuple[TradeDirection, dict] | None:
    """Resolve CALL ou PUT por score composto; retorna None apenas se sem probabilidade."""
    if is_technically_blocked(entry):
        return None
    dl_dir = infer_dl_direction(entry)
    if dl_dir is None:
        return None
    metrics = dict(entry.get("metrics") or {})
    cfg = exec_cfg if isinstance(exec_cfg, dict) else {}
    weights = _scoring_weights(cfg)
    call_score, put_score, hints = accumulate_direction_scores(
        entry,
        metrics,
        weights,
        recovery_active=recovery_active,
        bias_fns=(_trend_bias, _exhaustion_bias, _indicator_regime_bias),
        dl_scores_fn=_dl_call_put_scores,
        val_bias_fn=_val_accuracy_bias,
        low_val_fn=_low_val_accuracy_bias,
    )
    exec_dir = TradeDirection.CALL if call_score + 1e-9 >= put_score else TradeDirection.PUT
    finalize_direction_metrics(
        metrics,
        call_score=call_score,
        put_score=put_score,
        hints=hints,
        dl_dir=dl_dir,
        exec_dir=exec_dir,
        clamp01=_clamp01,
    )
    return exec_dir, metrics
