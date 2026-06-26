"""Motor unificado de direcao CALL/PUT por scoring de indicadores e DL."""

from __future__ import annotations

import contextlib

from src.domain.models.trade import TradeDirection


def infer_dl_direction(entry: dict) -> TradeDirection | None:
    """Obtem direcao prevista pelo DL ou infere a partir de raw_prob."""
    direction = entry.get("direction")
    if direction is not None:
        return direction
    metrics = entry.get("metrics") or {}
    raw = metrics.get("raw_prob")
    if raw is None:
        return None
    return TradeDirection.CALL if float(raw) > 0.5 else TradeDirection.PUT


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
    """Calcula contribuicao lateralizada de raw_prob no score CALL/PUT."""
    metrics = entry.get("metrics") or {}
    raw = metrics.get("raw_prob")
    if raw is None:
        dl_dir = infer_dl_direction(entry)
        if dl_dir is None:
            return 0.5, 0.5
        raw = 0.55 if dl_dir == TradeDirection.CALL else 0.45
    raw_f = _clamp01(float(raw))
    w = float(weights["dl_raw_weight"])
    return 0.5 + (raw_f - 0.5) * w, 0.5 + (0.5 - raw_f) * w


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
    """Resolve CALL ou PUT por score composto; retorna None apenas se sem raw_prob."""
    if is_technically_blocked(entry):
        return None
    dl_dir = infer_dl_direction(entry)
    if dl_dir is None:
        return None
    metrics = dict(entry.get("metrics") or {})
    cfg = exec_cfg if isinstance(exec_cfg, dict) else {}
    weights = _scoring_weights(cfg)

    call_score = 0.0
    put_score = 0.0
    hints: list[str] = []

    dl_call, dl_put = _dl_call_put_scores(entry, weights)
    call_score += dl_call
    put_score += dl_put

    val_call, val_put = _val_accuracy_bias(metrics, weights)
    call_score += val_call - 0.5
    put_score += val_put - 0.5

    for bias_fn in (_trend_bias, _exhaustion_bias, _indicator_regime_bias):
        c_bias, p_bias, hint = bias_fn(metrics, weights)
        call_score += c_bias - 0.5
        put_score += p_bias - 0.5
        if hint and hint not in hints:
            hints.append(hint)

    c_low, p_low, low_hint = _low_val_accuracy_bias(entry, metrics, weights)
    call_score += c_low - 0.5
    put_score += p_low - 0.5
    if low_hint and low_hint not in hints:
        hints.append(low_hint)

    if recovery_active:
        trend_str = metrics.get("trend_direction")
        if trend_str:
            with contextlib.suppress(KeyError, ValueError):
                trend_dir = TradeDirection[str(trend_str).upper()]
                w = float(weights["trend_weight"]) * 0.5
                if trend_dir == TradeDirection.CALL:
                    call_score += w
                else:
                    put_score += w

    exec_dir = TradeDirection.CALL if call_score + 1e-9 >= put_score else TradeDirection.PUT
    chosen = max(call_score, put_score)
    metrics["direction_call_score"] = call_score
    metrics["direction_put_score"] = put_score
    metrics["direction_margin"] = abs(call_score - put_score)
    metrics["direction_hints"] = hints
    metrics["direction_hint"] = hints[0] if hints else None
    metrics["dl_direction"] = dl_dir.name
    metrics["exec_direction"] = exec_dir.name
    metrics["direction_inverted"] = dl_dir != exec_dir
    side_strength = _clamp01(chosen)
    if metrics.get("trade_score") is None:
        metrics["trade_score"] = side_strength
    metrics["resolved_conviction"] = side_strength
    return exec_dir, metrics
