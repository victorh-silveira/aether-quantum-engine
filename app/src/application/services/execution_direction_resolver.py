"""Motor de direcao linear puro: segue estritamente o sinal do Deep Learning (TCN)."""

from __future__ import annotations

from src.domain.models.trade import TradeDirection


_TECHNICAL_BLOCKS = frozenset({"data", "predict_error", "training"})


def _direction_prob(entry: dict) -> float | None:
    """Retorna probabilidade calibrada de CALL surfaçada pelo decision_bridge."""
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
    """Obtem direcao prevista pelo DL (P(CALL) > P(PUT) => CALL, senao PUT)."""
    direction = entry.get("direction")
    if direction is not None:
        return direction
    metrics = entry.get("metrics") or {}
    prob = _direction_prob(entry)
    if prob is None:
        return None
    pivot = _direction_pivot(metrics)
    return TradeDirection.CALL if float(prob) > pivot else TradeDirection.PUT


def is_technically_blocked(entry: dict) -> bool:
    """Indica bloqueio absoluto por falha tecnica de dados ou treino."""
    metrics = entry.get("metrics") or {}
    if metrics.get("deploy_ok") is False:
        return True
    gate = str(metrics.get("gate_reason") or "")
    return gate in _TECHNICAL_BLOCKS


def _clamp01(value: float) -> float:
    """Limita valor ao intervalo [0, 1]."""
    return max(0.0, min(1.0, float(value)))


def resolve_execution_direction(
    entry: dict,
    *,
    exec_cfg: dict | None = None,
    calibration_cfg: dict | None = None,
    recovery_active: bool = False,
    symbol: str | None = None,
    corr_matrix: dict[tuple[str, str], float] | None = None,
) -> tuple[TradeDirection, dict] | None:
    """Resolve direcao seguindo estritamente o DL; trade_score e a probabilidade calibrada bruta."""
    _ = (exec_cfg, calibration_cfg, recovery_active, symbol, corr_matrix)
    if is_technically_blocked(entry):
        return None
    dl_dir = infer_dl_direction(entry)
    if dl_dir is None:
        return None
    metrics = dict(entry.get("metrics") or {})
    prob = _direction_prob(entry)
    if prob is None:
        prob = 0.55 if dl_dir == TradeDirection.CALL else 0.45
    prob = _clamp01(prob)
    call_score = prob
    put_score = 1.0 - prob
    score = call_score if dl_dir == TradeDirection.CALL else put_score
    metrics["dl_direction"] = dl_dir.name
    metrics["exec_direction"] = dl_dir.name
    metrics["resolved_direction"] = dl_dir.name
    metrics["direction_inverted"] = False
    metrics["direction_call_score"] = call_score
    metrics["direction_put_score"] = put_score
    metrics["direction_margin"] = abs(call_score - put_score)
    metrics["trade_score"] = score
    metrics["conviction"] = score
    return dl_dir, metrics
