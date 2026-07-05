"""Motor de direcao com inversao micro orientada pelo meta-classificador LightGBM."""

from __future__ import annotations

from src.application.services.meta_classifier_stacking import (
    apply_meta_payoff_to_metrics,
    resolve_meta_payoff_score,
)
from src.application.services.meta_direction_flip import apply_meta_direction_flip
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


def _seed_direction_metrics(
    metrics: dict,
    *,
    dl_dir: TradeDirection,
    prob: float,
) -> float:
    """Inicializa scores laterais TCN antes do refinamento meta-classificador."""
    call_score = prob
    put_score = 1.0 - prob
    score = call_score if dl_dir == TradeDirection.CALL else put_score
    metrics["dl_direction"] = dl_dir.name
    metrics["exec_direction"] = dl_dir.name
    metrics["resolved_direction"] = dl_dir.name
    metrics["direction_inverted"] = False
    metrics["meta_direction_flip"] = False
    metrics["direction_call_score"] = call_score
    metrics["direction_put_score"] = put_score
    metrics["direction_margin"] = abs(call_score - put_score)
    return score


def resolve_execution_direction(
    entry: dict,
    *,
    exec_cfg: dict | None = None,
    calibration_cfg: dict | None = None,
    recovery_active: bool = False,
    symbol: str | None = None,
    corr_matrix: dict[tuple[str, str], float] | None = None,
    infra_cfg: dict | None = None,
) -> tuple[TradeDirection, dict] | None:
    """Resolve direcao micro com autonomia do meta-classificador para inversao em exaustao."""
    _ = (exec_cfg, calibration_cfg, recovery_active, corr_matrix)
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
    score = _seed_direction_metrics(metrics, dl_dir=dl_dir, prob=prob)
    payoff_score, meta_applied = resolve_meta_payoff_score(
        symbol=symbol,
        metrics=metrics,
        direction=dl_dir,
        tcn_probability=prob,
        base_score=score,
        config={"infra": infra_cfg} if infra_cfg else None,
    )
    exec_dir, final_score = apply_meta_direction_flip(
        dl_dir,
        metrics,
        payoff_score,
        meta_applied=meta_applied,
        tcn_probability=prob,
    )
    if exec_dir == dl_dir:
        if meta_applied or payoff_score != score:
            apply_meta_payoff_to_metrics(
                metrics,
                direction=dl_dir,
                tcn_probability=prob,
                payoff_score=payoff_score,
                meta_applied=meta_applied,
            )
        else:
            metrics["trade_score"] = score
            metrics["conviction"] = score
    else:
        _ = final_score
    return exec_dir, metrics
