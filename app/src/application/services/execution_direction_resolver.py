"""Motor de direcao com refinamento de payoff continuo pelo meta-regressor LightGBM."""

from __future__ import annotations

from typing import Any

from src.application.services.execution_quality_gate import (
    direction_margin_from_probability,
    ensure_direction_margin,
    passes_execution_quality,
)
from src.application.services.meta_classifier_stacking import resolve_meta_payoff_edge
from src.application.services.meta_direction_flip import SIGNAL_SUSPENDED
from src.application.services.meta_payoff_regression import apply_meta_regression_edge
from src.application.services.payoff_edge_zscore import attach_payoff_edge_zscore_metrics
from src.domain.models.trade import TradeDirection


D_SQUEEZE_BB_WIDTH_ANOMALY_RATIO = 0.55


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
    """Inicializa scores laterais TCN antes do refinamento meta-regressor."""
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
    metrics["direction_margin"] = direction_margin_from_probability(prob, direction=dl_dir.name)
    return score


def _sync_entry_metrics(entry: dict, metrics: dict) -> None:
    """Propaga metricas resolvidas de volta ao entry de decisao."""
    entry_metrics = entry.get("metrics")
    if isinstance(entry_metrics, dict):
        entry_metrics.update(metrics)
    else:
        entry["metrics"] = metrics


def _reject_on_quality_gate(
    entry: dict,
    metrics: dict,
    gate_probe: dict,
    exec_cfg_dict: dict,
    *,
    risk_manager: Any | None = None,
) -> bool:
    """Retorna True quando o gate de alta conviccao suspende o candidato."""
    if passes_execution_quality(
        gate_probe,
        exec_cfg=exec_cfg_dict,
        risk_manager=risk_manager,
    ):
        return False
    metrics.update(
        {
            key: gate_probe[key]
            for key in ("regime_skip_cycle", "direction_margin", "quality_gate_reason")
            if key in gate_probe
        },
    )
    metrics["signal_status"] = SIGNAL_SUSPENDED
    metrics["quality_guard_reject"] = True
    _sync_entry_metrics(entry, metrics)
    return True


def resolve_execution_direction(
    entry: dict,
    *,
    exec_cfg: dict | None = None,
    calibration_cfg: dict | None = None,
    recovery_active: bool = False,
    symbol: str | None = None,
    corr_matrix: dict[tuple[str, str], float] | None = None,
    infra_cfg: dict | None = None,
    peer_entry: dict | None = None,
    cycle_id: int = 0,
    risk_manager: Any | None = None,
) -> tuple[TradeDirection, dict] | None:
    """Resolve direcao micro fiel ao sinal TCN/DL com telemetria meta-regressor."""
    _ = (calibration_cfg, recovery_active, corr_matrix, peer_entry, cycle_id)
    exec_cfg_dict = exec_cfg if isinstance(exec_cfg, dict) else {}
    dl_dir = infer_dl_direction(entry)
    if is_technically_blocked(entry) or dl_dir is None:
        return None
    metrics = dict(entry.get("metrics") or {})
    metrics["bb_width_anomaly_ratio"] = D_SQUEEZE_BB_WIDTH_ANOMALY_RATIO
    prob = _direction_prob(entry)
    if prob is None:
        prob = 0.55 if dl_dir == TradeDirection.CALL else 0.45
    prob = _clamp01(prob)
    score = _seed_direction_metrics(metrics, dl_dir=dl_dir, prob=prob)
    predicted_edge, meta_applied = resolve_meta_payoff_edge(
        symbol=symbol,
        metrics=metrics,
        direction=dl_dir,
        tcn_probability=prob,
        _base_score=score,
        config={"infra": infra_cfg} if infra_cfg else None,
    )
    gate_probe = dict(metrics)
    gate_probe["predicted_payoff_edge"] = float(predicted_edge)
    gate_probe["meta_classifier_applied"] = bool(meta_applied)
    if _reject_on_quality_gate(entry, metrics, gate_probe, exec_cfg_dict, risk_manager=risk_manager):
        return None
    exec_dir, _final_score = apply_meta_regression_edge(
        dl_dir,
        metrics,
        predicted_edge,
        meta_applied=meta_applied,
        base_score=score,
        symbol=symbol,
    )
    attach_payoff_edge_zscore_metrics(metrics, float(metrics.get("predicted_payoff_edge", 0.0)))
    metrics["exec_direction"] = exec_dir.name
    metrics["resolved_direction"] = exec_dir.name
    metrics["direction_inverted"] = exec_dir != dl_dir
    metrics["tcn_score"] = prob
    ensure_direction_margin(metrics)
    _sync_entry_metrics(entry, metrics)
    return exec_dir, metrics
