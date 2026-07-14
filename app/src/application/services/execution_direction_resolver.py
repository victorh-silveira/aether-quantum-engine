"""Motor de direcao com refinamento de payoff continuo pelo meta-regressor LightGBM."""

from __future__ import annotations

from typing import Any

from src.application.services.execution_quality_gate import (
    direction_margin_from_probability,
    ensure_direction_margin,
    passes_execution_quality,
)
from src.application.services.execution_quality_gate_meta import evaluate_meta_payoff_quality
from src.application.services.meta_classifier_stacking import resolve_meta_payoff_edge
from src.application.services.meta_direction_flip import SIGNAL_SUSPENDED
from src.application.services.meta_payoff_regression import (
    apply_meta_regression_edge,
    veto_calibration_neutral_drift,
)
from src.application.services.meta_payoff_veto_gate import (
    apply_meta_payoff_negative_zscore_veto,
    should_veto_meta_payoff_negative_zscore,
)
from src.application.services.payoff_edge_zscore import attach_payoff_edge_zscore_metrics
from src.domain.models.trade import TradeDirection
from src.domain.risk.stake_sizing import metric_float


D_SQUEEZE_BB_WIDTH_ANOMALY_RATIO = 0.55
_STRONG_NEGATIVE_ZSCORE = -0.20


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


def _has_meta_zscore_telemetry(metrics: dict) -> bool:
    """True quando Z-Score meta ja foi anexado e o buffer tem amostra suficiente."""
    if metrics.get("meta_payoff_edge_zscore") is None and metrics.get("edge_zscore") is None:
        return False
    samples = metrics.get("edge_zscore_samples")
    return samples is None or int(samples) >= 2


def _meta_zscore_soft_ok(metrics: dict) -> bool:
    """True quando Z meta nao e fortemente negativo (piso soft de quality AND)."""
    return metric_float(metrics, "meta_payoff_edge_zscore", "edge_zscore", default=0.0) >= _STRONG_NEGATIVE_ZSCORE


def _reject_on_quality_gate(
    entry: dict,
    metrics: dict,
    gate_probe: dict,
    exec_cfg_dict: dict,
    *,
    risk_manager: Any | None = None,
    recovery_active: bool = False,
) -> bool:
    """Retorna True quando quality gate reprova o candidato conforme regime."""
    has_meta = _has_meta_zscore_telemetry(gate_probe)
    meta_passed = False
    if has_meta:
        meta_passed = evaluate_meta_payoff_quality(
            gate_probe,
            exec_cfg=exec_cfg_dict,
            risk_manager=risk_manager,
        )
    tcn_passed = passes_execution_quality(
        gate_probe,
        exec_cfg=exec_cfg_dict,
        risk_manager=risk_manager,
    )
    meta_soft_ok = (not has_meta) or _meta_zscore_soft_ok(gate_probe)
    waived = bool(gate_probe.get("meta_payoff_veto_waived"))
    if recovery_active or waived:
        accepted = meta_passed or tcn_passed
    elif has_meta:
        accepted = (tcn_passed and meta_soft_ok) or meta_passed
    else:
        accepted = tcn_passed
    if accepted:
        for key in ("execution_gate_state", "quality_gate_regime", "direction_margin"):
            if key in gate_probe:
                metrics[key] = gate_probe[key]
        if meta_passed and not tcn_passed:
            metrics["execution_gate_state"] = "meta_zscore_pass"
        elif tcn_passed and not meta_passed:
            metrics.pop("quality_guard_reject", None)
            metrics.pop("regime_skip_cycle", None)
            metrics.pop("quality_gate_reason", None)
            if metrics.get("execution_gate_state") == "meta_zscore_reject":
                metrics.pop("execution_gate_state", None)
        return False
    metrics.update(
        {
            key: gate_probe[key]
            for key in (
                "regime_skip_cycle",
                "direction_margin",
                "quality_gate_reason",
                "execution_gate_state",
            )
            if key in gate_probe
        },
    )
    metrics["signal_status"] = SIGNAL_SUSPENDED
    metrics["quality_guard_reject"] = True
    _sync_entry_metrics(entry, metrics)
    return True


def _recovery_soft_quality_continue(metrics: dict) -> bool:
    """True quando recovery pode seguir apesar de quality aspiracional (Z >= -0.20)."""
    return metric_float(metrics, "meta_payoff_edge_zscore", "edge_zscore", default=0.0) >= _STRONG_NEGATIVE_ZSCORE


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
    _ = (calibration_cfg, corr_matrix, peer_entry, cycle_id)
    exec_cfg_dict = exec_cfg if isinstance(exec_cfg, dict) else {}
    dl_dir = infer_dl_direction(entry)
    if is_technically_blocked(entry) or dl_dir is None:
        return None
    metrics = dict(entry.get("metrics") or {})
    if veto_calibration_neutral_drift(metrics):
        _sync_entry_metrics(entry, metrics)
        return None
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
    if metrics.get("meta_payoff_edge_zscore") is None and metrics.get("edge_zscore") is None:
        attach_payoff_edge_zscore_metrics(
            metrics,
            float(metrics.get("predicted_payoff_edge", predicted_edge)),
            symbol=symbol,
        )
    if should_veto_meta_payoff_negative_zscore(metrics, direction=dl_dir, risk_manager=risk_manager):
        apply_meta_payoff_negative_zscore_veto(metrics)
        _sync_entry_metrics(entry, metrics)
        return None
    gate_probe = dict(metrics)
    gate_probe["predicted_payoff_edge"] = float(predicted_edge)
    gate_probe["meta_classifier_applied"] = bool(meta_applied)
    if _reject_on_quality_gate(
        entry,
        metrics,
        gate_probe,
        exec_cfg_dict,
        risk_manager=risk_manager,
        recovery_active=recovery_active,
    ):
        if not (recovery_active and _recovery_soft_quality_continue(metrics)):
            return None
        metrics.pop("signal_status", None)
        entry_metrics = entry.get("metrics")
        if isinstance(entry_metrics, dict):
            entry_metrics.pop("signal_status", None)
    require_meta = bool(exec_cfg_dict.get("require_meta_for_execution", False))
    if require_meta and not recovery_active and not bool(meta_applied):
        metrics["gate_reason"] = "meta_unavailable"
        metrics["quality_guard_reject"] = True
        _sync_entry_metrics(entry, metrics)
        return None
    exec_dir, _final_score = apply_meta_regression_edge(
        dl_dir,
        metrics,
        predicted_edge,
        meta_applied=meta_applied,
        base_score=score,
        symbol=symbol,
    )
    metrics["exec_direction"] = exec_dir.name
    metrics["resolved_direction"] = exec_dir.name
    metrics["direction_inverted"] = exec_dir != dl_dir
    metrics["tcn_score"] = prob
    ensure_direction_margin(metrics)
    _sync_entry_metrics(entry, metrics)
    return exec_dir, metrics
