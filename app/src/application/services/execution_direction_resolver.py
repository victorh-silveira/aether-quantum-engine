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
    """Retorna probabilidade calibrada de CALL."""
    m = entry.get("metrics") or {}
    cal = m.get("calibrated_prob")
    if cal is not None:
        return float(cal)
    raw = m.get("raw_prob")
    return float(raw) if raw is not None else None


def _direction_pivot(metrics: dict) -> float:
    """Retorna o pivot CALL/PUT."""
    c, p = metrics.get("dynamic_call_threshold"), metrics.get("dynamic_put_threshold")
    return (float(c) + float(p)) * 0.5 if c is not None and p is not None else 0.5


def infer_dl_direction(entry: dict) -> TradeDirection | None:
    """Obtem direcao prevista pelo DL."""
    d = entry.get("direction")
    if d is not None:
        return d
    p = _direction_prob(entry)
    if p is None:
        return None
    return TradeDirection.CALL if float(p) > _direction_pivot(entry.get("metrics") or {}) else TradeDirection.PUT


def is_technically_blocked(entry: dict) -> bool:
    """Indica bloqueio absoluto por falha tecnica."""
    m = entry.get("metrics") or {}
    return m.get("deploy_ok") is False or str(m.get("gate_reason") or "") in _TECHNICAL_BLOCKS


def _clamp01(v: float) -> float:
    """Limita valor ao intervalo [0, 1]."""
    return max(0.0, min(1.0, float(v)))


def _seed_direction_metrics(metrics: dict, *, dl_dir: TradeDirection, prob: float) -> float:
    """Inicializa scores laterais TCN."""
    call, put = prob, 1.0 - prob
    metrics.update(
        {
            "dl_direction": dl_dir.name,
            "exec_direction": dl_dir.name,
            "resolved_direction": dl_dir.name,
            "direction_inverted": False,
            "meta_direction_flip": False,
            "direction_call_score": call,
            "direction_put_score": put,
            "direction_margin": direction_margin_from_probability(prob, direction=dl_dir.name),
        }
    )
    return call if dl_dir == TradeDirection.CALL else put


def _sync_entry_metrics(entry: dict, metrics: dict) -> None:
    """Propaga metricas resolvidas de volta."""
    entry["metrics"].update(metrics) if isinstance(entry.get("metrics"), dict) else entry.setdefault("metrics", metrics)


def _has_meta_zscore_telemetry(metrics: dict) -> bool:
    """True quando Z-Score meta ja foi anexado."""
    if metrics.get("meta_payoff_edge_zscore") is None and metrics.get("edge_zscore") is None:
        return False
    s = metrics.get("edge_zscore_samples")
    return s is None or int(s) >= 2


def _meta_zscore_soft_ok(metrics: dict) -> bool:
    """True quando Z meta nao e fortemente negativo."""
    return metric_float(metrics, "meta_payoff_edge_zscore", "edge_zscore", default=0.0) >= _STRONG_NEGATIVE_ZSCORE


def _reject_on_quality_gate(
    entry: dict,
    metrics: dict,
    gate_probe: dict,
    exec_cfg_dict: dict,
    *,
    risk_manager: Any | None = None,
    recovery_active: bool = False,
    skipped_cycles_counter: int | None = None,
    orch: Any | None = None,
) -> bool:
    """Retorna True quando quality gate reprova."""
    kw = {
        "exec_cfg": exec_cfg_dict,
        "risk_manager": risk_manager,
        "skipped_cycles_counter": skipped_cycles_counter,
        "orch": orch,
    }
    has_meta = _has_meta_zscore_telemetry(gate_probe)
    meta_passed = has_meta and evaluate_meta_payoff_quality(gate_probe, **kw)
    tcn_passed = passes_execution_quality(gate_probe, **kw)
    meta_soft_ok = (not has_meta) or _meta_zscore_soft_ok(gate_probe)
    waived = bool(gate_probe.get("meta_payoff_veto_waived"))
    accepted = (
        (meta_passed or tcn_passed)
        if (recovery_active or waived)
        else (((tcn_passed and meta_soft_ok) or meta_passed) if has_meta else tcn_passed)
    )
    if accepted:
        for k in ("execution_gate_state", "quality_gate_regime", "direction_margin"):
            if k in gate_probe:
                metrics[k] = gate_probe[k]
        if meta_passed and not tcn_passed:
            metrics["execution_gate_state"] = "meta_zscore_pass"
        elif tcn_passed and not meta_passed:
            for k in ("quality_guard_reject", "regime_skip_cycle", "quality_gate_reason", "execution_gate_state"):
                metrics.pop(k, None)
        return False
    metrics.update(
        {
            k: gate_probe[k]
            for k in ("regime_skip_cycle", "direction_margin", "quality_gate_reason", "execution_gate_state")
            if k in gate_probe
        }
    )
    metrics.update({"signal_status": SIGNAL_SUSPENDED, "quality_guard_reject": True})
    _sync_entry_metrics(entry, metrics)
    return True


def _apply_technical_agreement(
    metrics: dict, dl_dir: TradeDirection, prob: float, exec_cfg: dict
) -> tuple[float, bool]:
    """Aplica acordo ou desacordo tecnico de votos."""
    call_votes, put_votes = int(metrics.get("call_votes", 0)), int(metrics.get("put_votes", 0))
    total = call_votes + put_votes
    if total <= 0:
        return prob, False
    opp = put_votes / total if dl_dir == TradeDirection.CALL else call_votes / total
    if opp >= 0.75 and exec_cfg.get("discordance_veto_enabled", True):
        return prob, True
    adjusted = prob
    if (1.0 - opp) >= 0.80:
        adjusted = min(1.0, prob + 0.05) if dl_dir == TradeDirection.CALL else max(0.0, prob - 0.05)
    return adjusted, False


def _initial_direction_checks(entry: dict, exec_cfg_dict: dict) -> tuple[TradeDirection, dict, float] | None:
    """Executa checagens tecnicas preliminares."""
    dl_dir = infer_dl_direction(entry)
    if is_technically_blocked(entry) or dl_dir is None:
        return None
    metrics = dict(entry.get("metrics") or {})
    if veto_calibration_neutral_drift(metrics):
        _sync_entry_metrics(entry, metrics)
        return None
    metrics["bb_width_anomaly_ratio"] = D_SQUEEZE_BB_WIDTH_ANOMALY_RATIO
    prob = _direction_prob(entry)
    if prob is None:  # pragma: no cover
        prob = 0.55 if dl_dir == TradeDirection.CALL else 0.45
    prob, should_veto = _apply_technical_agreement(metrics, dl_dir, _clamp01(prob), exec_cfg_dict)
    if should_veto:
        metrics.update({"gate_reason": "technical_discordance", "quality_guard_reject": True})
        _sync_entry_metrics(entry, metrics)
        return None
    return dl_dir, metrics, prob


def _finalize_execution_metrics(
    entry: dict,
    metrics: dict,
    dl_dir: TradeDirection,
    prob: float,
    predicted_edge: float,
    *,
    meta_applied: bool,
    score: float,
    symbol: str | None,
) -> tuple[TradeDirection, dict]:
    """Aplica decisao de execucao final."""
    exec_dir, _final_score = apply_meta_regression_edge(
        dl_dir,
        metrics,
        predicted_edge,
        meta_applied=meta_applied,
        base_score=score,
        symbol=symbol,
    )
    metrics.update(
        {
            "exec_direction": exec_dir.name,
            "resolved_direction": exec_dir.name,
            "direction_inverted": exec_dir != dl_dir,
            "tcn_score": prob,
        }
    )
    ensure_direction_margin(metrics)
    _sync_entry_metrics(entry, metrics)
    return exec_dir, metrics


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
    skipped_cycles_counter: int | None = None,
    orch: Any | None = None,
) -> tuple[TradeDirection, dict] | None:
    """Resolve direcao micro fiel ao sinal TCN/DL com telemetria meta-regressor."""
    _ = (calibration_cfg, corr_matrix, peer_entry, cycle_id)
    exec_cfg_dict = exec_cfg if isinstance(exec_cfg, dict) else {}
    checks = _initial_direction_checks(entry, exec_cfg_dict)
    if checks is None:
        return None
    dl_dir, metrics, prob = checks
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
            metrics, float(metrics.get("predicted_payoff_edge", predicted_edge)), symbol=symbol
        )
    if should_veto_meta_payoff_negative_zscore(metrics, direction=dl_dir, risk_manager=risk_manager):
        apply_meta_payoff_negative_zscore_veto(metrics)
        _sync_entry_metrics(entry, metrics)
        return None
    gate_probe = dict(metrics)
    gate_probe["predicted_payoff_edge"] = float(predicted_edge)
    gate_probe["meta_classifier_applied"] = bool(meta_applied)
    kw = {
        "risk_manager": risk_manager,
        "recovery_active": recovery_active,
        "skipped_cycles_counter": skipped_cycles_counter,
        "orch": orch,
    }
    if _reject_on_quality_gate(entry, metrics, gate_probe, exec_cfg_dict, **kw):
        if not (recovery_active and _meta_zscore_soft_ok(metrics)):
            return None
        metrics.pop("signal_status", None)
        entry_metrics = entry.get("metrics")
        if isinstance(entry_metrics, dict):
            entry_metrics.pop("signal_status", None)
    require_meta = bool(exec_cfg_dict.get("require_meta_for_execution", False))
    if require_meta and not recovery_active and not bool(meta_applied):
        metrics.update({"gate_reason": "meta_unavailable", "quality_guard_reject": True})
        _sync_entry_metrics(entry, metrics)
        return None
    return _finalize_execution_metrics(
        entry,
        metrics,
        dl_dir,
        prob,
        predicted_edge,
        meta_applied=meta_applied,
        score=score,
        symbol=symbol,
    )
