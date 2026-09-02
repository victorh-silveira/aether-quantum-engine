"""Inferencia de direcao TCN e bloqueios tecnicos (sem vetos de sinal)."""

from __future__ import annotations

from typing import Any

from src.application.services.force_trade_mode import force_trade_every_cycle, synthesize_force_direction
from src.domain.models.trade import TradeDirection
from src.domain.risk.soft_recovery_policy import negative_zscore_veto_floor_for_risk
from src.domain.risk.stake_sizing import metric_float


_TECHNICAL_BLOCKS = frozenset({"data", "predict_error", "training"})
_NEUTRAL_PIVOT_EPS = 1e-9


def direction_prob(entry: dict) -> float | None:
    """Le probabilidade calibrada ou bruta do candidato."""
    m = entry.get("metrics") or {}
    cal = m.get("calibrated_prob")
    if cal is not None:
        return float(cal)
    raw = m.get("raw_prob")
    return float(raw) if raw is not None else None


def direction_pivot(metrics: dict) -> float:
    """Calcula pivo neutro a partir dos limiares dinamicos CALL/PUT."""
    c, p = metrics.get("dynamic_call_threshold"), metrics.get("dynamic_put_threshold")
    return (float(c) + float(p)) * 0.5 if c is not None and p is not None else 0.5


def infer_dl_direction(entry: dict) -> TradeDirection | None:
    """Infere CALL/PUT a partir da probabilidade, ou None se indefinido."""
    metrics = entry.get("metrics") or {}
    if metrics.get("calibration_mode") == "neutral_zone" or metrics.get("gate_reason") == "neutral_zone":
        return None
    d = entry.get("direction")
    if d is not None:
        return d
    p = direction_prob(entry)
    if p is None:
        return None
    pivot = direction_pivot(metrics)
    return TradeDirection.CALL if float(p) + _NEUTRAL_PIVOT_EPS >= float(pivot) else TradeDirection.PUT


def is_technically_blocked(entry: dict) -> bool:
    """True quando deploy falhou ou o gate_reason e bloqueio tecnico."""
    m = entry.get("metrics") or {}
    return m.get("deploy_ok") is False or str(m.get("gate_reason") or "") in _TECHNICAL_BLOCKS


def clamp01(v: float) -> float:
    """Restringe valor ao intervalo unitario [0, 1]."""
    return max(0.0, min(1.0, float(v)))


def seed_direction_metrics(metrics: dict, *, dl_dir: TradeDirection, prob: float) -> float:
    """Inicializa scores e margem direcional a partir da probabilidade TCN."""
    cal = float(prob)
    call, put = cal, 1.0 - cal
    raw_existing = metrics.get("raw_prob")
    try:
        raw = float(raw_existing) if raw_existing is not None else cal
    except (TypeError, ValueError):
        raw = cal
    metrics.update(
        {
            "dl_direction": dl_dir.name,
            "resolved_direction": dl_dir.name,
            "raw_prob": raw,
            "raw_call_prob": raw,
            "raw_put_prob": 1.0 - raw,
            "raw_margin": abs(raw - 0.5),
            "calibrated_prob": cal,
            "calibrated_call_prob": call,
            "calibrated_put_prob": put,
            "direction_margin": abs(cal - 0.5),
        }
    )
    return max(call, put)


def sync_entry_metrics(entry: dict, metrics: dict) -> None:
    """Propaga metricas resolvidas de volta ao entry do ciclo."""
    entry["metrics"].update(metrics) if isinstance(entry.get("metrics"), dict) else entry.setdefault("metrics", metrics)


def meta_zscore_soft_ok(metrics: dict, *, risk_manager: Any | None = None) -> bool:
    """True quando o Z-Score meta esta acima do piso soft de recovery."""
    floor = negative_zscore_veto_floor_for_risk(risk_manager)
    return metric_float(metrics, "meta_payoff_edge_zscore", "edge_zscore", default=0.0) >= floor


def initial_direction_checks(
    entry: dict,
    exec_cfg_dict: dict,
    *,
    orch: Any | None = None,
    skipped_cycles_counter: int | None = None,
) -> tuple[TradeDirection, dict, float] | None:
    """Resolve lado TCN; so bloqueia por falta de direcao ou bloqueio tecnico."""
    _ = (orch, skipped_cycles_counter)
    metrics = dict(entry.get("metrics") or {})
    for sticky in (
        "side_eq_gate_done",
        "side_eq_blocked",
        "side_eq_flipped",
        "side_eq_flip_from",
        "side_eq_margin_boost",
        "execution_candidate_ready",
        "quality_guard_reject",
        "regime_skip_cycle",
    ):
        metrics.pop(sticky, None)
    force = force_trade_every_cycle(exec_cfg_dict)
    dl_dir = infer_dl_direction(entry)
    if force and dl_dir is None:
        dl_dir = synthesize_force_direction(entry)
    if is_technically_blocked(entry) or dl_dir is None:
        sync_entry_metrics(entry, metrics)
        return None
    prob = direction_prob(entry)
    if prob is None:
        prob = 0.55 if dl_dir == TradeDirection.CALL else 0.45
    metrics["dl_direction"] = dl_dir.name
    metrics["resolved_direction"] = dl_dir.name
    sync_entry_metrics(entry, metrics)
    return dl_dir, metrics, clamp01(float(prob))
