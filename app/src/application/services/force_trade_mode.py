"""Modo force-trade: sintetiza candidatos e stake minima quando o ciclo exige ordem."""

from __future__ import annotations

from typing import Any

from src.domain.models.trade import TradeDirection
from src.domain.risk.stake_sizing import enrich_metrics_conviction, metric_float


_TECHNICAL_BLOCKS = frozenset({"data", "predict_error", "training"})
_FORCE_MIN_TRADE_SCORE = 0.51


def force_trade_every_cycle(exec_cfg: dict | None) -> bool:
    """Indica se o bloco de execucao exige trade a cada ciclo."""
    if not isinstance(exec_cfg, dict):
        return False
    return bool(exec_cfg.get("force_trade_every_cycle", False))


def force_trade_from_config(config: dict | None) -> bool:
    """Le force_trade_every_cycle a partir da config raiz do motor."""
    if not isinstance(config, dict):
        return False
    orch = config.get("orchestrator")
    if not isinstance(orch, dict):
        return False
    execution = orch.get("execution")
    return force_trade_every_cycle(execution if isinstance(execution, dict) else None)


def force_trade_from_orch(orch: Any | None) -> bool:
    """Indica force-trade a partir da instancia do orquestrador."""
    if orch is None:
        return False
    return force_trade_from_config(getattr(orch, "config", None))


def resolve_force_min_stake(config: dict | None) -> float:
    """Resolve stake minima de force-trade a partir de risk_management.params."""
    if not isinstance(config, dict):
        return 1.0
    risk = config.get("risk_management")
    if not isinstance(risk, dict):
        return 1.0
    params = risk.get("params")
    if not isinstance(params, dict):
        return 1.0
    try:
        return max(0.35, float(params.get("stake_min", 1.0)))
    except (TypeError, ValueError):
        return 1.0


def _entry_prob(entry: dict) -> float | None:
    """Extrai probabilidade calibrada ou bruta do entry de decisao."""
    metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
    for key in ("calibrated_prob", "raw_prob"):
        raw = metrics.get(key)
        if raw is None:
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return None


def synthesize_force_direction(entry: dict) -> TradeDirection | None:
    """Deriva CALL/PUT forçado a partir da direcao existente ou da probabilidade."""
    metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
    if metrics.get("deploy_ok") is False:
        return None
    if str(metrics.get("gate_reason") or "") in _TECHNICAL_BLOCKS:
        return None
    direction = entry.get("direction")
    if direction is not None and hasattr(direction, "name"):
        return direction
    prob = _entry_prob(entry)
    if prob is None:
        return None
    return TradeDirection.CALL if float(prob) + 1e-12 >= 0.5 else TradeDirection.PUT


def synthesize_force_trade_candidate(
    trade_symbols: list[str] | tuple[str, ...],
    decisions: dict,
    orch: Any | None = None,
) -> tuple[str, TradeDirection, dict] | None:
    """Monta o primeiro candidato elegivel sob force-trade no pool de simbolos."""
    if not isinstance(decisions, dict):
        return None
    for symbol in trade_symbols:
        entry = decisions.get(symbol)
        if not isinstance(entry, dict):
            continue
        direction = synthesize_force_direction(entry)
        if direction is None:
            continue
        metrics = dict(entry.get("metrics") or {})
        metrics.pop("quality_guard_reject", None)
        metrics.pop("regime_skip_cycle", None)
        metrics.pop("quality_gate_reason", None)
        metrics.pop("persistence_guard_skip", None)
        if str(metrics.get("gate_reason") or "") == "neutral_clamp":
            metrics["gate_reason"] = None
        if str(metrics.get("calibration_mode") or "") == "neutral_clamp":
            metrics["calibration_mode"] = "calibrated"
        metrics["execute"] = True
        metrics["force_trade_every_cycle"] = True
        metrics["exec_direction"] = direction.name
        metrics["resolved_direction"] = direction.name
        metrics["dl_direction"] = direction.name
        score = max(
            _FORCE_MIN_TRADE_SCORE,
            metric_float(metrics, "trade_score", "conviction", default=_FORCE_MIN_TRADE_SCORE),
        )
        metrics["trade_score"] = score
        metrics["conviction"] = score
        enrich_metrics_conviction(metrics)
        entry["direction"] = direction
        entry["metrics"] = metrics
        _ = orch
        return symbol, direction, metrics
    return None
