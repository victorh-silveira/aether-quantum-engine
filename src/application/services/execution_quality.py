"""Qualidade direcional, margem e veredicto de gate (HARD_SKIP | SOFT_SIZE | ALLOW)."""

from __future__ import annotations

from typing import Any


VERDICT_HARD_SKIP = "HARD_SKIP"
VERDICT_SOFT_SIZE = "SOFT_SIZE"
VERDICT_ALLOW = "ALLOW"

_SOFT_FLAGS = (
    "loss_clf_soft",
    "cal_margin_soft",
    "mini_pair_soft",
    "regime_chop_soft",
    "fusion_weak_ev_soft",
    "anti_loss_soft",
)

__all__ = [
    "VERDICT_ALLOW",
    "VERDICT_HARD_SKIP",
    "VERDICT_SOFT_SIZE",
    "apply_quality_penalty_to_metrics",
    "blocks_single_strike",
    "direction_margin_from_probability",
    "ensure_direction_margin",
    "is_skip_signal_status",
    "is_soft_size",
    "metrics_block_execution",
    "read_risk_session_state",
    "stamp_allow",
    "stamp_edge_without_direction",
    "stamp_hard_skip",
    "stamp_soft_size",
    "sync_direction_margin",
]


def direction_margin_from_probability(call_probability: float, *, direction: str | None = None) -> float:
    """Calcula distancia da confianca lateral escolhida ao centro neutro 0.50."""
    call_score = max(0.0, min(1.0, float(call_probability)))
    side_probability = call_score
    if str(direction or "").upper() == "PUT":
        side_probability = 1.0 - call_score
    return abs(side_probability - 0.5)


def ensure_direction_margin(metrics: dict) -> float:
    """Garante direction_margin a partir da probabilidade calibrada ou bruta."""
    prob = metrics.get("calibrated_prob", metrics.get("raw_prob"))
    if prob is not None:
        direction = metrics.get("exec_direction") or metrics.get("resolved_direction") or metrics.get("dl_direction")
        margin = direction_margin_from_probability(
            float(prob),
            direction=str(direction) if direction is not None else None,
        )
    else:
        stored = metrics.get("direction_margin")
        margin = float(stored) if stored is not None else 0.0
    metrics["direction_margin"] = margin
    return margin


def sync_direction_margin(metrics: dict, *, direction: str) -> float:
    """Atualiza direction_margin a partir da probabilidade calibrada ou scores laterais."""
    prob = metrics.get("calibrated_prob", metrics.get("raw_prob"))
    if prob is not None:
        margin = direction_margin_from_probability(float(prob), direction=direction)
    else:
        margin = abs(float(metrics["direction_call_score"]) - float(metrics["direction_put_score"]))
    metrics["direction_margin"] = margin
    return margin


def stamp_edge_without_direction(
    metrics: dict[str, Any],
    *,
    margin_floor: float,
    score_factor: float = 0.85,
) -> None:
    """Marca Edge meta nao acionavel quando a margem TCN falha e comprime score."""
    metrics["edge_without_direction"] = True
    metrics["edge_without_direction_margin_floor"] = float(margin_floor)
    edge_raw = metrics.get("predicted_payoff_edge")
    if edge_raw is not None:
        metrics["edge_without_direction_edge"] = float(edge_raw)
    base = metrics.get("trade_score")
    if base is None:
        base = metrics.get("conviction")
    if base is None:
        return
    factor = max(0.0, min(1.0, float(score_factor)))
    compressed = float(base) * factor
    metrics["trade_score"] = compressed
    metrics["conviction"] = compressed
    metrics["edge_without_direction_penalty"] = max(0.0, float(base) - compressed)


def read_risk_session_state(
    risk_manager: Any | None,
    *,
    linear: int | None = None,
    pending_loss_total: float | None = None,
) -> tuple[int, float]:
    """Extrai perdas lineares e passivo pendente do RiskManager ou overrides explicitos."""
    if linear is not None:
        session_linear = int(linear)
    elif risk_manager is not None:
        session_linear = int(
            getattr(risk_manager, "consecutive_losses_linear", getattr(risk_manager, "linear", 0)) or 0,
        )
    else:
        session_linear = 0
    if pending_loss_total is not None:
        pending = float(pending_loss_total)
    elif risk_manager is not None:
        total_fn = getattr(risk_manager, "pending_loss_total", None)
        if callable(total_fn):
            pending = float(total_fn())
        else:
            pending_map = getattr(risk_manager, "pending_loss", None)
            pending = float(sum(pending_map.values())) if isinstance(pending_map, dict) else 0.0
    else:
        pending = 0.0
    return session_linear, pending


def apply_quality_penalty_to_metrics(
    metrics: dict,
    *,
    exec_cfg: dict | None = None,
    risk_manager: Any | None = None,
    skipped_cycles_counter: int | None = None,
    orch: Any | None = None,
    **kwargs,
) -> float:
    """Telemetria de margem sem rejeicao de qualidade."""
    _ = (exec_cfg, risk_manager, skipped_cycles_counter, orch, kwargs)
    ensure_direction_margin(metrics)
    return 0.0


def stamp_hard_skip(metrics: dict[str, Any], reason: str) -> None:
    """Marca veto tecnico HARD; limpa soft de sizing do mesmo ciclo."""
    metrics["gate_verdict"] = VERDICT_HARD_SKIP
    metrics["gate_verdict_reason"] = str(reason or "hard_skip")
    metrics["execution_candidate_ready"] = False
    metrics["execute"] = False
    metrics["gate_reason"] = str(reason or "hard_skip")


def stamp_soft_size(metrics: dict[str, Any], reason: str) -> None:
    """Marca EXEC permitido com sizing comprimido; bloqueia Single-Strike."""
    if str(metrics.get("gate_verdict") or "").upper() == VERDICT_HARD_SKIP:
        return
    metrics["gate_verdict"] = VERDICT_SOFT_SIZE
    metrics["gate_verdict_reason"] = str(reason or "soft_size")


def stamp_allow(metrics: dict[str, Any], reason: str = "allow") -> None:
    """Marca ciclo liberado para Single-Strike quando nenhum soft/hard ativo."""
    if str(metrics.get("gate_verdict") or "").upper() in {VERDICT_HARD_SKIP, VERDICT_SOFT_SIZE}:
        return
    metrics["gate_verdict"] = VERDICT_ALLOW
    metrics["gate_verdict_reason"] = str(reason or "allow")


def is_soft_size(metrics: dict[str, Any] | None) -> bool:
    """True quando o ciclo esta em SOFT_SIZE ou ha flag soft de sinal."""
    if not isinstance(metrics, dict):
        return False
    if str(metrics.get("gate_verdict") or "").strip().upper() == VERDICT_SOFT_SIZE:
        return True
    return any(bool(metrics.get(flag)) for flag in _SOFT_FLAGS)


def blocks_single_strike(metrics: dict[str, Any] | None) -> bool:
    """True quando Single-Strike / stop-win boost nao deve elevar stake."""
    return is_soft_size(metrics)


def is_skip_signal_status(status: object) -> bool:
    """True para SKIP tecnico (SKIP / SKIP:REASON)."""
    token = str(status or "").strip().upper()
    return token == "SKIP" or token.startswith("SKIP:")


def metrics_block_execution(metrics: dict[str, Any] | None) -> bool:
    """True quando candidato nao pode EXEC (ready=False, HARD_SKIP ou SKIP)."""
    if not isinstance(metrics, dict):
        return False
    if metrics.get("execution_candidate_ready") is False:
        return True
    if str(metrics.get("gate_verdict") or "").upper() == VERDICT_HARD_SKIP:
        return True
    return is_skip_signal_status(metrics.get("signal_status"))
