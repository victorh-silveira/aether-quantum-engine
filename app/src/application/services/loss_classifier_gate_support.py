"""Helpers de limpeza e p_eff do gate loss-classifier (FLIP no piso)."""

from __future__ import annotations

from typing import Any

from src.application.services.loss_classifier_vectors import store_loss_flip_ctx
from src.domain.models.trade import TradeDirection


_STALE_LOSS_CLF_KEYS = (
    "loss_clf_hard",
    "loss_clf_flip",
    "loss_clf_p_loss",
    "loss_clf_p_eff",
    "loss_clf_model_version",
    "loss_clf_n_train",
    "loss_clf_auto_learn",
    "loss_clf_veto_ready",
    "loss_clf_veto_mode",
    "loss_clf_hard_p_loss_floor",
    "loss_clf_flip_floor",
    "loss_clf_feature_vector",
    "loss_clf_cycle_id",
    "loss_clf_tape_discord",
    "loss_clf_young_shrink",
    "loss_clf_bootstrap",
    "loss_clf_flip_blocked",
)


def clear_stale_loss_clf_metrics(metrics: dict[str, Any]) -> None:
    """Remove telemetria loss-clf do ciclo anterior antes do predict fresco."""
    for key in _STALE_LOSS_CLF_KEYS:
        metrics.pop(key, None)
    reason = str(metrics.get("gate_reason") or "").strip()
    if reason in {"loss_clf", "loss_clf_hard"}:
        metrics.pop("gate_reason", None)
    status = str(metrics.get("signal_status") or "").strip().upper()
    if status in {"SKIP:LOSS_CLF", "SKIP:LOSS_CLF_HARD"}:
        metrics.pop("signal_status", None)


def resolve_tcn_ref(metrics: dict[str, Any], exec_dir: TradeDirection) -> TradeDirection:
    """Ancora features no TCN; fallback no lado EXEC se TCN ausente."""
    name = str(metrics.get("tcn_direction") or metrics.get("dl_direction") or "").strip().upper()
    if name == TradeDirection.CALL.name:
        return TradeDirection.CALL
    if name == TradeDirection.PUT.name:
        return TradeDirection.PUT
    return exec_dir


def resolve_scale_tape(metrics: dict[str, Any]) -> str | None:
    """Tape CALL/PUT da scale vision, ou None se ausente/invalido."""
    raw = str(metrics.get("scale_tape_consensus") or "").strip().upper()
    if raw in {TradeDirection.CALL.name, TradeDirection.PUT.name}:
        return raw
    return None


def compute_loss_clf_p_eff(
    p_loss: float,
    *,
    n_train: int,
    flip_trust_n: int,
    flip_young_shrink: float,
    hard_floor: float,
    flip_young_p_eff_floor: float,
    tcn_ref: TradeDirection,
    tape: str | None,
) -> tuple[float, bool, bool, float]:
    """Calcula p_eff (shrink young) e piso de decisao; tape so telemetria.

    Returns:
        (p_eff, young_shrink_applied, tape_discord, flip_floor)
    """
    young = int(n_train) < int(flip_trust_n)
    p_eff = 0.5 + (float(p_loss) - 0.5) * float(flip_young_shrink) if young else float(p_loss)
    tape_discord = bool(tape) and tape != tcn_ref.name
    flip_floor = float(flip_young_p_eff_floor) if young else float(hard_floor)
    return p_eff, young, tape_discord, flip_floor


def stamp_loss_clf_flip_ctx(orch: Any, symbol: str | None, metrics: dict[str, Any]) -> None:
    """Persiste telemetria de FLIP para QUALITY no settle."""
    if orch is None or not symbol:
        return
    store_loss_flip_ctx(
        orch,
        str(symbol),
        {
            "flip": bool(metrics.get("loss_clf_flip")),
            "young": bool(metrics.get("loss_clf_young_shrink")),
            "p_loss": float(metrics.get("loss_clf_p_loss") or 0.5),
            "p_eff": float(metrics.get("loss_clf_p_eff") or 0.5),
            "n_train": int(metrics.get("loss_clf_n_train") or 0),
            "blocked": str(metrics.get("loss_clf_flip_blocked") or ""),
        },
    )
