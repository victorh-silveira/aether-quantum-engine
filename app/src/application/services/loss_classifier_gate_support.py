"""Helpers de limpeza do gate loss-classifier (HARD SKIP)."""

from __future__ import annotations

from typing import Any

from src.domain.models.trade import TradeDirection


_STALE_LOSS_CLF_KEYS = (
    "loss_clf_hard",
    "loss_clf_p_loss",
    "loss_clf_model_version",
    "loss_clf_n_train",
    "loss_clf_auto_learn",
    "loss_clf_veto_ready",
    "loss_clf_veto_mode",
    "loss_clf_hard_p_loss_floor",
    "loss_clf_feature_vector",
    "loss_clf_cycle_id",
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
