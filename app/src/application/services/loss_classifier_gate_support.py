"""Helpers de limpeza/telemetria soft do gate loss-classifier."""

from __future__ import annotations

import logging
from typing import Any

from src.application.services.log_dedupe import log_debug_if_changed
from src.application.services.loss_classifier_flip import apply_soft_kelly, resolve_soft_kelly_mult
from src.domain.models.trade import TradeDirection


logger = logging.getLogger("AETH")

_STALE_LOSS_CLF_KEYS = (
    "loss_clf_hard",
    "loss_clf_flip",
    "loss_clf_flip_ref",
    "loss_clf_flip_blocked",
    "loss_clf_flip_reason",
    "loss_clf_flip_edge",
    "loss_clf_flip_edge_floor",
    "loss_clf_flip_seed_discord",
    "loss_clf_flip_seed_cal_discord",
    "loss_clf_flip_cal_overrides_scale",
    "loss_clf_flip_candle_waive_scale",
    "loss_clf_flip_candle_waive_edge",
    "loss_clf_flip_candle_floor",
    "loss_clf_flip_scale_p_override",
    "loss_clf_flip_seed_p_override",
    "loss_clf_flip_p_ovr_waive_edge",
    "loss_clf_flip_block_tcn_pos_edge",
    "loss_clf_tcn_side_edge",
    "loss_clf_soft",
    "loss_clf_soft_waived_pending",
    "loss_clf_soft_kelly_mult",
    "loss_clf_soft_max_stake_pct",
    "loss_clf_p_loss",
    "loss_clf_model_version",
    "loss_clf_n_train",
    "loss_clf_auto_learn",
    "loss_clf_veto_ready",
    "loss_clf_veto_mode",
    "loss_clf_collapsed",
    "loss_clf_hard_p_loss_floor",
    "loss_clf_flip_p_loss_floor",
    "loss_clf_feature_vector",
    "loss_clf_cycle_id",
)


def clear_stale_loss_clf_metrics(metrics: dict[str, Any]) -> None:
    """Remove telemetria loss-clf do ciclo anterior antes do predict fresco."""
    for key in _STALE_LOSS_CLF_KEYS:
        metrics.pop(key, None)
    reason = str(metrics.get("gate_reason") or "").strip()
    if reason in {"loss_clf_hard", "loss_clf_flip"}:
        metrics.pop("gate_reason", None)
    status = str(metrics.get("signal_status") or "").strip().upper()
    if status in {"SKIP:LOSS_CLF_HARD", "FLIP:LOSS_CLF"}:
        metrics.pop("signal_status", None)


def resolve_tcn_ref(metrics: dict[str, Any], exec_dir: TradeDirection) -> TradeDirection:
    """Ancora features/FLIP no TCN; fallback no lado pos-SCALE se TCN ausente."""
    name = str(metrics.get("tcn_direction") or "").strip().upper()
    if name == TradeDirection.CALL.name:
        return TradeDirection.CALL
    if name == TradeDirection.PUT.name:
        return TradeDirection.PUT
    return exec_dir


def emit_loss_clf_soft(
    orch: Any,
    metrics: dict[str, Any],
    *,
    cfg: dict[str, Any],
    response: dict[str, Any],
    p_loss: float,
    pending: float,
    cycle_id: int,
    auto_flag: int,
) -> None:
    """Aplica soft Kelly ou waive com pending material."""
    soft_mult = resolve_soft_kelly_mult(p_loss, cfg)
    material_pending = float(pending) + 1e-12 >= 0.5
    if material_pending:
        metrics["loss_clf_soft"] = True
        metrics["loss_clf_soft_kelly_mult"] = float(soft_mult)
        metrics["loss_clf_soft_waived_pending"] = True
        log_debug_if_changed(
            orch,
            logger,
            f"loss_clf_soft_pending:{cycle_id}",
            f"{response['model_version']}:{p_loss:.5f}:waive:{auto_flag}",
            "LOSS_CLF || SOFT_WAIVE_PENDING auto_learn=%d ver=%s n=%d p_loss=%.5f cover=1",
            auto_flag,
            response["model_version"],
            int(response["n_train"]),
            p_loss,
        )
        return
    apply_soft_kelly(metrics, soft_mult, p_loss=p_loss, cfg=cfg)
    log_debug_if_changed(
        orch,
        logger,
        f"loss_clf_soft:{cycle_id}",
        f"{response['model_version']}:{p_loss:.5f}:{soft_mult:.2f}:{auto_flag}",
        "LOSS_CLF || SOFT auto_learn=%d ver=%s n=%d p_loss=%.5f kelly_mult=%.2f",
        auto_flag,
        response["model_version"],
        int(response["n_train"]),
        p_loss,
        float(soft_mult),
    )
