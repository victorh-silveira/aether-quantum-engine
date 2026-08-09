"""Aplica soft Kelly e FLIP CALL/PUT ancorado no TCN apos signal_skip."""

from __future__ import annotations

import logging
from typing import Any

from src.application.services.execution_quality_gate import read_risk_session_state
from src.application.services.log_dedupe import log_debug_if_changed
from src.application.services.loss_classifier_features import build_loss_feature_vector
from src.application.services.loss_classifier_flip import (
    apply_loss_flip,
    apply_soft_kelly,
    cal_disagrees_ref,
    is_collapsed_p_loss,
    is_seed_model,
    resolve_soft_kelly_mult,
    scale_confirms_ref,
)
from src.application.services.loss_classifier_vectors import store_loss_feature_vector
from src.domain.models.trade import TradeDirection
from src.infrastructure.inference.loss_classifier_client import (
    loss_classifier_enabled,
    resolve_loss_classifier_config,
)
from src.infrastructure.inference.loss_classifier_pool import predict_loss_via_config_sync


logger = logging.getLogger("AETH")

__all__ = ("apply_loss_classifier_gate", "resolve_soft_kelly_mult")

_STALE_LOSS_CLF_KEYS = (
    "loss_clf_hard",
    "loss_clf_flip",
    "loss_clf_flip_ref",
    "loss_clf_flip_blocked",
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


def _clear_stale_loss_clf_metrics(metrics: dict[str, Any]) -> None:
    """Remove telemetria loss-clf do ciclo anterior antes do predict fresco."""
    for key in _STALE_LOSS_CLF_KEYS:
        metrics.pop(key, None)
    reason = str(metrics.get("gate_reason") or "").strip()
    if reason in {"loss_clf_hard", "loss_clf_flip"}:
        metrics.pop("gate_reason", None)
    status = str(metrics.get("signal_status") or "").strip().upper()
    if status in {"SKIP:LOSS_CLF_HARD", "FLIP:LOSS_CLF"}:
        metrics.pop("signal_status", None)


def _resolve_tcn_ref(metrics: dict[str, Any], exec_dir: TradeDirection) -> TradeDirection:
    """Ancora features/FLIP no TCN; fallback no lado pos-SCALE se TCN ausente."""
    name = str(metrics.get("tcn_direction") or "").strip().upper()
    if name == TradeDirection.CALL.name:
        return TradeDirection.CALL
    if name == TradeDirection.PUT.name:
        return TradeDirection.PUT
    return exec_dir


def _emit_soft(
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


def apply_loss_classifier_gate(
    metrics: dict[str, Any],
    exec_dir: TradeDirection,
    *,
    orch: Any | None = None,
    force: bool = False,
    symbol: str | None = None,
) -> bool:
    """Consulta loss-clf; flip so com auto_learn e sem consenso SCALE; soft na faixa media."""
    if force or orch is None:
        return False
    _clear_stale_loss_clf_metrics(metrics)
    if str(metrics.get("gate_reason") or "").strip():
        return False
    config = getattr(orch, "config", None)
    if not loss_classifier_enabled(config if isinstance(config, dict) else None):
        return False
    cfg = resolve_loss_classifier_config(None)
    cycle_id = int(getattr(orch, "_active_cycle_id", 0) or 0)
    metrics["loss_clf_cycle_id"] = cycle_id
    risk_manager = getattr(orch, "risk_manager", None)
    linear, pending = read_risk_session_state(risk_manager)
    bankroll = 0.0
    state = getattr(orch, "state", None)
    try:
        bankroll = float(getattr(state, "balance", 0.0) or 0.0)
    except (TypeError, ValueError):
        bankroll = 0.0
    if bankroll <= 1e-9 and risk_manager is not None:
        try:
            bankroll = float(getattr(risk_manager, "bankroll", getattr(risk_manager, "initial_bankroll", 0.0)) or 0.0)
        except (TypeError, ValueError):
            bankroll = 0.0
    ref_dir = _resolve_tcn_ref(metrics, exec_dir)
    vector = build_loss_feature_vector(
        metrics,
        ref_dir,
        pending=pending,
        linear=linear,
        bankroll=bankroll,
    )
    metrics["loss_clf_feature_vector"] = list(vector)
    if symbol:
        store_loss_feature_vector(orch, str(symbol), list(vector))
    response = predict_loss_via_config_sync(
        config if isinstance(config, dict) else {},
        {
            "feature_vector": vector,
            "symbol": str(symbol or ""),
            "direction": ref_dir.name,
            "veto_p_loss_floor": float(cfg["veto_p_loss_floor"]),
        },
    )
    p_loss = float(response["p_loss"])
    metrics["loss_clf_p_loss"] = p_loss
    metrics["loss_clf_model_version"] = str(response["model_version"])
    metrics["loss_clf_n_train"] = int(response["n_train"])
    metrics["loss_clf_auto_learn"] = bool(response["auto_learn_applied"])
    veto_ready = bool(response["veto_ready"])
    if is_collapsed_p_loss(response):
        veto_ready = False
        metrics["loss_clf_collapsed"] = True
    metrics["loss_clf_veto_ready"] = veto_ready
    metrics["loss_clf_veto_mode"] = "soft"
    auto_flag = 1 if response["auto_learn_applied"] else 0
    flip_floor = float(cfg["hard_p_loss_floor"])
    soft_floor = float(cfg["veto_p_loss_floor"])
    seed_block = is_seed_model(response, require_auto_learn=bool(cfg.get("flip_require_auto_learn", True)))
    scale_block = scale_confirms_ref(metrics, ref_dir)
    cal_discord = cal_disagrees_ref(metrics, ref_dir)
    if seed_block and not scale_block and bool(cfg.get("flip_allow_seed_on_scale_discord", True)):
        seed_block = False
        metrics["loss_clf_flip_seed_discord"] = True
    if seed_block and cal_discord and bool(cfg.get("flip_allow_seed_on_cal_discord", True)):
        seed_block = False
        metrics["loss_clf_flip_seed_cal_discord"] = True
    if scale_block and cal_discord and bool(cfg.get("flip_allow_seed_on_cal_discord", True)):
        scale_block = False
        metrics["loss_clf_flip_cal_overrides_scale"] = True
    can_flip = bool(veto_ready) and p_loss + 1e-12 >= flip_floor and not seed_block and not scale_block
    if can_flip:
        flipped = apply_loss_flip(metrics, ref_dir, cfg=cfg)
        log_debug_if_changed(
            orch,
            logger,
            f"loss_clf_flip:{cycle_id}",
            f"{response['model_version']}:{ref_dir.name}:{flipped.name}:{p_loss:.5f}:{flip_floor:.2f}:{auto_flag}",
            "LOSS_CLF || FLIP auto_learn=%d ver=%s n=%d from=%s to=%s p_loss=%.5f floor=%.2f",
            auto_flag,
            response["model_version"],
            int(response["n_train"]),
            ref_dir.name,
            flipped.name,
            p_loss,
            flip_floor,
        )
        return False
    if bool(veto_ready) and p_loss + 1e-12 >= flip_floor and (seed_block or scale_block):
        metrics["loss_clf_flip_blocked"] = "seed" if seed_block else "scale_consensus"
        log_debug_if_changed(
            orch,
            logger,
            f"loss_clf_flip_block:{cycle_id}",
            f"{response['model_version']}:{metrics['loss_clf_flip_blocked']}:{ref_dir.name}:{p_loss:.5f}",
            "LOSS_CLF || FLIP_BLOCK reason=%s side=%s p_loss=%.5f keep_tcn=1",
            metrics["loss_clf_flip_blocked"],
            ref_dir.name,
            p_loss,
        )
    if bool(veto_ready) and p_loss + 1e-12 >= soft_floor:
        _emit_soft(
            orch,
            metrics,
            cfg=cfg,
            response=response,
            p_loss=p_loss,
            pending=pending,
            cycle_id=cycle_id,
            auto_flag=auto_flag,
        )
        return False
    ver = str(response.get("model_version") or "none")
    if metrics.get("loss_clf_collapsed"):
        log_debug_if_changed(
            orch,
            logger,
            f"loss_clf_degen:{cycle_id}",
            f"{ver}:{p_loss:.5f}:{auto_flag}",
            "LOSS_CLF || DEGEN auto_learn=%d ver=%s n=%d p_loss=%.5f keep_seed=1",
            auto_flag,
            ver,
            int(response["n_train"]),
            p_loss,
        )
        return False
    log_debug_if_changed(
        orch,
        logger,
        f"loss_clf_ok:{cycle_id}",
        f"{ver}:{p_loss:.5f}:{auto_flag}:{1 if veto_ready else 0}",
        "LOSS_CLF || OK auto_learn=%d ver=%s n=%d p_loss=%.5f veto_ready=%d",
        auto_flag,
        ver,
        int(response["n_train"]),
        p_loss,
        1 if veto_ready else 0,
    )
    return False
