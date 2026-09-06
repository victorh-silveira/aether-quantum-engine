"""Aplica soft Kelly e FLIP CALL/PUT ancorado no TCN apos signal_skip."""

from __future__ import annotations

import logging
from typing import Any

from src.application.services.execution_quality_gate import read_risk_session_state
from src.application.services.log_dedupe import log_debug_if_changed
from src.application.services.loss_classifier_features import build_loss_feature_vector
from src.application.services.loss_classifier_flip import (
    apply_loss_flip,
    is_collapsed_p_loss,
    post_flip_edge_ok,
    resolve_flip_p_loss_floor,
    resolve_flip_waivers,
    resolve_soft_kelly_mult,
    revert_loss_flip,
    seed_candle_blocks_flip,
    tcn_pos_edge_blocks_flip,
)
from src.application.services.loss_classifier_gate_support import (
    apply_flip_guards_p_override,
    clear_stale_loss_clf_metrics,
    emit_loss_clf_soft,
    resolve_tcn_ref,
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
    clear_stale_loss_clf_metrics(metrics)
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
    ref_dir = resolve_tcn_ref(metrics, exec_dir)
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
    flip_floor = resolve_flip_p_loss_floor(metrics, ref_dir, cfg=cfg)
    soft_floor = float(cfg["veto_p_loss_floor"])
    seed_block, scale_block = resolve_flip_waivers(metrics, response, ref_dir, cfg=cfg, p_loss=p_loss)
    pos_edge_block = tcn_pos_edge_blocks_flip(metrics, ref_dir, cfg=cfg)
    seed_candle_block = seed_candle_blocks_flip(metrics, response, ref_dir, cfg=cfg)
    seed_block, scale_block, pos_edge_block, seed_candle_block, flip_floor = apply_flip_guards_p_override(
        metrics,
        p_loss=p_loss,
        cfg=cfg,
        seed_block=seed_block,
        scale_block=scale_block,
        pos_edge_block=pos_edge_block,
        seed_candle_block=seed_candle_block,
        flip_floor=flip_floor,
    )
    can_flip = (
        bool(veto_ready)
        and p_loss + 1e-12 >= flip_floor
        and not seed_block
        and not scale_block
        and not pos_edge_block
        and not seed_candle_block
    )
    if can_flip:
        flipped = apply_loss_flip(metrics, ref_dir, cfg=cfg)
        if not post_flip_edge_ok(metrics, flipped, cfg=cfg):
            revert_loss_flip(metrics, ref_dir, reason="neg_edge")
            log_debug_if_changed(
                orch,
                logger,
                f"loss_clf_flip_block:{cycle_id}",
                f"{response['model_version']}:neg_edge:{ref_dir.name}:{p_loss:.5f}",
                "LOSS_CLF || FLIP_BLOCK reason=%s side=%s p_loss=%.5f keep_tcn=1",
                "neg_edge",
                ref_dir.name,
                p_loss,
            )
            emit_loss_clf_soft(
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
    if (
        bool(veto_ready)
        and p_loss + 1e-12 >= flip_floor
        and (seed_block or scale_block or pos_edge_block or seed_candle_block)
    ):
        if pos_edge_block:
            metrics["loss_clf_flip_blocked"] = "tcn_pos_edge"
        elif seed_candle_block:
            metrics["loss_clf_flip_blocked"] = "seed_candle"
        else:
            metrics["loss_clf_flip_blocked"] = "scale_consensus" if scale_block else "seed"
        log_debug_if_changed(
            orch,
            logger,
            f"loss_clf_flip_block:{cycle_id}",
            f"{response['model_version']}:{metrics['loss_clf_flip_blocked']}:{ref_dir.name}:{p_loss:.5f}",
            "LOSS_CLF || FLIP_BLOCK reason=%s side=%s p_loss=%.5f veto_abstain=1",
            metrics["loss_clf_flip_blocked"],
            ref_dir.name,
            p_loss,
        )
        is_mature = (
            bool(veto_ready)
            and bool(response.get("auto_learn_applied"))
            and int(response.get("n_train", 0)) >= 24
            and bool(cfg.get("hard_blocks_flip_block", False))
        )
        if is_mature:
            metrics["execution_candidate_ready"] = False
            metrics["gate_reason"] = "loss_clf_hard"
            metrics["signal_status"] = "SKIP:LOSS_CLF_HARD"
            return True
    if bool(veto_ready) and p_loss + 1e-12 >= soft_floor:
        emit_loss_clf_soft(
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
