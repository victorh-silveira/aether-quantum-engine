"""Aplica soft Kelly do loss-classifier apos signal_skip (sem hard SKIP)."""

from __future__ import annotations

import logging
from typing import Any

from src.application.services.execution_quality_gate import read_risk_session_state
from src.application.services.log_dedupe import log_info_if_changed
from src.application.services.loss_classifier_features import build_loss_feature_vector
from src.application.services.loss_classifier_vectors import store_loss_feature_vector
from src.domain.models.trade import TradeDirection
from src.infrastructure.inference.loss_classifier_client import (
    loss_classifier_enabled,
    resolve_loss_classifier_config,
)
from src.infrastructure.inference.loss_classifier_pool import predict_loss_via_config_sync


logger = logging.getLogger("AETH")


def resolve_soft_kelly_mult(p_loss: float, cfg: dict[str, Any]) -> float:
    """Interpola soft Kelly entre floor e p_loss alto (maior risco → menor mult)."""
    floor = float(cfg["veto_p_loss_floor"])
    high = float(cfg["soft_p_loss_high"])
    mult_lo = float(cfg["soft_kelly_mult"])
    mult_hi = float(cfg["soft_kelly_mult_high"])
    p = float(p_loss)
    if p >= high:
        return mult_hi
    if p <= floor or high <= floor:
        return mult_lo
    t = (p - floor) / (high - floor)
    return mult_lo + t * (mult_hi - mult_lo)


def _apply_soft_kelly(metrics: dict[str, Any], mult: float, *, p_loss: float, cfg: dict[str, Any]) -> None:
    """Atenua kelly_fraction_scale e marca teto de stake abaixo do piso explore."""
    scale = float(metrics.get("kelly_fraction_scale", 1.0) or 1.0)
    metrics["kelly_fraction_scale"] = max(0.05, scale * float(mult))
    metrics["loss_clf_soft"] = True
    metrics["loss_clf_soft_kelly_mult"] = float(mult)
    metrics["loss_clf_soft_max_stake_pct"] = float(cfg["soft_max_stake_pct_high"])
    _ = p_loss


def apply_loss_classifier_gate(
    metrics: dict[str, Any],
    exec_dir: TradeDirection,
    *,
    orch: Any | None = None,
    force: bool = False,
    symbol: str | None = None,
) -> bool:
    """Consulta loss-clf; sempre soft Kelly quando veto. Nunca hard SKIP. Retorna False."""
    if force or orch is None:
        return False
    if str(metrics.get("gate_reason") or "").strip():
        return False
    config = getattr(orch, "config", None)
    if not loss_classifier_enabled(config if isinstance(config, dict) else None):
        return False
    cfg = resolve_loss_classifier_config(None)
    risk_manager = getattr(orch, "risk_manager", None)
    linear, pending = read_risk_session_state(risk_manager)
    bankroll = 0.0
    state = getattr(orch, "state", None)
    try:
        bankroll = float(getattr(state, "balance", 0.0) or 0.0)
    except (TypeError, ValueError):
        bankroll = 0.0
    vector = build_loss_feature_vector(
        metrics,
        exec_dir,
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
            "direction": exec_dir.name,
            "veto_p_loss_floor": float(cfg["veto_p_loss_floor"]),
        },
    )
    metrics["loss_clf_p_loss"] = float(response["p_loss"])
    metrics["loss_clf_model_version"] = str(response["model_version"])
    metrics["loss_clf_n_train"] = int(response["n_train"])
    metrics["loss_clf_auto_learn"] = bool(response["auto_learn_applied"])
    metrics["loss_clf_veto_ready"] = bool(response["veto_ready"])
    metrics["loss_clf_veto_mode"] = "soft"
    auto_flag = 1 if response["auto_learn_applied"] else 0
    if response["veto"] and response["veto_ready"]:
        soft_mult = resolve_soft_kelly_mult(float(response["p_loss"]), cfg)
        material_pending = float(pending) + 1e-12 >= 0.5
        if material_pending:
            metrics["loss_clf_soft"] = True
            metrics["loss_clf_soft_kelly_mult"] = float(soft_mult)
            metrics["loss_clf_soft_waived_pending"] = True
            log_info_if_changed(
                orch,
                logger,
                "loss_clf_soft_pending",
                f"{response['model_version']}:{response['p_loss']:.4f}:waive:{auto_flag}",
                "LOSS_CLF || SOFT_WAIVE_PENDING auto_learn=%d ver=%s n=%d p_loss=%.4f cover=1",
                auto_flag,
                response["model_version"],
                int(response["n_train"]),
                float(response["p_loss"]),
            )
        else:
            _apply_soft_kelly(metrics, soft_mult, p_loss=float(response["p_loss"]), cfg=cfg)
            log_info_if_changed(
                orch,
                logger,
                "loss_clf_soft",
                f"{response['model_version']}:{response['p_loss']:.4f}:{soft_mult:.2f}:{auto_flag}",
                "LOSS_CLF || SOFT auto_learn=%d ver=%s n=%d p_loss=%.4f kelly_mult=%.2f",
                auto_flag,
                response["model_version"],
                int(response["n_train"]),
                float(response["p_loss"]),
                float(soft_mult),
            )
        return False
    log_info_if_changed(
        orch,
        logger,
        "loss_clf_ok",
        f"{response['model_version']}:{response['p_loss']:.4f}:{auto_flag}",
        "LOSS_CLF || OK auto_learn=%d ver=%s n=%d p_loss=%.4f veto_ready=%d",
        auto_flag,
        response["model_version"],
        int(response["n_train"]),
        float(response["p_loss"]),
        1 if response["veto_ready"] else 0,
    )
    return False
