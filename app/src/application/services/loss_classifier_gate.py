"""HARD SKIP por P_LOSS do aether-loss-classifier (sem FLIP, sem Soft Kelly)."""

from __future__ import annotations

import logging
from typing import Any

from src.application.services.execution_gate_verdict import stamp_hard_skip
from src.application.services.execution_quality_gate import read_risk_session_state
from src.application.services.log_dedupe import log_debug_if_changed
from src.application.services.loss_classifier_features import build_loss_feature_vector
from src.application.services.loss_classifier_gate_support import (
    clear_stale_loss_clf_metrics,
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

__all__ = ("apply_loss_classifier_gate",)


def apply_loss_classifier_gate(
    metrics: dict[str, Any],
    exec_dir: TradeDirection,
    *,
    orch: Any | None = None,
    force: bool = False,
    symbol: str | None = None,
) -> bool:
    """Consulta loss-clf; HARD SKIP se p_loss >= hard_p_loss_floor."""
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
    hard_floor = float(cfg["hard_p_loss_floor"])
    metrics["loss_clf_hard_p_loss_floor"] = hard_floor
    response = predict_loss_via_config_sync(
        config if isinstance(config, dict) else {},
        {
            "feature_vector": vector,
            "symbol": str(symbol or ""),
            "direction": ref_dir.name,
            "veto_p_loss_floor": hard_floor,
        },
    )
    p_loss = float(response["p_loss"])
    metrics["loss_clf_p_loss"] = p_loss
    metrics["loss_clf_model_version"] = str(response["model_version"])
    metrics["loss_clf_n_train"] = int(response["n_train"])
    metrics["loss_clf_auto_learn"] = bool(response["auto_learn_applied"])
    metrics["loss_clf_veto_ready"] = bool(response["veto_ready"])
    metrics["loss_clf_veto_mode"] = "hard"
    auto_flag = 1 if response["auto_learn_applied"] else 0
    ver = str(response.get("model_version") or "none")
    if p_loss + 1e-12 >= hard_floor:
        metrics["execution_candidate_ready"] = False
        metrics["gate_reason"] = "loss_clf"
        metrics["signal_status"] = "SKIP:LOSS_CLF"
        metrics["loss_clf_hard"] = True
        stamp_hard_skip(metrics, "loss_clf")
        log_debug_if_changed(
            orch,
            logger,
            f"loss_clf_hard:{cycle_id}",
            f"{ver}:{p_loss:.5f}:{hard_floor:.2f}:{auto_flag}",
            "LOSS_CLF || HARD auto_learn=%d ver=%s n=%d p_loss=%.5f floor=%.2f",
            auto_flag,
            ver,
            int(response["n_train"]),
            p_loss,
            hard_floor,
        )
        return True
    log_debug_if_changed(
        orch,
        logger,
        f"loss_clf_ok:{cycle_id}",
        f"{ver}:{p_loss:.5f}:{auto_flag}:{1 if response['veto_ready'] else 0}",
        "LOSS_CLF || OK auto_learn=%d ver=%s n=%d p_loss=%.5f veto_ready=%d",
        auto_flag,
        ver,
        int(response["n_train"]),
        p_loss,
        1 if response["veto_ready"] else 0,
    )
    return False
