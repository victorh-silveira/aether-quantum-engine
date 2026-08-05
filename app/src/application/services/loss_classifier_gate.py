"""Aplica veto do loss-classifier apos signal_skip (telemetria + log)."""

from __future__ import annotations

import logging
from typing import Any

from src.application.services.execution_quality_gate import read_risk_session_state
from src.application.services.log_dedupe import log_info_if_changed
from src.application.services.loss_classifier_features import build_loss_feature_vector
from src.domain.models.trade import TradeDirection
from src.infrastructure.inference.loss_classifier_client import (
    loss_classifier_enabled,
    resolve_loss_classifier_config,
)
from src.infrastructure.inference.loss_classifier_pool import predict_loss_via_config_sync


logger = logging.getLogger("AETH")
_REASON = "loss_clf_veto"


def apply_loss_classifier_gate(
    metrics: dict[str, Any],
    exec_dir: TradeDirection,
    *,
    orch: Any | None = None,
    force: bool = False,
    symbol: str | None = None,
) -> bool:
    """Consulta loss-clf; True se vetou. Nao chama se ja houver gate_reason."""
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
        store = getattr(orch, "_loss_clf_vectors", None)
        if not isinstance(store, dict):
            store = {}
            orch._loss_clf_vectors = store
        store[str(symbol)] = list(vector)
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
    auto_flag = 1 if response["auto_learn_applied"] else 0
    if response["veto"] and response["veto_ready"]:
        metrics["execution_candidate_ready"] = False
        metrics["gate_reason"] = _REASON
        metrics["signal_skip_reason"] = _REASON
        metrics["signal_status"] = f"SKIP:{_REASON.upper()}"
        log_info_if_changed(
            orch,
            logger,
            "loss_clf_veto",
            f"{response['model_version']}:{response['p_loss']:.4f}:{auto_flag}",
            "LOSS_CLF || VETO auto_learn=%d ver=%s n=%d p_loss=%.4f reason=%s",
            auto_flag,
            response["model_version"],
            int(response["n_train"]),
            float(response["p_loss"]),
            _REASON,
        )
        return True
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
