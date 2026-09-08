"""Token compacto [GATES] — LOSS_CLF HARD SKIP por P_LOSS."""

from __future__ import annotations

from typing import Any

from src.application.services.market_audit_log_helpers import metric_float


def format_gates_audit_line(metrics: dict[str, Any]) -> str:
    """Uma linha [GATES] com LOSS_CLF HARD|OK|off."""
    p_loss = metric_float(metrics, "loss_clf_p_loss", default=-1.0)
    hard = bool(metrics.get("loss_clf_hard")) or str(metrics.get("gate_reason") or "").strip().lower() == "loss_clf"
    n_train = int(metrics.get("loss_clf_n_train") or 0)
    auto_learn = 1 if metrics.get("loss_clf_auto_learn") else 0
    floor = metric_float(metrics, "loss_clf_hard_p_loss_floor", default=0.9)
    verdict = str(metrics.get("gate_verdict") or "").strip().upper()
    verdict_tok = f" | verdict={verdict}" if verdict else ""
    if hard and p_loss >= 0.0:
        loss_tok = f"HARD auto={auto_learn} p={p_loss:.5f} floor={floor:.2f} n={n_train}"
        skip = "loss_clf"
    elif p_loss >= 0.0:
        ready = 1 if metrics.get("loss_clf_veto_ready") else 0
        ver = str(metrics.get("loss_clf_model_version") or "-")
        loss_tok = f"OK auto={auto_learn} p={p_loss:.5f} ready={ready} n={n_train} ver={ver}"
        skip = "-"
    else:
        loss_tok = "OFF"
        skip = "-"
    return f"[GATES] || LOSS_CLF: {loss_tok} | skip={skip}{verdict_tok}"
