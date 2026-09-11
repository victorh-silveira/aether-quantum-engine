"""Token compacto [GATES] — LOSS_CLF FLIP por p_eff."""

from __future__ import annotations

from typing import Any

from src.application.services.market_audit_log_helpers import metric_float


def format_gates_audit_line(metrics: dict[str, Any]) -> str:
    """Uma linha [GATES] com LOSS_CLF FLIP|OK|off (p= cru, pe= efetivo se diferir)."""
    p_loss = metric_float(metrics, "loss_clf_p_loss", default=-1.0)
    p_eff = metric_float(metrics, "loss_clf_p_eff", default=-1.0)
    flipped = bool(metrics.get("loss_clf_flip"))
    n_train = int(metrics.get("loss_clf_n_train") or 0)
    auto_learn = 1 if metrics.get("loss_clf_auto_learn") else 0
    floor = metric_float(metrics, "loss_clf_flip_floor", default=-1.0)
    if floor < 0.0:
        floor = metric_float(metrics, "loss_clf_hard_p_loss_floor", default=0.58)
    verdict = str(metrics.get("gate_verdict") or "").strip().upper()
    verdict_tok = f" | verdict={verdict}" if verdict else ""
    pe_tok = ""
    if p_loss >= 0.0 and p_eff >= 0.0 and abs(p_eff - p_loss) > 1e-9:
        pe_tok = f" pe={p_eff:.5f}"
    if flipped and p_loss >= 0.0:
        loss_tok = f"FLIP auto={auto_learn} p={p_loss:.5f}{pe_tok} floor={floor:.2f} n={n_train}"
        skip = "-"
    elif p_loss >= 0.0:
        ready = 1 if metrics.get("loss_clf_veto_ready") else 0
        ver = str(metrics.get("loss_clf_model_version") or "-")
        blocked = str(metrics.get("loss_clf_flip_blocked") or "").strip()
        blocked_tok = f" blocked={blocked}" if blocked else ""
        boot_tok = ""
        if blocked == "bootstrap":
            buffer_n = int(metrics.get("loss_clf_buffer_n") or 0)
            exit_n = max(1, int(metrics.get("loss_clf_bootstrap_exit_n") or 4))
            boot_tok = f" boot={buffer_n}/{exit_n}"
            ready_note = " ready_seed=1" if ready else " ready_seed=0"
            loss_tok = (
                f"OK auto={auto_learn} p={p_loss:.5f}{pe_tok}{blocked_tok}{boot_tok}{ready_note} n={n_train} ver={ver}"
            )
        else:
            loss_tok = f"OK auto={auto_learn} p={p_loss:.5f}{pe_tok}{blocked_tok} ready={ready} n={n_train} ver={ver}"
        skip = "-"
    else:
        loss_tok = "OFF"
        skip = "-"
    reason = str(metrics.get("gate_reason") or "").strip()
    if reason:
        skip = reason
    return f"[GATES] || LOSS_CLF: {loss_tok} | skip={skip}{verdict_tok}"
