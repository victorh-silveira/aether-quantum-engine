"""Entradas de decisao, cooldown e reexportes do bridge Deep Learning."""

from src.application.services.deep_learning.dl_outcomes import is_symbol_session_paused
from src.application.services.deep_learning.dl_params import optional_float, parse_dl_params
from src.application.services.deep_learning.dl_post_loss import post_loss_block_reason


def build_decision_entry(
    direction,
    prob: float,
    *,
    execute: bool,
    val_accuracy: float,
    edge: float,
    train_loss: float | None,
    trade_score: float | None = None,
    raw_prob: float | None = None,
    val_brier: float | None = None,
    val_ece: float | None = None,
) -> dict:
    """Monta entrada de decisao com metricas de conviccao e gating."""
    score = float(trade_score if trade_score is not None else prob)
    note = f"DL score={score:.2f} edge={edge:.2f} val_acc={val_accuracy:.2f}"
    if val_brier is not None:
        note = f"{note} brier={val_brier:.2f}"
    if train_loss is not None:
        note = f"{note} loss={train_loss:.4f}"
    metrics = {
        "conviction": score,
        "trade_score": score,
        "execute": execute,
        "duration": 1,
        "llm_note": note,
        "val_accuracy": val_accuracy,
        "edge": edge,
    }
    if raw_prob is not None:
        metrics["raw_prob"] = float(raw_prob)
        metrics["raw_conviction"] = max(float(raw_prob), 1.0 - float(raw_prob))
    if val_brier is not None:
        metrics["val_brier"] = float(val_brier)
    if val_ece is not None:
        metrics["val_ece"] = float(val_ece)
    return {"direction": direction, "metrics": metrics}


def recovery_gating_active(orch) -> bool:
    """Indica se ha perda pendente no risk manager para usar gating de recuperacao."""
    pending = getattr(getattr(orch, "risk_manager", None), "pending_loss", None)
    if not pending:
        return False
    return sum(float(v) for v in pending.values()) > 0.0


def pending_loss_total(orch) -> float:
    """Soma perdas pendentes registradas no risk manager."""
    pending = getattr(getattr(orch, "risk_manager", None), "pending_loss", None)
    if not pending:
        return 0.0
    return sum(float(v) for v in pending.values())


def apply_symbol_loss_cooldown(orch, symbol: str, entry: dict) -> dict:
    """Bloqueia execucao por cooldown, sessao ou veto pos-loss."""
    rm = getattr(orch, "risk_manager", None)
    dl_cfg = orch.config.get("deep_learning", {}) if hasattr(orch, "config") else {}
    flip_min = float(dl_cfg.get("post_loss_flip_raw_min", 0.58))
    if entry["metrics"].get("execute"):
        repeat = post_loss_block_reason(
            orch,
            symbol,
            entry.get("direction"),
            raw_prob=entry["metrics"].get("raw_prob"),
            flip_raw_min=flip_min,
        )
        if repeat:
            entry["metrics"]["execute"] = False
            entry["metrics"]["gate_reason"] = repeat
    if entry["metrics"].get("execute") and rm is not None and rm.is_symbol_on_loss_cooldown(symbol):
        entry["metrics"]["execute"] = False
        entry["metrics"]["gate_reason"] = "cooldown"
    if entry["metrics"].get("execute") and is_symbol_session_paused(orch, symbol):
        entry["metrics"]["execute"] = False
        entry["metrics"]["gate_reason"] = "session_pause"
    return entry


__all__ = [
    "apply_symbol_loss_cooldown",
    "build_decision_entry",
    "optional_float",
    "parse_dl_params",
    "pending_loss_total",
    "recovery_gating_active",
]
