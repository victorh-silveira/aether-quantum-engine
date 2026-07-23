"""Entradas de decisao, cooldown e reexportes do bridge Deep Learning."""

import numpy as np

from src.application.services.deep_learning.dl_params import optional_float, parse_dl_params


def resample_m1_to_m15(
    prices: np.ndarray,
    open_: np.ndarray | None,
    high: np.ndarray | None,
    low: np.ndarray | None,
) -> tuple[np.ndarray, np.ndarray | None, np.ndarray | None, np.ndarray | None]:
    """Resamples M1 (60s) arrays to non-overlapping M15 (900s) arrays ending at current candle."""
    n = len(prices)
    if n < 15:
        return prices, open_, high, low

    indices = list(range(n - 1, 13, -15))[::-1]

    resampled_close = prices[indices]
    resampled_open = open_[np.array(indices) - 14] if open_ is not None else None

    if high is not None:
        resampled_high = np.array([np.max(high[idx - 14 : idx + 1]) for idx in indices], dtype=np.float64)
    else:
        resampled_high = None

    if low is not None:
        resampled_low = np.array([np.min(low[idx - 14 : idx + 1]) for idx in indices], dtype=np.float64)
    else:
        resampled_low = None

    return resampled_close, resampled_open, resampled_high, resampled_low


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
    contract_duration: int | None = None,
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
    if contract_duration is not None:
        metrics["duration"] = int(contract_duration)
    return {"direction": direction, "metrics": metrics}


def recovery_gating_active(orch) -> bool:
    """Indica se ha perda pendente ou linear ativo para gating de recuperacao."""
    rm = getattr(orch, "risk_manager", None)
    if rm is None:
        return False
    raw_linear = getattr(rm, "consecutive_losses_linear", 0)
    linear = int(raw_linear) if isinstance(raw_linear, (int, float)) and not isinstance(raw_linear, bool) else 0
    if linear > 0:
        return True
    pending = getattr(rm, "pending_loss", None)
    if not isinstance(pending, dict) or not pending:
        return False
    try:
        return sum(float(v) for v in pending.values()) > 0.0
    except (TypeError, ValueError):
        return False


def pending_loss_total(orch) -> float:
    """Soma perdas pendentes registradas no risk manager."""
    pending = getattr(getattr(orch, "risk_manager", None), "pending_loss", None)
    if not pending:
        return 0.0
    return sum(float(v) for v in pending.values())


def apply_symbol_loss_cooldown(orch, symbol: str, entry: dict) -> dict:
    """Retorna a entrada sem aplicar cooldown ou pausa (desativado)."""
    _ = orch
    _ = symbol
    return entry


__all__ = [
    "apply_symbol_loss_cooldown",
    "build_decision_entry",
    "optional_float",
    "parse_dl_params",
    "pending_loss_total",
    "recovery_gating_active",
    "resample_m1_to_m15",
]
