"""Formatos unificados de auditoria de mercado para terminal e monitoramento."""

from __future__ import annotations

import logging
from typing import Any

from src.application.services.market_audit_cycle import (
    format_execution_ticket_line,
    format_gates_audit_line,
    format_indicators_audit_line,
    format_kelly_audit_line,
    format_settlement_audit_line,
)
from src.application.services.market_audit_log_helpers import (
    cluster_symbol_token,
    pop_contract_audit,
    resolve_cluster_timeframe,
    resolve_edge_breakeven_p,
    resolve_meta_payoff_zscore,
    resolve_predicted_edge,
    resolve_raw_predicted_edge,
    store_contract_audit,
)


__all__ = [
    "emit_audit_info",
    "format_cluster_audit_line",
    "format_execution_ticket_line",
    "format_gates_audit_line",
    "format_indicators_audit_line",
    "format_kelly_audit_line",
    "format_settlement_audit_line",
    "pop_contract_audit",
    "resolve_cluster_timeframe",
    "resolve_edge_breakeven_p",
    "resolve_meta_payoff_zscore",
    "resolve_predicted_edge",
    "resolve_raw_predicted_edge",
    "resolve_settlement_tag",
    "resolve_stake_audit_context",
    "resolve_stake_mode_tag",
    "store_contract_audit",
]


def emit_audit_info(logger: logging.Logger, message: str) -> None:
    """Emite cada linha de auditoria como INFO separado (prefixo [cN|SYM] por linha)."""
    for part in str(message or "").splitlines():
        line = part.strip()
        if line:
            logger.info("%s", line)


def format_cluster_audit_line(
    decisions: dict[str, Any],
    *,
    timeframe: str = "M5",
) -> str:
    """Monta linha compacta de decisao do cluster por simbolo."""
    if not isinstance(decisions, dict) or not decisions:
        return f"[CLUSTER] || {timeframe} || EMPTY"
    tokens = [
        cluster_symbol_token(str(symbol), entry if isinstance(entry, dict) else {})
        for symbol, entry in decisions.items()
    ]
    return f"[CLUSTER] || {timeframe} || " + " || ".join(tokens)


def resolve_stake_mode_tag(
    mode_tag: str,
    linear_losses: int,
    *,
    stake_regime: str | None = None,
) -> str:
    """Compacta modo de sizing em rotulo curto (EXPLORE|RECOVER + KELLY/DAL_Ln)."""
    tag = str(mode_tag or "KELLY").upper()
    if tag.startswith("EXPLORE_") or tag.startswith("RECOVER_"):
        return tag
    compact = f"DAL_L{max(0, int(linear_losses))}" if ("ALEMBERT" in tag or tag.startswith("DAL")) else "KELLY"
    regime = str(stake_regime or "EXPLORE").upper()
    if regime not in ("EXPLORE", "RECOVER"):
        regime = "EXPLORE"
    return f"{regime}_{compact}"


def resolve_stake_audit_context(rm: Any, *, balance_fallback: float | None = None) -> dict[str, Any]:
    """Le modo, pendencia, banca e telemetria financeira do ultimo sizing."""
    audit = getattr(rm, "_last_stake_audit", None)
    mode_tag = "EXPLORE_KELLY"
    pending = float(rm.pending_loss_total()) if callable(getattr(rm, "pending_loss_total", None)) else 0.0
    bankroll = float(getattr(rm, "bankroll", getattr(rm, "initial_bankroll", 0.0)) or 0.0)
    linear = int(getattr(rm, "consecutive_losses_linear", 0) or 0)
    cap = 0.0
    recovery_infeasible = False
    if isinstance(audit, dict):
        return {
            "mode_tag": str(audit.get("mode_tag") or mode_tag),
            "pending": float(audit.get("pending", pending)),
            "bankroll": float(audit.get("bankroll", bankroll)),
            "linear": int(audit.get("linear_losses", linear)),
            "cap": float(audit.get("cap", cap)),
            "recovery_infeasible": bool(audit.get("recovery_infeasible", False)),
        }
    if isinstance(balance_fallback, (int, float)):
        bankroll = float(balance_fallback)
    return {
        "mode_tag": mode_tag,
        "pending": pending,
        "bankroll": bankroll,
        "linear": linear,
        "cap": cap,
        "recovery_infeasible": recovery_infeasible,
    }


def resolve_settlement_tag(*, profit: float, linear_before: int) -> str:
    """Resolve sufixo de liquidacao (RESET_LINEAR / COOLDOWN_Ln)."""
    if float(profit) >= 0.0:
        return "RESET_LINEAR" if int(linear_before) > 0 else "FLAT_KEEP"
    return f"COOLDOWN_L{max(1, int(linear_before) + 1)}"
