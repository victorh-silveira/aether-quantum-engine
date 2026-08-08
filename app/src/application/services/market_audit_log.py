"""Formatos unificados de auditoria de mercado para terminal e monitoramento."""

from __future__ import annotations

import logging
from typing import Any

from src.application.services.execution_scale_vision import format_scale_ind_token
from src.application.services.market_audit_log_helpers import (
    cluster_symbol_token,
    indicator_snapshot,
    metric_float,
    pop_contract_audit,
    resolve_cluster_timeframe,
    resolve_meta_payoff_zscore,
    resolve_predicted_edge,
    store_contract_audit,
)


__all__ = [
    "emit_audit_info",
    "format_cluster_audit_line",
    "format_execution_ticket_line",
    "format_indicators_audit_line",
    "format_settlement_audit_line",
    "pop_contract_audit",
    "resolve_cluster_timeframe",
    "resolve_meta_payoff_zscore",
    "resolve_predicted_edge",
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
        return f"[CLUSTER] {timeframe} || EMPTY"
    tokens = [
        cluster_symbol_token(str(symbol), entry if isinstance(entry, dict) else {})
        for symbol, entry in decisions.items()
    ]
    return f"[CLUSTER] {timeframe} || " + " || ".join(tokens)


def _indicator_float(snap: dict[str, Any], key: str, default: float = 0.0) -> float:
    """Extrai float seguro de um snapshot de indicadores."""
    val = snap.get(key)
    if val is None:
        return default
    try:
        return float(val)
    except (ValueError, TypeError):
        return default


def format_indicators_audit_line(cycle_id: int, symbol: str, metrics: dict[str, Any]) -> str:
    """Monta telemetria IND em multiplas linhas curtas."""
    _ = symbol
    snap = indicator_snapshot(metrics)
    rsi = _indicator_float(snap, "rsi")
    adx = _indicator_float(snap, "adx")
    hurst = _indicator_float(snap, "hurst")
    atr = _indicator_float(snap, "atr_norm")
    bbw = _indicator_float(snap, "bb_width")
    vol_r = _indicator_float(snap, "vol_ratio") or _indicator_float(snap, "vol_ratio_short_long")
    z_edge = metric_float(metrics, "edge_zscore", "meta_payoff_edge_zscore", default=0.0)
    acc = metric_float(metrics, "val_accuracy", default=0.0)
    margin = metric_float(metrics, "direction_margin", default=0.0)
    cal_edge = metric_float(metrics, "cal_side_edge", default=resolve_predicted_edge(metrics))
    neutral = str(metrics.get("calibration_mode") or metrics.get("gate_reason") or "na")
    if (
        str(metrics.get("calibration_mode") or "") == "neutral_clamp"
        or str(metrics.get("gate_reason") or "") == "neutral_clamp"
    ):
        neutral = "neutral_clamp"
    elif neutral not in {"neutral_clamp", "tcn_macro_override", "raw_extreme", "calibrated", "neutral_zone"}:
        neutral = "na"
    meta_veto = str(metrics.get("meta_veto_mode") or "none")
    scale_tok = format_scale_ind_token(metrics)
    prefix = f"[C{int(cycle_id):04d}] IND"
    return (
        f"{prefix} || RSI: {rsi:>7.4f} | ADX: {adx:>7.4f} | HURST: {hurst:>7.4f}\n"
        f"{prefix} || ATR: {atr:>8.4f} | BBW: {bbw:>8.4f} | VOL_R: {vol_r:>7.4f}\n"
        f"{prefix} || Z: {z_edge:>+6.2f} | ACC: {acc:>6.4f} | MARGIN: {margin:>5.3f} | "
        f"CAL_EDGE: {cal_edge:>+.3f}\n"
        f"{prefix} || NEUTRAL: {neutral} | META_VETO: {meta_veto} || {scale_tok}"
    )


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


def format_execution_ticket_line(
    cycle_id: int,
    *,
    direction: str,
    symbol: str,
    stake: float,
    mode_tag: str,
    pending: float,
    bankroll: float,
    contract_id: int,
    payout: float,
    linear: int = 0,
    cap: float = 0.0,
    recovery_infeasible: bool = False,
) -> str:
    """Monta boleta EXEC em multiplas linhas curtas."""
    infeas = " | RECOVERY_INFEASIBLE" if recovery_infeasible else ""
    prefix = f"[C{int(cycle_id):04d}] EXEC"
    return (
        f"{prefix} || {direction} [{symbol}] || STAKE: {float(stake):.2f} ({mode_tag}){infeas}\n"
        f"{prefix} || PEND: {float(pending):.2f} | LIN: {int(linear)} | CAP: {float(cap):.2f} | "
        f"BANCA: {float(bankroll):.2f}\n"
        f"{prefix} || CID: {int(contract_id)} | PAY: {float(payout):.2f}"
    )


def resolve_settlement_tag(*, profit: float, linear_before: int) -> str:
    """Resolve sufixo de liquidacao (RESET_LINEAR / COOLDOWN_Ln)."""
    if float(profit) >= 0.0:
        return "RESET_LINEAR" if int(linear_before) > 0 else "FLAT_KEEP"
    return f"COOLDOWN_L{max(1, int(linear_before) + 1)}"


def format_settlement_audit_line(
    cycle_id: int,
    outcome: str,
    profit: float,
    direction: str,
    symbol: str,
    edge: float,
    *,
    settlement_tag: str | None = None,
    pending: float | None = None,
    linear: int | None = None,
    mode_tag: str | None = None,
    recovery_infeasible: bool = False,
) -> str:
    """Monta linha padronizada de liquidacao RESOLVED."""
    _ = (direction, symbol, edge)
    tag = settlement_tag or resolve_settlement_tag(profit=profit, linear_before=0)
    extras = ""
    if pending is not None or linear is not None or mode_tag is not None:
        pend_s = f"{float(pending):.2f}" if pending is not None else "n/a"
        lin_s = str(int(linear)) if linear is not None else "n/a"
        mode_s = str(mode_tag) if mode_tag else "n/a"
        infeas = " | RECOVERY_INFEASIBLE" if recovery_infeasible else ""
        extras = f" | PEND: {pend_s} | LIN: {lin_s} | MODE: {mode_s}{infeas}"
    return f"[C{int(cycle_id):04d}] RESOLVED || STATUS: {str(outcome):<4} | P&L: {float(profit):>+7.2f} | {tag}{extras}"
