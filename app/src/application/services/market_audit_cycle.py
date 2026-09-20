"""Formatadores do pacote de auditoria por ciclo."""

from __future__ import annotations

from typing import Any

from src.application.services.execution_scale_vision import format_scale_ind_token
from src.application.services.market_audit_gate_tokens import format_gates_audit_line
from src.application.services.market_audit_log_helpers import (
    indicator_snapshot,
    metric_float,
    resolve_predicted_edge,
)


__all__ = [
    "format_execution_ticket_line",
    "format_gates_audit_line",
    "format_indicators_audit_line",
    "format_kelly_audit_line",
    "format_settlement_audit_line",
]


def _f(metrics: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    """Le float de metrics com fallback."""
    return metric_float(metrics, *keys, default=default)


def _snap_f(snap: dict[str, Any], key: str, default: float = 0.0) -> float:
    """Le float do snapshot de indicadores."""
    raw = snap.get(key)
    if raw is None:
        return default
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def format_indicators_audit_line(cycle_id: int, symbol: str, metrics: dict[str, Any]) -> str:
    """Telemetria IND em uma linha com chave [IND]."""
    _ = (cycle_id, symbol)
    snap = indicator_snapshot(metrics)
    rsi = _snap_f(snap, "rsi")
    adx = _snap_f(snap, "adx")
    hurst = _snap_f(snap, "hurst")
    atr = _snap_f(snap, "atr_norm")
    bbw = _snap_f(snap, "bb_width")
    vol_r = _snap_f(snap, "vol_ratio") or _snap_f(snap, "vol_ratio_short_long")
    z_edge = _f(metrics, "edge_zscore", "meta_payoff_edge_zscore", default=0.0)
    acc = _f(metrics, "val_accuracy", default=0.0)
    margin = _f(metrics, "direction_margin", default=0.0)
    cal_edge = _f(metrics, "cal_side_edge", default=resolve_predicted_edge(metrics))
    neutral = str(metrics.get("calibration_mode") or metrics.get("gate_reason") or "na")
    if (
        str(metrics.get("calibration_mode") or "") == "neutral_clamp"
        or str(metrics.get("gate_reason") or "") == "neutral_clamp"
    ):
        neutral = "neutral_clamp"
    elif neutral not in {"neutral_clamp", "tcn_macro_override", "raw_extreme", "calibrated", "neutral_zone"}:
        neutral = "na"
    meta_veto = str(metrics.get("meta_veto_mode") or "none")
    meta_applied = bool(metrics.get("meta_classifier_applied"))
    meta_edge = metrics.get("predicted_payoff_edge")
    if meta_edge is None:
        meta_edge = metrics.get("meta_payoff_edge")
    if meta_applied and meta_edge is not None:
        try:
            edge_f = float(meta_edge)
            sat_tok = " sat=1" if abs(edge_f - 0.85) < 1e-6 else ""
            z_raw = metrics.get("meta_payoff_edge_zscore", metrics.get("edge_zscore"))
            z_tok = ""
            if z_raw is not None:
                try:
                    z_tok = f" z={float(z_raw):+.2f}"
                except (TypeError, ValueError):
                    z_tok = ""
            meta_tok = f"META: applied=1 edge={edge_f:+.3f}{sat_tok}{z_tok}"
        except (TypeError, ValueError):
            meta_tok = "META: applied=1 edge=n/a"
    elif meta_applied:
        meta_tok = "META: applied=1 edge=n/a"
    else:
        meta_tok = "META: applied=0"
    scale_tok = format_scale_ind_token(metrics)
    return (
        f"[IND] || RSI: {rsi:>7.4f} | ADX: {adx:>7.4f} | HURST: {hurst:>7.4f}\n"
        f"[IND] || ATR: {atr:>8.4f} | BBW: {bbw:>8.4f} | VOL_R: {vol_r:>7.4f}\n"
        f"[IND] || Z: {z_edge:>+6.2f} | ACC: {acc:>6.4f} | MARGIN: {margin:>5.3f} | "
        f"CAL_EDGE: {cal_edge:>+.3f}\n"
        f"[IND] || NEUTRAL: {neutral} | META_VETO: {meta_veto} | {meta_tok} || {scale_tok}"
    )


def format_kelly_audit_line(
    metrics: dict[str, Any],
    *,
    stake: float,
    mode_tag: str,
    audit: dict[str, Any] | None = None,
) -> str:
    """Linha [KELLY] com p/live/mode e stake final."""
    p = _f(metrics, "conviction", "trade_score", "kelly_p", default=0.0)
    live_wr = metrics.get("live_wr")
    live_n = int(metrics.get("live_n", 0) or 0)
    f_star = _f(metrics, "kelly_fraction", "f_star", default=0.0)
    mode = str((audit or {}).get("mode_tag") or mode_tag or "explore").lower()
    if not (mode.startswith("explore") or mode.startswith("recover")):
        mode = str(metrics.get("stake_regime") or "explore").lower()
    wr_s = f"{float(live_wr):.4f}" if live_wr is not None else "n/a"
    infeas = " | RECOVERY_INFEASIBLE" if (audit or {}).get("recovery_infeasible") else ""
    return (
        f"[KELLY] || p={p:.4f} | live_wr={wr_s} | live_n={live_n} | f*={f_star:.6f} | "
        f"mode={mode} | stake={float(stake):.2f} ({mode_tag}){infeas}"
    )


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
    """Boleta [EXEC] em uma linha."""
    _ = cycle_id
    infeas = " | RECOVERY_INFEASIBLE" if recovery_infeasible else ""
    return (
        f"[EXEC] || {direction} [{symbol}] || STAKE: {float(stake):.2f} ({mode_tag}){infeas} | "
        f"PEND: {float(pending):.2f} | LIN: {int(linear)} | CAP: {float(cap):.2f} | "
        f"BANCA: {float(bankroll):.2f} | CID: {int(contract_id)} | PAY: {float(payout):.2f}"
    )


def _settlement_tag(*, profit: float, linear_before: int) -> str:
    """Resolve sufixo de liquidacao (RESET_LINEAR / COOLDOWN_Ln)."""
    if float(profit) >= 0.0:
        return "RESET_LINEAR" if int(linear_before) > 0 else "FLAT_KEEP"
    return f"COOLDOWN_L{max(1, int(linear_before) + 1)}"


def format_decision_origin_line(symbol: str, direction: Any, metrics: dict[str, Any]) -> str:
    """Linha [DECISION] com direcao e origem (TCN_DIRECT / FLIP_LOSS_CLF / FLIP_ANTI_TREND_LOCK)."""
    dir_name = direction.name if hasattr(direction, "name") else str(direction).upper()
    origin = str(metrics.get("direction_origin") or "TCN_DIRECT")
    tcn_dir = str(metrics.get("tcn_direction") or dir_name).upper()
    prob = _f(metrics, "conviction", "calibrated_prob", "tcn_probability", default=0.5)
    edge = _f(metrics, "cal_side_edge", "edge", default=0.0)
    trend = str(metrics.get("trend_direction") or "-").upper()
    if origin == "FLIP_ANTI_TREND_LOCK":
        from_dir = str(metrics.get("anti_trend_lock_from") or tcn_dir)
        detail = f"FLIP anti_trend_lock ({from_dir}->{dir_name}) | trend={trend} | p_orig={prob:.3f}"
    elif origin == "FLIP_LOSS_CLF":
        pe = _f(metrics, "loss_clf_p_eff", "loss_clf_p_loss", default=0.58)
        detail = f"FLIP loss_clf ({tcn_dir}->{dir_name}) | pe={pe:.3f} | trend={trend}"
    else:
        detail = f"TCN_DIRECT | p={prob:.3f} | edge={edge:+.3f} | trend={trend}"
    return f"[DECISION] || {dir_name} [{symbol}] || ORIGEM: {detail}"


def format_market_summary_line(symbol: str, metrics: dict[str, Any]) -> str:
    """Linha [MARKET] com indicadores chave e candle M5 em formato conciso."""
    snap = indicator_snapshot(metrics)
    rsi = _snap_f(snap, "rsi", default=_f(metrics, "rsi", default=0.5))
    adx = _snap_f(snap, "adx", default=_f(metrics, "adx", default=0.0))
    atr = _snap_f(
        snap,
        "atr_norm",
        default=_snap_f(snap, "atr_raw", default=_snap_f(snap, "atr", default=_f(metrics, "atr_norm", default=0.0))),
    )
    bbw = _snap_f(snap, "bb_width", default=_f(metrics, "bb_width", default=0.0))
    trend = str(metrics.get("trend_direction") or "-").upper()
    candle = str(metrics.get("closed_micro_candle_dir") or "-").upper()
    regime = str(metrics.get("scale_micro_regime") or "-").lower()
    return (
        f"[MARKET] || {symbol} || RSI: {rsi:.3f} | ADX: {adx:.3f} | ATR: {atr:.2f} | "
        f"BB_W: {bbw:.4f} | CANDLE: {candle} | TREND: {trend} | REGIME: {regime}"
    )


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
    learn_detail: str | None = None,
    session_pnl: float | None = None,
    target_pnl: float | None = None,
) -> str:
    """Linha [RESOLVED] com opcional LEARN e progresso da sessao embutidos."""
    _ = (direction, symbol, edge, cycle_id)
    tag = settlement_tag or _settlement_tag(profit=profit, linear_before=0)
    pend_s = f"{float(pending):.2f}" if pending is not None else "n/a"
    lin_s = str(int(linear)) if linear is not None else "n/a"
    mode_s = str(mode_tag) if mode_tag else "n/a"
    infeas = " | RECOVERY_INFEASIBLE" if recovery_infeasible else ""
    learn = f" | LEARN: {learn_detail}" if learn_detail else ""
    sess_tok = ""
    if session_pnl is not None:
        sess_tok = f" | SESSAO: {float(session_pnl):>+7.2f}"
        if target_pnl is not None and float(target_pnl) > 0.0:
            pct = (float(session_pnl) / float(target_pnl)) * 100.0
            sess_tok += f" | ALVO: ${float(target_pnl):.2f} ({pct:.1f}%)"
    return (
        f"[RESOLVED] || STATUS: {str(outcome):<4} | P&L: {float(profit):>+7.2f}{sess_tok} | {tag} | "
        f"PEND: {pend_s} | LIN: {lin_s} | MODE: {mode_s}{infeas}{learn}"
    )
