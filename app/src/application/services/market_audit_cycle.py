"""Formatadores do pacote de 6 linhas de auditoria por ciclo."""

from __future__ import annotations

from typing import Any

from src.application.services.execution_scale_vision import format_scale_ind_token
from src.application.services.market_audit_log_helpers import (
    indicator_snapshot,
    metric_float,
    resolve_predicted_edge,
)


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


def format_gates_audit_line(metrics: dict[str, Any]) -> str:
    """Compacta LOSS_CLF + CHOP + NEG_EDGE em uma linha [GATES]."""
    p_loss = _f(metrics, "loss_clf_p_loss", default=-1.0)
    soft = bool(metrics.get("loss_clf_soft"))
    flipped = bool(metrics.get("loss_clf_flip"))
    blocked = str(metrics.get("loss_clf_flip_blocked") or "").strip()
    auto_learn = 1 if metrics.get("loss_clf_auto_learn") else 0
    n_train = int(metrics.get("loss_clf_n_train") or 0)
    ver = str(metrics.get("loss_clf_model_version") or "-")
    veto_ready = 1 if metrics.get("loss_clf_veto_ready") else 0
    soft_mult = _f(metrics, "loss_clf_soft_kelly_mult", default=0.0)
    if flipped:
        loss_tok = f"FLIP auto={auto_learn} p={p_loss:.5f} n={n_train}"
    elif blocked:
        loss_tok = f"FLIP_BLOCK:{blocked} auto={auto_learn} p={p_loss:.5f} soft={soft_mult:.2f}"
    elif soft and p_loss >= 0.0:
        loss_tok = f"SOFT auto={auto_learn} p={p_loss:.5f} mult={soft_mult:.2f} n={n_train}"
    elif p_loss >= 0.0:
        loss_tok = f"OK auto={auto_learn} p={p_loss:.5f} ready={veto_ready} n={n_train} ver={ver}"
    else:
        loss_tok = "OFF"
    if bool(metrics.get("regime_chop_soft")):
        adx = _f(metrics, "regime_chop_adx", default=0.0)
        hurst = _f(metrics, "regime_chop_hurst", default=0.0)
        scale_chop = "yes" if bool(metrics.get("regime_chop_via_scale")) else "band"
        chop_tok = f"CHOP adx={adx:.4f} hurst={hurst:.4f} scale={scale_chop}"
    else:
        chop_tok = "CHOP off"
    edge = _f(metrics, "cal_side_edge", default=resolve_predicted_edge(metrics))
    floor = _f(metrics, "cal_side_edge_floor", default=0.0)
    waived = str(metrics.get("signal_skip_waived") or "")
    side = str(metrics.get("exec_direction") or metrics.get("resolved_direction") or "-")
    if waived == "neg_edge_soft" or floor > 1e-12:
        neg_tok = f"NEG_EDGE side={side} edge={edge:+.4f} floor={floor:.4f}"
    else:
        neg_tok = "NEG_EDGE off"
    skip = waived if waived else "-"
    return f"[GATES] || LOSS_CLF: {loss_tok} | {chop_tok} | {neg_tok} | skip={skip}"


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
    scale_tok = format_scale_ind_token(metrics)
    return (
        f"[IND] || RSI: {rsi:>7.4f} | ADX: {adx:>7.4f} | HURST: {hurst:>7.4f}\n"
        f"[IND] || ATR: {atr:>8.4f} | BBW: {bbw:>8.4f} | VOL_R: {vol_r:>7.4f}\n"
        f"[IND] || Z: {z_edge:>+6.2f} | ACC: {acc:>6.4f} | MARGIN: {margin:>5.3f} | "
        f"CAL_EDGE: {cal_edge:>+.3f}\n"
        f"[IND] || NEUTRAL: {neutral} | META_VETO: {meta_veto} || {scale_tok}"
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
) -> str:
    """Linha [RESOLVED] com opcional LEARN embutido."""
    _ = (direction, symbol, edge, cycle_id)
    tag = settlement_tag or _settlement_tag(profit=profit, linear_before=0)
    pend_s = f"{float(pending):.2f}" if pending is not None else "n/a"
    lin_s = str(int(linear)) if linear is not None else "n/a"
    mode_s = str(mode_tag) if mode_tag else "n/a"
    infeas = " | RECOVERY_INFEASIBLE" if recovery_infeasible else ""
    learn = f" | LEARN: {learn_detail}" if learn_detail else ""
    return (
        f"[RESOLVED] || STATUS: {str(outcome):<4} | P&L: {float(profit):>+7.2f} | {tag} | "
        f"PEND: {pend_s} | LIN: {lin_s} | MODE: {mode_s}{infeas}{learn}"
    )
