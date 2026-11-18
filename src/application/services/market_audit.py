"""Formatadores de auditoria de movimento, contrato broker e tokens [GATES]."""

from __future__ import annotations

import math
from typing import Any

from src.application.services.market_audit_log_helpers import metric_float
from src.application.services.market_forming_candle import LiveCandleSnapshot


def format_live_candle_line(snapshot: LiveCandleSnapshot, *, boundary_seconds: int = 300) -> str:
    """Monta linha compacta de auditoria da vela em formacao ao vivo."""
    delta = snapshot.spot_price - snapshot.open_price
    tf = f"M{max(1, int(boundary_seconds) // 60)}" if int(boundary_seconds) % 60 == 0 else f"{boundary_seconds}s"
    return (
        f"[LIVE_CANDLE] || {tf} || {snapshot.symbol}: {snapshot.current_side} ({delta:+.5f}) | "
        f"spot={snapshot.spot_price:.5f} o={snapshot.open_price:.5f} h={snapshot.high_price:.5f} l={snapshot.low_price:.5f} | "
        f"{snapshot.elapsed_seconds}s/{boundary_seconds}s ({snapshot.progress_pct * 100.0:.0f}%) | "
        f"wicks: U={snapshot.upper_wick_ratio * 100.0:.0f}% L={snapshot.lower_wick_ratio * 100.0:.0f}%"
    )


def format_next_movement_line(symbol: str, metrics: dict[str, Any]) -> str:
    """Monta linha de projecao quantitativa do proximo movimento do preco."""
    delta_raw = metrics.get("predicted_movement_delta")
    if delta_raw is None:
        return f"[NEXT_MOVE] || {str(symbol).upper()} || N/A"
    try:
        delta = float(delta_raw)
    except (TypeError, ValueError):
        return f"[NEXT_MOVE] || {str(symbol).upper()} || N/A"
    if not math.isfinite(delta):
        return f"[NEXT_MOVE] || {str(symbol).upper()} || N/A"
    delta_pct = delta * 100.0
    side = str(metrics.get("predicted_movement_side") or ("CALL" if delta > 0 else "PUT"))
    confluence = "ALIGNED" if bool(metrics.get("movement_confluence", True)) else "DIVERGENT"
    atr_r = metrics.get("movement_atr_ratio")
    atr_tok = f"{float(atr_r):.2f}x" if atr_r is not None and math.isfinite(float(atr_r)) else "n/a"
    return (
        f"[NEXT_MOVE] || {str(symbol).upper()} || EXP_DELTA: {delta_pct:+.4f}% | "
        f"SIDE: {side} | CONFLUENCE: {confluence} | ATR_RATIO: {atr_tok}"
    )


def broker_price_audit(payload: dict, direction: str | None) -> str | None:
    """Resume deslocamento real sem inferir outcome a partir da vela M15."""
    if str(payload.get("audit_source") or "broker") != "broker":
        return None
    raw_entry = (
        payload.get("entry_spot")
        or payload.get("entry_tick")
        or payload.get("barrier")
        or payload.get("entry_tick_display_value")
    )
    raw_exit = (
        payload.get("exit_spot")
        or payload.get("exit_tick")
        or payload.get("sell_spot")
        or payload.get("current_spot")
        or payload.get("exit_tick_display_value")
    )
    try:
        entry = float(raw_entry)
        exit_ = float(raw_exit)
    except (TypeError, ValueError):
        return None
    if not all(math.isfinite(value) and value > 0.0 for value in (entry, exit_)):
        return None
    side = str(direction or "").upper()
    if side not in {"CALL", "PUT"}:
        return None
    delta = exit_ - entry
    observed = "CALL" if delta > 0.0 else "PUT" if delta < 0.0 else "FLAT"
    phase = ""
    raw_time = payload.get("entry_spot_time", payload.get("entry_tick_time"))
    try:
        entry_time = int(raw_time)
    except (TypeError, ValueError):
        entry_time = 0
    if entry_time > 0:
        phase = f" | FASE_M15: {entry_time % 900}s"
    return f"SPOT: {entry:.2f}->{exit_:.2f} ({delta:+.2f}) | REAL: {observed}{phase}"


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
        loss_tok = "OBSERVE" if metrics.get("loss_clf_observe_only") else "OFF"
        skip = "-"
    reason = str(metrics.get("gate_reason") or "").strip()
    if reason == "neg_edge":
        edge_raw = metrics.get("cal_side_edge", metrics.get("edge"))
        try:
            edge_f = float(edge_raw)
            floor_f = float(metrics.get("min_edge_floor", 0.0) or 0.0)
            if edge_f > 0.0 and floor_f > 0.0:
                skip = f"min_edge({edge_f:+.3f}<{floor_f:.3f})"
            elif edge_f <= 0.0:
                skip = f"neg_edge({edge_f:+.3f})"
            else:
                skip = reason
        except (TypeError, ValueError):
            skip = reason
    elif reason:
        skip = reason
    trigger = str(metrics.get("market_trigger_status") or "")
    trigger_tok = ""
    if trigger:
        candidate = str(metrics.get("market_trigger_candidate") or "-")
        trigger_tok = f" | trigger={trigger} candidate={candidate}"
    return f"[GATES] || LOSS_CLF: {loss_tok} | skip={skip}{verdict_tok}{trigger_tok}"
