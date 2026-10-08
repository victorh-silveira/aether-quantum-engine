"""Logs de bloqueio de execucao quando nenhuma ordem e enviada."""

from src.application.services.force_trade_mode import force_trade_from_orch
from src.application.services.log_dedupe import clear_log_channel, log_info_if_changed
from src.application.services.market_audit_log import emit_audit_info, format_gates_audit_line


_TECHNICAL_REASONS = frozenset(
    {
        "training",
        "data",
        "deploy",
        "predict_error",
        "neg_edge",
        "call_top_rejection",
        "call_down_continuation",
        "put_bottom_rejection",
        "put_up_continuation",
        "min_edge",
        "chop_congestion",
        "trend_discord",
        "exhaustion",
        "exhaustion_call",
        "exhaustion_put",
        "directional_momentum_discord",
        "bearish_momentum_discord",
        "bullish_momentum_discord",
        "two_bar_counter_trend",
        "wick_rejection",
        "wick_rejection_call",
        "wick_rejection_put",
        "climactic_blowoff",
        "climactic_blowoff_call",
        "climactic_blowoff_put",
        "opposing_bearish_marubozu",
        "opposing_bullish_marubozu",
        "adverse_tick_flow_call",
        "adverse_tick_flow_put",
        "neg_edge_zscore_panic",
        "anti_loss_seed_discord",
        "anti_loss_ema_trend",
        "anti_loss_ema_slope",
        "live_exec_discord",
        "loss_clf_hard",
        "acc_floor",
        "exec_vs_candle",
        "scale_rescue",
        "scale_candle_conflict",
        "doji",
    }
)


def _candidate_block_reason(metrics: dict) -> str | None:
    """Extrai motivo tecnico ou SKIP de sinal para EXEC_EMPTY."""
    reason = metrics.get("gate_reason") or metrics.get("skip_reason")
    if isinstance(reason, str) and reason.strip():
        token = reason.strip()
        if token == "neg_edge":
            edge_raw = metrics.get("cal_side_edge", metrics.get("edge"))
            try:
                edge_f = float(edge_raw)
                floor_f = float(metrics.get("min_edge_floor", 0.0) or 0.0)
                if edge_f > 0.0 and floor_f > 0.0:
                    return "min_edge"
            except (TypeError, ValueError):
                pass
        if token in _TECHNICAL_REASONS:
            return token
    if metrics.get("deploy_ok") is False:
        return "deploy"
    if bool(metrics.get("execution_candidate_ready")):
        return "ready_not_selected"
    if metrics.get("signal_status") == "SIGNAL_SUSPENDED":
        return "SIGNAL_SUSPENDED"
    return None


def log_execution_blockers(executor, decisions: dict, *, pending: float = 0.0) -> None:
    """Registra treino em andamento e bloqueios tecnicos sem ordem."""
    if force_trade_from_orch(executor.orch):
        return
    cid = f"C{int(executor.orch._active_cycle_id):04d}"
    training: list[str] = []
    blocked: list[str] = []
    for symbol in executor._trade_symbols():
        entry = decisions.get(symbol)
        if not entry:
            blocked.append(f"{symbol}:no_decision")
            continue
        metrics = entry.get("metrics") or {}
        if metrics.get("gate_reason") == "training":
            training.append(symbol)
            continue
        reason = _candidate_block_reason(metrics)
        if reason:
            blocked.append(f"{symbol}:{reason}")
            if isinstance(metrics, dict):
                emit_audit_info(executor.logger, format_gates_audit_line(metrics))
        else:
            blocked.append(f"{symbol}:no_candidate")
            if isinstance(metrics, dict) and (
                metrics.get("tcn_direction") or metrics.get("gate_reason") or metrics.get("skip_reason")
            ):
                emit_audit_info(executor.logger, format_gates_audit_line(metrics))
    if training:
        log_info_if_changed(
            executor.orch,
            executor.logger,
            "dl_treino",
            " ".join(training),
            "[%s] DL_TREINO || %s | primeiro treino em andamento | trades suspensos",
            cid,
            " ".join(training),
        )
    elif clear_log_channel(executor.orch, "dl_treino"):
        executor.logger.info("[%s] DL_TREINO || concluido | todos os modelos treinados", cid)
    if blocked:
        payload = " ".join(blocked)
        log_info_if_changed(
            executor.orch,
            executor.logger,
            f"exec_empty_blocks:{cid}",
            payload,
            "[%s] EXEC_EMPTY || sem ordem | %s",
            cid,
            payload,
        )
    if float(pending) > 0.0:
        log_info_if_changed(
            executor.orch,
            executor.logger,
            f"exec_empty_recovery:{cid}",
            f"{float(pending):.2f}:{int(getattr(executor.orch.risk_manager, 'consecutive_losses_linear', 0))}",
            "[%s] EXEC_EMPTY || recovery sem ordem | pend=$%.2f | linear=%d",
            cid,
            float(pending),
            int(getattr(executor.orch.risk_manager, "consecutive_losses_linear", 0)),
        )
