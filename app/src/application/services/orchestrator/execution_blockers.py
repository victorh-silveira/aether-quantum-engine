"""Logs de bloqueio de execucao quando nenhuma ordem e enviada."""

from src.application.services.log_dedupe import clear_log_channel, log_info_if_changed


_BRIEF_METRIC_KEYS = (("s", "trade_score"), ("r", "raw_prob"), ("v", "val_accuracy"), ("b", "val_brier"))


def blocked_metrics_brief(metrics: dict) -> str:
    """Formata valores compactos das metricas DL para diagnostico de bloqueio."""
    parts: list[str] = []
    for tag, key in _BRIEF_METRIC_KEYS:
        value = metrics.get(key)
        if value is None:
            continue
        num = float(value)
        if key == "raw_prob":
            num = max(num, 1.0 - num)
        parts.append(f"{tag}{num:.2f}")
    if not parts:
        return ""
    return " " + " ".join(parts)


def _no_direction_reason(symbol: str, metrics: dict) -> str:
    """Formata motivo de bloqueio quando a decisao nao tem direcao definida."""
    gate = metrics.get("gate_reason")
    if gate == "data":
        return f"{symbol}:dados"
    raw = metrics.get("raw_prob")
    if gate == "direction_margin" and raw is not None:
        return f"{symbol}:sem_direcao:r{float(raw):.2f}"
    return f"{symbol}:sem_direcao"


def log_execution_blockers(executor, decisions: dict) -> None:
    """Registra motivo quando nenhuma ordem foi montada apesar de decisoes no ciclo."""
    cid = f"C{int(executor.orch._active_cycle_id):04d}"
    reasons: list[str] = []
    training: list[str] = []
    bankroll_snapshot = float(executor.orch.state.balance)
    for symbol in executor._trade_symbols():
        entry = decisions.get(symbol)
        if not entry:
            continue
        metrics = entry["metrics"]
        direction = entry["direction"]
        if metrics.get("gate_reason") == "training":
            training.append(symbol)
            continue
        if direction is None:
            reasons.append(_no_direction_reason(symbol, metrics))
            continue
        if not metrics.get("execute", True):
            block_reason = str(metrics.get("gate_reason") or metrics.get("llm_block_reason") or "execute_false")
            reasons.append(f"{symbol}:{block_reason}{blocked_metrics_brief(metrics)}")
            continue
        dl_cfg = executor.orch.config.get("deep_learning", {})
        stake = executor.orch.risk_manager.calculate_stake(
            bankroll_snapshot,
            symbol,
            conviction=float(metrics.get("conviction", 0.60)),
            silent=True,
            cycle_id=int(executor.orch._active_cycle_id),
            dl_metrics=metrics,
            order_direction=direction.name,
            max_val_brier=float(dl_cfg.get("max_val_brier_execute", 0.28)),
        )
        block = executor.orch.risk_manager.stake_block_reason(
            bankroll_snapshot,
            symbol,
            conviction=float(metrics.get("conviction", 0.60)),
            cycle_id=int(executor.orch._active_cycle_id),
            dl_metrics=metrics,
            order_direction=direction.name,
            max_val_brier=float(dl_cfg.get("max_val_brier_execute", 0.28)),
        )
        if stake <= 0:
            reasons.append(f"{symbol}:{block or 'stake_zero'}")
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
    if reasons:
        log_info_if_changed(
            executor.orch,
            executor.logger,
            "exec_none",
            " | ".join(reasons),
            "[%s] EXEC_NONE || %s",
            cid,
            " | ".join(reasons),
        )
