"""Logs de bloqueio de execucao quando nenhuma ordem e enviada."""


def log_execution_blockers(executor, decisions: dict) -> None:
    """Registra motivo quando nenhuma ordem foi montada apesar de decisoes no ciclo."""
    cid = f"C{int(executor.orch._active_cycle_id):04d}"
    reasons: list[str] = []
    bankroll_snapshot = float(executor.orch.state.balance)
    for symbol in executor._trade_symbols():
        entry = decisions.get(symbol)
        if not entry:
            continue
        metrics = entry["metrics"]
        direction = entry["direction"]
        if direction is None:
            gate = metrics.get("gate_reason")
            if gate == "data":
                reasons.append(f"{symbol}:dados")
            elif gate == "direction_margin":
                raw = metrics.get("raw_prob")
                if raw is not None:
                    reasons.append(f"{symbol}:sem_direcao:r{float(raw):.2f}")
                else:
                    reasons.append(f"{symbol}:sem_direcao")
            else:
                reasons.append(f"{symbol}:sem_direcao")
            continue
        if not metrics.get("execute", True):
            block_reason = str(metrics.get("gate_reason") or metrics.get("llm_block_reason") or "execute_false")
            reasons.append(f"{symbol}:{block_reason}")
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
    if reasons:
        executor.logger.info("[%s] EXEC_NONE || %s", cid, " | ".join(reasons))
