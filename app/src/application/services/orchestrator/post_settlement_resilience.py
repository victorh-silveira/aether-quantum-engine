"""Recuperacao transparente e timeouts resilientes do ciclo pos-liquidacao."""

from __future__ import annotations

from typing import Any


_POST_SETTLEMENT_SAFE_CYCLE_TIMEOUT_SECONDS = 600.0
_PATIENT_QUALITY_GATE_REGIMES = frozenset({"meta_zscore_reject", "mandatory_continuous"})
_TRANSPARENT_LOOP_RECOVERY_LOG = "[AETHER] Loop reinicializado de forma transparente para manter persistencia continua"


def resolve_post_settlement_cycle_timeout(orch: Any, orch_cfg: dict) -> float | None:
    """Retorna None quando o motor deve aguardar o mercado sem teto rigido."""
    regime = str(getattr(orch, "_last_quality_gate_regime", "") or "")
    if regime in _PATIENT_QUALITY_GATE_REGIMES:
        return None
    config = getattr(orch, "config", {})
    exec_chunk = config.get("orchestrator", {}).get("execution", {}) if isinstance(config, dict) else {}
    if isinstance(exec_chunk, dict) and exec_chunk.get("mandatory_trade_each_cycle"):
        return None
    raw = orch_cfg.get("post_settlement_cycle_timeout_seconds", _POST_SETTLEMENT_SAFE_CYCLE_TIMEOUT_SECONDS)
    return float(raw)


def clear_post_settlement_polling_state(orch: Any) -> None:
    """Limpa slot de trading e tasks de polling presas apos recuperacao."""
    orch.is_trading = False
    poll_task = getattr(orch, "_trading_slot_poll_task", None)
    if poll_task is not None and not poll_task.done():
        poll_task.cancel()
    orch._trading_slot_poll_task = None
    wake = getattr(orch, "_post_settlement_wake", None)
    if wake is not None:
        wake.clear()


def recover_post_settlement_loop_transparently(orch: Any) -> None:
    """Reinicializa contadores pos-liquidacao sem encerrar o processo do host."""
    streak = int(getattr(orch, "_post_settlement_incomplete_streak", 0))
    deadlock = bool(getattr(orch, "_post_settlement_deadlock", False))
    if not deadlock and streak <= 0:
        return
    orch._post_settlement_incomplete_streak = 0
    orch._post_settlement_deadlock = False
    clear_post_settlement_polling_state(orch)
    orch.logger.info(_TRANSPARENT_LOOP_RECOVERY_LOG)
