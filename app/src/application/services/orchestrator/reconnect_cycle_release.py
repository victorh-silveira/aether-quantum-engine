"""Libera ciclos de trading apos reconexao do WebSocket."""

from __future__ import annotations

import asyncio
from typing import Any

from src.application.services.execution_quality_gate import read_risk_session_state
from src.application.services.orchestrator.warm_up_buffer_guard import resolve_stream_warm_up_delay_seconds


_RECOVERY_WARM_UP_DELAY_SECONDS = 5.0
_RECOVERY_PENDING_WARM_UP_MAX_SECONDS = 12.0


def _pending_loss_total(risk_manager: Any | None) -> float:
    """Resolve passivo pendente total a partir do RiskManager."""
    _, pending = read_risk_session_state(risk_manager)
    return pending


def resolve_post_reconnect_warm_up_delay_seconds(orch: Any) -> float:
    """Reduz aquecimento micro quando ha passivo pendente para nao travar recovery."""
    pending = _pending_loss_total(getattr(orch, "risk_manager", None))
    if pending > 0.0:
        return min(_RECOVERY_PENDING_WARM_UP_MAX_SECONDS, _RECOVERY_WARM_UP_DELAY_SECONDS)
    return resolve_stream_warm_up_delay_seconds(getattr(orch, "config", {}) or {})


def schedule_post_reconnect_warm_up_barrier(orch: Any) -> float:
    """Registra barreira de aquecimento ajustada ao estado de recovery."""
    delay = resolve_post_reconnect_warm_up_delay_seconds(orch)
    loop = asyncio.get_running_loop()
    orch._stream_warmed_up_at = loop.time() + delay
    orch._warm_up_logged_until = 0.0
    orch._warm_up_waiver_applied = False
    return delay


def release_trading_cycle_after_reconnect(orch: Any) -> None:
    """Invalida assinatura e epoch processado para retomar ciclos apos queda de rede."""
    orch.last_data_signature = ""
    orch._signature_invalidation_logged_key = ""
    orch._last_processed_epoch = 0
    orch._quality_guard_logged_cycle_id = -1
    schedule_post_reconnect_warm_up_barrier(orch)
    linear, pending = read_risk_session_state(getattr(orch, "risk_manager", None))
    logger = getattr(orch, "logger", None)
    if logger is not None:
        logger.info(
            "RECOV: ciclo liberado | linear=%d | pend=$%.2f | warm_up=%.0fs",
            int(linear),
            float(pending),
            resolve_post_reconnect_warm_up_delay_seconds(orch),
        )
