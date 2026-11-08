"""Pool de recovery e bloqueio de simbolos do cluster."""

from typing import Any

from src.domain.models.trade import TradeDirection


def pending_recovery_active(pending_loss: dict, consecutive_losses_linear: int = 0) -> bool:
    """True quando ha perda pendente ou linear D'Alembert ativo."""
    linear = (
        int(consecutive_losses_linear)
        if isinstance(consecutive_losses_linear, (int, float)) and not isinstance(consecutive_losses_linear, bool)
        else 0
    )
    if linear > 0:
        return True
    if not isinstance(pending_loss, dict):
        return False
    try:
        return sum(float(v) for v in pending_loss.values()) > 0.0
    except (TypeError, ValueError):
        return False


def recovery_blocked_symbols(risk_manager: Any, kelly_config: dict) -> frozenset[str]:
    """Simbolos bloqueados por rotacao apos perdas consecutivas."""
    linear = int(getattr(risk_manager, "consecutive_losses_linear", 0))
    rotation_cycles = int(kelly_config.get("symbol_loss_rotation_cycles", 1))
    last = getattr(risk_manager, "last_loss_symbol", None)
    if linear >= rotation_cycles and last:
        return frozenset({str(last)})
    return frozenset()


def recovery_candidate_pool(
    candidates: list[tuple[str, TradeDirection, dict]],
    *,
    last_loss_symbol: str | None,
    recovery_active: bool,
    skip_symbols: frozenset[str] | None = None,
) -> list[tuple[str, TradeDirection, dict]]:
    """Filtra o pool de candidatos sob recovery ativo."""
    _ = last_loss_symbol
    pool = list(candidates)
    if not recovery_active:
        return pool
    skip = skip_symbols or frozenset()
    if skip:
        pool = [item for item in pool if item[0] not in skip]
    return pool
