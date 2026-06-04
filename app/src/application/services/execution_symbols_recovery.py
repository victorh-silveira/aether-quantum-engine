"""Pool de recovery, hedge forcado e candidatos de recuperacao no par."""

from src.application.services.execution_direction import (
    build_forced_direction_candidate,
    recovery_hedge_target,
)
from src.domain.models.trade import TradeDirection


def pending_recovery_active(pending_loss: dict) -> bool:
    """Indica se ha perda pendente ativando modo de recuperacao na selecao."""
    return sum(float(v) for v in pending_loss.values()) > 0.0


def recovery_candidate_pool(
    candidates: list[tuple[str, TradeDirection, dict]],
    *,
    last_loss_symbol: str | None,
    last_loss_direction: str | None,
    recovery_active: bool,
) -> list[tuple[str, TradeDirection, dict]]:
    """Restringe candidatos em recovery: execute=true, hedge do par e sem repetir loss."""
    pool = list(candidates)
    if not recovery_active:
        return pool
    approved = [item for item in pool if item[2].get("execute")]
    if approved:
        pool = approved
    hedge = recovery_hedge_target(last_loss_symbol, last_loss_direction)
    if hedge is not None:
        sym, direction = hedge
        hedged = [item for item in pool if item[0] == sym and item[1] == direction]
        if hedged:
            pool = hedged
        else:
            pool = [item for item in candidates if item[0] == sym and item[1] == direction]
            if not pool:
                pool = list(candidates)
    if last_loss_symbol:
        filtered = [item for item in pool if item[0] != last_loss_symbol]
        if filtered:
            pool = filtered
    return pool


def inject_recovery_hedge_candidates(
    candidates: list[tuple[str, TradeDirection, dict]],
    decisions: dict,
    *,
    last_loss_symbol: str | None,
    last_loss_direction: str | None,
) -> list[tuple[str, TradeDirection, dict]]:
    """Acrescenta candidato de hedge no par quando DL nao gera a direcao correta."""
    hedge = recovery_hedge_target(last_loss_symbol, last_loss_direction)
    if hedge is None:
        return candidates
    sym, direction = hedge
    if any(item[0] == sym and item[1] == direction for item in candidates):
        return candidates
    entry = decisions.get(sym)
    if not entry:
        return candidates
    forced = build_forced_direction_candidate(sym, entry, direction)
    if forced is None:
        return candidates
    return list(candidates) + [forced]


def has_recovery_hedge_candidate(
    candidates: list[tuple[str, TradeDirection, dict]],
    *,
    last_loss_symbol: str | None,
    last_loss_direction: str | None,
) -> bool:
    """True se existe candidato alinhado ao alvo de hedge ou hedge nao se aplica."""
    hedge = recovery_hedge_target(last_loss_symbol, last_loss_direction)
    if hedge is None:
        return True
    sym, direction = hedge
    return any(item[0] == sym and item[1] == direction for item in candidates)
