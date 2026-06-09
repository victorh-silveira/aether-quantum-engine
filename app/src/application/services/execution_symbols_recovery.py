"""Pool de recovery e candidatos na mesma direcao do cluster."""

from src.application.services.execution_direction import recovery_execution_eligible
from src.domain.models.trade import TradeDirection


_CLUSTER_CORE = frozenset({"R_50", "R_75"})


def pending_recovery_active(pending_loss: dict) -> bool:
    """Indica se ha perda pendente ativando modo de recuperacao na selecao."""
    return sum(float(v) for v in pending_loss.values()) > 0.0


def _matches_loss_direction(
    item: tuple[str, TradeDirection, dict],
    last_loss_direction: str | None,
) -> bool:
    """Indica se a direcao do candidato coincide com a do ultimo loss."""
    if not last_loss_direction:
        return True
    return item[1].name == str(last_loss_direction).upper()


def recovery_candidate_pool(
    candidates: list[tuple[str, TradeDirection, dict]],
    *,
    last_loss_symbol: str | None,
    last_loss_direction: str | None,
    recovery_active: bool,
) -> list[tuple[str, TradeDirection, dict]]:
    """Restringe recovery a mesma direcao CALL/PUT e simbolos centrais do cluster."""
    pool = list(candidates)
    if not recovery_active:
        return pool
    if last_loss_direction:
        aligned = [item for item in pool if _matches_loss_direction(item, last_loss_direction)]
        if not aligned:
            return []
        pool = aligned
    approved = [item for item in pool if item[2].get("execute")]
    if approved:
        pool = approved
    else:
        quality = [item for item in pool if recovery_execution_eligible({"direction": item[1], "metrics": item[2]})]
        if quality:
            pool = quality
    core = [item for item in pool if item[0] in _CLUSTER_CORE]
    if core:
        pool = core
    if last_loss_symbol:
        alt = [item for item in pool if item[0] != last_loss_symbol]
        if alt:
            pool = alt
    return pool


def inject_recovery_hedge_candidates(
    candidates: list[tuple[str, TradeDirection, dict]],
    _decisions: dict,
    *,
    last_loss_symbol: str | None,
    last_loss_direction: str | None,
) -> list[tuple[str, TradeDirection, dict]]:
    """Mantem candidatos sem injetar hedge oposto no par Range."""
    _ = (last_loss_symbol, last_loss_direction)
    return candidates


def has_recovery_hedge_candidate(
    candidates: list[tuple[str, TradeDirection, dict]],
    *,
    last_loss_symbol: str | None,
    last_loss_direction: str | None,
) -> bool:
    """True se existe candidato na mesma direcao do ultimo loss."""
    _ = last_loss_symbol
    if not last_loss_direction:
        return True
    target = str(last_loss_direction).upper()
    return any(item[1].name == target for item in candidates)
