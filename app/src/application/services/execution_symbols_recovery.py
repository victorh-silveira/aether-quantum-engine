"""Pool de recovery, hedge forcado e candidatos de recuperacao no par."""

from src.application.services.execution_direction import (
    build_forced_direction_candidate,
    recovery_hedge_target,
)
from src.domain.models.trade import TradeDirection
from src.domain.symbols.range_symbols import hedge_peer


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
    """Restringe candidatos em recovery priorizando hedge no par apos loss."""
    _ = last_loss_direction
    pool = list(candidates)
    if not recovery_active:
        return pool
    approved = [item for item in pool if item[2].get("execute")]
    if approved:
        pool = approved
    if last_loss_symbol:
        peer = hedge_peer(last_loss_symbol)
        if peer:
            peered = [item for item in pool if item[0] == peer]
            if peered:
                pool = peered
            else:
                peered = [item for item in candidates if item[0] == peer]
                if peered:
                    pool = peered
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
    """Acrescenta candidato com a direcao de hedge do ultimo loss para o par (peer)."""
    if not last_loss_symbol or not last_loss_direction:
        return candidates
    target = recovery_hedge_target(last_loss_symbol, last_loss_direction)
    if target is None:
        return candidates
    peer, forced_dir = target
    entry = decisions.get(peer) if peer else None
    if not entry:
        return candidates
    if any(item[0] == peer and item[1] == forced_dir for item in candidates):
        return candidates
    built = build_forced_direction_candidate(peer, entry, forced_dir)
    if built is None:
        return candidates
    clean_candidates = [item for item in candidates if item[0] != peer]
    return clean_candidates + [built]


def has_recovery_hedge_candidate(
    candidates: list[tuple[str, TradeDirection, dict]],
    *,
    last_loss_symbol: str | None,
    last_loss_direction: str | None,
) -> bool:
    """True se existe candidato do par (peer) correspondente ao ultimo loss."""
    _ = last_loss_direction
    if not last_loss_symbol:
        return True
    peer = hedge_peer(last_loss_symbol)
    if not peer:
        return True
    return any(item[0] == peer for item in candidates)
