"""Pool de recovery, hedge forcado e candidatos de recuperacao no par."""

from src.application.services.execution_direction import (
    build_execution_candidate,
    infer_dl_direction,
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
    """Restringe candidatos em recovery: execute=true, par (peer) do ultimo loss e sem repetir loss."""
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
    """Acrescenta candidato com a direcao prevista pelo DL para o par (peer) do ultimo loss."""
    _ = last_loss_direction
    if not last_loss_symbol:
        return candidates
    peer = hedge_peer(last_loss_symbol)
    entry = decisions.get(peer) if peer else None
    if not entry:
        return candidates
    dl_dir = infer_dl_direction(entry)
    if dl_dir is None:
        return candidates
    if any(item[0] == peer and item[1] == dl_dir for item in candidates):
        return candidates
    built = build_execution_candidate(peer, entry, invert_dl_direction=False)
    return list(candidates) + [built]


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
