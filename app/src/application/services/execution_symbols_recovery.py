"""Pool de recovery e ranking de candidatos do cluster."""

from typing import Any

from src.domain.models.trade import TradeDirection


def pending_recovery_active(pending_loss: dict) -> bool:
    """Indica se ha perda pendente ativando modo de recuperacao na selecao."""
    return sum(float(v) for v in pending_loss.values()) > 0.0


def recovery_blocked_symbols(risk_manager: Any, kelly_config: dict) -> frozenset[str]:
    """Exclui simbolo da ultima loss durante streak linear ativa."""
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
    last_loss_direction: str | None,
    recovery_active: bool,
    skip_symbols: frozenset[str] | None = None,
) -> list[tuple[str, TradeDirection, dict]]:
    """Aplica apenas bloqueios duros; ranking define direcao e simbolo."""
    _ = (last_loss_symbol, last_loss_direction)
    pool = list(candidates)
    if not recovery_active:
        return pool
    skip = skip_symbols or frozenset()
    if skip:
        pool = [item for item in pool if item[0] not in skip]
    return pool


def recovery_rank_score(
    item: tuple[str, TradeDirection, dict],
    *,
    last_loss_symbol: str | None = None,
    last_loss_direction: str | None = None,
    base_score: float,
) -> float:
    """Pontua candidato em recovery priorizando melhor tendencia e diversificacao."""
    score = float(base_score)
    metrics = item[2]
    if last_loss_symbol and item[0] == last_loss_symbol:
        score -= 0.20
    elif last_loss_symbol and item[0] != last_loss_symbol:
        score += 0.08
    if last_loss_direction:
        ld = str(last_loss_direction).upper()
        if item[1].name == ld:
            score -= 0.12
        else:
            score += 0.06
    raw = metrics.get("raw_prob")
    if raw is not None and item[1] == TradeDirection.CALL and float(raw) > 0.5:
        score += 0.02
    if raw is not None and item[1] == TradeDirection.PUT and float(raw) <= 0.5:
        score += 0.02
    if metrics.get("execute"):
        score += 0.03
    return score


def inject_recovery_hedge_candidates(
    candidates: list[tuple[str, TradeDirection, dict]],
    _decisions: dict,
    *,
    last_loss_symbol: str | None = None,
    last_loss_direction: str | None = None,
) -> list[tuple[str, TradeDirection, dict]]:
    """Mantem pool original; hedge estrutural desativado."""
    _ = (last_loss_symbol, last_loss_direction)
    return candidates


def has_recovery_hedge_candidate(
    candidates: list[tuple[str, TradeDirection, dict]],
    *,
    last_loss_symbol: str | None,
    last_loss_direction: str | None,
) -> bool:
    """Indica se existe candidato na mesma direcao do ultimo loss."""
    _ = last_loss_symbol
    if not last_loss_direction:
        return True
    target = str(last_loss_direction).upper()
    return any(item[1].name == target for item in candidates)


def apply_recovery_direction_flip(
    best: tuple[str, TradeDirection, dict] | None,
    _decisions: dict,
    *,
    recovery_active: bool = False,
    last_loss_symbol: str | None = None,
    last_loss_direction: str | None = None,
    flip_enabled: bool = False,
    flip_max_conviction: float = 0.56,
    consecutive_losses: int = 0,
    flip_use_trend: bool = False,
) -> tuple[str, TradeDirection, dict] | None:
    """Mantem direcao resolvida pelo DL sem inversao pos-loss."""
    _ = (
        recovery_active,
        last_loss_symbol,
        last_loss_direction,
        flip_enabled,
        flip_max_conviction,
        consecutive_losses,
        flip_use_trend,
    )
    return best
