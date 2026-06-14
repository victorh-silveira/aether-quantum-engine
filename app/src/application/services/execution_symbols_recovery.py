"""Pool de recovery e ranking de candidatos do cluster."""

from typing import Any

from src.application.services.execution_direction import build_forced_direction_candidate, recovery_hedge_target
from src.domain.models.trade import TradeDirection


_CLUSTER_CORE = frozenset({"R_50", "R_75"})


def pending_recovery_active(pending_loss: dict) -> bool:
    """Indica se ha perda pendente ativando modo de recuperacao na selecao."""
    return sum(float(v) for v in pending_loss.values()) > 0.0


def recovery_blocked_symbols(risk_manager: Any, kelly_config: dict) -> frozenset[str]:
    """Simbolos excluidos do recovery por cooldown ou sequencia de losses em martingale."""
    blocked: set[str] = set()
    max_streak = int(kelly_config.get("recovery_martingale_max_losses_per_symbol", 2))
    streaks = getattr(risk_manager, "recovery_symbol_loss_streak", {}) or {}
    for symbol, count in streaks.items():
        if int(count) >= max_streak:
            blocked.add(str(symbol))
    cooldowns = getattr(risk_manager, "symbol_loss_cooldown", {}) or {}
    for symbol, remaining in cooldowns.items():
        if int(remaining) > 0:
            blocked.add(str(symbol))
    return frozenset(blocked)


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
    if item[0] in _CLUSTER_CORE:
        score += 0.04
    if last_loss_symbol and item[0] == last_loss_symbol:
        score -= 0.12
    elif last_loss_symbol and item[0] != last_loss_symbol:
        score += 0.05
    if last_loss_direction:
        ld = str(last_loss_direction).upper()
        if item[1].name == ld:
            score -= 0.08
        else:
            score += 0.07
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
    decisions: dict,
    *,
    last_loss_symbol: str | None,
    last_loss_direction: str | None,
) -> list[tuple[str, TradeDirection, dict]]:
    """Inclui candidato hedge estrutural do par Range quando ausente no pool."""
    target = recovery_hedge_target(last_loss_symbol, last_loss_direction)
    if target is None:
        return candidates
    peer, hedge_dir = target
    if any(item[0] == peer and item[1] == hedge_dir for item in candidates):
        return candidates
    entry = decisions.get(peer)
    if not entry:
        return candidates
    built = build_forced_direction_candidate(peer, entry, hedge_dir)
    if built is None:
        return candidates
    return list(candidates) + [built]


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
