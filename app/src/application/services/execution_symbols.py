"""Simbolos elegiveis e ranking de candidatos para execucao."""

from src.application.services.execution_market_rank import market_decision_score
from src.application.services.execution_symbols_recovery import (
    has_recovery_hedge_candidate,
    inject_recovery_hedge_candidates,
    pending_recovery_active,
    recovery_candidate_pool,
)
from src.domain.models.trade import TradeDirection


__all__ = [
    "symbols_eligible_for_execution",
    "candidate_execution_score",
    "select_best_execution_candidate",
    "select_mandatory_execution_candidate",
    "pending_recovery_active",
    "inject_recovery_hedge_candidates",
    "has_recovery_hedge_candidate",
    "format_execution_alternates",
]


def symbols_eligible_for_execution(anchor: str, symbols: list[str], *, include_anchor: bool) -> list[str]:
    """Retorna simbolos do cluster que podem receber ordens de execucao."""
    eligible = []
    for symbol in symbols:
        if symbol == anchor and not include_anchor:
            continue
        eligible.append(symbol)
    return eligible


def candidate_execution_score(
    metrics: dict,
    *,
    recovery_active: bool,
    symbol: str | None = None,
    exec_direction: TradeDirection | None = None,
    last_loss_symbol: str | None = None,
    last_loss_direction: str | None = None,
) -> float:
    """Pontua candidato com score bruto, val_acc e edge."""
    direction = exec_direction
    if direction is None and metrics.get("exec_direction"):
        name = str(metrics["exec_direction"]).upper()
        direction = TradeDirection.CALL if name == "CALL" else TradeDirection.PUT
    return market_decision_score(
        metrics,
        exec_direction=direction,
        recovery_active=recovery_active,
        symbol=symbol,
        last_loss_symbol=last_loss_symbol,
        last_loss_direction=last_loss_direction,
    )


def select_best_execution_candidate(
    candidates: list[tuple[str, TradeDirection, dict]],
    *,
    last_loss_symbol: str | None,
    last_loss_direction: str | None = None,
    diversify_margin: float,
    recovery_active: bool,
    skip_symbols: frozenset[str] | None = None,
) -> tuple[str, TradeDirection, dict] | None:
    """Escolhe melhor candidato por score de mercado."""
    _ = diversify_margin
    pool = recovery_candidate_pool(
        candidates,
        last_loss_symbol=last_loss_symbol,
        last_loss_direction=last_loss_direction,
        recovery_active=recovery_active,
        skip_symbols=skip_symbols,
    )
    if not pool:
        return None

    def rank_key(item: tuple[str, TradeDirection, dict]) -> float:
        """Retorna score de ranking para ordenar candidatos de execucao."""
        return candidate_execution_score(
            item[2],
            recovery_active=recovery_active,
            symbol=item[0],
            exec_direction=item[1],
            last_loss_symbol=last_loss_symbol,
            last_loss_direction=last_loss_direction,
        )

    ranked = sorted(pool, key=rank_key, reverse=True)
    return ranked[0]


def select_mandatory_execution_candidate(
    _orch,
    candidates: list[tuple[str, TradeDirection, dict]],
    *,
    last_loss_symbol: str | None,
    last_loss_direction: str | None = None,
    diversify_margin: float,
    recovery_active: bool,
    skip_symbols: frozenset[str] | None = None,
) -> tuple[str, TradeDirection, dict] | None:
    """Escolhe candidato em modo obrigatorio priorizando melhor score."""
    pool = recovery_candidate_pool(
        candidates,
        last_loss_symbol=last_loss_symbol,
        last_loss_direction=last_loss_direction,
        recovery_active=recovery_active,
        skip_symbols=skip_symbols,
    )
    if not pool:
        return None
    return select_best_execution_candidate(
        pool,
        last_loss_symbol=last_loss_symbol,
        last_loss_direction=last_loss_direction,
        diversify_margin=diversify_margin,
        recovery_active=recovery_active,
        skip_symbols=skip_symbols,
    )


def format_execution_alternates(
    candidates: list[tuple[str, TradeDirection, dict]],
    *,
    exclude_symbol: str | None = None,
    limit: int = 2,
) -> str:
    """Formata simbolos alternativos ordenados por score para log EXEC_SEL."""
    ranked = sorted(
        candidates,
        key=lambda item: candidate_execution_score(item[2], recovery_active=False),
        reverse=True,
    )
    alts = [item for item in ranked if item[0] != exclude_symbol][:limit]
    return ", ".join(
        f"{symbol}({metrics.get('trade_score', metrics.get('raw_prob', 0.0)):.2f})" for symbol, _, metrics in alts
    )
