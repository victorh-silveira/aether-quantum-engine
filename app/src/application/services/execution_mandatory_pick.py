"""Selecao obrigatoria de candidatos por ranking de mercado."""

from src.application.services.execution_direction import build_forced_recovery_candidate
from src.application.services.execution_market_rank import (
    _trade_score,
    build_market_execution_candidate,
    mandatory_pool_eligible,
    market_decision_score,
    resolve_market_direction,
)
from src.domain.models.trade import TradeDirection


def _symbol_order(
    trade_symbols: list[str],
    last_loss_symbol: str | None,
    *,
    skip_symbols: frozenset[str],
) -> list[str]:
    """Ordena simbolos priorizando core e diversificacao apos loss."""
    eligible = [symbol for symbol in trade_symbols if symbol not in skip_symbols]
    core = [symbol for symbol in ("R_75", "R_50") if symbol in eligible]
    alt = [symbol for symbol in core if symbol != last_loss_symbol]
    if alt:
        core = alt + [symbol for symbol in core if symbol not in alt]
    tail = [symbol for symbol in eligible if symbol not in core and symbol != last_loss_symbol]
    if not tail:
        tail = [symbol for symbol in eligible if symbol not in core]
    return core + tail


def _rank_eligible_candidates(
    order: list[str],
    decisions: dict,
    *,
    recovery_active: bool,
    last_loss_symbol: str | None,
    last_loss_direction: str | None,
    min_signal: float,
    min_val: float,
    aligned_dir: TradeDirection | None,
) -> tuple[str, TradeDirection, dict] | None:
    """Rankeia candidatos elegiveis e retorna o melhor por score de mercado."""
    best = None
    best_score = -1.0
    for symbol in order:
        entry = decisions.get(symbol)
        if not entry or not mandatory_pool_eligible(entry):
            continue
        metrics = entry.get("metrics") or {}
        score = _trade_score(metrics)
        val = float(metrics.get("val_accuracy", 0.0))
        if score + 1e-9 < min_signal:
            continue
        if aligned_dir is not None:
            direction = resolve_market_direction(entry)
            if direction != aligned_dir:
                continue
            if val + 1e-9 < min_val:
                continue
        candidate = build_market_execution_candidate(symbol, entry)
        rank = market_decision_score(
            candidate[2],
            exec_direction=candidate[1],
            recovery_active=recovery_active,
            symbol=symbol,
            last_loss_symbol=last_loss_symbol,
            last_loss_direction=last_loss_direction,
        )
        if rank > best_score:
            best_score = rank
            best = candidate
    return best


def pick_best_mandatory_candidate(
    trade_symbols: list[str],
    decisions: dict,
    *,
    recovery_active: bool,
    last_loss_symbol: str | None,
    last_loss_direction: str | None,
    skip_symbols: frozenset[str] | None = None,
    min_signal: float = 0.0,
    min_val: float = 0.0,
) -> tuple[str, TradeDirection, dict] | None:
    """Escolhe melhor candidato obrigatorio por score de mercado."""
    skip = skip_symbols or frozenset()
    order = _symbol_order(trade_symbols, last_loss_symbol, skip_symbols=skip)
    aligned_dir = None
    if recovery_active and last_loss_direction:
        name = str(last_loss_direction).upper()
        if name == "CALL":
            aligned_dir = TradeDirection.CALL
        elif name == "PUT":
            aligned_dir = TradeDirection.PUT
    if aligned_dir is not None:
        aligned = _rank_eligible_candidates(
            order,
            decisions,
            recovery_active=recovery_active,
            last_loss_symbol=last_loss_symbol,
            last_loss_direction=last_loss_direction,
            min_signal=min_signal,
            min_val=min_val,
            aligned_dir=aligned_dir,
        )
        if aligned is not None:
            return aligned
    ranked = _rank_eligible_candidates(
        order,
        decisions,
        recovery_active=recovery_active,
        last_loss_symbol=last_loss_symbol,
        last_loss_direction=last_loss_direction,
        min_signal=min_signal,
        min_val=0.0,
        aligned_dir=None,
    )
    if ranked is not None:
        return ranked
    return pick_absolute_mandatory_candidate(
        trade_symbols,
        decisions,
        recovery_active=recovery_active,
        last_loss_symbol=last_loss_symbol,
        last_loss_direction=last_loss_direction,
        min_signal=min_signal,
    )


def pick_absolute_mandatory_candidate(
    trade_symbols: list[str],
    decisions: dict,
    *,
    recovery_active: bool,
    last_loss_symbol: str | None,
    last_loss_direction: str | None,
    min_signal: float = 0.0,
) -> tuple[str, TradeDirection, dict] | None:
    """Garante ordem quando filtros de recovery esgotam o pool."""
    order = _symbol_order(trade_symbols, last_loss_symbol, skip_symbols=frozenset())
    best = None
    best_score = -1.0
    for symbol in order:
        entry = decisions.get(symbol)
        if not entry or not mandatory_pool_eligible(entry):
            continue
        if _trade_score(entry.get("metrics") or {}) + 1e-9 < min_signal:
            continue
        direction = resolve_market_direction(entry)
        metrics = dict(entry.get("metrics") or {})
        rank = market_decision_score(
            metrics,
            exec_direction=direction,
            recovery_active=recovery_active,
            symbol=symbol,
            last_loss_symbol=last_loss_symbol,
            last_loss_direction=last_loss_direction,
        )
        if rank > best_score:
            best_score = rank
            best = build_forced_recovery_candidate(symbol, entry, direction)
    return best
