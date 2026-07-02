"""Selecao obrigatoria de candidatos por ranking de mercado."""

from src.application.services.execution_direction import (
    _entry_gate_blocked,
    build_execution_candidate,
    build_forced_direction_candidate,
    build_forced_recovery_candidate,
    meets_mandatory_signal_floor,
    recovery_hedge_target,
)
from src.application.services.execution_market_rank import (
    _trade_score,
    build_market_execution_candidate,
    mandatory_pool_eligible,
    market_decision_score,
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
    core = [symbol for symbol in ("RDBULL", "RDBEAR") if symbol in eligible]
    alt = [symbol for symbol in core if symbol != last_loss_symbol]
    if alt:
        core = alt + [symbol for symbol in core if symbol not in alt]
    tail = [symbol for symbol in eligible if symbol not in core and symbol != last_loss_symbol]
    if not tail:
        tail = [symbol for symbol in eligible if symbol not in core]
    return core + tail


def _recovery_hedge_pick(
    decisions: dict,
    *,
    last_loss_symbol: str | None,
    last_loss_direction: str | None,
    skip_symbols: frozenset[str],
    consecutive_losses: int = 0,
    mean_reversion_enabled: bool = True,
    low_accuracy_enabled: bool = True,
) -> tuple[str, TradeDirection, dict] | None:
    """Prioriza par Drift com direcao estrutural oposta ao ultimo loss."""
    _ = (consecutive_losses, mean_reversion_enabled, low_accuracy_enabled)
    target = recovery_hedge_target(last_loss_symbol, last_loss_direction)
    if target is None:
        return None
    peer, hedge_dir = target
    if peer in skip_symbols:
        return None
    entry = decisions.get(peer)
    if not entry or _entry_gate_blocked(entry.get("metrics") or {}):
        return None
    built = build_forced_direction_candidate(peer, entry, hedge_dir)
    if built is not None:
        return built
    return build_forced_recovery_candidate(peer, entry, hedge_dir)


def _rank_eligible_candidates(
    order: list[str],
    decisions: dict,
    *,
    recovery_active: bool,
    last_loss_symbol: str | None,
    last_loss_direction: str | None,
    min_signal: float,
    min_val: float,
    consecutive_losses: int = 0,
    mean_reversion_enabled: bool = True,
    low_accuracy_enabled: bool = True,
) -> tuple[str, TradeDirection, dict] | None:
    """Rankeia candidatos elegiveis e retorna o melhor por score de mercado."""
    best = None
    best_score = -1.0
    for symbol in order:
        entry = decisions.get(symbol)
        if not entry or not mandatory_pool_eligible(entry):
            continue
        metrics = entry.get("metrics") or {}
        if (min_signal > 0.0 or min_val > 0.0) and not meets_mandatory_signal_floor(
            metrics, min_signal=min_signal, min_val=min_val
        ):
            continue
        score = _trade_score(metrics)
        candidate = build_market_execution_candidate(
            symbol,
            entry,
            recovery_active=recovery_active,
            consecutive_losses=consecutive_losses,
            mean_reversion_enabled=mean_reversion_enabled,
            low_accuracy_enabled=low_accuracy_enabled,
        )
        if candidate is None:
            candidate = build_execution_candidate(
                symbol,
                entry,
                recovery_active=recovery_active,
            )
        if candidate is None:
            continue
        rank = market_decision_score(
            candidate[2],
            exec_direction=candidate[1],
            recovery_active=recovery_active,
            symbol=symbol,
            last_loss_symbol=last_loss_symbol,
            last_loss_direction=last_loss_direction,
        )
        if score + 1e-9 >= min_signal:
            rank += 0.08
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
    consecutive_losses: int = 0,
    mean_reversion_enabled: bool = True,
    low_accuracy_enabled: bool = True,
) -> tuple[str, TradeDirection, dict] | None:
    """Escolhe melhor candidato obrigatorio por score de mercado."""
    skip = skip_symbols or frozenset()
    if recovery_active:
        hedge = _recovery_hedge_pick(
            decisions,
            last_loss_symbol=last_loss_symbol,
            last_loss_direction=last_loss_direction,
            skip_symbols=skip,
            consecutive_losses=consecutive_losses,
            mean_reversion_enabled=mean_reversion_enabled,
            low_accuracy_enabled=low_accuracy_enabled,
        )
        if hedge is not None and meets_mandatory_signal_floor(hedge[2], min_signal=min_signal, min_val=min_val):
            return hedge
    order = _symbol_order(trade_symbols, last_loss_symbol, skip_symbols=skip)
    ranked = _rank_eligible_candidates(
        order,
        decisions,
        recovery_active=recovery_active,
        last_loss_symbol=last_loss_symbol,
        last_loss_direction=last_loss_direction,
        min_signal=min_signal,
        min_val=min_val,
        consecutive_losses=consecutive_losses,
        mean_reversion_enabled=mean_reversion_enabled,
        low_accuracy_enabled=low_accuracy_enabled,
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
        min_val=min_val,
        consecutive_losses=consecutive_losses,
        mean_reversion_enabled=mean_reversion_enabled,
        low_accuracy_enabled=low_accuracy_enabled,
    )


def pick_absolute_mandatory_candidate(
    trade_symbols: list[str],
    decisions: dict,
    *,
    recovery_active: bool,
    last_loss_symbol: str | None,
    last_loss_direction: str | None,
    min_signal: float = 0.0,
    min_val: float = 0.0,
    consecutive_losses: int = 0,
    mean_reversion_enabled: bool = True,
    low_accuracy_enabled: bool = True,
) -> tuple[str, TradeDirection, dict] | None:
    """Garante ordem quando filtros de recovery esgotam o pool."""
    order = _symbol_order(trade_symbols, last_loss_symbol, skip_symbols=frozenset())
    best = None
    best_score = -1.0
    for symbol in order:
        entry = decisions.get(symbol)
        if not entry or not mandatory_pool_eligible(entry):
            continue
        metrics = entry.get("metrics") or {}
        if (min_signal > 0.0 or min_val > 0.0) and not meets_mandatory_signal_floor(
            metrics, min_signal=min_signal, min_val=min_val
        ):
            continue
        candidate = build_market_execution_candidate(
            symbol,
            entry,
            recovery_active=recovery_active,
            consecutive_losses=consecutive_losses,
            mean_reversion_enabled=mean_reversion_enabled,
            low_accuracy_enabled=low_accuracy_enabled,
        )
        if candidate is None:
            candidate = build_execution_candidate(
                symbol,
                entry,
                recovery_active=recovery_active,
            )
        if candidate is None:
            continue
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
