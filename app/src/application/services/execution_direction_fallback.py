"""Fallback obrigatorio de execucao quando o pool DL fica vazio."""

from typing import Any

from src.application.services.execution_direction import (
    _entry_gate_blocked,
    _entry_signal_strength,
    build_execution_candidate,
)
from src.application.services.execution_mandatory_pick import pick_best_mandatory_candidate
from src.application.services.execution_market_rank import build_market_execution_candidate
from src.domain.models.trade import TradeDirection


def _scored_fallback_pick(
    order: list[str],
    decisions: dict,
    *,
    skip_symbols: frozenset[str] | None = None,
    min_signal: float = 0.0,
    min_val: float = 0.0,
    recovery_active: bool = False,
    consecutive_losses: int = 0,
    skipped_cycles_counter: int | None = None,
    orch: Any | None = None,
) -> tuple[str, TradeDirection, dict] | None:
    """Escolhe o melhor candidato de fallback por score de sinal."""
    _ = consecutive_losses
    skip = skip_symbols or frozenset()
    best = None
    best_score = -1.0
    for symbol in order:
        if symbol in skip:
            continue
        entry = decisions.get(symbol)
        if not entry or _entry_gate_blocked(entry.get("metrics") or {}):
            continue
        metrics = entry.get("metrics") or {}
        score, raw_side = _entry_signal_strength(metrics)
        if max(score, raw_side) + 1e-9 < min_signal:
            continue
        if min_val > 0.0 and float(metrics.get("val_accuracy", 0.0)) + 1e-9 < min_val:
            continue
        candidate = build_market_execution_candidate(symbol, entry, recovery_active=recovery_active)
        if candidate is None:
            candidate = build_execution_candidate(
                symbol, entry, skipped_cycles_counter=skipped_cycles_counter, orch=orch
            )
        if candidate is None or score < best_score:
            continue
        best_score = score
        best = candidate
    return best


def _last_resort_fallback_pick(
    trade_symbols: list[str],
    decisions: dict,
    *,
    skip_symbols: frozenset[str] | None = None,
    min_signal: float = 0.0,
    min_val: float = 0.0,
    recovery_active: bool = False,
    consecutive_losses: int = 0,
    skipped_cycles_counter: int | None = None,
    orch: Any | None = None,
) -> tuple[str, TradeDirection, dict] | None:
    """Ultimo recurso de fallback percorrendo simbolos em ordem."""
    _ = consecutive_losses
    skip = skip_symbols or frozenset()
    for symbol in trade_symbols:
        if symbol in skip:
            continue
        entry = decisions.get(symbol)
        if not entry or _entry_gate_blocked(entry.get("metrics") or {}):
            continue
        metrics = entry.get("metrics") or {}
        score, raw_side = _entry_signal_strength(metrics)
        if max(score, raw_side) + 1e-9 < min_signal:
            continue
        if min_val > 0.0 and float(metrics.get("val_accuracy", 0.0)) + 1e-9 < min_val:
            continue
        candidate = build_market_execution_candidate(symbol, entry, recovery_active=recovery_active)
        if candidate is None:
            candidate = build_execution_candidate(
                symbol, entry, recovery_active=recovery_active, skipped_cycles_counter=skipped_cycles_counter, orch=orch
            )
        if candidate is None:
            continue
        return candidate
    return None


def build_mandatory_fallback_candidate(
    trade_symbols: list[str],
    decisions: dict,
    *,
    recovery_active: bool,
    last_loss_symbol: str | None,
    skip_symbols: frozenset[str] | None = None,
    min_signal: float = 0.0,
    min_val: float = 0.0,
    consecutive_losses: int = 0,
    skipped_cycles_counter: int | None = None,
    orch: Any | None = None,
) -> tuple[str, TradeDirection, dict] | None:
    """Monta candidato obrigatorio quando o pool DL esta vazio."""
    ranked = pick_best_mandatory_candidate(
        trade_symbols,
        decisions,
        recovery_active=recovery_active,
        last_loss_symbol=last_loss_symbol,
        skip_symbols=skip_symbols,
        min_signal=min_signal,
        min_val=min_val,
        consecutive_losses=consecutive_losses,
    )
    if ranked is not None:
        return ranked
    scored = _scored_fallback_pick(
        trade_symbols,
        decisions,
        skip_symbols=skip_symbols,
        min_signal=min_signal,
        min_val=min_val,
        recovery_active=recovery_active,
        consecutive_losses=consecutive_losses,
        skipped_cycles_counter=skipped_cycles_counter,
        orch=orch,
    )
    if scored is not None:
        return scored
    return _last_resort_fallback_pick(
        trade_symbols,
        decisions,
        skip_symbols=skip_symbols,
        min_signal=min_signal,
        min_val=min_val,
        recovery_active=recovery_active,
        consecutive_losses=consecutive_losses,
        skipped_cycles_counter=skipped_cycles_counter,
        orch=orch,
    )
