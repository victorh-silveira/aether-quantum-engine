"""Fallback obrigatorio de execucao quando o pool DL fica vazio."""

from typing import Any

from src.application.services.execution_direction import (
    _entry_gate_blocked,
    _entry_signal_strength,
    build_execution_candidate,
    build_forced_recovery_candidate,
    infer_dl_direction,
)
from src.application.services.execution_mandatory_pick import pick_best_mandatory_candidate
from src.application.services.execution_market_rank import build_market_execution_candidate
from src.domain.models.trade import TradeDirection
from src.domain.symbols.drift_symbols import TRADING_SYMBOLS


def _recovery_metrics_eligible(metrics: dict, *, min_signal: float, min_val: float) -> bool:
    """Indica se metricas atendem pisos de recovery para execucao alinhada."""
    score, raw_side = _entry_signal_strength(metrics)
    if score + 1e-9 < min_signal and raw_side + 1e-9 < min_signal:
        return False
    return float(metrics.get("val_accuracy", 0.0)) + 1e-9 >= min_val


def _loss_direction(value: str | None) -> TradeDirection | None:
    """Converte direcao textual do ultimo loss para TradeDirection."""
    if not value:
        return None
    name = str(value).upper()
    if name == "CALL":
        return TradeDirection.CALL
    if name == "PUT":
        return TradeDirection.PUT
    return None


def _symbol_priority(
    symbols: list[str],
    last_loss_symbol: str | None,
    *,
    skip_symbols: frozenset[str] | None = None,
    recovery_core_only: bool = False,
) -> list[str]:
    """Ordena simbolos priorizando o universo operacional e evitando repetir o ultimo loss."""
    skip = skip_symbols or frozenset()
    eligible = [symbol for symbol in symbols if symbol not in skip]
    core = [symbol for symbol in TRADING_SYMBOLS if symbol in eligible]
    alt = [symbol for symbol in core if symbol != last_loss_symbol]
    if alt:
        core = alt + [symbol for symbol in core if symbol not in alt]
    if recovery_core_only:
        return core
    tail = [symbol for symbol in eligible if symbol not in core and symbol != last_loss_symbol]
    if not tail:
        tail = [symbol for symbol in eligible if symbol not in core]
    return core + tail


def _forced_recovery_pick(
    order: list[str],
    decisions: dict,
    forced_dir: TradeDirection,
    *,
    skip_symbols: frozenset[str] | None = None,
    min_signal: float = 0.0,
    min_val: float = 0.0,
) -> tuple[str, TradeDirection, dict] | None:
    """Seleciona simbolo com DL alinhado a direcao do loss e qualidade minima."""
    skip = skip_symbols or frozenset()
    aligned: list[tuple[str, TradeDirection, dict]] = []
    for symbol in order:
        if symbol in skip:
            continue
        entry = decisions.get(symbol)
        if not entry or _entry_gate_blocked(entry.get("metrics") or {}):
            continue
        metrics = entry.get("metrics") or {}
        if not _recovery_metrics_eligible(metrics, min_signal=min_signal, min_val=min_val):
            continue
        dl_dir = infer_dl_direction(entry)
        if dl_dir != forced_dir:
            continue
        aligned.append(build_forced_recovery_candidate(symbol, entry, forced_dir))
    if aligned:
        return aligned[0]
    return None


def _scored_fallback_pick(
    order: list[str],
    decisions: dict,
    *,
    skip_symbols: frozenset[str] | None = None,
    min_signal: float = 0.0,
    min_val: float = 0.0,
    recovery_active: bool = False,
    consecutive_losses: int = 0,
    mean_reversion_enabled: bool = True,
    low_accuracy_enabled: bool = True,
    skipped_cycles_counter: int | None = None,
    orch: Any | None = None,
) -> tuple[str, TradeDirection, dict] | None:
    """Escolhe candidato inferivel com maior trade_score no modo obrigatorio."""
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
                skipped_cycles_counter=skipped_cycles_counter,
                orch=orch,
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
    mean_reversion_enabled: bool = True,
    low_accuracy_enabled: bool = True,
    skipped_cycles_counter: int | None = None,
    orch: Any | None = None,
) -> tuple[str, TradeDirection, dict] | None:
    """Ultimo recurso de execucao obrigatoria usando raw_prob ou CALL padrao."""
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
                skipped_cycles_counter=skipped_cycles_counter,
                orch=orch,
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
    last_loss_direction: str | None,
    skip_symbols: frozenset[str] | None = None,
    min_signal: float = 0.0,
    min_val: float = 0.0,
    consecutive_losses: int = 0,
    mean_reversion_enabled: bool = True,
    low_accuracy_enabled: bool = True,
    skipped_cycles_counter: int | None = None,
    orch: Any | None = None,
) -> tuple[str, TradeDirection, dict] | None:
    """Garante ordem em modo obrigatorio quando o pool DL fica vazio."""
    ranked = pick_best_mandatory_candidate(
        trade_symbols,
        decisions,
        recovery_active=recovery_active,
        last_loss_symbol=last_loss_symbol,
        last_loss_direction=last_loss_direction,
        skip_symbols=skip_symbols,
        min_signal=min_signal,
        min_val=min_val,
        consecutive_losses=consecutive_losses,
        mean_reversion_enabled=mean_reversion_enabled,
        low_accuracy_enabled=low_accuracy_enabled,
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
        mean_reversion_enabled=mean_reversion_enabled,
        low_accuracy_enabled=low_accuracy_enabled,
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
        mean_reversion_enabled=mean_reversion_enabled,
        low_accuracy_enabled=low_accuracy_enabled,
        skipped_cycles_counter=skipped_cycles_counter,
        orch=orch,
    )
