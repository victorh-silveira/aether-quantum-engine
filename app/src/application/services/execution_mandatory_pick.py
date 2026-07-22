"""Selecao obrigatoria de candidatos por ranking de mercado."""

from typing import Any

from src.application.services.execution_direction import build_execution_candidate, meets_mandatory_signal_floor
from src.application.services.execution_market_rank import (
    _trade_score,
    build_market_execution_candidate,
    mandatory_pool_eligible,
    market_decision_score,
)
from src.domain.models.trade import TradeDirection
from src.domain.symbols.drift_symbols import TRADING_SYMBOLS


def _symbol_order(trade_symbols: list[str], last_loss_symbol: str | None, *, skip_symbols: frozenset[str]) -> list[str]:
    """Ordena simbolos priorizando core e diversificacao apos loss."""
    eligible = [symbol for symbol in trade_symbols if symbol not in skip_symbols]
    core = [symbol for symbol in TRADING_SYMBOLS if symbol in eligible]
    alt = [symbol for symbol in core if symbol != last_loss_symbol]
    if alt:
        core = alt + [symbol for symbol in core if symbol not in alt]
    tail = [symbol for symbol in eligible if symbol not in core and symbol != last_loss_symbol]
    if not tail:
        tail = [symbol for symbol in eligible if symbol not in core]
    return core + tail


def _candidate_kwargs(
    *,
    recovery_active: bool,
    orch: Any | None,
    cycle_id: int,
    risk_manager: Any | None,
    skipped_cycles_counter: int | None,
    infra_cfg: dict | None,
    decisions: dict | None,
    exec_cfg: dict | None,
) -> dict:
    """Monta kwargs comuns para resolucao de candidato obrigatorio."""
    return {
        "recovery_active": recovery_active,
        "orch": orch,
        "cycle_id": cycle_id,
        "risk_manager": risk_manager,
        "skipped_cycles_counter": skipped_cycles_counter,
        "infra_cfg": infra_cfg,
        "decisions": decisions,
        "exec_cfg": exec_cfg,
    }


def _rank_eligible_candidates(
    order: list[str],
    decisions: dict,
    *,
    recovery_active: bool,
    last_loss_symbol: str | None,
    min_signal: float,
    min_val: float,
    orch: Any | None = None,
    cycle_id: int = 0,
    risk_manager: Any | None = None,
    skipped_cycles_counter: int | None = None,
    infra_cfg: dict | None = None,
    exec_cfg: dict | None = None,
) -> tuple[str, TradeDirection, dict] | None:
    """Rankeia candidatos elegiveis e retorna o melhor por score de mercado."""
    best = None
    best_score = -1.0
    kw = _candidate_kwargs(
        recovery_active=recovery_active,
        orch=orch,
        cycle_id=cycle_id,
        risk_manager=risk_manager,
        skipped_cycles_counter=skipped_cycles_counter,
        infra_cfg=infra_cfg,
        decisions=decisions,
        exec_cfg=exec_cfg,
    )
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
        candidate = build_market_execution_candidate(symbol, entry, **kw)
        if candidate is None:
            candidate = build_execution_candidate(symbol, entry, **kw)
        if candidate is None:
            continue
        rank = market_decision_score(
            candidate[2],
            exec_direction=candidate[1],
            recovery_active=recovery_active,
            symbol=symbol,
            last_loss_symbol=last_loss_symbol,
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
    skip_symbols: frozenset[str] | None = None,
    min_signal: float = 0.0,
    min_val: float = 0.0,
    consecutive_losses: int = 0,
    orch: Any | None = None,
    cycle_id: int = 0,
    risk_manager: Any | None = None,
    skipped_cycles_counter: int | None = None,
    infra_cfg: dict | None = None,
    exec_cfg: dict | None = None,
) -> tuple[str, TradeDirection, dict] | None:
    """Escolhe melhor candidato obrigatorio por score de mercado."""
    _ = consecutive_losses
    skip = skip_symbols or frozenset()
    order = _symbol_order(trade_symbols, last_loss_symbol, skip_symbols=skip)
    ranked = _rank_eligible_candidates(
        order,
        decisions,
        recovery_active=recovery_active,
        last_loss_symbol=last_loss_symbol,
        min_signal=min_signal,
        min_val=min_val,
        orch=orch,
        cycle_id=cycle_id,
        risk_manager=risk_manager,
        skipped_cycles_counter=skipped_cycles_counter,
        infra_cfg=infra_cfg,
        exec_cfg=exec_cfg,
    )
    if ranked is not None:
        return ranked
    return pick_absolute_mandatory_candidate(
        trade_symbols,
        decisions,
        recovery_active=recovery_active,
        last_loss_symbol=last_loss_symbol,
        min_signal=min_signal,
        min_val=min_val,
        orch=orch,
        cycle_id=cycle_id,
        risk_manager=risk_manager,
        skipped_cycles_counter=skipped_cycles_counter,
        infra_cfg=infra_cfg,
        exec_cfg=exec_cfg,
    )


def pick_absolute_mandatory_candidate(
    trade_symbols: list[str],
    decisions: dict,
    *,
    recovery_active: bool,
    last_loss_symbol: str | None,
    min_signal: float = 0.0,
    min_val: float = 0.0,
    consecutive_losses: int = 0,
    orch: Any | None = None,
    cycle_id: int = 0,
    risk_manager: Any | None = None,
    skipped_cycles_counter: int | None = None,
    infra_cfg: dict | None = None,
    exec_cfg: dict | None = None,
) -> tuple[str, TradeDirection, dict] | None:
    """Garante ordem quando filtros de recovery esgotam o pool."""
    _ = consecutive_losses
    order = _symbol_order(trade_symbols, last_loss_symbol, skip_symbols=frozenset())
    best = None
    best_score = -1.0
    kw = _candidate_kwargs(
        recovery_active=recovery_active,
        orch=orch,
        cycle_id=cycle_id,
        risk_manager=risk_manager,
        skipped_cycles_counter=skipped_cycles_counter,
        infra_cfg=infra_cfg,
        decisions=decisions,
        exec_cfg=exec_cfg,
    )
    for symbol in order:
        entry = decisions.get(symbol)
        if not entry or not mandatory_pool_eligible(entry):
            continue
        metrics = entry.get("metrics") or {}
        if (min_signal > 0.0 or min_val > 0.0) and not meets_mandatory_signal_floor(
            metrics, min_signal=min_signal, min_val=min_val
        ):
            continue
        candidate = build_market_execution_candidate(symbol, entry, **kw)
        if candidate is None:
            candidate = build_execution_candidate(symbol, entry, **kw)
        if candidate is None:
            continue
        rank = market_decision_score(
            candidate[2],
            exec_direction=candidate[1],
            recovery_active=recovery_active,
            symbol=symbol,
            last_loss_symbol=last_loss_symbol,
        )
        if rank > best_score:
            best_score = rank
            best = candidate
    return best
