"""Helpers de contexto binario e gates pos-predicao para dl_predict."""

from typing import Any

from src.application.services.deep_learning.dl_binary_direction import (
    apply_mean_reversion_override,
    binary_direction_veto,
    build_binary_context,
)
from src.application.services.deep_learning.dl_gating import strong_signal_bypasses_val_acc
from src.application.services.deep_learning.dl_outcomes import live_win_rate
from src.application.services.deep_learning.dl_regime import direction_aligns_with_regime
from src.domain.symbols.range_symbols import hedge_peer, sym_is_low_barrier


def prepare_binary_direction(
    symbol: str,
    direction,
    raw_prob: float,
    trade_score: float,
    prices,
    gran: int,
    pair_prices,
    open_,
    high,
    low,
    params: dict[str, Any],
):
    """Monta contexto binario e aplica override estatistico sobre a direcao do DL."""
    peer = hedge_peer(str(symbol))
    sym_is_bull = sym_is_low_barrier(str(symbol), peer) if peer else False
    binary_ctx = build_binary_context(
        prices,
        granularity=gran,
        pair_prices=pair_prices,
        sym_is_bull=sym_is_bull,
        open_=open_,
        high=high,
        low=low,
    )
    direction, stat_override, raw_prob = apply_mean_reversion_override(
        direction,
        raw_prob,
        binary_ctx,
        params,
    )
    if stat_override:
        trade_score = max(float(raw_prob), 1.0 - float(raw_prob))
    return direction, stat_override, raw_prob, trade_score, sym_is_bull, binary_ctx


def resolve_execution_gates(
    *,
    execute: bool,
    block: str | None,
    direction,
    prices,
    binary_ctx: dict,
    params: dict[str, Any],
    sym_is_bull: bool,
    orch,
    symbol: str,
) -> tuple[bool, str | None, float | None]:
    """Aplica gates de regime, sinal binario e live win rate apos o scoring do modelo."""
    regime_required = bool(params.get("require_regime_alignment", True))
    if (
        execute
        and regime_required
        and not direction_aligns_with_regime(
            direction,
            prices,
            min_strength=float(params.get("min_regime_strength", 0.0)),
            rsi_overbought=float(params.get("rsi_overbought_threshold", 1.01)),
            rsi_oversold=float(params.get("rsi_oversold_threshold", -0.01)),
        )
    ):
        return False, "regime", live_win_rate(orch, symbol)
    if execute:
        binary_block = binary_direction_veto(
            direction,
            binary_ctx,
            params,
            sym_is_bull=sym_is_bull,
        )
        if binary_block is not None:
            return False, binary_block, live_win_rate(orch, symbol)
    live_wr = live_win_rate(orch, symbol)
    if execute and live_wr is not None and live_wr + 1e-9 < float(params.get("min_live_win_rate", 0.42)):
        return False, "live_wr", live_wr
    return execute, block, live_wr


def val_accuracy_bypass_flag(
    *,
    execute: bool,
    val_accuracy: float,
    min_val_accuracy: float,
    allow_bypass: bool,
    raw_side: float,
    edge: float,
    params: dict[str, Any],
) -> bool:
    """Indica se sinal forte ou moderado contorna o piso de val_accuracy."""
    if not execute or val_accuracy + 1e-9 >= min_val_accuracy:
        return False
    strong = (
        params.get("bypass_min_conviction") is not None
        and params.get("bypass_min_edge") is not None
        and allow_bypass
        and strong_signal_bypasses_val_acc(
            raw_side,
            edge,
            bypass_min_conviction=params["bypass_min_conviction"],
            bypass_min_edge=params["bypass_min_edge"],
        )
    )
    moderate = (
        params.get("moderate_min_conviction") is not None
        and params.get("moderate_min_edge") is not None
        and allow_bypass
        and strong_signal_bypasses_val_acc(
            raw_side,
            edge,
            bypass_min_conviction=params["moderate_min_conviction"],
            bypass_min_edge=params["moderate_min_edge"],
        )
    )
    return bool(strong or moderate)
