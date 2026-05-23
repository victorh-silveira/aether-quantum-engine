"""Sinais Medallion quantitativos e selecao de trades para backtest."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from src.application.services.llm.cluster_direction import cluster_direction_from_tag
from src.application.services.llm.cluster_statarb_select import (
    resolve_statarb_cluster_config,
    select_cluster_symbols_by_statarb,
)
from src.application.services.llm.global_macro_confluence import MacroSnapshot
from src.application.services.llm.llm_cluster_exclusive import (
    cluster_region_for_symbol,
    exclusive_cluster_by_macro_enabled,
    resolve_exclusive_cluster_region,
)
from src.application.services.llm.llm_macro_confluence_guards import apply_macro_confluence_guard
from src.application.services.llm.macro_cluster_align import cluster_trade_direction
from src.application.services.llm.macro_config import resolve_macro_config
from src.domain.models.trade import TradeDirection


@dataclass(frozen=True)
class BacktestOrder:
    """Ordem simulada em uma barra M15."""

    bar_index: int
    symbol: str
    direction: TradeDirection
    conviction: float
    macro_tag: str
    active_region: str
    index_note: str


def derive_quant_cluster_tags(
    snapshot: MacroSnapshot, macro_cfg: dict[str, Any] | None
) -> tuple[str | None, str | None]:
    """Surrogate quant para US_CLUSTER e EU_CLUSTER (sem LLM)."""
    cfg = resolve_macro_config(macro_cfg if isinstance(macro_cfg, dict) else None)
    floor = float(cfg["confluence_conviction_floor"])
    tag = snapshot.tag
    us_s = float(snapshot.us_strength)
    eu_s = float(snapshot.eu_strength)

    if tag == "risk_on" and min(us_s, eu_s) >= floor:
        return "CALL", "CALL"
    if tag == "risk_off" and min(us_s, eu_s) >= floor:
        return "PUT", "PUT"

    us_tok = cluster_trade_direction(snapshot.us_dir)
    eu_tok = cluster_trade_direction(snapshot.eu_dir)
    us_out = us_tok if us_tok and us_s >= floor else None
    eu_out = eu_tok if eu_tok and eu_s >= floor else None
    return us_out, eu_out


def _base_conviction(snapshot: MacroSnapshot) -> float:
    """Conviccao base a partir das forcas de cluster."""
    return max(float(snapshot.us_strength), float(snapshot.eu_strength), 0.55)


def _exclusive_enabled(config: dict[str, Any]) -> bool:
    """Le flag exclusive_cluster_by_macro da config."""

    class _Cfg:
        def __init__(self, c: dict[str, Any]):
            self.config = c

    return exclusive_cluster_by_macro_enabled(_Cfg(config))


def _cluster_allowed(
    *,
    candidates: list[str],
    tag: str | None,
    spreads_map: dict[str, float],
    hmm_state: int,
    statarb_cfg: dict[str, Any],
) -> tuple[TradeDirection | None, set[str], str]:
    """Direcao e indices permitidos para um cluster."""
    direction, _ = cluster_direction_from_tag(tag)
    if direction is None:
        return None, set(), ""
    allowed, note = select_cluster_symbols_by_statarb(
        candidates, direction, spreads_map, hmm_state=hmm_state, cfg=statarb_cfg
    )
    return direction, allowed, note


def _try_order(
    *,
    bar_index: int,
    sym: str,
    target_direction: TradeDirection | None,
    allowed: set[str],
    note: str,
    snapshot: MacroSnapshot,
    macro_cfg: dict[str, Any] | None,
    metrics: dict[str, Any],
    min_conv: float,
    active_region: str,
) -> BacktestOrder | None:
    """Monta ordem se guardrails e conviccao permitirem."""
    if target_direction is None or sym not in allowed:
        return None
    conviction = float(metrics["conviction"])
    direction, conviction, _, _, execute = apply_macro_confluence_guard(
        target_direction,
        conviction,
        snapshot,
        macro_cfg if isinstance(macro_cfg, dict) else None,
        sym=sym,
    )
    if not execute or direction is None or conviction < min_conv:
        return None
    return BacktestOrder(
        bar_index=bar_index,
        symbol=sym,
        direction=direction,
        conviction=conviction,
        macro_tag=snapshot.tag,
        active_region=active_region,
        index_note=note,
    )


def resolve_orders_from_cluster_tags(
    *,
    bar_index: int,
    snapshot: MacroSnapshot,
    config: dict[str, Any],
    us_symbols: list[str],
    eu_symbols: list[str],
    all_symbols: list[str],
    anchor: str,
    us_tag: str | None,
    eu_tag: str | None,
    conviction: float,
) -> list[BacktestOrder]:
    """Gera ordens a partir de tags US/EU (LLM ou surrogate quant)."""
    strategy = config.get("strategy", {})
    corr_cfg = strategy.get("correlation") if isinstance(strategy.get("correlation"), dict) else {}
    macro_cfg = strategy.get("macro") if isinstance(strategy.get("macro"), dict) else {}
    llm_cfg = config.get("llm") if isinstance(config.get("llm"), dict) else {}
    min_conv = float(llm_cfg.get("min_conviction_execute", 0.60))

    us_targets = tuple(us_symbols)
    eu_targets = tuple(eu_symbols)
    metrics: dict[str, Any] = {
        "macro_sentiment": snapshot.tag,
        "macro_confluence_tag": snapshot.tag,
        "macro_us_strength_quant": snapshot.us_strength,
        "macro_eu_strength_quant": snapshot.eu_strength,
        "us_cluster": us_tag,
        "eu_cluster": eu_tag,
        "statarb_spreads": dict(snapshot.statarb_spreads or {}),
        "hmm_state": int(snapshot.hmm_state),
        "hmm_prob": float(snapshot.hmm_prob),
        "conviction": conviction,
    }

    exclusive = _exclusive_enabled(config)
    active_region = resolve_exclusive_cluster_region(metrics) if exclusive else None
    if exclusive and active_region is None:
        return []

    statarb_cfg = resolve_statarb_cluster_config(corr_cfg, macro_cfg)
    spreads_map = dict(snapshot.statarb_spreads or {})
    hmm_state = int(snapshot.hmm_state)
    anchor_in_us = anchor in us_targets
    anchor_in_eu = anchor in eu_targets

    us_cands = [s for s in all_symbols if s in us_targets and s != anchor and not anchor_in_us]
    eu_cands = [s for s in all_symbols if s in eu_targets and s != anchor and not anchor_in_eu]
    us_dir, us_allowed, us_note = _cluster_allowed(
        candidates=us_cands, tag=us_tag, spreads_map=spreads_map, hmm_state=hmm_state, statarb_cfg=statarb_cfg
    )
    eu_dir, eu_allowed, eu_note = _cluster_allowed(
        candidates=eu_cands, tag=eu_tag, spreads_map=spreads_map, hmm_state=hmm_state, statarb_cfg=statarb_cfg
    )

    region_label = active_region or ""
    for sym in all_symbols:
        if sym == anchor:
            continue
        sym_region = cluster_region_for_symbol(sym, us_targets=us_targets, eu_targets=eu_targets)
        if exclusive and active_region and sym_region and sym_region != active_region:
            continue
        if sym in us_targets and not anchor_in_us:
            target_direction, allowed, note = us_dir, us_allowed, us_note
        elif sym in eu_targets and not anchor_in_eu:
            target_direction, allowed, note = eu_dir, eu_allowed, eu_note
        else:
            continue
        order = _try_order(
            bar_index=bar_index,
            sym=sym,
            target_direction=target_direction,
            allowed=allowed,
            note=note,
            snapshot=snapshot,
            macro_cfg=macro_cfg if isinstance(macro_cfg, dict) else None,
            metrics=metrics,
            min_conv=min_conv,
            active_region=region_label,
        )
        if order is not None:
            return [order]
    return []


def resolve_orders_at_bar(
    *,
    bar_index: int,
    snapshot: MacroSnapshot,
    config: dict[str, Any],
    us_symbols: list[str],
    eu_symbols: list[str],
    all_symbols: list[str],
    anchor: str,
) -> list[BacktestOrder]:
    """Gera ordens do ciclo Medallion na barra (surrogate quant, sem EURUSD)."""
    macro_cfg = config.get("strategy", {}).get("macro") if isinstance(config.get("strategy"), dict) else {}
    us_tag, eu_tag = derive_quant_cluster_tags(snapshot, macro_cfg if isinstance(macro_cfg, dict) else None)
    return resolve_orders_from_cluster_tags(
        bar_index=bar_index,
        snapshot=snapshot,
        config=config,
        us_symbols=us_symbols,
        eu_symbols=eu_symbols,
        all_symbols=all_symbols,
        anchor=anchor,
        us_tag=us_tag,
        eu_tag=eu_tag,
        conviction=_base_conviction(snapshot),
    )
