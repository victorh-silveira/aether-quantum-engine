"""Sinais Medallion e propagacao de cluster (mesmo pipeline do motor live)."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from scripts.backtest.backtest_cluster_runtime import BacktestClusterRuntime
from src.application.services.llm.cluster_refresh_execute_policy import cluster_refresh_may_execute
from src.application.services.llm.global_macro_confluence import MacroSnapshot
from src.application.services.llm.llm_cluster_propagate import propagate_cluster_decisions
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
    """Surrogate quant para US_CLUSTER e EU_CLUSTER (modo sem API Gemini)."""
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
    return max(float(snapshot.us_strength), float(snapshot.eu_strength), 0.55)


def _metrics_from_snapshot(
    snapshot: MacroSnapshot,
    *,
    us_tag: str | None,
    eu_tag: str | None,
    conviction: float,
) -> dict[str, Any]:
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
    m5_dirs = getattr(snapshot, "index_m5_dir_by_symbol", None)
    if isinstance(m5_dirs, dict) and m5_dirs:
        metrics["index_m5_dir_by_symbol"] = dict(m5_dirs)
    return metrics


def apply_backtest_refresh_execute_gate(
    runtime: BacktestClusterRuntime,
    decisions: dict[str, dict],
) -> None:
    refresh_without_llm = bool(getattr(runtime, "_cluster_refresh_without_llm", False))
    may_exec, _reason = cluster_refresh_may_execute(
        runtime,
        decisions,
        refresh_without_llm=refresh_without_llm,
    )
    if refresh_without_llm and not may_exec:
        for sym, entry in list(decisions.items()):
            if sym == runtime.anchor:
                continue
            if not isinstance(entry, dict):
                continue
            metrics = entry.get("metrics")
            if isinstance(metrics, dict):
                metrics["execute"] = False


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
    runtime: BacktestClusterRuntime,
) -> list[BacktestOrder]:
    _ = (config, us_symbols, eu_symbols, all_symbols)
    """Propaga tags US/EU com propagate_cluster_decisions (guardrails live)."""
    metrics = _metrics_from_snapshot(snapshot, us_tag=us_tag, eu_tag=eu_tag, conviction=conviction)
    decisions: dict[str, dict] = {}
    propagate_cluster_decisions(
        runtime,
        anchor_sym=anchor,
        direction=TradeDirection.CALL,
        metrics=metrics,
        decisions=decisions,
        cid=f"C{bar_index:04d}",
    )
    apply_backtest_refresh_execute_gate(runtime, decisions)
    region = str(metrics.get("cluster_active_region") or "")
    orders: list[BacktestOrder] = []
    for sym, entry in decisions.items():
        if sym == anchor:
            continue
        m = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
        if not m.get("execute"):
            continue
        direction = entry.get("direction")
        if not isinstance(direction, TradeDirection):
            continue
        note = str(m.get("llm_note") or "")
        orders.append(
            BacktestOrder(
                bar_index=bar_index,
                symbol=sym,
                direction=direction,
                conviction=float(m.get("conviction", conviction)),
                macro_tag=snapshot.tag,
                active_region=region,
                index_note=note,
            )
        )
    if len(orders) > 1:
        orders.sort(key=lambda o: o.conviction, reverse=True)
        return orders[:1]
    return orders


def resolve_orders_at_bar(
    *,
    bar_index: int,
    snapshot: MacroSnapshot,
    config: dict[str, Any],
    us_symbols: list[str],
    eu_symbols: list[str],
    all_symbols: list[str],
    anchor: str,
    runtime: BacktestClusterRuntime,
) -> list[BacktestOrder]:
    """Gera ordens na barra via surrogate quant e pipeline de cluster live."""
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
        runtime=runtime,
    )
