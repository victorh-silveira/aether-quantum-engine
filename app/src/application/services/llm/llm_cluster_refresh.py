"""Repropaga cluster com macro/StatArb frescos mantendo tags LLM em cache."""

from __future__ import annotations

import time
from typing import Any

from src.application.services.llm.llm_cluster_propagate import propagate_cluster_decisions
from src.application.services.llm.macro_config import MacroSnapshot
from src.domain.models.trade import TradeDirection


def resolve_cluster_refresh_interval_seconds(config: dict[str, Any]) -> int:
    """Intervalo de reanalise do cluster; default = cycle_interval_seconds do orquestrador."""
    orch_cfg = config.get("orchestrator") if isinstance(config.get("orchestrator"), dict) else {}
    explicit = orch_cfg.get("cluster_refresh_interval_seconds")
    if explicit is not None:
        return max(0, int(explicit))
    return max(0, int(orch_cfg.get("cycle_interval_seconds") or 0))


def cluster_refresh_due(
    orch: Any,
    *,
    now_epoch: float,
    interval_seconds: int,
) -> bool:
    """True quando passou o intervalo desde o ultimo refresh do cluster."""
    if interval_seconds <= 0:
        return False
    last = getattr(orch, "_last_cluster_refresh_epoch", None)
    if last is None:
        return True
    return (now_epoch - float(last)) >= float(interval_seconds)


def merge_macro_snapshot_into_metrics(metrics: dict[str, Any], snapshot: MacroSnapshot) -> dict[str, Any]:
    """Atualiza metricas da ancora com macro e StatArb do snapshot atual."""
    out = dict(metrics)
    out["macro_sentiment"] = snapshot.tag
    out["macro_confluence_tag"] = snapshot.tag
    out["eurusd_bias_quant"] = snapshot.eurusd_bias
    out["macro_us_dir_quant"] = snapshot.us_dir
    out["macro_eu_dir_quant"] = snapshot.eu_dir
    out["macro_us_strength_quant"] = float(snapshot.us_strength)
    out["macro_eu_strength_quant"] = float(snapshot.eu_strength)
    out["statarb_spreads"] = dict(snapshot.statarb_spreads or {})
    out["hmm_state"] = int(getattr(snapshot, "hmm_state", 0))
    out["hmm_prob"] = float(getattr(snapshot, "hmm_prob", 1.0))
    return out


def refresh_cluster_decisions_from_cache(
    orch: Any,
    macro_snapshot: MacroSnapshot,
    cid: str,
) -> dict[str, dict] | None:
    """Repropaga cluster com tags LLM em cache e macro/StatArb atualizados."""
    cached = getattr(orch, "_last_llm_decisions", None)
    if not isinstance(cached, dict) or not cached:
        return None
    anchor_sym = orch.anchor
    entry = cached.get(anchor_sym)
    if not isinstance(entry, dict):
        return dict(cached)
    direction = entry.get("direction")
    if not isinstance(direction, TradeDirection):
        return dict(cached)
    raw_metrics = entry.get("metrics")
    metrics = merge_macro_snapshot_into_metrics(
        dict(raw_metrics) if isinstance(raw_metrics, dict) else {},
        macro_snapshot,
    )
    decisions: dict[str, dict] = {anchor_sym: {"direction": direction, "metrics": metrics}}
    propagate_cluster_decisions(
        orch,
        anchor_sym=anchor_sym,
        direction=direction,
        metrics=metrics,
        decisions=decisions,
        cid=cid,
    )
    now_epoch = time.time()
    orch._last_llm_decisions = dict(decisions)
    orch._last_llm_macro_tag = macro_snapshot.tag
    orch._last_cluster_refresh_epoch = now_epoch
    orch.logger.info(
        "[%s] CLUSTER_REFRESH macro=%s | us=%s eu=%s",
        cid,
        macro_snapshot.tag,
        metrics.get("us_cluster"),
        metrics.get("eu_cluster"),
    )
    return decisions
