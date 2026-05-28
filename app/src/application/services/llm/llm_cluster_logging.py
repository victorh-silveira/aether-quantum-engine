"""Logs de propagacao CLUSTER_PROP, CLUSTER_INVERT e CLUSTER_BLOCK."""

from __future__ import annotations

from typing import Any

from src.domain.models.trade import TradeDirection


def log_cluster_propagation_results(
    orch: Any,
    *,
    cid: str,
    anchor_sym: str,
    corr_cfg: dict[str, Any],
    macro_tag: str,
    active_region: str | None,
    us_dir: TradeDirection | None,
    eu_dir: TradeDirection | None,
    us_note: str,
    eu_note: str,
    propagated_tags: list[str],
    blocked_tags: list[str],
    inverted_tags: list[str],
) -> None:
    """Emite logs CLUSTER_INVERT, CLUSTER_PROP, CLUSTER_BLOCK ou vazio."""
    if inverted_tags:
        orch.logger.info(
            "[%s] CLUSTER_INVERT || %s >> %s",
            cid,
            anchor_sym,
            ", ".join(inverted_tags),
        )
    if propagated_tags:
        label = "CLUSTER_BEST" if bool(corr_cfg.get("best_symbol_only", False)) else "CLUSTER_PROP"
        orch.logger.info(
            "[%s] %s || %s >> %s",
            cid,
            label,
            anchor_sym,
            ", ".join(propagated_tags),
        )
    if blocked_tags:
        orch.logger.info(
            "[%s] CLUSTER_BLOCK || %s >> %s",
            cid,
            anchor_sym,
            ", ".join(blocked_tags),
        )
    elif not propagated_tags and not blocked_tags and (us_dir is not None or eu_dir is not None):
        orch.logger.info(
            "[%s] CLUSTER_PROP vazio || macro=%s exclusive=%s us=%s eu=%s | %s | %s",
            cid,
            macro_tag,
            active_region or "-",
            us_dir.name if us_dir else "-",
            eu_dir.name if eu_dir else "-",
            us_note or "-",
            eu_note or "-",
        )
