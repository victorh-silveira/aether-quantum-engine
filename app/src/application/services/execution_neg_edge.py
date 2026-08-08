"""Soft Kelly quando o Edge calibrado do lado executado esta abaixo do piso SSOT."""

from __future__ import annotations

import logging
from typing import Any

from src.application.services.execution_signal_skip import apply_kelly_soft
from src.application.services.log_dedupe import log_info_if_changed
from src.application.services.market_audit_log_helpers import resolve_predicted_edge
from src.domain.config_knobs import merge_settings_block, require_float, require_keys


logger = logging.getLogger("AETH")


def parse_neg_edge_soft_config(raw: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve neg_edge_soft_kelly_mult em orchestrator.execution.signal_skip."""
    block = merge_settings_block(("orchestrator", "execution", "signal_skip"), raw)
    require_keys(block, ("neg_edge_soft_kelly_mult",), "orchestrator.execution.signal_skip")
    soft_mult = require_float(block, "neg_edge_soft_kelly_mult")
    if soft_mult <= 0.0 or soft_mult > 1.0:
        raise ValueError("orchestrator.execution.signal_skip.neg_edge_soft_kelly_mult deve estar em (0, 1]")
    return {"neg_edge_soft_kelly_mult": soft_mult}


def _payout_from_orch(orch: Any | None) -> float:
    """Le payout_estimate do risco SSOT; fallback operacional 0.72."""
    if orch is None:
        return 0.72
    config = getattr(orch, "config", None)
    if not isinstance(config, dict):
        return 0.72
    risk = config.get("risk_management") if isinstance(config.get("risk_management"), dict) else {}
    params = risk.get("params") if isinstance(risk.get("params"), dict) else {}
    try:
        return max(0.01, float(params.get("payout_estimate", 0.72)))
    except (TypeError, ValueError):
        return 0.72


def _min_edge_from_orch(orch: Any | None) -> float:
    """Le deep_learning.min_edge_execute; default 0.0."""
    if orch is None:
        return 0.0
    config = getattr(orch, "config", None)
    if not isinstance(config, dict):
        return 0.0
    dl = config.get("deep_learning") if isinstance(config.get("deep_learning"), dict) else {}
    try:
        return max(0.0, float(dl.get("min_edge_execute", 0.0)))
    except (TypeError, ValueError):
        return 0.0


def _soft_mult_from_orch(orch: Any | None, override: float | None = None) -> float:
    """Le neg_edge_soft_kelly_mult do SSOT signal_skip."""
    if override is not None:
        return max(0.05, min(1.0, float(override)))
    if orch is None:
        return float(parse_neg_edge_soft_config(None)["neg_edge_soft_kelly_mult"])
    config = getattr(orch, "config", None)
    raw = None
    if isinstance(config, dict):
        orch_ex = config.get("orchestrator", {})
        if isinstance(orch_ex, dict):
            ex = orch_ex.get("execution", {})
            if isinstance(ex, dict):
                raw = ex.get("signal_skip")
    return float(parse_neg_edge_soft_config(raw if isinstance(raw, dict) else None)["neg_edge_soft_kelly_mult"])


def apply_negative_cal_edge_pause(
    metrics: dict[str, Any],
    *,
    orch: Any | None = None,
    force: bool = False,
    min_edge: float | None = None,
    payout: float | None = None,
    soft_mult: float | None = None,
) -> bool:
    """Soft Kelly quando Edge Cal do lado < min_edge_execute. True se atenuou."""
    if force:
        return False
    if metrics.get("execution_candidate_ready") is False:
        return False
    status = str(metrics.get("signal_status") or "").strip().upper()
    if status == "SKIP" or status.startswith("SKIP:"):
        return False
    direction = str(metrics.get("exec_direction") or metrics.get("resolved_direction") or "").upper()
    if direction not in {"CALL", "PUT"}:
        return False
    pay = float(payout) if payout is not None else _payout_from_orch(orch)
    floor = float(min_edge) if min_edge is not None else _min_edge_from_orch(orch)
    edge = float(resolve_predicted_edge(metrics, direction=direction, payout=pay))
    metrics["cal_side_edge"] = edge
    metrics["cal_side_edge_floor"] = floor
    if edge + 1e-12 >= floor:
        return False
    mult = _soft_mult_from_orch(orch, soft_mult)
    apply_kelly_soft(metrics, mult, waived="neg_edge_soft", flag="neg_edge_soft")
    metrics.pop("neg_edge_pause", None)
    if orch is not None:
        log_info_if_changed(
            orch,
            logger,
            "neg_edge_soft",
            f"{direction}:{edge:.4f}:{floor:.4f}:{mult:.2f}",
            "EDGE || NEG_SOFT side=%s edge=%+.4f floor=%.4f kelly_mult=%.2f",
            direction,
            edge,
            floor,
            mult,
        )
    return True
