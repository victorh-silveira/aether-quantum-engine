"""Bloqueio ou soft Kelly quando Edge Cal do lado fica abaixo do piso SSOT."""

from __future__ import annotations

import logging
from typing import Any

from src.application.services.execution_signal_skip import apply_kelly_soft
from src.application.services.loss_classifier_flip import closed_micro_candle_side
from src.application.services.market_audit_log_helpers import resolve_predicted_edge
from src.domain.config_knobs import merge_settings_block, require_bool, require_float, require_keys


logger = logging.getLogger("AETH")


def parse_neg_edge_soft_config(raw: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve knobs neg_edge em orchestrator.execution.signal_skip."""
    block = merge_settings_block(("orchestrator", "execution", "signal_skip"), raw)
    require_keys(
        block,
        (
            "neg_edge_soft_kelly_mult",
            "neg_edge_hard_skip",
            "neg_edge_soft_when_closed_candle_agree",
            "neg_edge_soft_min_edge",
        ),
        "orchestrator.execution.signal_skip",
    )
    soft_mult = require_float(block, "neg_edge_soft_kelly_mult")
    if soft_mult <= 0.0 or soft_mult > 1.0:
        raise ValueError("orchestrator.execution.signal_skip.neg_edge_soft_kelly_mult deve estar em (0, 1]")
    soft_min = require_float(block, "neg_edge_soft_min_edge")
    if soft_min > 0.0 or soft_min < -1.0:
        raise ValueError("orchestrator.execution.signal_skip.neg_edge_soft_min_edge deve estar em [-1, 0]")
    return {
        "neg_edge_soft_kelly_mult": soft_mult,
        "neg_edge_hard_skip": require_bool(block, "neg_edge_hard_skip"),
        "neg_edge_soft_when_closed_candle_agree": require_bool(block, "neg_edge_soft_when_closed_candle_agree"),
        "neg_edge_soft_min_edge": soft_min,
    }


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


def _signal_skip_raw(orch: Any | None) -> dict[str, Any] | None:
    """Extrai bloco signal_skip do orch.config se presente."""
    if orch is None:
        return None
    config = getattr(orch, "config", None)
    if not isinstance(config, dict):
        return None
    orch_ex = config.get("orchestrator", {})
    if not isinstance(orch_ex, dict):
        return None
    ex = orch_ex.get("execution", {})
    if not isinstance(ex, dict):
        return None
    raw = ex.get("signal_skip")
    return raw if isinstance(raw, dict) else None


def _neg_edge_cfg(orch: Any | None) -> dict[str, Any]:
    """Resolve hard_skip + soft mult do SSOT (merge com override parcial do orch)."""
    return parse_neg_edge_soft_config(_signal_skip_raw(orch))


def _soft_mult_from_orch(orch: Any | None, override: float | None = None) -> float:
    """Le neg_edge_soft_kelly_mult do SSOT signal_skip."""
    if override is not None:
        return max(0.05, min(1.0, float(override)))
    return float(_neg_edge_cfg(orch)["neg_edge_soft_kelly_mult"])


def _apply_neg_edge_hard(metrics: dict[str, Any], *, direction: str, edge: float, floor: float) -> None:
    """Marca EXEC_EMPTY tecnico por Edge Cal abaixo do floor."""
    metrics["execution_candidate_ready"] = False
    metrics["gate_reason"] = "neg_edge"
    metrics["signal_status"] = "SKIP:NEG_EDGE"
    metrics.pop("neg_edge_soft", None)
    metrics.pop("neg_edge_soft_kelly_mult", None)
    metrics.pop("neg_edge_candle_soft", None)
    metrics.pop("neg_edge_p_ovr_soft", None)
    if str(metrics.get("signal_skip_waived") or "") == "neg_edge_soft":
        metrics.pop("signal_skip_waived", None)
    metrics.pop("neg_edge_pause", None)
    logger.debug(
        "EDGE || NEG_HARD side=%s edge=%+.4f floor=%.4f",
        direction,
        edge,
        floor,
    )


def apply_negative_cal_edge_pause(
    metrics: dict[str, Any],
    *,
    orch: Any | None = None,
    force: bool = False,
    min_edge: float | None = None,
    payout: float | None = None,
    soft_mult: float | None = None,
) -> bool:
    """Hard-skip (mandato) ou soft Kelly se Edge Cal do lado < min_edge_execute."""
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
    cfg = _neg_edge_cfg(orch)
    candle_agree = False
    if bool(cfg.get("neg_edge_soft_when_closed_candle_agree", False)):
        candle_agree = closed_micro_candle_side(metrics) == direction
    p_ovr_flip = bool(metrics.get("loss_clf_flip")) and (
        bool(metrics.get("loss_clf_flip_scale_p_override")) or bool(metrics.get("loss_clf_flip_seed_p_override"))
    )
    soft_min = float(cfg.get("neg_edge_soft_min_edge", -1.0))
    soft_too_deep = edge + 1e-12 < soft_min
    allow_soft = (candle_agree or p_ovr_flip) and not soft_too_deep
    if bool(cfg["neg_edge_hard_skip"]) and not allow_soft:
        _apply_neg_edge_hard(metrics, direction=direction, edge=edge, floor=floor)
        return True
    mult = _soft_mult_from_orch(orch, soft_mult)
    apply_kelly_soft(metrics, mult, waived="neg_edge_soft", flag="neg_edge_soft")
    if candle_agree:
        metrics["neg_edge_candle_soft"] = True
    if p_ovr_flip:
        metrics["neg_edge_p_ovr_soft"] = True
    metrics.pop("neg_edge_pause", None)
    if orch is not None:
        logger.debug(
            "EDGE || NEG_SOFT side=%s edge=%+.4f floor=%.4f kelly_mult=%.2f candle=%d",
            direction,
            edge,
            floor,
            mult,
            1 if candle_agree else 0,
        )
    return True
