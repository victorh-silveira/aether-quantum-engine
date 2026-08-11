"""Inverte CALL/PUT so quando a fusao escolhe ev_call (experimento SSOT)."""

from __future__ import annotations

from typing import Any

from src.domain.config_knobs import merge_settings_block, require_bool
from src.domain.models.trade import TradeDirection


def invert_exec_side_enabled(raw: dict[str, Any] | None = None) -> bool:
    """Le orchestrator.execution.invert_exec_side do SSOT."""
    block = merge_settings_block(("orchestrator", "execution"), raw if isinstance(raw, dict) else None)
    if "invert_exec_side" not in block:
        return False
    return require_bool(block, "invert_exec_side")


def _invert_from_orch(orch: Any | None) -> bool:
    """Resolve invert_exec_side a partir do orch.config."""
    if orch is None:
        return invert_exec_side_enabled(None)
    config = getattr(orch, "config", None)
    if not isinstance(config, dict):
        return invert_exec_side_enabled(None)
    orch_block = config.get("orchestrator")
    if not isinstance(orch_block, dict):
        return invert_exec_side_enabled(None)
    execution = orch_block.get("execution")
    return invert_exec_side_enabled(execution if isinstance(execution, dict) else None)


def _should_invert_for_fusion(metrics: dict[str, Any], exec_dir: TradeDirection) -> bool:
    """True so se fusao escolheu CALL por EV (ev_call); preserva ev_put."""
    if not bool(metrics.get("fusion_applied")):
        metrics["invert_skipped_reason"] = "no_fusion"
        return False
    why = str(metrics.get("fusion_reason") or "").strip().lower()
    if why != "ev_call":
        metrics["invert_skipped_reason"] = why or "not_ev_call"
        return False
    if exec_dir != TradeDirection.CALL:
        metrics["invert_skipped_reason"] = "side_not_call"
        return False
    return True


def apply_invert_exec_side(
    metrics: dict[str, Any],
    exec_dir: TradeDirection,
    *,
    orch: Any | None = None,
) -> TradeDirection:
    """Inverte CALL->PUT quando fusao ev_call; nao toca em ev_put."""
    if not _invert_from_orch(orch):
        return exec_dir
    metrics.setdefault("fusion_side_pre_invert", exec_dir.name)
    if not _should_invert_for_fusion(metrics, exec_dir):
        metrics["invert_exec_side"] = False
        return exec_dir
    flipped = TradeDirection.PUT
    metrics["invert_exec_side"] = True
    metrics["invert_from"] = exec_dir.name
    metrics["exec_direction"] = flipped.name
    metrics["resolved_direction"] = flipped.name
    gate = str(metrics.get("gate_reason") or "").strip().lower()
    status = str(metrics.get("signal_status") or "").strip().upper()
    if gate == "neg_edge" or status.startswith("SKIP:NEG_EDGE"):
        metrics["execution_candidate_ready"] = True
        metrics.pop("gate_reason", None)
        metrics.pop("signal_status", None)
        metrics.pop("neg_edge_bootstrap_deep", None)
        metrics.pop("neg_edge_soft", None)
        metrics["invert_cleared_signal_empty"] = True
    return flipped
