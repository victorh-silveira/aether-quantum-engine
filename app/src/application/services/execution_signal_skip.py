"""Catalogo minimo de SKIP de sinal (escopo 1.1, mandato explicito)."""

from __future__ import annotations

from typing import Any

from src.domain.config_knobs import merge_settings_block, require_bool, require_float, require_keys
from src.domain.models.trade import TradeDirection


_VALID = {TradeDirection.CALL.name, TradeDirection.PUT.name}
_REASON_CAL_MARGIN = "cal_margin"
_REASON_MINI_PAIR = "mini_pair_oppose"
SIGNAL_SKIP_REASONS = frozenset({_REASON_CAL_MARGIN, _REASON_MINI_PAIR})


def parse_signal_skip_config(raw: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve orchestrator.execution.signal_skip com merge SSOT."""
    block = merge_settings_block(("orchestrator", "execution", "signal_skip"), raw)
    require_keys(
        block,
        (
            "enabled",
            "min_direction_margin",
            "waive_margin_on_pending",
            "mini_pair_oppose_exec",
            "pending_dust",
        ),
        "orchestrator.execution.signal_skip",
    )
    return {
        "enabled": require_bool(block, "enabled"),
        "min_direction_margin": require_float(block, "min_direction_margin"),
        "waive_margin_on_pending": require_bool(block, "waive_margin_on_pending"),
        "mini_pair_oppose_exec": require_bool(block, "mini_pair_oppose_exec"),
        "pending_dust": require_float(block, "pending_dust"),
    }


def _side(value: object) -> str | None:
    """Normaliza CALL/PUT ou None."""
    side = str(value or "").strip().upper()
    return side if side in _VALID else None


def _pending_total(metrics: dict[str, Any], orch: Any | None) -> float:
    """Le pending material de metrics ou RiskManager do orquestrador."""
    for key in ("pending_loss_total", "pending_total", "recovery_pending"):
        raw = metrics.get(key)
        if raw is None:
            continue
        try:
            return max(0.0, float(raw))
        except (TypeError, ValueError):
            continue
    if orch is None:
        return 0.0
    risk_manager = getattr(orch, "risk_manager", None)
    if risk_manager is None:
        return 0.0
    total_fn = getattr(risk_manager, "pending_loss_total", None)
    if callable(total_fn):
        try:
            return max(0.0, float(total_fn()))
        except (TypeError, ValueError):
            return 0.0
    pending_map = getattr(risk_manager, "pending_loss", None)
    if isinstance(pending_map, dict):
        try:
            return max(0.0, float(sum(pending_map.values())))
        except (TypeError, ValueError):
            return 0.0
    return 0.0


def _mini_pair_opposes_exec(metrics: dict[str, Any], exec_dir: TradeDirection) -> bool:
    """True se par MINI unanime existe e discrepa do lado executado."""
    prev = _side(metrics.get("scale_mini_prev_bar_dir"))
    curr = _side(metrics.get("scale_mini_bar_dir"))
    if prev is None or curr is None or prev != curr:
        return False
    return prev != exec_dir.name


def _stamp_skip(metrics: dict[str, Any], reason: str) -> None:
    """Marca candidato como SKIP de sinal nomeado."""
    metrics["execution_candidate_ready"] = False
    metrics["gate_reason"] = reason
    metrics["signal_skip_reason"] = reason
    metrics["signal_status"] = f"SKIP:{reason.upper()}"


def apply_signal_skip_gates(
    metrics: dict[str, Any],
    exec_dir: TradeDirection,
    *,
    orch: Any | None = None,
    force: bool = False,
    cfg: dict[str, Any] | None = None,
) -> bool:
    """Aplica catálogo minimo; True se SKIP. Force-trade bypassa."""
    metrics.setdefault("signal_skip_reason", None)
    if force:
        return False
    vision = cfg if isinstance(cfg, dict) else parse_signal_skip_config(None)
    if not bool(vision.get("enabled", True)):
        return False
    if bool(vision.get("mini_pair_oppose_exec", True)) and _mini_pair_opposes_exec(metrics, exec_dir):
        _stamp_skip(metrics, _REASON_MINI_PAIR)
        return True
    floor = float(vision.get("min_direction_margin", 0.022))
    try:
        margin = float(metrics.get("direction_margin") or 0.0)
    except (TypeError, ValueError):
        margin = 0.0
    if margin + 1e-12 >= floor:
        return False
    pending = _pending_total(metrics, orch)
    dust = float(vision.get("pending_dust", 0.25))
    if bool(vision.get("waive_margin_on_pending", True)) and pending + 1e-12 >= dust:
        metrics["signal_skip_waived"] = "cal_margin_pending"
        return False
    _stamp_skip(metrics, _REASON_CAL_MARGIN)
    return True
