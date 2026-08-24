"""Bloqueio ou soft Kelly quando Edge Cal do lado fica abaixo do piso SSOT."""

from __future__ import annotations

import logging
from typing import Any

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
            "neg_edge_bootstrap_soft_kelly_mult",
            "neg_edge_deep_edge_floor",
        ),
        "orchestrator.execution.signal_skip",
    )
    soft_mult = require_float(block, "neg_edge_soft_kelly_mult")
    if soft_mult <= 0.0 or soft_mult > 1.0:
        raise ValueError("orchestrator.execution.signal_skip.neg_edge_soft_kelly_mult deve estar em (0, 1]")
    soft_min = require_float(block, "neg_edge_soft_min_edge")
    if soft_min > 0.0 or soft_min < -1.0:
        raise ValueError("orchestrator.execution.signal_skip.neg_edge_soft_min_edge deve estar em [-1, 0]")
    boot_mult = require_float(block, "neg_edge_bootstrap_soft_kelly_mult")
    if boot_mult <= 0.0 or boot_mult > 1.0:
        raise ValueError("orchestrator.execution.signal_skip.neg_edge_bootstrap_soft_kelly_mult deve estar em (0, 1]")
    deep_floor = require_float(block, "neg_edge_deep_edge_floor")
    if deep_floor > 0.0 or deep_floor < -1.0:
        raise ValueError("orchestrator.execution.signal_skip.neg_edge_deep_edge_floor deve estar em [-1, 0]")
    return {
        "neg_edge_soft_kelly_mult": soft_mult,
        "neg_edge_hard_skip": require_bool(block, "neg_edge_hard_skip"),
        "neg_edge_soft_when_closed_candle_agree": require_bool(block, "neg_edge_soft_when_closed_candle_agree"),
        "neg_edge_soft_min_edge": soft_min,
        "neg_edge_bootstrap_soft_kelly_mult": boot_mult,
        "neg_edge_deep_edge_floor": deep_floor,
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


def _is_recovery_active(orch: Any | None, metrics: dict[str, Any] | None = None) -> bool:
    """True se recovery esta ativo no orch ou metrics."""
    if metrics is not None and bool(metrics.get("recovery_mode")):
        return True
    if orch is None:
        return False
    risk_mgr = getattr(orch, "risk_manager", None)
    if risk_mgr is None:
        return False
    pending_total = 0.0
    pending_dict = getattr(risk_mgr, "pending_loss", None)
    if isinstance(pending_dict, dict):
        try:
            pending_total = sum(float(v) for v in pending_dict.values())
        except (TypeError, ValueError):
            pass
    consec_losses = 0
    raw_consec = getattr(risk_mgr, "consecutive_losses_linear", 0)
    try:
        consec_losses = int(raw_consec)
    except (TypeError, ValueError):
        pass
    return pending_total > 0.0 or consec_losses > 0


def _min_edge_from_orch(orch: Any | None, metrics: dict[str, Any] | None = None) -> float:
    """Le piso escalonado de Edge para modo Normal (explore) vs Recuperacao (recovery)."""
    if orch is None:
        return 0.0
    config = getattr(orch, "config", None)
    if not isinstance(config, dict):
        return 0.0
    dl = config.get("deep_learning") if isinstance(config.get("deep_learning"), dict) else {}
    signal_skip = config.get("orchestrator", {}).get("execution", {}).get("signal_skip", {}) if isinstance(config.get("orchestrator"), dict) else {}
    is_rec = _is_recovery_active(orch, metrics)
    if is_rec:
        rec_floor = signal_skip.get("min_edge_recovery", dl.get("min_edge_recovery"))
        if rec_floor is not None:
            try:
                return max(0.0, float(rec_floor))
            except (TypeError, ValueError):
                pass
    else:
        exp_floor = signal_skip.get("min_edge_explore", dl.get("min_edge_explore"))
        if exp_floor is not None:
            try:
                return max(0.0, float(exp_floor))
            except (TypeError, ValueError):
                pass
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


def _stamp_fusion_p_eff(metrics: dict[str, Any]) -> None:
    """Grava fusion_p_eff so para telemetria; nao alimenta o gate."""
    if not bool(metrics.get("fusion_applied")):
        return
    raw = metrics.get("fusion_p_eff")
    try:
        p_eff = float(raw)
    except (TypeError, ValueError):
        return
    if 0.0 < p_eff < 1.0:
        metrics["neg_edge_fusion_p_eff"] = p_eff


def _resolve_neg_side_edge(metrics: dict[str, Any], direction: str, pay: float) -> float:
    """Edge do lado pretendido (usa fusion_p_eff se a fusao foi aplicada, ou Cal TCN)."""
    _stamp_fusion_p_eff(metrics)
    if bool(metrics.get("fusion_applied")) and metrics.get("fusion_p_eff") is not None:
        try:
            p = float(metrics["fusion_p_eff"])
            edge = float((p * (1.0 + pay)) - 1.0)
            metrics["neg_edge_tcn_cal_edge"] = edge
            return edge
        except (TypeError, ValueError):
            pass
    edge = float(resolve_predicted_edge(metrics, direction=direction, payout=pay))
    metrics["neg_edge_tcn_cal_edge"] = edge
    return edge


def apply_negative_cal_edge_pause(
    metrics: dict[str, Any],
    *,
    orch: Any | None = None,
    force: bool = False,
    min_edge: float | None = None,
    payout: float | None = None,
    soft_mult: float | None = None,
) -> bool:
    """Veto de Edge negativo ou Z-Score panico."""
    _ = soft_mult
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
    floor = float(min_edge) if min_edge is not None else _min_edge_from_orch(orch, metrics=metrics)
    edge = _resolve_neg_side_edge(metrics, direction, pay)
    metrics["cal_side_edge"] = edge
    metrics["cal_side_edge_floor"] = floor
    z_val = metrics.get("edge_zscore", metrics.get("meta_payoff_edge_zscore"))
    if z_val is not None:
        try:
            z_float = float(z_val)
            if direction == "CALL" and z_float < -2.0:
                metrics["execution_candidate_ready"] = False
                metrics["gate_reason"] = "neg_edge_zscore_panic"
                metrics["signal_status"] = "SKIP:NEG_EDGE_ZSCORE_PANIC"
                return True
            if direction == "PUT" and z_float > 2.0:
                metrics["execution_candidate_ready"] = False
                metrics["gate_reason"] = "neg_edge_zscore_panic"
                metrics["signal_status"] = "SKIP:NEG_EDGE_ZSCORE_PANIC"
                return True
        except (TypeError, ValueError):
            pass
    cfg = _neg_edge_cfg(orch)
    auto_learn = bool(metrics.get("loss_clf_auto_learn"))
    deep_floor = float(cfg.get("neg_edge_deep_edge_floor", -0.12))
    hard_skip = bool(cfg.get("neg_edge_hard_skip", False))
    if edge + 1e-12 <= 0.0 or hard_skip:
        if edge + 1e-12 < floor or edge + 1e-12 <= 0.0:
            _apply_neg_edge_hard(metrics, direction=direction, edge=edge, floor=floor)
            if (not auto_learn) and edge + 1e-12 < deep_floor:
                metrics["neg_edge_bootstrap_deep"] = True
            else:
                metrics["neg_edge_nonpositive_hard"] = True
            return True
    elif edge + 1e-12 < floor:
        soft_kelly = float(cfg.get("neg_edge_soft_kelly_mult", 0.55))
        metrics["neg_edge_soft"] = True
        metrics["neg_edge_soft_kelly_mult"] = soft_kelly
        metrics["signal_skip_waived"] = "neg_edge_soft"
        cur_scale = float(metrics.get("kelly_fraction_scale", 1.0))
        metrics["kelly_fraction_scale"] = cur_scale * soft_kelly
        return False
    return False
