"""Fusao MACRO/MICRO/MINI/MILI + Cal/meta/loss-clf por EV esperado CALL vs PUT."""

from __future__ import annotations

import math
from typing import Any

from src.application.services.execution_direction_fusion_config import parse_direction_fusion_config
from src.application.services.execution_signal_skip import apply_kelly_soft
from src.application.services.loss_classifier_flip import tcn_pos_edge_blocks_flip
from src.application.services.market_audit_log_helpers import resolve_predicted_edge
from src.application.services.market_audit_ops_window import ops_window_candle_side
from src.domain.models.trade import TradeDirection
from src.domain.risk.kelly_runtime_config import load_kelly_runtime_from_settings


__all__ = (
    "apply_direction_fusion",
    "parse_direction_fusion_config",
)


def _clip01(p: float, *, eps: float = 1e-6) -> float:
    """Satura probabilidade em (eps, 1-eps)."""
    return max(eps, min(1.0 - eps, float(p)))


def _logit(p: float) -> float:
    """Logit de probabilidade saturada."""
    x = _clip01(p)
    return math.log(x / (1.0 - x))


def _sigmoid(z: float) -> float:
    """Sigmoid numerica com saturação em caudas extremas."""
    if z >= 30.0:
        return 1.0 - 1e-6
    if z <= -30.0:
        return 1e-6
    return 1.0 / (1.0 + math.exp(-z))


def _read_p_call(metrics: dict[str, Any]) -> float | None:
    """Le probabilidade CALL calibrada (fallback raw) das metrics."""
    raw = metrics.get("calibrated_prob")
    if raw is None:
        raw = metrics.get("raw_prob")
    if raw is None:
        return None
    try:
        return float(raw)
    except (TypeError, ValueError):
        return None


def _shrink_near_half(p_call: float, shrink: float) -> float:
    """Aproxima p_call de 0.5 quando margem e pequena."""
    if shrink <= 0.0:
        return p_call
    margin = abs(p_call - 0.5)
    damp = shrink * max(0.0, 1.0 - margin / 0.12)
    return 0.5 + (p_call - 0.5) * (1.0 - damp)


def _side_token(metrics: dict[str, Any], key: str) -> str | None:
    """Normaliza token CALL/PUT de uma chave de metrics."""
    side = str(metrics.get(key) or "").strip().upper()
    if side in {TradeDirection.CALL.name, TradeDirection.PUT.name}:
        return side
    return None


def _evidence(metrics: dict[str, Any], side: str, cfg: dict[str, Any]) -> float:
    """Soma pesos de evidencia multi-escala a favor do lado."""
    total = 0.0
    pairs = (
        ("scale_macro_dir", float(cfg["fusion_w_macro"])),
        ("scale_mini_dir", float(cfg["fusion_w_mini"])),
        ("scale_mini_bar_dir", float(cfg["fusion_w_mini"]) * 0.5),
        ("scale_mili_dir", float(cfg["fusion_w_mili"])),
        ("scale_tape_consensus", float(cfg["fusion_w_tape"])),
    )
    for key, weight in pairs:
        token = _side_token(metrics, key)
        if token is None or weight <= 0.0:
            continue
        total += weight if token == side else -weight
    candle = ops_window_candle_side(metrics)
    w_bar = float(cfg["fusion_w_micro_bar"])
    if candle is not None and w_bar > 0.0:
        total += w_bar if candle == side else -w_bar
    return total


def _loss_logit_bonus(metrics: dict[str, Any], side: str, cfg: dict[str, Any]) -> float:
    """Bonus logit continuo do loss-clf a favor do lado oposto ao ref."""
    weight = float(cfg["fusion_loss_weight"])
    if weight <= 0.0 or metrics.get("loss_clf_p_loss") is None:
        return 0.0
    if metrics.get("loss_clf_collapsed"):
        return 0.0
    requires_auto = bool(cfg.get("fusion_loss_requires_auto_learn", True))
    auto = bool(metrics.get("loss_clf_auto_learn"))
    if requires_auto and not auto:
        seed_mult = float(cfg.get("fusion_loss_seed_weight_mult", 0.0))
        if seed_mult <= 0.0:
            return 0.0
        weight = weight * seed_mult
    try:
        p_loss = float(metrics.get("loss_clf_p_loss"))
    except (TypeError, ValueError):
        return 0.0
    ref = str(
        metrics.get("loss_clf_flip_ref") or metrics.get("tcn_direction") or metrics.get("scale_micro_dir") or ""
    ).upper()
    if ref not in {TradeDirection.CALL.name, TradeDirection.PUT.name}:
        return 0.0
    strength = max(0.0, p_loss - 0.5) * 2.0 * weight
    if side == ref:
        return -strength
    return strength


def _apply_tcn_candle_agree_guard(
    metrics: dict[str, Any],
    *,
    vision: dict[str, Any],
    tcn_dir: TradeDirection,
    chosen: TradeDirection,
    p_effs: dict[str, float],
) -> TradeDirection:
    """Mantem TCN quando a janela ops concorda e a fusao tentaria inverter."""
    if not bool(vision.get("fusion_block_when_tcn_candle_agree", True)):
        return chosen
    candle = ops_window_candle_side(metrics)
    if candle is None or candle != tcn_dir.name or chosen == tcn_dir:
        return chosen
    metrics["fusion_blocked_tcn_candle"] = True
    metrics["fusion_reason"] = "tcn_candle_agree"
    metrics["fusion_p_eff"] = float(p_effs[tcn_dir.name])
    return tcn_dir


def _payout(orch: Any | None) -> float:
    """Resolve payout R_10 a partir do orch ou fallback Kelly SSOT."""
    rt = load_kelly_runtime_from_settings()
    if orch is None:
        return float(rt.get("payout_fallback", 0.72))
    config = getattr(orch, "config", None)
    if not isinstance(config, dict):
        return float(rt.get("payout_fallback", 0.72))
    risk = config.get("risk_management") if isinstance(config.get("risk_management"), dict) else {}
    params = risk.get("params") if isinstance(risk.get("params"), dict) else {}
    try:
        return max(0.01, float(params.get("payout_estimate", rt.get("payout_fallback", 0.72))))
    except (TypeError, ValueError):
        return float(rt.get("payout_fallback", 0.72))


def _fusion_raw(orch: Any | None) -> dict[str, Any] | None:
    """Extrai override parcial de scale_vision do orch.config."""
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
    raw = ex.get("scale_vision")
    return raw if isinstance(raw, dict) else None


def apply_direction_fusion(
    metrics: dict[str, Any],
    exec_dir: TradeDirection,
    *,
    orch: Any | None = None,
    cfg: dict[str, Any] | None = None,
) -> TradeDirection:
    """Escolhe CALL/PUT por argmax EV com p_eff (Cal + fita + loss + meta)."""
    vision = cfg if isinstance(cfg, dict) else parse_direction_fusion_config(_fusion_raw(orch))
    metrics["fusion_applied"] = False
    metrics.setdefault("fusion_reason", "idle")
    if not bool(vision.get("fusion_enabled", True)):
        metrics["fusion_reason"] = "disabled"
        return exec_dir
    p_call = _read_p_call(metrics)
    if p_call is None:
        metrics["fusion_reason"] = "no_cal"
        return exec_dir
    tcn_name = str(metrics.get("tcn_direction") or metrics.get("scale_micro_dir") or exec_dir.name).upper()
    tcn_dir = TradeDirection[tcn_name] if tcn_name in {"CALL", "PUT"} else exec_dir
    block_cfg = {
        "flip_block_when_tcn_pos_edge": bool(vision.get("fusion_block_when_tcn_pos_edge", True)),
        "flip_min_edge_execute": float(vision.get("fusion_min_edge_execute", 0.04)),
        "flip_tcn_pos_edge_raw_floor": float(vision.get("fusion_min_edge_execute", 0.04)),
        "flip_waive_tcn_pos_edge_on_discord": bool(vision.get("flip_waive_tcn_pos_edge_on_discord", False)),
    }
    if tcn_pos_edge_blocks_flip(metrics, tcn_dir, cfg=block_cfg):
        metrics["fusion_reason"] = "tcn_pos_edge"
        metrics["fusion_blocked_tcn_pos_edge"] = True
        metrics["fusion_ev_call"] = float(resolve_predicted_edge(metrics, direction="CALL", payout=_payout(orch)))
        metrics["fusion_ev_put"] = float(resolve_predicted_edge(metrics, direction="PUT", payout=_payout(orch)))
        metrics["fusion_p_call"] = float(p_call)
        metrics["fusion_p_put"] = float(1.0 - p_call)
        metrics["fusion_p_eff"] = float(p_call if tcn_dir == TradeDirection.CALL else 1.0 - p_call)
        metrics["fusion_applied"] = True
        metrics["exec_direction"] = tcn_dir.name
        metrics["resolved_direction"] = tcn_dir.name
        return tcn_dir
    shrink = float(vision.get("fusion_tcn_shrink_near_half", 0.0))
    p_anchor = _shrink_near_half(float(p_call), shrink)
    pay = _payout(orch)
    scores: dict[str, float] = {}
    p_effs: dict[str, float] = {}
    for side in (TradeDirection.CALL.name, TradeDirection.PUT.name):
        p_side = p_anchor if side == TradeDirection.CALL.name else 1.0 - p_anchor
        z = _logit(p_side) + _evidence(metrics, side, vision) + _loss_logit_bonus(metrics, side, vision)
        p_eff = max(1e-6, min(1.0 - 1e-6, _sigmoid(z)))
        p_effs[side] = p_eff
        scores[side] = p_eff * (1.0 + pay) - 1.0
    meta_w = float(vision.get("fusion_meta_ev_weight", 0.0))
    if meta_w > 0.0:
        try:
            meta_e = float(metrics.get("meta_payoff_edge", metrics.get("predicted_payoff_edge", 0.0)) or 0.0)
        except (TypeError, ValueError):
            meta_e = 0.0
        anchor = TradeDirection.CALL.name if float(p_call) + 1e-12 >= 0.5 else TradeDirection.PUT.name
        scores[anchor] = float(scores[anchor]) + meta_w * meta_e
    ev_call = float(scores[TradeDirection.CALL.name])
    ev_put = float(scores[TradeDirection.PUT.name])
    metrics["fusion_ev_call"] = ev_call
    metrics["fusion_ev_put"] = ev_put
    metrics["fusion_p_call"] = float(p_effs[TradeDirection.CALL.name])
    metrics["fusion_p_put"] = float(p_effs[TradeDirection.PUT.name])
    if max(ev_call, ev_put) <= 0.0:
        metrics["fusion_reason"] = "negative_ev_abstain"
        metrics["execution_candidate_ready"] = False
        metrics["signal_status"] = "SKIP:FUSION_NEGATIVE_EV"
        metrics["gate_reason"] = "fusion_negative_ev"
        metrics["fusion_applied"] = True
        return tcn_dir
    if abs(ev_call - ev_put) <= 1e-12:
        chosen = TradeDirection.CALL if float(p_call) + 1e-12 >= 0.5 else TradeDirection.PUT
        reason = "tie_cal"
    elif ev_call > ev_put:
        chosen = TradeDirection.CALL
        reason = "ev_call"
    else:
        chosen = TradeDirection.PUT
        reason = "ev_put"
    metrics["fusion_reason"] = reason
    metrics["fusion_p_eff"] = float(p_effs[chosen.name])
    chosen = _apply_tcn_candle_agree_guard(metrics, vision=vision, tcn_dir=tcn_dir, chosen=chosen, p_effs=p_effs)
    metrics["fusion_applied"] = True
    if chosen != exec_dir:
        metrics["fusion_switched"] = True
        metrics["fusion_from"] = exec_dir.name
    else:
        metrics["fusion_switched"] = False
    metrics["exec_direction"] = chosen.name
    metrics["resolved_direction"] = chosen.name
    metrics["execution_candidate_ready"] = True
    chosen_ev = float(ev_call if chosen == TradeDirection.CALL else ev_put)
    metrics["fusion_chosen_ev"] = chosen_ev
    if chosen_ev + 1e-12 < float(vision.get("fusion_min_edge_execute", 0.04)):
        soft = float(vision.get("fusion_weak_ev_soft_kelly_mult", 0.40))
        apply_kelly_soft(metrics, soft, waived="fusion_weak_ev", flag="fusion_weak_ev_soft")
        metrics["fusion_weak_ev"] = chosen_ev
    return chosen
