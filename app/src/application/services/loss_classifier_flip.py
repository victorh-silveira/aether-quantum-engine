"""Helpers de FLIP/soft do loss-classifier (seed e consenso SCALE)."""

from __future__ import annotations

from typing import Any

from src.domain.models.trade import TradeDirection


def resolve_soft_kelly_mult(p_loss: float, cfg: dict[str, Any]) -> float:
    """Interpola soft Kelly entre floor e p_loss alto (maior risco → menor mult)."""
    floor = float(cfg["veto_p_loss_floor"])
    high = float(cfg["soft_p_loss_high"])
    mult_lo = float(cfg["soft_kelly_mult"])
    mult_hi = float(cfg["soft_kelly_mult_high"])
    p = float(p_loss)
    if p >= high:
        return mult_hi
    if p <= floor or high <= floor:
        return mult_lo
    t = (p - floor) / (high - floor)
    return mult_lo + t * (mult_hi - mult_lo)


def is_seed_model(response: dict[str, Any], *, require_auto_learn: bool) -> bool:
    """True se FLIP exige auto_learn e o artefato ainda nao retrainou via /learn."""
    if not require_auto_learn:
        return False
    return not bool(response.get("auto_learn_applied"))


def scale_confirms_ref(metrics: dict[str, Any], ref_dir: TradeDirection) -> bool:
    """True se tape ou maioria de votos confirma o lado TCN (nao inverter)."""
    tape = str(metrics.get("scale_tape_consensus") or "").strip().upper()
    if tape == ref_dir.name:
        return True
    try:
        vc = int(metrics.get("scale_vote_call_n") or 0)
        vp = int(metrics.get("scale_vote_put_n") or 0)
    except (TypeError, ValueError):
        return False
    if ref_dir == TradeDirection.CALL and vc >= vp + 2:
        return True
    return ref_dir == TradeDirection.PUT and vp >= vc + 2


def apply_loss_flip(metrics: dict[str, Any], ref_dir: TradeDirection, *, cfg: dict[str, Any]) -> TradeDirection:
    """Inverte CALL↔PUT relativo ao TCN quando p_loss alto; mantem candidato EXEC."""
    flipped = TradeDirection.PUT if ref_dir == TradeDirection.CALL else TradeDirection.CALL
    metrics["exec_direction"] = flipped.name
    metrics["resolved_direction"] = flipped.name
    metrics["loss_clf_flip"] = True
    metrics["loss_clf_veto_mode"] = "flip"
    metrics["loss_clf_flip_ref"] = ref_dir.name
    metrics["loss_clf_flip_p_loss_floor"] = float(cfg["hard_p_loss_floor"])
    metrics["loss_clf_hard_p_loss_floor"] = float(cfg["hard_p_loss_floor"])
    metrics["loss_clf_soft_waived_pending"] = False
    metrics["execution_candidate_ready"] = True
    return flipped


def apply_soft_kelly(metrics: dict[str, Any], mult: float, *, p_loss: float, cfg: dict[str, Any]) -> None:
    """Atenua kelly_fraction_scale e marca teto de stake abaixo do piso explore."""
    scale = float(metrics.get("kelly_fraction_scale", 1.0) or 1.0)
    metrics["kelly_fraction_scale"] = max(0.05, scale * float(mult))
    metrics["loss_clf_soft"] = True
    metrics["loss_clf_soft_kelly_mult"] = float(mult)
    metrics["loss_clf_soft_max_stake_pct"] = float(cfg["soft_max_stake_pct_high"])
    _ = p_loss
