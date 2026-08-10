"""Helpers de FLIP/soft do loss-classifier (seed e consenso SCALE)."""

from __future__ import annotations

from typing import Any

from src.application.services.market_audit_log_helpers import resolve_predicted_edge
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


def is_collapsed_p_loss(response: dict[str, Any], *, eps: float = 0.02) -> bool:
    """True quando auto_learn devolve p_loss ~0.5 (modelo live degenerado)."""
    if not bool(response.get("auto_learn_applied")):
        return False
    try:
        p_loss = float(response.get("p_loss"))
    except (TypeError, ValueError):
        return True
    return abs(p_loss - 0.5) < float(eps)


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


def cal_disagrees_ref(
    metrics: dict[str, Any],
    ref_dir: TradeDirection,
    *,
    margin: float = 0.0,
) -> bool:
    """True se Cal aponta o lado oposto ao TCN com margem minima |cal-0.5|."""
    if "calibrated_prob" not in metrics or metrics.get("calibrated_prob") is None:
        return False
    try:
        cal = float(metrics["calibrated_prob"])
    except (TypeError, ValueError):
        return False
    cal_dir = TradeDirection.CALL if cal + 1e-12 >= 0.5 else TradeDirection.PUT
    if cal_dir == ref_dir:
        return False
    return abs(cal - 0.5) + 1e-12 >= float(margin)


def resolve_flip_waivers(
    metrics: dict[str, Any],
    response: dict[str, Any],
    ref_dir: TradeDirection,
    *,
    cfg: dict[str, Any],
) -> tuple[bool, bool]:
    """Aplica waivers de seed/scale; Cal so anula SCALE com auto_learn."""
    seed_block = is_seed_model(response, require_auto_learn=bool(cfg.get("flip_require_auto_learn", True)))
    scale_block = scale_confirms_ref(metrics, ref_dir)
    cal_margin = float(cfg.get("flip_cal_discord_margin", 0.03))
    cal_discord = cal_disagrees_ref(metrics, ref_dir, margin=cal_margin)
    if seed_block and not scale_block and bool(cfg.get("flip_allow_seed_on_scale_discord", True)):
        seed_block = False
        metrics["loss_clf_flip_seed_discord"] = True
    if seed_block and cal_discord and bool(cfg.get("flip_allow_seed_on_cal_discord", True)):
        seed_block = False
        metrics["loss_clf_flip_seed_cal_discord"] = True
    if (
        scale_block
        and cal_discord
        and bool(response.get("auto_learn_applied"))
        and bool(cfg.get("flip_allow_seed_on_cal_discord", True))
    ):
        scale_block = False
        metrics["loss_clf_flip_cal_overrides_scale"] = True
    return seed_block, scale_block


def flip_reason_token(metrics: dict[str, Any]) -> str:
    """Razao curta de FLIP para telemetria GATES."""
    if metrics.get("loss_clf_flip_cal_overrides_scale"):
        return "cal_ovr"
    if metrics.get("loss_clf_flip_seed_discord"):
        return "seed_discord"
    if metrics.get("loss_clf_flip_seed_cal_discord"):
        return "seed_cal"
    return "ok"


def post_flip_edge_ok(metrics: dict[str, Any], flipped: TradeDirection, *, cfg: dict[str, Any]) -> bool:
    """False se o lado pos-FLIP tem edge Cal abaixo do floor de execucao."""
    if not bool(cfg.get("flip_require_pos_edge", True)):
        return True
    floor = float(cfg.get("flip_min_edge_execute", 0.04))
    edge = resolve_predicted_edge(metrics, direction=flipped.name)
    metrics["loss_clf_flip_edge"] = float(edge)
    metrics["loss_clf_flip_edge_floor"] = float(floor)
    return float(edge) + 1e-12 >= floor


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
    metrics["loss_clf_flip_reason"] = flip_reason_token(metrics)
    return flipped


def revert_loss_flip(metrics: dict[str, Any], ref_dir: TradeDirection, *, reason: str) -> None:
    """Desfaz FLIP e marca bloqueio (ex.: edge pos-FLIP abaixo do floor)."""
    metrics["exec_direction"] = ref_dir.name
    metrics["resolved_direction"] = ref_dir.name
    metrics.pop("loss_clf_flip", None)
    metrics.pop("loss_clf_flip_reason", None)
    metrics["loss_clf_veto_mode"] = "soft"
    metrics["loss_clf_flip_blocked"] = str(reason)
    metrics["loss_clf_flip_ref"] = ref_dir.name
    metrics["execution_candidate_ready"] = True


def apply_soft_kelly(metrics: dict[str, Any], mult: float, *, p_loss: float, cfg: dict[str, Any]) -> None:
    """Atenua kelly_fraction_scale e marca teto de stake abaixo do piso explore."""
    scale = float(metrics.get("kelly_fraction_scale", 1.0) or 1.0)
    metrics["kelly_fraction_scale"] = max(0.05, scale * float(mult))
    metrics["loss_clf_soft"] = True
    metrics["loss_clf_soft_kelly_mult"] = float(mult)
    metrics["loss_clf_soft_max_stake_pct"] = float(cfg["soft_max_stake_pct_high"])
    _ = p_loss
