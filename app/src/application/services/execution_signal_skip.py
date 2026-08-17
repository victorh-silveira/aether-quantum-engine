"""Catalogo minimo de atenuacao de sinal (escopo 1.1); soft Kelly sem flip de lado."""

from __future__ import annotations

from typing import Any

from src.domain.config_knobs import merge_settings_block, require_bool, require_float, require_keys
from src.domain.models.trade import TradeDirection


_VALID = {TradeDirection.CALL.name, TradeDirection.PUT.name}


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
            "waive_mini_pair_min_margin",
            "mini_pair_soft_kelly_mult",
            "cal_margin_soft_kelly_mult",
            "pending_dust",
            "chop_pause_enabled",
            "chop_adx_max",
            "chop_hurst_min",
            "chop_hurst_max",
            "chop_soft_kelly_mult",
            "neg_edge_soft_kelly_mult",
            "neg_edge_hard_skip",
            "neg_edge_soft_when_closed_candle_agree",
            "neg_edge_soft_min_edge",
            "neg_edge_bootstrap_soft_kelly_mult",
            "neg_edge_deep_edge_floor",
            "anti_loss_seed_discord_enabled",
            "anti_loss_p_loss_floor",
            "anti_loss_require_seed",
            "anti_loss_hard_skip_explore",
            "anti_loss_recover_soft_kelly_mult",
            "anti_loss_require_tcn_pos_edge",
        ),
        "orchestrator.execution.signal_skip",
    )
    soft_mult = require_float(block, "mini_pair_soft_kelly_mult")
    if soft_mult <= 0.0 or soft_mult > 1.0:
        raise ValueError("orchestrator.execution.signal_skip.mini_pair_soft_kelly_mult deve estar em (0, 1]")
    cal_soft = require_float(block, "cal_margin_soft_kelly_mult")
    if cal_soft <= 0.0 or cal_soft > 1.0:
        raise ValueError("orchestrator.execution.signal_skip.cal_margin_soft_kelly_mult deve estar em (0, 1]")
    chop_soft = require_float(block, "chop_soft_kelly_mult")
    if chop_soft <= 0.0 or chop_soft > 1.0:
        raise ValueError("orchestrator.execution.signal_skip.chop_soft_kelly_mult deve estar em (0, 1]")
    neg_soft = require_float(block, "neg_edge_soft_kelly_mult")
    if neg_soft <= 0.0 or neg_soft > 1.0:
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
    anti_floor = require_float(block, "anti_loss_p_loss_floor")
    if anti_floor < 0.0 or anti_floor > 1.0:
        raise ValueError("orchestrator.execution.signal_skip.anti_loss_p_loss_floor deve estar em [0, 1]")
    anti_soft = require_float(block, "anti_loss_recover_soft_kelly_mult")
    if anti_soft <= 0.0 or anti_soft > 1.0:
        raise ValueError("orchestrator.execution.signal_skip.anti_loss_recover_soft_kelly_mult deve estar em (0, 1]")
    hurst_min = require_float(block, "chop_hurst_min")
    hurst_max = require_float(block, "chop_hurst_max")
    if hurst_max < hurst_min:
        raise ValueError("orchestrator.execution.signal_skip.chop_hurst_max deve ser >= chop_hurst_min")
    return {
        "enabled": require_bool(block, "enabled"),
        "min_direction_margin": require_float(block, "min_direction_margin"),
        "waive_margin_on_pending": require_bool(block, "waive_margin_on_pending"),
        "mini_pair_oppose_exec": require_bool(block, "mini_pair_oppose_exec"),
        "waive_mini_pair_min_margin": require_float(block, "waive_mini_pair_min_margin"),
        "mini_pair_soft_kelly_mult": soft_mult,
        "cal_margin_soft_kelly_mult": cal_soft,
        "pending_dust": require_float(block, "pending_dust"),
        "chop_pause_enabled": require_bool(block, "chop_pause_enabled"),
        "chop_adx_max": require_float(block, "chop_adx_max"),
        "chop_hurst_min": hurst_min,
        "chop_hurst_max": hurst_max,
        "chop_soft_kelly_mult": chop_soft,
        "neg_edge_soft_kelly_mult": neg_soft,
        "neg_edge_hard_skip": require_bool(block, "neg_edge_hard_skip"),
        "neg_edge_soft_when_closed_candle_agree": require_bool(block, "neg_edge_soft_when_closed_candle_agree"),
        "neg_edge_soft_min_edge": soft_min,
        "neg_edge_bootstrap_soft_kelly_mult": boot_mult,
        "neg_edge_deep_edge_floor": deep_floor,
        "anti_loss_seed_discord_enabled": require_bool(block, "anti_loss_seed_discord_enabled"),
        "anti_loss_p_loss_floor": anti_floor,
        "anti_loss_require_seed": require_bool(block, "anti_loss_require_seed"),
        "anti_loss_hard_skip_explore": require_bool(block, "anti_loss_hard_skip_explore"),
        "anti_loss_recover_soft_kelly_mult": anti_soft,
        "anti_loss_require_tcn_pos_edge": require_bool(block, "anti_loss_require_tcn_pos_edge"),
    }


def apply_kelly_soft(metrics: dict[str, Any], soft_mult: float, *, waived: str, flag: str) -> None:
    """Atenua kelly_fraction_scale sem bloquear EXEC nem inverter CALL/PUT."""
    scale = float(metrics.get("kelly_fraction_scale", 1.0) or 1.0)
    metrics["kelly_fraction_scale"] = max(0.05, scale * float(soft_mult))
    metrics[flag] = True
    metrics[f"{flag}_kelly_mult"] = float(soft_mult)
    metrics["signal_skip_waived"] = waived


def _side(value: object) -> str | None:
    """Normaliza CALL/PUT ou None."""
    side = str(value or "").strip().upper()
    return side if side in _VALID else None


def _direction_margin(metrics: dict[str, Any]) -> float:
    """Le direction_margin numerica; 0.0 se ausente/invalida."""
    try:
        return float(metrics.get("direction_margin") or 0.0)
    except (TypeError, ValueError):
        return 0.0


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


def is_skip_signal_status(status: object) -> bool:
    """True para SKIP tecnico legado (SKIP / SKIP:REASON)."""
    token = str(status or "").strip().upper()
    return token == "SKIP" or token.startswith("SKIP:")


def metrics_block_execution(metrics: dict[str, Any] | None) -> bool:
    """True so para bloqueio tecnico (ready=False ou SKIP legado). Sinal = soft Kelly."""
    if not isinstance(metrics, dict):
        return False
    if metrics.get("execution_candidate_ready") is False:
        return True
    return is_skip_signal_status(metrics.get("signal_status"))


def apply_signal_skip_gates(
    metrics: dict[str, Any],
    exec_dir: TradeDirection,
    *,
    orch: Any | None = None,
    force: bool = False,
    cfg: dict[str, Any] | None = None,
    symbol: str | None = None,
) -> bool:
    """Aplica atenuacao soft Kelly; nunca inverte CALL/PUT."""
    _ = symbol
    metrics.setdefault("signal_skip_reason", None)
    if force:
        return False
    vision = cfg if isinstance(cfg, dict) else parse_signal_skip_config(None)
    if not bool(vision.get("enabled", True)):
        return False
    if bool(vision.get("mini_pair_oppose_exec", True)) and _mini_pair_opposes_exec(metrics, exec_dir):
        apply_kelly_soft(
            metrics,
            float(vision.get("mini_pair_soft_kelly_mult", 0.55)),
            waived="mini_pair_soft",
            flag="mini_pair_soft",
        )
    margin = _direction_margin(metrics)
    floor = float(vision.get("min_direction_margin", 0.022))
    pending = _pending_total(metrics, orch)
    dust = float(vision.get("pending_dust", 0.25))
    if margin + 1e-12 >= floor:
        return False
    if bool(vision.get("waive_margin_on_pending", True)) and pending + 1e-12 >= dust:
        metrics["signal_skip_waived"] = "cal_margin_pending"
        return False
    apply_kelly_soft(
        metrics,
        float(vision.get("cal_margin_soft_kelly_mult", 0.55)),
        waived="cal_margin_soft",
        flag="cal_margin_soft",
    )
    return False
