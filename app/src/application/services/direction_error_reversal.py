"""Servico de aplicacao para Alpha Flip Adaptativo baseado em Brier Score."""

from __future__ import annotations

import logging
from typing import Any

from src.domain.math.error_driven_reversal import (
    calculate_trade_brier_score,
    is_error_driven_reversal_armed,
    resolve_target_reversal_direction,
)
from src.domain.models.trade import TradeDirection


logger = logging.getLogger("AETH")


def _get_reversal_config(orch: Any, override_cfg: dict | None = None) -> dict:
    """Extrai configuracao de error_driven_reversal do orquestrador com defaults seguros."""
    if isinstance(override_cfg, dict) and override_cfg:
        return override_cfg
    config = getattr(orch, "config", None)
    if isinstance(config, dict):
        exec_cfg = config.get("orchestrator", {}).get("execution", {})
        rev_cfg = exec_cfg.get("error_driven_reversal")
        if isinstance(rev_cfg, dict):
            return rev_cfg
    return {"enabled": True, "brier_threshold": 0.40, "min_conviction": 0.60}


def record_error_reversal_on_settlement(
    orch: Any,
    symbol: str,
    *,
    won: bool,
    raw_prob: float | None,
    direction: str | None,
    cfg: dict | None = None,
) -> None:
    """Avalia o resultado liquidado e arma o Alpha Flip caso o choque de erro supere o limiar."""
    if orch is None:
        return
    bag = getattr(orch, "_pending_alpha_reversal", None)
    if bag is None:
        orch._pending_alpha_reversal = {}
        bag = orch._pending_alpha_reversal

    sym = str(symbol)
    if won:
        bag.pop(sym, None)
        return

    rev_cfg = _get_reversal_config(orch, cfg)
    if not bool(rev_cfg.get("enabled", True)):
        bag.pop(sym, None)
        return

    if raw_prob is None or not direction:
        return

    dir_clean = str(direction).upper().strip()
    if dir_clean not in {"CALL", "PUT"}:
        return

    brier, eps = calculate_trade_brier_score(float(raw_prob), won=won, direction=dir_clean)
    threshold = float(rev_cfg.get("brier_threshold", 0.40))
    conviction = float(rev_cfg.get("min_conviction", 0.60))

    if is_error_driven_reversal_armed(brier, float(raw_prob), threshold=threshold, min_conviction=conviction):
        target_dir = resolve_target_reversal_direction(dir_clean)
        cycle_id = int(getattr(orch, "_active_cycle_id", 0) or 0)
        bag[sym] = {
            "brier": float(brier),
            "residual": float(eps),
            "from_dir": dir_clean,
            "target_dir": target_dir,
            "armed_cycle": cycle_id,
        }
        logger.info(
            "ALPHA_FLIP || Armado para %s: BS=%.4f eps=%+.4f dir=%s -> inv=%s ciclo=%d",
            sym,
            brier,
            eps,
            dir_clean,
            target_dir,
            cycle_id,
        )


def apply_error_reversal_to_direction(
    orch: Any,
    symbol: str,
    current_dir: TradeDirection,
    metrics: dict,
    exec_cfg: dict | None = None,
) -> tuple[TradeDirection, bool]:
    """Aplica o Alpha Flip na resolucao direcional do ciclo t+1 e consome a flag."""
    if orch is None:
        return current_dir, False

    bag = getattr(orch, "_pending_alpha_reversal", None)
    if not isinstance(bag, dict):
        return current_dir, False

    sym = str(symbol)
    info = bag.pop(sym, None)
    if not isinstance(info, dict):
        return current_dir, False

    rev_cfg = _get_reversal_config(orch, (exec_cfg or {}).get("error_driven_reversal"))
    if not bool(rev_cfg.get("enabled", True)):
        return current_dir, False

    target_name = str(info.get("target_dir") or "").upper()
    if target_name not in {TradeDirection.CALL.name, TradeDirection.PUT.name}:
        flipped = TradeDirection.PUT if current_dir == TradeDirection.CALL else TradeDirection.CALL
    else:
        flipped = TradeDirection[target_name]

    metrics["alpha_flip_applied"] = True
    metrics["alpha_flip_brier"] = float(info.get("brier", 0.0))
    metrics["alpha_flip_residual"] = float(info.get("residual", 0.0))
    metrics["alpha_flip_from"] = str(info.get("from_dir", current_dir.name))
    metrics["alpha_flip_to"] = flipped.name
    metrics["direction_origin"] = "FLIP_ERROR_DRIVEN_ALPHA"

    logger.info(
        "ALPHA_FLIP || Disparo Adaptativo em %s: BS=%.4f eps=%+.4f de %s para %s",
        sym,
        metrics["alpha_flip_brier"],
        metrics["alpha_flip_residual"],
        current_dir.name,
        flipped.name,
    )
    return flipped, True
