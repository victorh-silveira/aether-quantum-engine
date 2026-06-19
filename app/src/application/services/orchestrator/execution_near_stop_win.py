"""Pausa execucao obrigatoria fraca quando a sessao esta perto do stop win."""

from __future__ import annotations

from typing import Any

from src.application.services.execution_direction import _entry_gate_blocked, _entry_signal_strength
from src.domain.risk.stop_win_target import resolve_stop_win_target


def best_blocked_signal_strength(decisions: dict[str, dict]) -> float:
    """Maior forca de sinal entre entradas DL bloqueadas por execute=false."""
    best = 0.0
    for entry in decisions.values():
        metrics = entry.get("metrics") or {}
        if _entry_gate_blocked(metrics):
            continue
        score, raw_side = _entry_signal_strength(metrics)
        best = max(best, score, raw_side)
    return best


def decisions_all_dl_blocked(decisions: dict[str, dict]) -> bool:
    """True quando nenhum simbolo passou no gate execute=true do DL."""
    if not decisions:
        return True
    for entry in decisions.values():
        metrics = entry.get("metrics") or {}
        if metrics.get("execute"):
            return False
    return True


def near_stop_win_mandatory_pause(
    risk_manager: Any,
    risk_config: dict,
    kelly_config: dict,
) -> bool:
    """True quando Kelly obrigatorio deve pausar por proximidade da meta diaria."""
    if sum(float(v) for v in risk_manager.pending_loss.values()) > 0.0:
        return False
    progress = float(kelly_config.get("near_stop_win_mandatory_pause_fraction", 0.90))
    if progress <= 0.0:
        return False
    target = resolve_stop_win_target(risk_config, float(risk_manager.initial_bankroll))
    if target <= 0.0:
        return False
    pnl = float(risk_manager.total_session_profit)
    return pnl >= target * progress


def should_pause_weak_mandatory(
    exec_mgr: Any,
    decisions: dict[str, dict],
    *,
    recovery_active: bool,
) -> bool:
    """Indica pausa de fallback obrigatorio com todos os simbolos bloqueados pelo DL."""
    orch = exec_mgr.orch
    risk_cfg = orch.config.get("risk_management", {}) if isinstance(orch.config, dict) else {}
    kelly_cfg = risk_cfg.get("kelly", {}) if isinstance(risk_cfg, dict) else {}
    if recovery_active:
        if decisions:
            min_val = float(kelly_cfg.get("recovery_min_val_accuracy", 0.50))
            if min_val > 0.0:
                all_below = True
                for entry in decisions.values():
                    val = float(entry.get("metrics", {}).get("val_accuracy", 0.0))
                    if val >= min_val:
                        all_below = False
                        break
                if all_below:
                    exec_mgr.logger.warning(
                        "RISK REC PAUSE: Todos os simbolos com val_accuracy abaixo de recovery_min_val_accuracy=%.2f",
                        min_val,
                    )
                    return True
        return False
    if not decisions_all_dl_blocked(decisions):
        return False
    min_signal = float(kelly_cfg.get("mandatory_min_trade_score", 0.45))
    if best_blocked_signal_strength(decisions) + 1e-9 < min_signal:
        return True
    return near_stop_win_mandatory_pause(orch.risk_manager, risk_cfg, kelly_cfg)
