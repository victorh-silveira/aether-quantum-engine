"""Pisos de qualidade de sinal para execucao em modo recovery."""

from __future__ import annotations

from src.application.services.execution_direction import _MANDATORY_HARD_BLOCKS, mandatory_execution_eligible
from src.domain.risk.stake_sizing import raw_side_from_metrics


def recovery_min_signal(
    kelly_config: dict,
    *,
    recovery_active: bool,
    pending_total: float = 0.0,
    consecutive_losses: int = 0,
) -> float:
    """Piso de trade_score para pool e fallback obrigatorio, escalando com perdas consecutivas."""
    floor = float(kelly_config.get("mandatory_min_trade_score", 0.45))
    if not recovery_active:
        return floor
    recovery_floor = float(kelly_config.get("recovery_min_trade_score", floor))
    force_min = float(kelly_config.get("recovery_force_min_trade_score", recovery_floor))
    force_pending = float(kelly_config.get("recovery_force_pending_min", 0.0))
    sig_floor = recovery_floor
    if force_pending > 0.0 and float(pending_total) + 1e-9 >= force_pending:
        sig_floor = min(recovery_floor, force_min)

    # Escalonamento dinamico com base em perdas consecutivas globais
    losses = int(consecutive_losses)
    if losses == 2:
        sig_floor = max(sig_floor, 0.53)
    elif losses == 3:
        sig_floor = max(sig_floor, 0.55)
    elif losses >= 4:
        sig_floor = max(sig_floor, 0.58)

    return sig_floor


def recovery_min_val_accuracy(
    kelly_config: dict,
    *,
    consecutive_losses: int = 0,
) -> float:
    """Piso de val_accuracy para candidatos de recovery obrigatorio, escalando com perdas consecutivas."""
    base_val = float(kelly_config.get("recovery_min_val_accuracy", 0.50))

    # Escalonamento dinamico com base em perdas consecutivas globais
    losses = int(consecutive_losses)
    if losses == 2:
        base_val = max(base_val, 0.52)
    elif losses == 3:
        base_val = max(base_val, 0.53)
    elif losses >= 4:
        base_val = max(base_val, 0.55)

    return base_val


def cluster_entry_eligible(
    entry: dict,
    *,
    mandatory: bool,
    recovery_active: bool,
    recovery_cfg: dict,  # noqa: ARG001
    min_signal: float,
    min_val: float,
) -> bool:
    """Indica se entrada DL pode entrar no pool de candidatos do ciclo."""
    if recovery_active:
        # Em modo de recuperacao, a validacao de qualidade e obrigatoria e rigorosa,
        # impedindo o bypass de sinais fracos mesmo que marcados como executaveis pelo DL.
        metrics = entry.get("metrics") or {}
        if metrics.get("deploy_ok") is False:
            return False

        # Filtros de seguranca absolutos (hard blocks)
        if metrics.get("gate_reason") in _MANDATORY_HARD_BLOCKS:
            return False

        score = float(metrics.get("trade_score", metrics.get("conviction", 0.0)))
        raw_side = raw_side_from_metrics(metrics)
        effective_signal = max(score, raw_side)
        val = float(metrics.get("val_accuracy", 0.0))

        return effective_signal + 1e-9 >= min_signal and not (min_val > 0.0 and val + 1e-9 < min_val)

    may_execute = bool(entry.get("metrics", {}).get("execute", False))
    if may_execute:
        return True
    return mandatory and mandatory_execution_eligible(entry, min_signal=min_signal, min_val_accuracy=min_val)
