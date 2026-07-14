"""Filtros e penalidades de protecao contra padroes recorrentes de loss."""

from __future__ import annotations

from typing import Any

from src.domain.models.trade import TradeDirection
from src.domain.risk.recovery_hurst_decay import resolve_effective_hurst_min
from src.domain.risk.recovery_hurst_gate import recovery_pool_has_persistence


def _directional_calibrated_side(metrics: dict, direction: TradeDirection | None = None) -> float:
    """Retorna conviccao calibrada no lado da direcao de execucao."""
    calibrated = metrics.get("calibrated_prob", metrics.get("tcn_score", metrics.get("raw_prob", 0.5)))
    try:
        value = float(calibrated)
    except (TypeError, ValueError):
        return 0.0
    if direction == TradeDirection.PUT:
        return 1.0 - value
    if direction == TradeDirection.CALL:
        return value
    return max(value, 1.0 - value)


def _directional_raw_side(metrics: dict, direction: TradeDirection | None = None) -> float:
    """Retorna probabilidade bruta TCN no lado da direcao de execucao."""
    raw = metrics.get("raw_prob")
    if raw is None:
        return 0.0
    try:
        value = float(raw)
    except (TypeError, ValueError):
        return 0.0
    if direction == TradeDirection.PUT:
        return 1.0 - value
    if direction == TradeDirection.CALL:
        return value
    return max(value, 1.0 - value)


def edge_conviction_disconnect_penalty(
    metrics: dict,
    *,
    exec_direction: TradeDirection | None = None,
) -> float:
    """Penaliza score quando meta-edge inflado diverge da conviccao direcional."""
    try:
        edge = float(metrics.get("edge", 0.0))
        margin = float(metrics.get("direction_margin", 0.0))
        z_edge = float(metrics.get("edge_zscore", 0.0))
    except (TypeError, ValueError):
        return 0.0
    calibrated_side = _directional_calibrated_side(metrics, exec_direction)
    raw_side = _directional_raw_side(metrics, exec_direction)
    penalty = 0.0
    if edge >= 0.35 and margin < 0.22:
        penalty = max(penalty, 0.16)
    if edge >= 0.50 and margin < 0.26:
        penalty = max(penalty, 0.12)
    if z_edge >= 0.85 and margin < 0.24:
        penalty = max(penalty, 0.14)
    if edge >= 0.40 and calibrated_side < 0.32:
        penalty = max(penalty, 0.18)
    if edge >= 0.35 and raw_side < 0.30:
        penalty = max(penalty, 0.15)
    return penalty


def apply_loss_protection_penalties(metrics: dict, *, exec_direction: TradeDirection | None = None) -> None:
    """Registra penalidade de desconexao edge-conviccao nas metricas do candidato."""
    direction = exec_direction
    if direction is None and metrics.get("exec_direction"):
        name = str(metrics["exec_direction"]).upper()
        direction = TradeDirection.CALL if name == "CALL" else TradeDirection.PUT
    penalty = edge_conviction_disconnect_penalty(metrics, exec_direction=direction)
    if penalty > 0.0:
        metrics["loss_protection_penalty"] = penalty


def _loss_protection_cfg(exec_cfg: dict | None) -> dict[str, Any]:
    """Extrai bloco loss_protection da configuracao de execucao."""
    if not isinstance(exec_cfg, dict):
        return {}
    nested = exec_cfg.get("loss_protection")
    return nested if isinstance(nested, dict) else {}


def _loss_protection_recovery_blocks(
    metrics: dict,
    *,
    recovery_active: bool,
    linear: int,
    min_margin_recovery: float,
    recovery_min_hurst: float,
    margin: float,
) -> bool:
    """True quando recovery ou Hurst minimo bloqueiam o candidato."""
    if recovery_active and linear >= 2 and margin + 1e-9 < min_margin_recovery:
        return True
    if recovery_active and linear >= 1:
        indicators = metrics.get("indicators") or {}
        micro = metrics.get("micro_indicators") or {}
        raw_hurst = indicators.get("hurst")
        if raw_hurst is None and isinstance(micro, dict):
            raw_hurst = micro.get("hurst")
        if raw_hurst is None:
            return False
        return float(raw_hurst) + 1e-9 < recovery_min_hurst
    return False


def _loss_protection_signal_blocks(
    metrics: dict,
    direction: TradeDirection,
    *,
    margin: float,
    edge: float,
    z_edge: float,
    min_margin: float,
    max_edge_low_margin: float,
    max_z_low_margin: float,
) -> bool:
    """True quando margem, edge ou desconexao edge-conviccao bloqueiam o candidato."""
    if margin + 1e-9 < min_margin and edge >= max_edge_low_margin:
        return True
    if margin + 1e-9 < min_margin and z_edge >= max_z_low_margin:
        return True
    return edge_conviction_disconnect_penalty(metrics, exec_direction=direction) >= 0.18


def candidate_passes_loss_protection(
    item: tuple[str, TradeDirection, dict],
    *,
    exec_cfg: dict | None,
    recovery_active: bool,
    consecutive_losses: int,
) -> bool:
    """True quando candidato atende piso de conviccao direcional para entrada."""
    if not isinstance(item, tuple) or len(item) < 3:
        return False
    metrics = item[2]
    if not isinstance(metrics, dict):
        return False
    cfg = _loss_protection_cfg(exec_cfg)
    try:
        margin = float(metrics.get("direction_margin", 0.0))
        edge = float(metrics.get("edge", 0.0))
        z_edge = float(metrics.get("edge_zscore", 0.0))
    except (TypeError, ValueError):
        return False
    min_margin = float(cfg.get("min_direction_margin", 0.18))
    min_margin_recovery = float(cfg.get("recovery_min_direction_margin", 0.20))
    recovery_min_hurst = float(cfg.get("recovery_min_hurst", 0.50))
    max_edge_low_margin = float(cfg.get("max_edge_without_margin", 0.40))
    max_z_low_margin = float(cfg.get("max_zscore_without_margin", 0.85))
    linear = int(consecutive_losses)
    if _loss_protection_recovery_blocks(
        metrics,
        recovery_active=recovery_active,
        linear=linear,
        min_margin_recovery=min_margin_recovery,
        recovery_min_hurst=recovery_min_hurst,
        margin=margin,
    ):
        return False
    return not _loss_protection_signal_blocks(
        metrics,
        item[1],
        margin=margin,
        edge=edge,
        z_edge=z_edge,
        min_margin=min_margin,
        max_edge_low_margin=max_edge_low_margin,
        max_z_low_margin=max_z_low_margin,
    )


def filter_loss_protection_candidates(
    candidates: list[tuple[str, TradeDirection, dict]],
    *,
    exec_cfg: dict | None,
    recovery_active: bool,
    consecutive_losses: int,
) -> list[tuple[str, TradeDirection, dict]]:
    """Remove candidatos com edge inflado e conviccao direcional insuficiente."""
    if not candidates:
        return []
    filtered = [
        item
        for item in candidates
        if candidate_passes_loss_protection(
            item,
            exec_cfg=exec_cfg,
            recovery_active=recovery_active,
            consecutive_losses=consecutive_losses,
        )
    ]
    if filtered:
        return filtered
    return list(candidates)


def filter_recovery_hurst_candidates(
    candidates: list[tuple[str, TradeDirection, dict]],
    *,
    kelly_cfg: dict | None,
    consecutive_losses: int,
    recovery_skip_counter: int = 0,
    session_drawdown: float = 0.0,
) -> list[tuple[str, TradeDirection, dict]]:
    """Em recovery N2+, prioriza candidatos com Hurst de persistencia."""
    if not candidates or int(consecutive_losses) < 2:
        return list(candidates)
    cfg = kelly_cfg if isinstance(kelly_cfg, dict) else {}
    hurst_min = resolve_effective_hurst_min(
        cfg,
        int(recovery_skip_counter),
        consecutive_losses=int(consecutive_losses),
        session_drawdown=float(session_drawdown),
    )
    if recovery_pool_has_persistence(
        candidates,
        consecutive_losses=int(consecutive_losses),
        hurst_min=hurst_min,
    ):
        persistent = []
        threshold = float(hurst_min)
        for item in candidates:
            if not isinstance(item, tuple) or len(item) < 3:
                continue
            metrics = item[2]
            if not isinstance(metrics, dict):
                continue
            indicators = metrics.get("indicators") or {}
            if float(indicators.get("hurst", 0.0)) + 1e-9 >= threshold:
                persistent.append(item)
        if persistent:
            return persistent
    return list(candidates)
