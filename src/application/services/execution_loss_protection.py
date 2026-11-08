"""Filtros e penalidades de protecao contra padroes recorrentes de loss."""

from __future__ import annotations

from typing import Any

from src.application.services.execution_runtime_config import resolve_loss_protection_config
from src.domain.models.trade import TradeDirection


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
    disconnect = resolve_loss_protection_config()["disconnect"]
    soft = disconnect["edge_margin_soft"]
    hard = disconnect["edge_margin_hard"]
    zrule = disconnect["zscore_margin"]
    cal = disconnect["edge_calibrated_side"]
    raw = disconnect["edge_raw_side"]
    penalty = 0.0
    if edge >= float(soft["edge_min"]) and margin < float(soft["margin_max"]):
        penalty = max(penalty, float(soft["score"]))
    if edge >= float(hard["edge_min"]) and margin < float(hard["margin_max"]):
        penalty = max(penalty, float(hard["score"]))
    if z_edge >= float(zrule["z_min"]) and margin < float(zrule["margin_max"]):
        penalty = max(penalty, float(zrule["score"]))
    if edge >= float(cal["edge_min"]) and calibrated_side < float(cal["calibrated_side_max"]):
        penalty = max(penalty, float(cal["score"]))
    if edge >= float(raw["edge_min"]) and raw_side < float(raw["raw_side_max"]):
        penalty = max(penalty, float(raw["score"]))
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
    block = float(resolve_loss_protection_config()["disconnect"]["block_threshold"])
    return edge_conviction_disconnect_penalty(metrics, exec_direction=direction) >= block


def candidate_passes_loss_protection(
    item: tuple[str, TradeDirection, dict],
    *,
    exec_cfg: dict | None,
    recovery_active: bool,
    consecutive_losses: int,
) -> bool:
    """Filtro de loss protection desativado; qualquer candidato tipado passa."""
    _ = (exec_cfg, recovery_active, consecutive_losses)
    if not isinstance(item, tuple) or len(item) < 3:
        return False
    return isinstance(item[2], dict)


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
    """Filtro Hurst de recovery desativado; retorna o pool integral."""
    _ = (kelly_cfg, consecutive_losses, recovery_skip_counter, session_drawdown)
    return list(candidates)
