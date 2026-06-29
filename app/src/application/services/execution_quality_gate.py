"""Pisos de qualidade para filtrar candidatos antes da execucao."""

from src.application.services.deep_learning.dl_gating import resolve_calibrated_edge
from src.application.services.execution_squeeze_gate import passes_squeeze_gate
from src.application.services.orchestrator.execution_recovery_gate import effective_signal, recovery_min_signal


def quality_gate_params(exec_cfg: dict) -> dict[str, float]:
    """Le limites de qualidade configurados em orchestrator.execution.quality_gate."""
    chunk = exec_cfg.get("quality_gate") if isinstance(exec_cfg, dict) else {}
    if not isinstance(chunk, dict):
        chunk = {}
    return {
        "min_direction_margin": float(chunk.get("min_direction_margin", 0.05)),
        "inverted_min_score": float(chunk.get("inverted_min_score", 0.74)),
        "min_adx_normal": float(chunk.get("min_adx_normal", 0.18)),
    }


def _effective_edge(metrics: dict) -> float:
    """Edge operacional priorizando probabilidade calibrada."""
    calibrated_edge = metrics.get("calibrated_edge")
    if calibrated_edge is not None:
        return float(calibrated_edge)
    edge = metrics.get("edge")
    if edge is not None:
        return float(edge)
    return resolve_calibrated_edge(
        metrics.get("calibrated_prob"),
        raw_prob=metrics.get("raw_prob"),
    )


def _effective_min_edge(metrics: dict, min_edge: float) -> float:
    """Piso de edge global ou dinamico por candidato."""
    dynamic = metrics.get("dynamic_min_edge")
    if dynamic is None:
        return float(min_edge)
    return max(float(min_edge), float(dynamic))


def _quality_failures(
    metrics: dict,
    *,
    min_signal: float,
    min_val: float,
    min_edge: float,
    min_direction_margin: float,
    inverted_min_score: float,
    min_adx_normal: float,
    recovery_active: bool,
    exhaustion_gate_cfg: dict | None = None,
    recovery_kelly_cfg: dict | None = None,
    consecutive_losses: int = 0,
) -> bool:
    """Retorna True quando algum piso de qualidade nao e atendido."""
    eff = effective_signal(metrics)
    indicators = metrics.get("indicators") or {}
    adx = float(indicators.get("adx", 0.0))
    margin = float(metrics.get("direction_margin", 0.0))
    edge_floor = _effective_min_edge(metrics, min_edge)
    edge_val = _effective_edge(metrics)
    gate = exhaustion_gate_cfg if isinstance(exhaustion_gate_cfg, dict) else {}
    min_exhaustion = float(gate.get("min_penalty_skip", 0.12))
    exhaustion_fail = (
        bool(metrics.get("exhaustion_conflict"))
        and float(metrics.get("exhaustion_penalty", 0.0)) + 1e-9 >= min_exhaustion
    )
    signal_floor = float(min_signal)
    if recovery_active and isinstance(recovery_kelly_cfg, dict):
        hurst = float(indicators.get("hurst", 0.5))
        signal_floor = recovery_min_signal(
            recovery_kelly_cfg,
            recovery_active=True,
            consecutive_losses=consecutive_losses,
            hurst=hurst,
        )
    checks = [
        eff + 1e-9 < signal_floor,
        min_val > 0.0 and float(metrics.get("val_accuracy", 0.0)) + 1e-9 < min_val,
        edge_floor > 0.0 and edge_val + 1e-9 < edge_floor,
        min_direction_margin > 0.0 and margin + 1e-9 < min_direction_margin,
        metrics.get("direction_inverted") and inverted_min_score > 0.0 and eff + 1e-9 < inverted_min_score,
        not recovery_active and min_adx_normal > 0.0 and adx + 1e-9 < min_adx_normal,
        exhaustion_fail,
    ]
    return any(checks)


def passes_execution_quality(
    metrics: dict,
    *,
    min_signal: float,
    min_val: float,
    min_edge: float,
    min_direction_margin: float = 0.0,
    inverted_min_score: float = 0.0,
    min_adx_normal: float = 0.0,
    recovery_active: bool = False,
    dynamic_threshold_cfg: dict | None = None,
    exhaustion_gate_cfg: dict | None = None,
    recovery_kelly_cfg: dict | None = None,
    consecutive_losses: int = 0,
) -> bool:
    """Indica se metricas pos-resolucao atendem pisos de conviccao e clareza direcional."""
    if not passes_squeeze_gate(metrics, cfg=dynamic_threshold_cfg):
        return False
    return not _quality_failures(
        metrics,
        min_signal=min_signal,
        min_val=min_val,
        min_edge=min_edge,
        min_direction_margin=min_direction_margin,
        inverted_min_score=inverted_min_score,
        min_adx_normal=min_adx_normal,
        recovery_active=recovery_active,
        exhaustion_gate_cfg=exhaustion_gate_cfg,
        recovery_kelly_cfg=recovery_kelly_cfg,
        consecutive_losses=consecutive_losses,
    )
