"""Pisos de qualidade para filtrar candidatos antes da execucao."""

from src.application.services.orchestrator.execution_recovery_gate import effective_signal


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
) -> bool:
    """Retorna True quando algum piso de qualidade nao e atendido."""
    eff = effective_signal(metrics)
    indicators = metrics.get("indicators") or {}
    adx = float(indicators.get("adx", 0.0))
    margin = float(metrics.get("direction_margin", 0.0))
    checks = [
        eff + 1e-9 < min_signal,
        min_val > 0.0 and float(metrics.get("val_accuracy", 0.0)) + 1e-9 < min_val,
        min_edge > 0.0 and float(metrics.get("edge", 0.0)) + 1e-9 < min_edge,
        min_direction_margin > 0.0 and margin + 1e-9 < min_direction_margin,
        metrics.get("direction_inverted") and inverted_min_score > 0.0 and eff + 1e-9 < inverted_min_score,
        not recovery_active and min_adx_normal > 0.0 and adx + 1e-9 < min_adx_normal,
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
) -> bool:
    """Indica se metricas pos-resolucao atendem pisos de conviccao e clareza direcional."""
    return not _quality_failures(
        metrics,
        min_signal=min_signal,
        min_val=min_val,
        min_edge=min_edge,
        min_direction_margin=min_direction_margin,
        inverted_min_score=inverted_min_score,
        min_adx_normal=min_adx_normal,
        recovery_active=recovery_active,
    )
