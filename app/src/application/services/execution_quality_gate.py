"""Quality gate neutralizado: boletamento continuo sem vetos nem skips de ciclo."""

MANDATORY_MIN_TRADE_SCORE_DEFAULT = 0.72

__all__ = [
    "apply_quality_penalty_to_metrics",
    "passes_execution_quality",
    "quality_gate_params",
]


def quality_gate_params(exec_cfg: dict) -> dict[str, float]:
    """Le limites configurados em orchestrator.execution.quality_gate (sem efeito de veto)."""
    chunk = exec_cfg.get("quality_gate") if isinstance(exec_cfg, dict) else {}
    if not isinstance(chunk, dict):
        chunk = {}
    return {
        "min_direction_margin": float(chunk.get("min_direction_margin", 0.0)),
        "inverted_min_score": float(chunk.get("inverted_min_score", 0.0)),
        "min_adx_normal": float(chunk.get("min_adx_normal", 0.0)),
    }


def passes_execution_quality(metrics: dict, **_kwargs) -> bool:
    """Sinal matematicamente valido sempre opera: nenhum veto de qualidade e nenhum skip."""
    metrics["regime_skip_cycle"] = False
    return True


def apply_quality_penalty_to_metrics(metrics: dict, **_kwargs) -> float:
    """Sem penalidade de qualidade: mantem trade_score bruto e garante regime_skip_cycle=False."""
    metrics["regime_skip_cycle"] = False
    return 0.0
