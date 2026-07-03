"""Pisos de qualidade para filtrar candidatos antes da execucao."""

from src.application.services.deep_learning.dl_gating import resolve_calibrated_edge
from src.application.services.execution_quality_asymmetric_gate import (
    NEUTRAL_REGIME_TOKEN,
    validate_recovery_asymmetric_gate,
)
from src.application.services.execution_squeeze_gate import passes_squeeze_gate
from src.application.services.execution_universal_regime_gate import UniversalRegimeEvaluator
from src.application.services.execution_universal_regime_types import RegimeState
from src.application.services.orchestrator.execution_recovery_gate import effective_signal, recovery_min_signal
from src.domain.models.trade import TradeDirection
from src.domain.risk.recovery_hurst_decay import resolve_effective_hurst_min


MANDATORY_MIN_TRADE_SCORE_DEFAULT = 0.72
MACRO_VOL_EXPANSION_MIN = 1.15
MICRO_VOL_DIVERGENCE_MAX = 0.50


def quality_gate_params(exec_cfg: dict) -> dict[str, float]:
    """Le limites de qualidade configurados em orchestrator.execution.quality_gate."""
    chunk = exec_cfg.get("quality_gate") if isinstance(exec_cfg, dict) else {}
    if not isinstance(chunk, dict):
        chunk = {}
    return {
        "min_direction_margin": float(chunk.get("min_direction_margin", 0.06)),
        "inverted_min_score": float(chunk.get("inverted_min_score", 0.76)),
        "min_adx_normal": float(chunk.get("min_adx_normal", 0.20)),
    }


def _macro_vol_ratio(metrics: dict) -> float:
    """Le vol_ratio macro M15 com fallback para indicadores micro."""
    macro = metrics.get("macro_indicators")
    if isinstance(macro, dict) and macro.get("vol_ratio") is not None:
        return float(macro["vol_ratio"])
    indicators = metrics.get("indicators") or {}
    return float(indicators.get("vol_ratio", 1.0))


def _micro_vol_ratio(metrics: dict) -> float:
    """Le vol_ratio micro M1 dos indicadores de execucao."""
    indicators = metrics.get("indicators") or {}
    return float(indicators.get("vol_ratio", _macro_vol_ratio(metrics)))


def apply_vol_cohesion_entropic_downgrade(metrics: dict) -> bool:
    """Rebaixa para ENTROPIC_NOISE quando M1 comprime sob macro M15 expandido."""
    macro_vol = _macro_vol_ratio(metrics)
    micro_vol = _micro_vol_ratio(metrics)
    if macro_vol + 1e-9 < MACRO_VOL_EXPANSION_MIN or micro_vol + 1e-9 >= MICRO_VOL_DIVERGENCE_MAX:
        return False
    regime = RegimeState.ENTROPIC_NOISE.value
    metrics["universal_regime"] = regime
    metrics["universal_regime_scenario"] = regime
    metrics["gate_penalty"] = "noise"
    metrics["regime_skip_cycle"] = True
    metrics["vol_cohesion_divergence"] = True
    metrics["vol_cohesion_macro_ratio"] = macro_vol
    metrics["vol_cohesion_micro_ratio"] = micro_vol
    return True


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
    recovery_skip_counter: int = 0,
    session_drawdown: float = 0.0,
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
    vol_cohesion_fail = bool(metrics.get("vol_cohesion_divergence"))
    asymmetric_fail = bool(metrics.get("recovery_asymmetric_gate"))
    signal_floor = float(min_signal)
    if recovery_active and isinstance(recovery_kelly_cfg, dict):
        hurst = float(indicators.get("hurst", 0.5))
        hurst_min = resolve_effective_hurst_min(
            recovery_kelly_cfg,
            recovery_skip_counter,
            consecutive_losses=consecutive_losses,
            session_drawdown=session_drawdown,
        )
        signal_floor = recovery_min_signal(
            recovery_kelly_cfg,
            recovery_active=True,
            consecutive_losses=consecutive_losses,
            hurst=hurst,
            hurst_persistence_min=hurst_min,
        )
    checks = [
        eff + 1e-9 < signal_floor,
        min_val > 0.0 and float(metrics.get("val_accuracy", 0.0)) + 1e-9 < min_val,
        edge_floor > 0.0 and edge_val + 1e-9 < edge_floor,
        min_direction_margin > 0.0 and margin + 1e-9 < min_direction_margin,
        metrics.get("direction_inverted") and inverted_min_score > 0.0 and eff + 1e-9 < inverted_min_score,
        not recovery_active and min_adx_normal > 0.0 and adx + 1e-9 < min_adx_normal,
        exhaustion_fail,
        vol_cohesion_fail,
        asymmetric_fail,
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
    recovery_skip_counter: int = 0,
    session_drawdown: float = 0.0,
) -> bool:
    """Indica se metricas atendem pisos de qualidade (legado para testes)."""
    apply_vol_cohesion_entropic_downgrade(metrics)
    validate_recovery_asymmetric_gate(metrics)
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
        recovery_skip_counter=recovery_skip_counter,
        session_drawdown=session_drawdown,
    )


def quality_gate_penalty(
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
    recovery_skip_counter: int = 0,
    session_drawdown: float = 0.0,
) -> float:
    """Penalidade composta em [0, 1] quando pisos de qualidade nao sao atingidos."""
    penalties: list[float] = []
    if not passes_squeeze_gate(metrics, cfg=dynamic_threshold_cfg):
        penalties.append(0.35)
    if _quality_failures(
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
        recovery_skip_counter=recovery_skip_counter,
        session_drawdown=session_drawdown,
    ):
        penalties.append(0.45)
    if not penalties:
        return 0.0
    return min(1.0, sum(penalties))


def apply_quality_penalty_to_metrics(
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
    recovery_skip_counter: int = 0,
    session_drawdown: float = 0.0,
) -> float:
    """Aplica penalidade de qualidade ao trade_score sem bloquear execucao."""
    dl_raw = metrics.get("dl_direction") or metrics.get("resolved_direction")
    exec_raw = metrics.get("exec_direction") or metrics.get("resolved_direction")
    dl_dir = None
    exec_dir = None
    try:
        if dl_raw:
            dl_dir = TradeDirection[str(dl_raw).upper()]
        if exec_raw:
            exec_dir = TradeDirection[str(exec_raw).upper()]
    except (KeyError, ValueError):
        dl_dir = None
        exec_dir = None
    if metrics.get("universal_regime") is None and dl_dir is not None and exec_dir is not None:
        apply_vol_cohesion_entropic_downgrade(metrics)
        if metrics.get("universal_regime") is None:
            evaluator = UniversalRegimeEvaluator(
                None,
                recovery_active=recovery_active,
                mandatory_min_signal=min_signal,
                kelly_cfg=recovery_kelly_cfg,
            )
            regime_eval = evaluator.evaluate(metrics, dl_dir=dl_dir, exec_dir=exec_dir)
            evaluator.apply(metrics, regime_eval, exec_dir, dl_dir=dl_dir)
            if regime_eval.regime is None and not metrics.get("universal_regime"):
                metrics["universal_regime"] = NEUTRAL_REGIME_TOKEN
                metrics["universal_regime_scenario"] = NEUTRAL_REGIME_TOKEN
    validate_recovery_asymmetric_gate(metrics)
    penalty = quality_gate_penalty(
        metrics,
        min_signal=min_signal,
        min_val=min_val,
        min_edge=min_edge,
        min_direction_margin=min_direction_margin,
        inverted_min_score=inverted_min_score,
        min_adx_normal=min_adx_normal,
        recovery_active=recovery_active,
        dynamic_threshold_cfg=dynamic_threshold_cfg,
        exhaustion_gate_cfg=exhaustion_gate_cfg,
        recovery_kelly_cfg=recovery_kelly_cfg,
        consecutive_losses=consecutive_losses,
        recovery_skip_counter=recovery_skip_counter,
        session_drawdown=session_drawdown,
    )
    if penalty <= 0.0:
        return 0.0
    metrics["quality_gate_penalty"] = penalty
    for key in ("trade_score", "conviction"):
        if key in metrics and metrics[key] is not None:
            metrics[key] = max(0.0, float(metrics[key]) * (1.0 - penalty))
    if penalty >= 0.35:
        metrics["execution_mode"] = metrics.get("execution_mode") or "EXEC_FALLBACK"
        metrics["fallback_reason"] = metrics.get("fallback_reason") or "qualidade_baixa"
    return penalty
