"""Volatility Overdrive Override: desvio tatico por conviccao assimétrica inter-symbol."""

from typing import Any

from src.application.services.execution_market_rank import market_decision_score
from src.application.services.execution_quality_gate import ensure_direction_margin, passes_execution_quality
from src.domain.models.trade import TradeDirection


NEUTRAL_MARGIN_EPSILON = 0.02
OVERDRIVE_MIN_DIRECTION_MARGIN = 0.15
OVERDRIVE_MIN_META_ZSCORE = 0.0


__all__ = [
    "try_volatility_overdrive_override",
    "volatility_overdrive_unblocks_cluster",
]


def _meta_payoff_edge_zscore(metrics: dict) -> float | None:
    """Le Z-Score meta-regressor do candidato para overdrive."""
    raw = metrics.get("meta_payoff_edge_zscore")
    if raw is not None:
        return float(raw)
    if metrics.get("meta_classifier_applied") or metrics.get("meta_applied"):
        return None
    fallback = metrics.get("edge_zscore")
    if fallback is None:
        return None
    return float(fallback)


def _passes_dynamic_quality_gate(metrics: dict, orch: Any) -> bool:
    """Indica se o candidato respeita o quality gate dinamico corrente."""
    config = getattr(orch, "config", {}) if orch is not None else {}
    exec_cfg = config.get("orchestrator", {}).get("execution", {}) if isinstance(config, dict) else {}
    risk_manager = getattr(orch, "risk_manager", None)
    return passes_execution_quality(metrics, exec_cfg=exec_cfg, risk_manager=risk_manager)


def _direction_margin(metrics: dict) -> float:
    """Retorna margem direcional TCN garantida no contêiner de métricas."""
    return ensure_direction_margin(metrics)


def _neutral_quality_failure(metrics: dict, orch: Any) -> bool:
    """True quando o quality gate reprova candidato com conviccao neutra."""
    if _passes_dynamic_quality_gate(metrics, orch):
        return False
    return _direction_margin(metrics) <= NEUTRAL_MARGIN_EPSILON


def _overdrive_conviction_eligible(metrics: dict) -> bool:
    """True quando par alternativo tem conviccao assimétrica e meta Z-Score nao degradado."""
    if _direction_margin(metrics) + 1e-12 < OVERDRIVE_MIN_DIRECTION_MARGIN:
        return False
    zscore = _meta_payoff_edge_zscore(metrics)
    if zscore is None:
        return False
    return zscore + 1e-12 >= OVERDRIVE_MIN_META_ZSCORE


def try_volatility_overdrive_override(
    orch: Any,
    ranked: list[tuple[str, TradeDirection, dict]],
) -> tuple[str, TradeDirection, dict] | None:
    """Substitui lider neutro reprovado pelo par com conviccao assimétrica no mesmo ciclo."""
    if len(ranked) < 2:
        return None
    leader_symbol, _, leader_metrics = ranked[0]
    if not _neutral_quality_failure(leader_metrics, orch):
        return None
    for alternate in ranked[1:]:
        alt_symbol, alt_direction, alt_metrics = alternate
        if not _overdrive_conviction_eligible(alt_metrics):
            continue
        alt_metrics["volatility_overdrive_selected"] = True
        alt_metrics["volatility_overdrive_ignored_symbol"] = leader_symbol
        call_prob = float(
            alt_metrics.get("calibrated_prob", alt_metrics.get("raw_prob", alt_metrics.get("tcn_score", 0.5)))
        )
        side_prob = call_prob if alt_direction.name == "CALL" else 1.0 - call_prob
        alt_metrics["volatility_overdrive_conviction"] = side_prob
        alt_metrics.pop("quality_guard_reject", None)
        alt_metrics.pop("regime_skip_cycle", None)
        alt_metrics.pop("quality_gate_reason", None)
        leader_metrics["volatility_overdrive_bypassed"] = True
        return alt_symbol, alt_direction, alt_metrics
    return None


def _decisions_ranked_pool(decisions: dict) -> list[tuple[str, TradeDirection, dict]]:
    """Monta pool ordenado por score a partir do mapa de decisoes DL do ciclo."""
    pool: list[tuple[str, TradeDirection, dict]] = []
    for symbol, entry in decisions.items():
        if not isinstance(entry, dict):
            continue
        metrics = entry.get("metrics")
        if not isinstance(metrics, dict):
            continue
        direction_name = str(metrics.get("exec_direction") or metrics.get("dl_direction") or "CALL").upper()
        direction = TradeDirection.CALL if direction_name == "CALL" else TradeDirection.PUT
        pool.append((str(symbol), direction, metrics))
    if not pool:
        return []
    return sorted(
        pool,
        key=lambda item: market_decision_score(
            item[2],
            exec_direction=item[1],
            recovery_active=False,
            symbol=item[0],
        ),
        reverse=True,
    )


def volatility_overdrive_unblocks_cluster(orch: Any, decisions: dict) -> bool:
    """Libera cluster quando overdrive substitui simbolo neutro por par convicto."""
    redirect = try_volatility_overdrive_override(orch, _decisions_ranked_pool(decisions))
    return redirect is not None
