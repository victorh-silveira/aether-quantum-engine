"""Politica de execucao de clusters US/EU conforme conviccao da LLM."""

from __future__ import annotations

from typing import Any

from src.domain.models.trade import TradeDirection


def cluster_follow_conviction_threshold(orch: Any) -> float:
    """Limiar minimo de conviccao para seguir cluster e predicao sem inversao."""
    llm = orch.config.get("llm", {}) if isinstance(getattr(orch, "config", None), dict) else {}
    ind = llm.get("indicator_config") if isinstance(llm.get("indicator_config"), dict) else {}
    raw = ind.get("cluster_follow_conviction_threshold")
    if raw is not None:
        return max(0.0, min(0.99, float(raw)))
    follow = ind.get("follow_threshold")
    if follow is not None:
        return max(0.0, min(0.99, float(follow)))
    return max(0.0, min(0.99, float(llm.get("min_conviction_execute", 0.85))))


def invert_cluster_tag(tag: str | None) -> str | None:
    """Inverte tag de cluster CALL/PUT."""
    if tag == "CALL":
        return "PUT"
    if tag == "PUT":
        return "CALL"
    return None


def cluster_execution_direction(
    cluster_tag: str | None,
    conviction: float,
    follow_threshold: float,
) -> tuple[TradeDirection | None, bool]:
    """Retorna direcao de execucao do cluster e se houve inversao pela conviccao."""
    if cluster_tag not in ("CALL", "PUT"):
        return None, False
    llm_dir = TradeDirection[cluster_tag]
    if float(conviction) >= float(follow_threshold):
        return llm_dir, False
    inverted = TradeDirection.PUT if llm_dir == TradeDirection.CALL else TradeDirection.CALL
    return inverted, True
