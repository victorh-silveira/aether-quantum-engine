"""Helpers de auditoria de mercado e resolução de métricas."""

from typing import Any


# Cache em memória para os contratos de auditoria
_CONTRACT_AUDIT_STORE: dict[str, dict[str, Any]] = {}


def resolve_meta_payoff_zscore(metrics: dict[str, Any] | None) -> float:
    """Resolve o z-score do meta payoff."""
    if not isinstance(metrics, dict):
        return 0.0
    return float(metrics.get("meta_payoff_edge_zscore", metrics.get("edge_zscore", 0.0)))


def resolve_predicted_edge(metrics: dict[str, Any], payout: float = 0.95) -> float:
    """Calcula o edge previsto baseado na probabilidade dominante (win probability)."""
    if not isinstance(metrics, dict):
        return 0.0
    prob = metrics.get("calibrated_prob", metrics.get("raw_prob", 0.5))
    if prob is None:
        return 0.0
    p = float(prob)
    p_win = max(p, 1.0 - p)
    return float((p_win * (1.0 + payout)) - 1.0)


def cluster_symbol_token(symbol: str | None) -> str:
    """Normaliza e retorna a tag/token do símbolo no cluster."""
    if not symbol:
        return "N/A"
    return str(symbol).upper().strip()


def resolve_cluster_timeframe(metrics: dict[str, Any] | None) -> str:
    """Resolve a string representativa do timeframe do cluster."""
    if not isinstance(metrics, dict):
        return "M5"
    return str(metrics.get("timeframe", metrics.get("tf", "M5")))


def indicator_snapshot(metrics: dict[str, Any] | None) -> dict[str, Any]:
    """Extrai um snapshot formatado dos indicadores técnicos."""
    if not isinstance(metrics, dict):
        return {}
    return metrics.get("indicators", metrics.get("indicator_snapshot", {}))


def metric_float(metrics: dict[str, Any] | None, *keys: str, default: float = 0.0) -> float:
    """Extrai valor float de uma lista de chaves possíveis no dicionário de métricas."""
    if not isinstance(metrics, dict):
        return default
    for k in keys:
        if k in metrics and metrics[k] is not None:
            try:
                return float(metrics[k])
            except (ValueError, TypeError):
                pass
    return default


def store_contract_audit(contract_id: str, audit_data: dict[str, Any]) -> None:
    """Armazena os dados de auditoria do contrato."""
    if contract_id:
        _CONTRACT_AUDIT_STORE[str(contract_id)] = audit_data


def pop_contract_audit(contract_id: str) -> dict[str, Any]:
    """Recupera e remove os dados de auditoria do contrato."""
    return _CONTRACT_AUDIT_STORE.pop(str(contract_id), {})
