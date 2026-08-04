"""Helpers de auditoria de mercado e resolução de métricas."""

from __future__ import annotations

from typing import Any


# Cache em memória para os contratos de auditoria
_CONTRACT_AUDIT_STORE: dict[str, Any] = {}


def resolve_meta_payoff_zscore(metrics: dict[str, Any] | None) -> float:
    """Resolve o z-score do meta payoff."""
    if not isinstance(metrics, dict):
        return 0.0
    return float(metrics.get("meta_payoff_edge_zscore", metrics.get("edge_zscore", 0.0)))


def resolve_predicted_edge(
    metrics: dict[str, Any],
    direction: str | None = None,
    payout: float = 0.95,
) -> float:
    """Calcula o edge previsto. Se direction for fornecido, edge é direcional (pode ser negativo)."""
    if not isinstance(metrics, dict):
        return 0.0
    prob = metrics.get("calibrated_prob", metrics.get("raw_prob", 0.5))
    if prob is None:
        return 0.0
    try:
        p = float(prob)
    except (TypeError, ValueError):
        return 0.0
    if direction and str(direction).upper() == "PUT":
        p = 1.0 - p
    elif not direction:
        p = max(p, 1.0 - p)
    return float((p * (1.0 + payout)) - 1.0)


def cluster_symbol_token(symbol: str | None, entry: dict[str, Any] | None = None) -> str:
    """Normaliza e retorna a tag/token do símbolo no cluster com dados da entry."""
    if not symbol:
        return "N/A"
    sym = str(symbol).upper().strip()
    if entry is None or not isinstance(entry, dict):
        return sym
    metrics = entry.get("metrics") or {}
    raw_p = _safe_float(metrics.get("raw_prob"), 0.5)
    cal_p = _safe_float(metrics.get("calibrated_prob"), 0.5)
    margin = abs(0.5 - cal_p)
    raw_dir = (
        metrics.get("exec_direction")
        or metrics.get("resolved_direction")
        or entry.get("direction")
        or metrics.get("dl_direction")
    )
    direction = str(raw_dir).replace("TradeDirection.", "").upper() if raw_dir is not None else "N/A"
    is_put = direction == "PUT"
    raw_p = 1.0 - raw_p if is_put else raw_p
    cal_p = 1.0 - cal_p if is_put else cal_p
    display_edge = resolve_predicted_edge(metrics, direction=direction)
    skip = _resolve_skip_reason(entry, metrics)
    if skip:
        return f"{sym}: {direction} (Prob: {raw_p:.3f} Cal: {cal_p:.3f} | {skip})"
    return f"{sym}: {direction} (Prob: {raw_p:.3f} Cal: {cal_p:.3f} Margin: {margin:.3f} Edge: {display_edge:+.3f})"


def _safe_float(value: Any, default: float) -> float:
    """Converte valor para float ou retorna default."""
    try:
        return float(value)
    except (ValueError, TypeError):
        return default


def _resolve_skip_reason(_entry: dict[str, Any], metrics: dict[str, Any]) -> str | None:
    """Determina a razao de skip do simbolo no cluster."""
    gate_reason = metrics.get("gate_reason")
    if gate_reason:
        return f"SKIP:{str(gate_reason).upper().strip()}"
    if metrics.get("quality_guard_reject"):
        return "NEUTRO_SKIP"
    if str(metrics.get("signal_status", "")).upper() == "SKIP":
        return "NEUTRO_SKIP"
    if not metrics.get("execute", True):
        return "NEUTRO_SKIP"
    return None


def resolve_cluster_timeframe(metrics: dict[str, Any] | None) -> str:
    """Resolve TF do CLUSTER priorizando o decisor micro (contrato)."""
    if not isinstance(metrics, dict):
        return "M2"
    data_handler = metrics.get("data_handler")
    if isinstance(data_handler, dict):
        micro = data_handler.get("micro_granularity")
        if micro is not None:
            return _granularity_to_tf(int(micro))
        granularity = data_handler.get("granularity")
        if granularity is not None:
            return _granularity_to_tf(int(granularity))
    return str(metrics.get("timeframe", metrics.get("tf", "M2")))


def _granularity_to_tf(seconds: int) -> str:
    """Converte granularidade em segundos para rotulo de timeframe."""
    if seconds >= 86400:
        return f"D{seconds // 86400}"
    if seconds >= 3600:
        return f"H{seconds // 3600}"
    if seconds >= 60:
        return f"M{seconds // 60}"
    return f"S{seconds}"


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


def store_contract_audit(*args: Any, **kwargs: Any) -> None:
    """Armazena dados de auditoria do contrato (suporta APIs legado e nova)."""
    if args and isinstance(args[0], str) and len(args) == 2 and isinstance(args[1], dict):
        _store_new(args[0], args[1])
    elif kwargs:
        _store_legacy(args, kwargs)
    elif args and len(args) >= 2:
        _store_legacy(args, {})


def _store_new(contract_id: str, audit_data: dict[str, Any]) -> None:
    """API nova: store_contract_audit(contract_id, audit_data)."""
    if contract_id:
        _CONTRACT_AUDIT_STORE[str(contract_id)] = audit_data


def _store_legacy(args: tuple[Any, ...], kwargs: dict[str, Any]) -> None:
    """API legado: store_contract_audit(orch, contract_id, symbol=..., direction=..., ...)."""
    if len(args) < 2:
        return
    contract_id = str(args[1])
    if not contract_id:
        return
    audit_data = {
        "symbol": kwargs.get("symbol"),
        "direction": kwargs.get("direction"),
        "edge": kwargs.get("edge"),
        "meta_payoff_edge_zscore": kwargs.get("meta_payoff_edge_zscore"),
        "raw_prob": kwargs.get("raw_prob"),
    }
    _CONTRACT_AUDIT_STORE[contract_id] = audit_data


def pop_contract_audit(*args: Any, **kwargs: Any) -> Any:
    """Recupera e remove dados de auditoria (suporta APIs legado e nova)."""
    _ = kwargs
    if len(args) == 1 and isinstance(args[0], str):
        return _pop_new(args[0])
    if len(args) >= 2:
        return _pop_legacy(args)
    return _pop_new(str(args[0]) if args else "")


def _pop_new(contract_id: str) -> dict[str, Any]:
    """API nova: pop_contract_audit(contract_id) -> dict."""
    return _CONTRACT_AUDIT_STORE.pop(str(contract_id), {})


def _pop_legacy(args: tuple[Any, ...]) -> tuple[Any, ...]:
    """API legado: pop_contract_audit(orch, contract_id) -> (symbol, direction, edge, zscore, raw_prob)."""
    contract_id = str(args[1])
    data = _CONTRACT_AUDIT_STORE.pop(contract_id, {})
    if not data:
        return ("", "", 0.0, None, None)
    return (
        data.get("symbol", ""),
        data.get("direction", ""),
        data.get("edge", 0.0),
        data.get("meta_payoff_edge_zscore"),
        data.get("raw_prob"),
    )
