"""Helpers de formatacao e persistencia para auditoria de mercado."""

from __future__ import annotations

from typing import Any


def resolve_predicted_edge(metrics: dict[str, Any]) -> float:
    """Extrai edge continuo do meta-regressor a partir das metricas do ciclo."""
    raw = metrics.get("predicted_payoff_edge", metrics.get("meta_calibrated_payoff_score", 0.0))
    return float(raw or 0.0)


def resolve_meta_payoff_zscore(metrics: dict[str, Any] | None) -> float | None:
    """Extrai z-score de payoff meta das metricas quando disponivel."""
    if not isinstance(metrics, dict):
        return None
    for key in ("meta_payoff_edge_zscore", "edge_zscore"):
        raw = metrics.get(key)
        if raw is None:
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return None


def resolve_cluster_timeframe(config: dict[str, Any] | None) -> str:
    """Resolve rotulo de timeframe do cluster a partir da granularidade configurada."""
    if not isinstance(config, dict):
        return "M5"
    data = config.get("data_handler")
    if not isinstance(data, dict):
        data = {}
    seconds = int(data.get("micro_granularity", data.get("granularity", 300)) or 300)
    if seconds >= 900:
        return "M15"
    if seconds >= 300:
        return "M5"
    if seconds >= 60:
        return f"M{max(1, seconds // 60)}"
    return f"S{seconds}"


def indicator_snapshot(metrics: dict[str, Any]) -> dict[str, float]:
    """Consolida indicadores macro e micro em um unico mapa numerico."""
    merged: dict[str, float] = {}
    for bucket in ("indicators", "macro_indicators", "micro_indicators"):
        chunk = metrics.get(bucket)
        if not isinstance(chunk, dict):
            continue
        for key, raw in chunk.items():
            if raw is None:
                continue
            try:
                merged[str(key)] = float(raw)
            except (TypeError, ValueError):
                continue
    return merged


def metric_float(metrics: dict[str, Any], *keys: str, default: float = 0.0) -> float:
    """Le o primeiro campo numerico disponivel nas metricas."""
    for key in keys:
        raw = metrics.get(key)
        if raw is None:
            continue
        try:
            return float(raw)
        except (TypeError, ValueError):
            continue
    return default


def veto_token(metrics: dict[str, Any]) -> str | None:
    """Resolve rotulo curto de veto de microestrutura ou quality gate."""
    for key in ("quality_gate_reason", "gate_reason"):
        raw = metrics.get(key)
        if not raw:
            continue
        token = str(raw).strip().upper().replace("-", "_")
        if token in {"NEUTRAL_CLAMP", "NEUTRO_CLAMP"}:
            return "NEUTRO_VETO"
        if token:
            return token
    if metrics.get("quality_guard_reject") or metrics.get("regime_skip_cycle"):
        return "NEUTRO_VETO"
    if metrics.get("execute") is False and metrics.get("deploy_ok") is not False:
        return "NEUTRO_VETO"
    return None


def cluster_symbol_token(symbol: str, entry: dict[str, Any]) -> str:
    """Formata token de simbolo no resumo CLUSTER."""
    metrics = entry.get("metrics") if isinstance(entry.get("metrics"), dict) else {}
    direction = entry.get("direction")
    if direction is not None and hasattr(direction, "name"):
        side = str(direction.name)
    else:
        side = str(
            metrics.get("exec_direction") or metrics.get("dl_direction") or metrics.get("resolved_direction") or "FLAT"
        ).upper()
    raw_prob = metric_float(metrics, "raw_prob", default=0.5)
    cal_prob = metric_float(metrics, "calibrated_prob", "raw_prob", default=raw_prob)
    margin_raw = metrics.get("direction_margin")
    if margin_raw is None:
        margin = abs(float(cal_prob) - 0.5)
    else:
        try:
            margin = float(margin_raw)
        except (TypeError, ValueError):
            margin = abs(float(cal_prob) - 0.5)
    edge = resolve_predicted_edge(metrics)
    detail = f"Prob: {raw_prob:0.3f} Cal: {cal_prob:0.3f} Margin: {margin:0.3f} Edge: {edge:+0.3f}"
    veto = veto_token(metrics)
    if veto is not None:
        return f"{symbol}: {side} ({detail} | {veto})"
    return f"{symbol}: {side} ({detail})"


def store_contract_audit(
    orch: Any,
    contract_id: int,
    *,
    symbol: str,
    direction: str,
    edge: float,
    meta_payoff_edge_zscore: float | None = None,
    raw_prob: float | None = None,
) -> None:
    """Persiste metadados de auditoria por contrato ate a liquidacao."""
    bag = getattr(orch, "_contract_audit", None)
    if bag is None:
        orch._contract_audit = {}
        bag = orch._contract_audit
    snap: dict[str, Any] = {
        "symbol": str(symbol),
        "direction": str(direction),
        "edge": float(edge),
    }
    if meta_payoff_edge_zscore is not None:
        snap["meta_payoff_edge_zscore"] = float(meta_payoff_edge_zscore)
    if raw_prob is not None:
        snap["raw_prob"] = float(raw_prob)
    bag[int(contract_id)] = snap


def pop_contract_audit(
    orch: Any,
    contract_id: int,
    *,
    contract: Any = None,
    symbol: str = "UNK",
) -> tuple[str, str, float, float | None, float | None]:
    """Recupera e remove metadados de auditoria de um contrato liquidado."""
    bag = getattr(orch, "_contract_audit", None) or {}
    snap = bag.pop(int(contract_id), None)
    if isinstance(snap, dict):
        z_raw = snap.get("meta_payoff_edge_zscore")
        z_score = float(z_raw) if z_raw is not None else None
        p_raw = snap.get("raw_prob")
        raw_prob = float(p_raw) if p_raw is not None else None
        return (
            str(snap.get("symbol", symbol)),
            str(snap.get("direction", "UNK")),
            float(snap.get("edge", 0.0)),
            z_score,
            raw_prob,
        )
    dir_name = "UNK"
    if contract is not None:
        loss_dir = getattr(contract, "direction", None)
        if loss_dir is not None:
            dir_name = loss_dir.name
    return str(symbol), dir_name, 0.0, None, None
