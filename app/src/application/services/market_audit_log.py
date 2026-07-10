"""Formatos unificados de auditoria de mercado para terminal e monitoramento."""

from __future__ import annotations

from typing import Any


_INDICATOR_KEYS = (
    "rsi",
    "macd_hist",
    "atr_norm",
    "bb_width",
    "hurst",
    "implied_vol_ratio",
    "vol_ratio",
    "momentum",
    "roc",
    "stoch_k",
    "adx",
    "cci",
)


def resolve_predicted_edge(metrics: dict[str, Any]) -> float:
    """Extrai edge continuo do meta-regressor a partir das metricas do ciclo."""
    raw = metrics.get("predicted_payoff_edge", metrics.get("meta_calibrated_payoff_score", 0.0))
    return float(raw or 0.0)


def _indicator_snapshot(metrics: dict[str, Any]) -> dict[str, float]:
    """Consolida indicadores macro e micro em um unico mapa numerico."""
    merged: dict[str, float] = {}
    for bucket in ("indicators", "micro_indicators"):
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


def format_settlement_audit_line(
    cycle_id: int,
    outcome: str,
    profit: float,
    direction: str,
    symbol: str,
    edge: float,
) -> str:
    """Monta linha padronizada de liquidacao WIN/LOSS."""
    return (
        f"[C{int(cycle_id):04d}] STATUS: {outcome} || "
        f"P&L: ${float(profit):+.2f} || {direction} || "
        f"sym={symbol} || edge={float(edge):.4f}"
    )


def format_direction_audit_line(
    cycle_id: int,
    direction: str,
    symbol: str,
    edge: float,
    *,
    dl_direction: str | None = None,
) -> str:
    """Monta linha padronizada de selecao de direcao micro."""
    flip = ""
    if dl_direction and dl_direction != direction:
        flip = f" || dl={dl_direction} inv"
    return f"[C{int(cycle_id):04d}] DIR_SEL || ord={direction}{flip} || sym={symbol} || edge={float(edge):.4f}"


def format_execution_audit_line(
    cycle_id: int,
    symbol: str,
    direction: str,
    tcn_score: float,
    edge: float,
    *,
    z_edge: float = 0.0,
) -> str:
    """Monta linha padronizada de boletamento EXEC_SEL com TCN, edge e Z-Score."""
    return (
        f"[C{int(cycle_id):04d}] EXEC_SEL | {symbol} | ord={direction} | "
        f"TCN={float(tcn_score):.2f} | edge={float(edge):.4f} | Z={float(z_edge):+.2f}"
    )


def format_indicators_audit_line(cycle_id: int, symbol: str, metrics: dict[str, Any]) -> str:
    """Monta linha com indicadores tecnicos e scores de direcao para debug."""
    snapshot = _indicator_snapshot(metrics)
    parts: list[str] = []
    for key in _INDICATOR_KEYS:
        if key in snapshot:
            parts.append(f"{key}={snapshot[key]:.4f}")
    for extra_key in ("raw_prob", "calibrated_prob", "direction_margin", "val_accuracy"):
        raw = metrics.get(extra_key)
        if raw is not None:
            parts.append(f"{extra_key}={float(raw):.4f}")
    dl_dir = metrics.get("dl_direction")
    if dl_dir:
        parts.append(f"dl={dl_dir}")
    exec_dir = metrics.get("exec_direction")
    if exec_dir:
        parts.append(f"exec={exec_dir}")
    body = " ".join(parts) if parts else "sem_indicadores"
    return f"[C{int(cycle_id):04d}] IND | {symbol} | {body}"


def store_contract_audit(
    orch: Any,
    contract_id: int,
    *,
    symbol: str,
    direction: str,
    edge: float,
) -> None:
    """Persiste metadados de auditoria por contrato ate a liquidacao."""
    bag = getattr(orch, "_contract_audit", None)
    if bag is None:
        orch._contract_audit = {}
        bag = orch._contract_audit
    bag[int(contract_id)] = {
        "symbol": str(symbol),
        "direction": str(direction),
        "edge": float(edge),
    }


def pop_contract_audit(
    orch: Any,
    contract_id: int,
    *,
    contract: Any = None,
    symbol: str = "UNK",
) -> tuple[str, str, float]:
    """Recupera e remove metadados de auditoria de um contrato liquidado."""
    bag = getattr(orch, "_contract_audit", None) or {}
    snap = bag.pop(int(contract_id), None)
    if isinstance(snap, dict):
        return str(snap.get("symbol", symbol)), str(snap.get("direction", "UNK")), float(snap.get("edge", 0.0))
    dir_name = "UNK"
    if contract is not None:
        loss_dir = getattr(contract, "direction", None)
        if loss_dir is not None:
            dir_name = loss_dir.name
    return str(symbol), dir_name, 0.0
