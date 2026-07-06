"""Formatos unificados de auditoria de mercado para terminal e monitoramento."""

from __future__ import annotations

from typing import Any

from src.application.services.payoff_edge_zscore import classify_edge_expectancy


def resolve_predicted_edge(metrics: dict[str, Any]) -> float:
    """Extrai edge continuo do meta-regressor a partir das metricas do ciclo."""
    raw = metrics.get("predicted_payoff_edge", metrics.get("meta_calibrated_payoff_score", 0.0))
    return float(raw or 0.0)


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
    expectancy: str | None = None,
) -> str:
    """Monta linha padronizada de boletamento EXEC_SEL com Z-Score e expectativa."""
    state = expectancy or classify_edge_expectancy(float(edge), float(z_edge))
    return (
        f"[C{int(cycle_id):04d}] EXEC_SEL | {symbol} | ord={direction} | "
        f"TCN={float(tcn_score):.2f} | edge={float(edge):.4f} (Z={float(z_edge):+.2f}) | {state}"
    )


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
