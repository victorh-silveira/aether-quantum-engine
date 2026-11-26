"""Cotacao preliminar e contexto financeiro do ciclo continuo Laya."""

from typing import Any

from src.domain.risk.kelly_recovery import sizing_from_config


async def quote_laya_sizing(orch: Any, decision: Any) -> Any:
    """Consulta payout real sem comprar e calcula stake inicial de referencia."""
    config = orch.config.get("risk_management", {})
    minimum = float(config.get("params", {}).get("stake_min", 1.0))
    quote = await orch.trade_handler.fetch_proposal_snapshot(
        decision.symbol, decision.direction, minimum, {"duration": 5, "duration_unit": "m"}
    )
    if quote is None:
        raise RuntimeError("LAYA_SIZING cotacao preliminar invalida")
    return sizing_from_config(
        float(orch.state.balance),
        decision.probability,
        float(quote["rate"]),
        float(orch.risk_manager.pending_loss),
        config,
    )
