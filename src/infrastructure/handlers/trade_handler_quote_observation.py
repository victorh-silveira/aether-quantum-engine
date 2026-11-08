"""Registro auditavel da cotacao antes da decisao de compra."""

from __future__ import annotations

from typing import Any

from src.application.services.rise_fall_quote_guard import unit_ev


async def record_indicator_quote(
    writer: Any,
    config: dict,
    symbol: str,
    direction: str,
    parameters: dict,
    proposal_id: str,
    price: float,
    payout: float | None,
    payout_rate: float | None,
) -> None:
    """Separa cotacoes confirmatorias por EV e margem apos haircut."""
    version = parameters.get("_indicator_model_version")
    if not version:
        return
    if writer is None or not hasattr(writer, "record_quote_observation"):
        raise RuntimeError("Captura da cotacao do indicador indisponivel")
    probability = parameters.get("_quote_guard_side_probability")
    setup = str(version).startswith("pullback-1-")
    edge = None if setup and config.get("trading", {}).get("mode") == "demo" else unit_ev(probability, payout_rate)
    confirmatory = (
        edge is not None
        and edge > 0
        and payout_rate is not None
        and float(probability) >= 1.0 / (1.0 + payout_rate) + 0.01
    )
    await writer.record_quote_observation(
        {
            "account_mode": str(config.get("trading", {}).get("mode", "demo")).lower(),
            "symbol": str(symbol),
            "period_seconds": int(parameters.get("_indicator_period_seconds") or 300),
            "candle_epoch": int(parameters.get("_signal_candle_epoch") or 0),
            "model_version": str(version),
            "side": str(direction),
            "proposal_id": str(proposal_id),
            "side_probability": None if setup and config.get("trading", {}).get("mode") == "demo" else probability,
            "quote_price": price,
            "quote_payout": payout,
            "quote_ev": edge,
            "cohort": "setup_pilot"
            if setup and config.get("trading", {}).get("mode") == "demo"
            else "confirmatory"
            if confirmatory
            else "exploratory",
        }
    )
