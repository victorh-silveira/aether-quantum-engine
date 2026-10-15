"""Auditoria de compra confirmada por versao e contrato."""

from __future__ import annotations

import math
from typing import Any


async def record_model_attribution(
    orch: Any, contract_id: int, symbol: str, direction: str, metrics: dict, contract: Any = None
) -> None:
    """Associa a compra confirmada ao checkpoint, sem repetir a ordem em falha de auditoria."""
    version = metrics.get("model_version")
    writer = getattr(orch, "market_writer", None)
    indicator_version = metrics.get("indicator_model_version")
    if writer is None or (
        (not isinstance(version, str) or len(version) != 64)
        and not (isinstance(indicator_version, str) and indicator_version)
    ):
        return
    quote_price = getattr(contract, "quote_price", None)
    quote_payout = getattr(contract, "quote_payout", None)
    quote_ev = getattr(contract, "quote_ev", None)
    try:
        quoted = (
            quote_price is not None
            and quote_payout is not None
            and quote_ev is not None
            and float(quote_price) > 0
            and float(quote_payout) > float(quote_price)
            and abs(float(quote_price) - float(contract.buy_price)) <= 0.01
            and float(quote_ev) > 0
            and float(quote_ev) >= 0.01 * float(quote_payout) / float(quote_price) - 1e-9
        )
    except (TypeError, ValueError, AttributeError):
        quoted = False
    probability = (
        None
        if str(indicator_version).startswith("pullback-1-")
        else metrics.get("tcn_calibrated_prob", metrics.get("calibrated_prob"))
    )
    try:
        probability = float(probability)
    except (TypeError, ValueError):
        probability = None
    if probability is not None and (not math.isfinite(probability) or not 0 <= probability <= 1):
        probability = None
    try:
        await writer.enqueue_contract_audit(
            {
                "contract_id": contract_id,
                "symbol": symbol,
                "account_mode": str(orch.config.get("trading", {}).get("mode", "demo")),
                "direction": direction,
                "settlement_source": "pending",
                "model_version": version if isinstance(version, str) and len(version) == 64 else None,
                "calibrated_call_prob": probability,
                "indicator_model_version": metrics.get("indicator_model_version"),
                "indicator_call_prob": metrics.get("indicator_call_probability"),
                "indicator_candle_epoch": metrics.get("indicator_candle_epoch"),
                "quote_price": getattr(contract, "quote_price", None),
                "quote_payout": getattr(contract, "quote_payout", None),
                "quote_ev": getattr(contract, "quote_ev", None),
                "indicator_period_seconds": metrics.get("indicator_period_seconds") or 300,
                "quote_cohort": (
                    "setup_pilot"
                    if str(indicator_version).startswith("pullback-1-")
                    and str(orch.config.get("trading", {}).get("mode", "demo")).lower() == "demo"
                    else ("confirmatory" if quoted else "exploratory")
                    if indicator_version
                    else None
                ),
            }
        )
    except Exception as exc:
        orch.logger.error("AUDIT: atribuicao de modelo falhou cid=%s erro=%s", contract_id, exc)
