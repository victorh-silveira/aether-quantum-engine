"""Normalizacao do payout cotado pela corretora para o risco da sessao."""

from __future__ import annotations

from typing import Any


def contract_profit_rate(payout: float, buy_price: float) -> float | None:
    """Converte payout bruto cotado em taxa liquida de lucro, ou None se invalido."""
    gross = float(payout)
    stake = float(buy_price)
    if gross <= stake or stake <= 0.0:
        return None
    return (gross / stake) - 1.0


def record_observed_payout(risk_params: dict[str, Any], *, payout: float, buy_price: float) -> float | None:
    """Atualiza somente a estimativa de payout da sessao com uma cotacao valida."""
    rate = contract_profit_rate(payout, buy_price)
    if rate is None:
        return None
    risk_params["payout_estimate"] = rate
    risk_params["observed_payout_rate"] = rate
    return rate
