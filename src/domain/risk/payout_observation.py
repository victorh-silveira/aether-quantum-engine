"""Normalizacao e teto do payout cotado pela corretora para o risco da sessao."""

from __future__ import annotations

import math
import re
from typing import Any


_MAX_PAYOUT_RE = re.compile(r"maximum\s+payout\s+of\s+([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)
_CUR_PAYOUT_RE = re.compile(r"current\s+payout\s+is\s+([0-9]+(?:\.[0-9]+)?)", re.IGNORECASE)


def parse_max_payout_rejection(message: str) -> tuple[float, float] | None:
    """Extrai teto e payout cotado de rejeicao Deriv por maximum payout."""
    if not message:
        return None
    max_match = _MAX_PAYOUT_RE.search(message)
    if max_match is None:
        return None
    max_payout = float(max_match.group(1))
    if max_payout <= 0.0:
        return None
    cur_match = _CUR_PAYOUT_RE.search(message)
    current_payout = float(cur_match.group(1)) if cur_match is not None else 0.0
    return max_payout, current_payout


def cap_stake_for_max_payout(
    stake: float,
    max_payout: float,
    current_payout_or_rate: float,
    stake_min: float,
    *,
    is_rate: bool = False,
) -> float:
    """Reduz stake proporcionalmente para respeitar o teto de payout do contrato."""
    if stake <= 0.0 or max_payout <= 0.0 or stake_min <= 0.0:
        return 0.0
    if is_rate:
        if current_payout_or_rate <= 0.0:
            return 0.0
        raw = (max_payout * 0.96) / (1.0 + current_payout_or_rate)
    else:
        if current_payout_or_rate <= max_payout:
            return 0.0
        raw = stake * (max_payout / current_payout_or_rate) * 0.95
    capped = min(float(stake), math.floor(raw * 100.0) / 100.0)
    return capped if capped + 1e-9 >= float(stake_min) else 0.0


def apply_broker_payout_cap(
    stake: float,
    pending_loss: float,
    quote_rate: float,
    metrics: dict[str, Any] | None,
    stake_min: float,
) -> tuple[float, float]:
    """Aplica teto de payout observado na pre-cotacao e recalcula amortizacao."""
    max_payout = float((metrics or {}).get("_broker_max_payout") or 0.0)
    effective = float(stake)
    if max_payout > 0.0 and quote_rate > 0.0:
        capped = cap_stake_for_max_payout(effective, max_payout, quote_rate, stake_min, is_rate=True)
        if capped > 0.0:
            effective = capped
    covered = (
        min(max(0.0, float(pending_loss)), effective * quote_rate) if pending_loss > 0.0 and quote_rate > 0.0 else 0.0
    )
    return effective, covered


def contract_profit_rate(payout: float, buy_price: float) -> float | None:
    """Converte payout bruto cotado em taxa liquida de lucro, ou None se invalido."""
    gross = float(payout)
    stake = float(buy_price)
    if not math.isfinite(gross) or not math.isfinite(stake) or gross <= stake or stake <= 0.0:
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
