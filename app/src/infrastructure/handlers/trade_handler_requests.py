"""Payloads de proposta e duracao de contratos Deriv."""

import time
from typing import Any

from src.domain.models.trade import TradeDirection


def resolve_api_contract_type(direction: TradeDirection, p_cfg: dict[str, Any]) -> str:
    """Mapeia direcao do motor para contract_type aceito na API Deriv."""
    raw_type = str(p_cfg.get("contract_type") or "").upper()
    if raw_type == "MULTIPLIER":
        return "MULTUP" if direction == TradeDirection.CALL else "MULTDOWN"
    return direction.value


def build_proposal_request(
    symbol: str, direction: TradeDirection, stake: float, p_cfg: dict[str, Any]
) -> dict[str, Any]:
    """Monta payload proposal com underlying_symbol (API Deriv atual)."""
    is_multiplier = p_cfg.get("contract_type") == "MULTIPLIER"
    request: dict[str, Any] = {
        "proposal": 1,
        "amount": round(float(stake), 2),
        "basis": "stake",
        "contract_type": resolve_api_contract_type(direction, p_cfg),
        "currency": "USD",
        "underlying_symbol": symbol,
    }

    if is_multiplier:
        request["multiplier"] = p_cfg.get("multiplier", 100)
        if "barrier" in p_cfg:
            request["barrier"] = p_cfg["barrier"]
        if "cancellation" in p_cfg:
            request["cancellation"] = p_cfg["cancellation"]
        if "limit_order" in p_cfg:
            request["limit_order"] = p_cfg["limit_order"]
    else:
        request["duration"] = p_cfg.get("duration", 5)
        request["duration_unit"] = p_cfg.get("duration_unit", "m")

    return request


def _contract_duration_seconds(parameters: dict) -> int:
    """Converte duration e duration_unit em segundos para expiry estimado."""
    dur = max(1, int(parameters.get("duration", 5)))
    unit = str(parameters.get("duration_unit", "m")).lower().strip()
    if unit == "m":
        return dur * 60
    if unit == "s":
        return dur
    if unit == "t":
        return dur * 2
    if unit == "d":
        return dur * 86400
    return dur * 60


def require_current_signal_candle(parameters: dict) -> None:
    """Recusa compra fora da abertura da vela prevista no periodo informado."""
    epoch = parameters.get("_signal_candle_epoch")
    if epoch is None:
        return
    period = int(parameters.get("_indicator_period_seconds") or 300)
    if period not in (300, 900, 3600):
        raise RuntimeError("Periodo do sinal invalido")
    now = time.time()
    expected = int(now // period) * period - period
    if int(epoch) != expected or now % period > 30:
        raise RuntimeError("Sinal expirou antes da compra; nova inferencia necessaria")
