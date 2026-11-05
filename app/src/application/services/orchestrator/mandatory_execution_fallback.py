"""Fallback direcional obrigatorio para politica de Zero Skips no setup M15."""

import logging
import time
from typing import Any

from src.domain.models.trade import TradeDirection
from src.domain.risk.pullback_stake import pullback_stake
from src.domain.symbols.drift_symbols import DEFAULT_ANCHOR, is_otc_exchange_open_brt


logger = logging.getLogger(__name__)


def _select_fallback_symbol(orch: Any) -> str:
    """Seleciona o ativo prioritario elegivel para execucao mandataria."""
    anchor = DEFAULT_ANCHOR
    now_ts = time.time()
    if is_otc_exchange_open_brt(anchor, now_ts):
        return anchor
    for sym in getattr(orch, "symbols", ()):
        if is_otc_exchange_open_brt(sym, now_ts):
            return sym
    return anchor


def _resolve_fallback_direction(orch: Any, symbol: str) -> TradeDirection:
    """Determina a direcao direcional mandataria por sinal ou buffer OHLC."""
    period = int(orch.config.get("pullback_strategy", {}).get("period_seconds", 900))
    sig = (
        getattr(orch, "_indicator_period_signals", {}).get((symbol, period))
        if isinstance(getattr(orch, "_indicator_period_signals", None), dict)
        else None
    )
    if sig is not None and getattr(sig, "side", "") in ("CALL", "PUT"):
        return TradeDirection[sig.side]
    buf_dict = getattr(orch, "data_buffer", None)
    if isinstance(buf_dict, dict):
        buf = buf_dict.get(symbol)
        if isinstance(buf, (list, tuple)) and buf:
            latest = buf[-1]
            c = getattr(latest, "close", None)
            o = getattr(latest, "open", None)
            if isinstance(c, (int, float)) and isinstance(o, (int, float)):
                return TradeDirection.CALL if c >= o else TradeDirection.PUT
    return TradeDirection.CALL


async def resolve_mandatory_fallback_order(orch: Any) -> list[tuple[str, TradeDirection, dict]]:
    """Gera e cota a ordem mandataria no broker quando nao ha setup confirmado."""
    cid = f"C{int(getattr(orch, '_active_cycle_id', 1)):04d}"
    symbol = _select_fallback_symbol(orch)
    direction = _resolve_fallback_direction(orch, symbol)
    period = int(orch.config.get("pullback_strategy", {}).get("period_seconds", 900))
    stake_min = float(orch.config.get("risk_management", {}).get("params", {}).get("stake_min", 1.0))
    balance = float(getattr(orch.state, "balance", 0.0) or 0.0)
    stake, _ = pullback_stake(balance, 0.0, 1.0, stake_min)
    if stake <= 0.0:
        logger.error("[%s] EXEC_ERROR_BALANCE || Saldo insuficiente para ordem mandataria: $%.2f", cid, balance)
        return []
    params = {"duration": period // 60, "duration_unit": "m"}
    try:
        quote = await orch.trade_handler.fetch_proposal_snapshot(symbol, direction, stake, params=params)
    except Exception as exc:
        logger.error("[%s] EXEC_ERROR_API || Falha na cotacao mandataria no broker: %s", cid, exc)
        return []
    if quote is None or not quote.get("proposal_id"):
        logger.error("[%s] EXEC_ERROR_BROKER || Proposta mandataria recusada pelo broker", cid)
        return []
    rate = float(quote.get("rate") or 0.70)
    metrics = {
        "indicator_candle_epoch": int(time.time()),
        "indicator_period_seconds": period,
        "indicator_model_version": "mandatory_institutional_v1",
        "indicator_quote_rate": rate,
        "trade_score": 0.55,
        "calibrated_prob": 0.55,
        "organic_decision_reason": "MANDATORY_TREND_EXECUTION",
        "indicator_demo_pilot": True,
        "_prequoted_proposal_id": str(quote["proposal_id"]),
        "_prequoted_ask_price": float(quote["price"]),
        "_prequoted_payout": float(quote["payout"]),
        "_prequoted_rate": rate,
        "_prequoted_timestamp": time.time(),
    }
    logger.info(
        "[%s] EXEC_MANDATORY || %s %dm %s stake=$%.2f payout=$%.2f rate=%.1f%%",
        cid,
        symbol,
        period // 60,
        direction.name,
        float(quote["price"]),
        float(quote["payout"]),
        rate * 100,
    )
    return [(symbol, direction, metrics)]
