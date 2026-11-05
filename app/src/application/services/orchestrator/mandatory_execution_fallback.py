"""Fallback direcional obrigatorio para politica de Zero Skips no setup M15."""

import logging
import time
from typing import Any

from src.application.services.orchestrator.indicator_period_filters import build_indicator_metrics
from src.application.services.orchestrator.indicator_period_orders import resolve_telemetry_side
from src.domain.models.trade import TradeDirection
from src.domain.risk.pullback_stake import pullback_stake
from src.domain.symbols.drift_symbols import DEFAULT_ANCHOR, is_otc_exchange_open_brt


logger = logging.getLogger(__name__)


def _candidate_fallback_symbols(orch: Any) -> list[str]:
    """Retorna lista priorizada de ativos elegiveis para execucao mandataria com rotacao."""
    anchor = DEFAULT_ANCHOR
    now_ts = time.time()
    candidates: list[str] = []
    if is_otc_exchange_open_brt(anchor, now_ts):
        candidates.append(anchor)
    for sym in getattr(orch, "symbols", ()):
        if sym not in candidates and is_otc_exchange_open_brt(sym, now_ts):
            candidates.append(sym)
    if not candidates:
        candidates.append(anchor)
        for sym in getattr(orch, "symbols", ()):
            if sym not in candidates:
                candidates.append(sym)
    return candidates


def _select_fallback_symbol(orch: Any) -> str:
    """Seleciona o ativo prioritario elegivel para execucao mandataria."""
    candidates = _candidate_fallback_symbols(orch)
    return candidates[0] if candidates else DEFAULT_ANCHOR


def _resolve_fallback_direction(orch: Any, symbol: str) -> tuple[TradeDirection, str, bool]:
    """Determina a direcao direcional mandataria por sinal ou buffer OHLC aplicando telemetria."""
    period = int(orch.config.get("pullback_strategy", {}).get("period_seconds", 900))
    sig = (
        getattr(orch, "_indicator_period_signals", {}).get((symbol, period))
        if isinstance(getattr(orch, "_indicator_period_signals", None), dict)
        else None
    )
    base_side = "CALL"
    if sig is not None and getattr(sig, "side", "") in ("CALL", "PUT"):
        base_side = sig.side
    else:
        buf_dict = getattr(orch, "data_buffer", None)
        if isinstance(buf_dict, dict):
            buf = buf_dict.get(symbol)
            if isinstance(buf, (list, tuple)) and buf:
                latest = buf[-1]
                c = getattr(latest, "close", None)
                o = getattr(latest, "open", None)
                if isinstance(c, (int, float)) and isinstance(o, (int, float)):
                    base_side = "CALL" if c >= o else "PUT"
    pb_cfg = orch.config.get("pullback_strategy", {}) if isinstance(getattr(orch, "config", None), dict) else {}
    resolved_side, decision_reason, is_flipped = resolve_telemetry_side(orch, symbol, base_side, pb_cfg)
    return TradeDirection[resolved_side], decision_reason, is_flipped


async def resolve_mandatory_fallback_order(orch: Any) -> list[tuple[str, TradeDirection, dict]]:
    """Gera e cota a ordem mandataria no broker quando nao ha setup confirmado com rotacao ativa."""
    cid = f"C{int(getattr(orch, '_active_cycle_id', 1)):04d}"
    period = int(orch.config.get("pullback_strategy", {}).get("period_seconds", 900))
    stake_min = float(orch.config.get("risk_management", {}).get("params", {}).get("stake_min", 1.0))
    balance = float(getattr(orch.state, "balance", 0.0) or 0.0)
    stake, _ = pullback_stake(balance, 0.0, 1.0, stake_min)
    if stake <= 0.0:
        logger.error("[%s] EXEC_ERROR_BALANCE || Saldo insuficiente para ordem mandataria: $%.2f", cid, balance)
        return []
    candidates = _candidate_fallback_symbols(orch)
    params = {"duration": period // 60, "duration_unit": "m"}
    for symbol in candidates:
        direction, decision_reason, is_flipped = _resolve_fallback_direction(orch, symbol)
        try:
            quote = await orch.trade_handler.fetch_proposal_snapshot(symbol, direction, stake, params=params)
        except Exception as exc:
            logger.warning("[%s] EXEC_ERROR_API || Falha na cotacao mandataria para %s: %s", cid, symbol, exc)
            continue
        if quote is None or not quote.get("proposal_id"):
            logger.warning("[%s] EXEC_MANDATORY_RETRY || Cotacao recusada para %s, rotacionando...", cid, symbol)
            continue
        now_ts = time.time()
        candle_epoch = int(now_ts // period) * period - period
        version = str(orch.config.get("pullback_strategy", {}).get("version", "senior_institutional"))
        rate = float(quote.get("rate") or 0.70)
        metrics = build_indicator_metrics(
            candle_epoch,
            period,
            version,
            direction.name,
            direction.name,
            f"MANDATORY_{decision_reason}",
            pilot=True,
            ev=None,
            prob=0.55,
        )
        metrics.update(
            {
                "direction_telemetry_flipped": is_flipped,
                "indicator_quote_rate": rate,
                "_prequoted_proposal_id": str(quote["proposal_id"]),
                "_prequoted_ask_price": float(quote["price"]),
                "_prequoted_payout": float(quote["payout"]),
                "_prequoted_rate": rate,
                "_prequoted_timestamp": now_ts,
            }
        )
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

    logger.error("[%s] EXEC_ERROR_BROKER || Proposta mandataria recusada para todos os candidatos", cid)
    return []
