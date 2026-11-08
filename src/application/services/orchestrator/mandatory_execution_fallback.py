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
    """Retorna lista priorizada de ativos elegiveis para execucao mandataria com rotacao elastica."""
    now_ts = time.time()
    raw_syms = getattr(orch, "symbols", ())
    sym_list = list(raw_syms) if isinstance(raw_syms, (list, tuple)) else []
    raw_symbols = list(dict.fromkeys([DEFAULT_ANCHOR, *sym_list]))
    raw_quarantine = getattr(orch, "_symbol_loss_quarantine", None)
    quarantine = raw_quarantine if isinstance(raw_quarantine, dict) else {}
    last_traded = str(getattr(orch, "_last_traded_symbol", "") or "")
    metrics_map = getattr(orch, "_latest_market_metrics", None)
    session_trk = getattr(orch, "_session_pair_tracker", None)

    def _score_symbol(sym: str) -> float:
        score = 10.0 if is_otc_exchange_open_brt(sym, now_ts) else 0.0
        m = metrics_map.get(sym, {}) if isinstance(metrics_map, dict) else {}
        adx = float(m.get("adx") or 0.0) if isinstance(m.get("adx"), (int, float)) else 0.0
        spread = float(m.get("spread") or 0.0) if isinstance(m.get("spread"), (int, float)) else 0.0
        score += 2.0 if adx >= 22.0 else 0.0
        score += 1.0 if abs(spread) > 0.01 else 0.0
        if session_trk is not None and hasattr(session_trk, "symbol_rotation_adjustment"):
            adj = session_trk.symbol_rotation_adjustment(sym)
            score += float(adj) * 5.0 if isinstance(adj, (int, float)) else 0.0
        q_ts = quarantine.get(sym, 0.0)
        score -= 20.0 if isinstance(q_ts, (int, float)) and now_ts < float(q_ts) else 0.0
        score -= 1.5 if sym == last_traded else 0.0
        return score

    return sorted(raw_symbols, key=_score_symbol, reverse=True)


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
    metrics_bag = getattr(orch, "_latest_market_metrics", None)
    context = metrics_bag.get(symbol, {}) if isinstance(metrics_bag, dict) else {}
    resolved_side, decision_reason, is_flipped = resolve_telemetry_side(orch, symbol, base_side, pb_cfg, context)
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
