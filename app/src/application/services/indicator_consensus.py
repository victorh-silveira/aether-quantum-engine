"""Consulta e aplica concordancia do indicador aprovado na execucao M5."""

import logging
import os
import time
from uuid import uuid4

from src.application.services.indicator_purchase_quarantine import purchase_account_key, quarantine_unknown_purchase
from src.domain.models.indicator_consensus import ApprovedSignal
from src.domain.models.trade import TradeDirection
from src.domain.risk.broker_ev_evidence import BrokerEvEvidence
from src.infrastructure.market.timescale_writer import TimescaleMarketWriter
from src.infrastructure.state.redis_state_store import RedisStateStore
from src.infrastructure.storage.indicator_broker_evidence_reader import (
    read_indicator_broker_evidence,
    read_indicator_real_evidence,
)
from src.infrastructure.storage.indicator_signal_reader import (
    read_approved_m5_signals,
    read_approved_period_signals,
    read_trade_signals,
)


logger = logging.getLogger("AETH")
__all__ = ("quarantine_unknown_purchase",)
DEFAULT_DSN = "postgresql://aether:aether@127.0.0.1:5432/aether"


def consensus_enabled(config: dict) -> bool:
    """Le a politica de concordancia entre TCN e indicador."""
    return bool(config.get("indicator_consensus", {}).get("enabled", False))


async def refresh_indicator_consensus(orch) -> None:
    """Busca um snapshot por ciclo sem reutilizar leituras de ciclos anteriores."""
    orch._indicator_signals = {}
    orch._indicator_broker_evidence = {}
    orch._indicator_real_evidence = {}
    orch._indicator_period_signals = {}
    orch._indicator_signal_status = {}
    orch._indicator_epoch = int(time.time() // 300) * 300 - 300
    if not consensus_enabled(orch.config):
        return
    dsn = os.environ.get("AETHER_INDICATOR_DSN", DEFAULT_DSN)
    try:
        multi_period = orch.config.get("indicator_consensus", {}).get("multi_period_trading", False)
        if multi_period and orch.config.get("indicator_consensus", {}).get("broker_ev_promotion", False):
            mode = str(orch.config.get("trading", {}).get("mode", "demo")).lower()
            orch._indicator_period_signals = await read_trade_signals(
                dsn,
                account_mode=mode,
                now=int(time.time()),
                setup_version=orch.config.get("pullback_strategy", {}).get("version"),
                status_out=orch._indicator_signal_status,
            )
            orch._indicator_signals = {
                s: v for (s, p), v in orch._indicator_period_signals.items() if p == 300 and s in orch.symbols
            }
        else:
            orch._indicator_signals = await read_approved_m5_signals(
                dsn, list(orch.symbols), expected_epoch=orch._indicator_epoch
            )
            if multi_period:
                orch._indicator_period_signals = await read_approved_period_signals(dsn, now=int(time.time()))
        evidence_symbols = sorted(set(orch.symbols) | {symbol for symbol, _ in orch._indicator_period_signals})
        if orch.config.get("indicator_consensus", {}).get("broker_ev_promotion", False):
            orch._indicator_broker_evidence = await read_indicator_broker_evidence(dsn, evidence_symbols)
            if str(orch.config.get("trading", {}).get("mode", "demo")).lower() == "real":
                orch._indicator_real_evidence = await read_indicator_real_evidence(dsn, evidence_symbols)
    except Exception as exc:
        orch._indicator_signals = {}
        orch._indicator_period_signals = {}
        orch._indicator_broker_evidence = {}
        orch._indicator_real_evidence = {}
        orch._indicator_signal_status = {}
        logger.warning("INDICADOR: leitura indisponivel; nenhuma ordem neste ciclo: %s", exc)


def candidate_agrees_with_indicator(orch, symbol: str, direction: str, metrics: dict) -> bool:
    """Confere se o indicador aprovado pode determinar a ordem nesta vela."""
    if not consensus_enabled(orch.config):
        return True
    signals: dict[str, ApprovedSignal] = getattr(orch, "_indicator_signals", {})
    signal = signals.get(symbol)
    epoch = int(getattr(orch, "_indicator_epoch", 0))
    if signal is None or signal.symbol != symbol or signal.candle_epoch != epoch:
        metrics["indicator_selection_reason"] = "indicator_no_approved_m5"
        return False
    if int(metrics.get("model_input_candle_epoch") or 0) != epoch:
        metrics["indicator_selection_reason"] = "indicator_candle_mismatch"
        return False
    metrics["indicator_model_version"] = signal.model_version
    metrics["indicator_call_probability"] = signal.probability
    metrics["indicator_candle_epoch"] = signal.candle_epoch
    metrics["indicator_period_seconds"] = 300
    metrics["indicator_execution_side"] = signal.side
    metrics["indicator_tcn_discord"] = str(metrics.get("tcn_direction") or direction).upper() != signal.side
    metrics["indicator_selection_reason"] = "indicator_selected"
    return True


def filter_indicator_candidates(orch, candidates: list[tuple]) -> list[tuple]:
    """Seleciona lado do indicador aprovado e evita contrato duplicado no ativo."""
    if not consensus_enabled(orch.config):
        return candidates
    active_ids = set(getattr(orch.state, "active_contracts", {}))
    contract_symbols = getattr(orch.risk_manager, "contract_to_symbol", {})
    active_symbols = {symbol for contract_id, symbol in contract_symbols.items() if contract_id in active_ids}
    accepted = []
    seen_symbols = set(active_symbols)
    for item in candidates:
        symbol, direction, metrics = item
        if symbol in seen_symbols:
            metrics["indicator_selection_reason"] = "indicator_contract_open"
            continue
        if not candidate_agrees_with_indicator(orch, symbol, direction.name, metrics):
            continue
        signal = orch._indicator_signals[symbol]
        if orch.config.get("indicator_consensus", {}).get("broker_ev_promotion", False):
            evidence = getattr(orch, "_indicator_broker_evidence", {}).get(
                (symbol, 300, signal.model_version), BrokerEvEvidence(0, ())
            )
            mode = str(orch.config.get("trading", {}).get("mode", "demo")).lower()
            pilot = mode == "demo" and evidence.pilot_available and not evidence.promoted
            writer = getattr(orch, "market_writer", None)
            if pilot and (not isinstance(writer, TimescaleMarketWriter) or writer._last_error is not None):
                metrics["indicator_selection_reason"] = "broker_audit_unavailable"
                logger.warning("INDICADOR: piloto DEMO sem captura auditada; compra suspensa para %s", symbol)
                continue
            if not evidence.promoted and not pilot:
                metrics["indicator_selection_reason"] = "broker_ev_not_qualified"
                logger.info(
                    "INDICADOR: %s versao=%s sem aptidao REAL | conta=%s compras_demo=%d liquidados=%d limite_95=%s",
                    symbol,
                    signal.model_version,
                    mode,
                    evidence.purchased,
                    evidence.settled,
                    evidence.lower_95,
                )
                continue
            if (
                mode == "real"
                and getattr(orch, "_indicator_real_evidence", {})
                .get((symbol, 300, signal.model_version), BrokerEvEvidence(0, ()))
                .suspended
            ):
                metrics["indicator_selection_reason"] = "real_return_lower_bound_nonpositive"
                continue
            metrics["indicator_demo_pilot"] = pilot
            metrics["indicator_broker_settled"] = evidence.settled
            metrics["indicator_broker_mean_return"] = evidence.mean_return
            metrics["indicator_broker_lower_95"] = evidence.lower_95
            metrics["indicator_exploratory_purchased"] = evidence.exploratory_purchased
            logger.info(
                "INDICADOR: %s versao=%s estado=%s compras_demo=%d liquidados=%d limite_95=%s",
                symbol,
                signal.model_version,
                "PILOTO_DEMO" if pilot else "APTO_REAL",
                evidence.purchased,
                evidence.settled,
                evidence.lower_95,
            )
        side_probability = signal.probability if signal.side == "CALL" else 1.0 - signal.probability
        metrics["tcn_calibrated_prob"] = metrics.get("calibrated_prob")
        metrics["tcn_raw_prob"] = metrics.get("raw_prob")
        metrics["calibrated_prob"] = signal.probability
        metrics["raw_prob"] = signal.probability
        metrics["trade_score"] = side_probability
        metrics["conviction"] = side_probability
        metrics["senior_trader_conviction"] = side_probability
        metrics["loss_clf_flip"] = False
        metrics["indicator_controls_execution"] = True
        metrics["direction_origin"] = "INDICATOR_APPROVED"
        metrics["exec_direction"] = signal.side
        metrics["resolved_direction"] = signal.side
        accepted.append((symbol, TradeDirection[signal.side], metrics))
        seen_symbols.add(symbol)
    return accepted


def multi_symbol_orders_enabled(orch, *, recovery_active: bool) -> bool:
    """Permite multiplos ativos no mesmo ciclo somente fora da recuperacao."""
    exec_cfg = orch.config.get("orchestrator", {}).get("execution", {})
    return consensus_enabled(orch.config) and not recovery_active and bool(exec_cfg.get("multi_symbol_orders", False))


def available_cluster_bankroll(bankroll: float, reserved: float, live_balance: float, *, multi_symbol: bool) -> float:
    """Reserva compras confirmadas antes de calcular a proxima stake."""
    if not multi_symbol:
        return bankroll
    return max(0.0, min(bankroll - reserved, live_balance))


def signal_epoch_current(config: dict, metrics: dict) -> bool:
    """Impede comprar apos a vela M5 de referencia deixar de ser a ultima fechada."""
    if consensus_enabled(config) and metrics.get("indicator_controls_execution"):
        period = int(metrics.get("indicator_period_seconds") or 300)
        now = time.time()
        expected = int(now // period) * period - period
        return (
            period in (300, 900, 3600)
            and int(metrics.get("indicator_candle_epoch") or 0) == expected
            and now % period <= 30
        )
    expected_epoch = int(time.time() // 300) * 300 - 300
    if consensus_enabled(config):
        return int(metrics.get("indicator_candle_epoch") or 0) == expected_epoch
    model_epoch = int(metrics.get("model_input_candle_epoch") or metrics.get("_order_candle_epoch") or 0)
    return model_epoch <= 0 or model_epoch == expected_epoch


def claim_indicator_order(orch, symbol: str, metrics: dict) -> bool:
    """Reserva uma compra por ativo e vela, inclusive entre ciclos concorrentes."""
    if metrics.get("indicator_controls_execution") and not metrics.get("indicator_candle_epoch"):
        return False
    fallback = int(time.time() // 300) * 300 - 300
    epoch = int(metrics.get("indicator_candle_epoch") or metrics.get("model_input_candle_epoch") or fallback)
    if epoch <= 0:
        return False
    metrics["_order_candle_epoch"] = epoch
    claims = getattr(orch, "_indicator_order_claims", None)
    if not isinstance(claims, set):
        claims = set()
        orch._indicator_order_claims = claims
    period = int(metrics.get("indicator_period_seconds") or 300)
    claims.intersection_update({item for item in claims if item[2] >= epoch})
    key = (str(symbol), period, epoch)
    if key in claims:
        return False
    claims.add(key)
    return True


async def claim_indicator_order_once(orch, symbol: str, metrics: dict) -> bool:
    """Reserva a vela localmente e no Redis antes de enviar a compra ao broker."""
    blocked = getattr(orch, "_unknown_purchase_until", None)
    if isinstance(blocked, dict) and float(blocked.get(str(symbol), 0.0)) > time.time():
        return False
    store = getattr(orch, "state_store", None)
    client = None
    if isinstance(store, RedisStateStore):
        try:
            client = await store._redis()
            if await client.exists(store._full_key(f"unknown_purchase:{purchase_account_key(orch, symbol)}")):
                return False
        except Exception as exc:
            logger.error("EXEC: estado de compra sem ACK indisponivel para %s: %s", symbol, exc)
            return False
    if not claim_indicator_order(orch, symbol, metrics):
        return False
    if client is None:
        return True
    epoch = int(metrics["_order_candle_epoch"])
    period = int(metrics.get("indicator_period_seconds") or 300)
    key = store._full_key(f"order_claim:{purchase_account_key(orch, symbol)}:{period}:{epoch}")
    token = uuid4().hex
    try:
        claimed = bool(await client.set(key, token, nx=True, ex=900))
        if claimed:
            metrics["_order_claim_token"] = token
            if orch.config.get("indicator_consensus", {}).get("multi_period_trading", False):
                symbol_key = store._full_key(f"order_symbol_claim:{purchase_account_key(orch, symbol)}")
                symbol_claimed = bool(await client.set(symbol_key, token, nx=True, ex=period + 30))
                if not symbol_claimed:
                    await release_unbought_order_claim(orch, symbol, metrics)
                    return False
                metrics["_symbol_claim_token"] = token
        return claimed
    except Exception as exc:
        logger.error("INDICADOR: reserva Redis indisponivel; compra suspensa para %s: %s", symbol, exc)
        if metrics.get("_order_claim_token"):
            await release_unbought_order_claim(orch, symbol, metrics)
        return False


async def release_unbought_order_claim(orch, symbol: str, metrics: dict) -> None:
    """Libera reserva propria apenas quando a proposta terminou sem enviar compra."""
    epoch = int(metrics.get("_order_candle_epoch") or 0)
    period = int(metrics.get("indicator_period_seconds") or 300)
    key = (str(symbol), period, epoch)
    token = metrics.get("_order_claim_token")
    store = getattr(orch, "state_store", None)
    if isinstance(store, RedisStateStore) and token:
        script = "if redis.call('GET', KEYS[1]) == ARGV[1] then return redis.call('DEL', KEYS[1]) else return 0 end"
        redis_key = store._full_key(f"order_claim:{purchase_account_key(orch, symbol)}:{period}:{epoch}")
        try:
            client = await store._redis()
            await client.eval(script, 1, redis_key, token)
            if orch.config.get("indicator_consensus", {}).get("multi_period_trading", False):
                symbol_key = store._full_key(f"order_symbol_claim:{purchase_account_key(orch, symbol)}")
                await client.eval(script, 1, symbol_key, token)
        except Exception as exc:
            logger.error("EXEC: falha ao liberar reserva sem compra para %s: %s", symbol, exc)
            return
    claims = getattr(orch, "_indicator_order_claims", None)
    if isinstance(claims, set):
        claims.discard(key)
