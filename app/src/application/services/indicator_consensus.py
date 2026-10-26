"""Consulta e aplica concordancia do indicador aprovado na execucao M5."""

import logging
import os
import time

from src.application.services.indicator_order_claims import (
    claim_indicator_order,
    claim_indicator_order_once,
    release_unbought_order_claim,
)
from src.application.services.indicator_purchase_quarantine import quarantine_unknown_purchase
from src.domain.models.indicator_consensus import ApprovedSignal
from src.domain.models.trade import TradeDirection
from src.domain.risk.broker_ev_evidence import BrokerEvEvidence
from src.infrastructure.market.timescale_writer import TimescaleMarketWriter
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
__all__ = (
    "claim_indicator_order",
    "claim_indicator_order_once",
    "quarantine_unknown_purchase",
    "release_unbought_order_claim",
)
DEFAULT_DSN = "postgresql://aether:aether@127.0.0.1:5432/aether"


def consensus_enabled(config: dict) -> bool:
    """Le a politica de concordancia entre TCN e indicador."""
    return bool(config.get("indicator_consensus", {}).get("enabled", False))


async def refresh_indicator_consensus(orch) -> None:
    """Busca um snapshot por ciclo sem reutilizar leituras de ciclos anteriores."""
    orch._indicator_signals, orch._indicator_period_signals = {}, {}
    orch._indicator_broker_evidence, orch._indicator_real_evidence, orch._indicator_signal_status = {}, {}, {}
    period_seconds = int(orch.config.get("pullback_strategy", {}).get("period_seconds", 300))
    orch._indicator_epoch = int(time.time() // period_seconds) * period_seconds - period_seconds
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
                s: v
                for (s, p), v in orch._indicator_period_signals.items()
                if p == period_seconds and s in orch.symbols
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
        orch._indicator_signals, orch._indicator_period_signals = {}, {}
        orch._indicator_broker_evidence, orch._indicator_real_evidence, orch._indicator_signal_status = {}, {}, {}
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
    period_seconds = int(orch.config.get("pullback_strategy", {}).get("period_seconds", 300))
    metrics["indicator_model_version"] = signal.model_version
    metrics["indicator_call_probability"] = signal.probability
    metrics["indicator_candle_epoch"] = signal.candle_epoch
    metrics["indicator_period_seconds"] = period_seconds
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
    accepted, seen_symbols = [], set(active_symbols)
    period = int(orch.config.get("pullback_strategy", {}).get("period_seconds", 300))
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
                (symbol, period, signal.model_version), BrokerEvEvidence(0, ())
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
                .get((symbol, period, signal.model_version), BrokerEvEvidence(0, ()))
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
        if signal.probability is not None:
            side_probability = signal.probability if signal.side == "CALL" else 1.0 - signal.probability
        else:
            side_probability = 0.5
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
    """Exige vela fechada vigente e entrada dentro da janela operacional."""
    pb_cfg = (config or {}).get("pullback_strategy", {})
    if pb_cfg.get("enabled") and (not consensus_enabled(config) or not metrics.get("indicator_controls_execution")):
        return False
    if consensus_enabled(config) and metrics.get("indicator_controls_execution"):
        period = int(metrics.get("indicator_period_seconds") or (0 if pb_cfg.get("enabled") else 300))
        if period <= 0:
            return False
        now = time.time()
        expected = int(now // period) * period - period
        window = int(pb_cfg.get("entry_window_seconds", 30))
        epoch_val = int(metrics.get("indicator_candle_epoch") or 0)
        allowed = (
            period == int(pb_cfg.get("period_seconds", 300)) if pb_cfg.get("enabled") else period in (300, 900, 3600)
        )
        return allowed and epoch_val == expected and now % period <= window
    p_fb = int(pb_cfg.get("period_seconds", 300))
    expected_epoch = int(time.time() // p_fb) * p_fb - p_fb
    if consensus_enabled(config):
        return int(metrics.get("indicator_candle_epoch") or 0) == expected_epoch
    model_epoch = int(metrics.get("model_input_candle_epoch") or metrics.get("_order_candle_epoch") or 0)
    return model_epoch <= 0 or model_epoch == expected_epoch
