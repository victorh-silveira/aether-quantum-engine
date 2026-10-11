"""Consulta e aplica concordancia do indicador aprovado na execucao M5."""

import logging
import os
import time

from src.domain.models.indicator_consensus import ApprovedSignal, agrees_with_execution
from src.infrastructure.storage.indicator_signal_reader import read_approved_m5_signals


logger = logging.getLogger("AETH")
DEFAULT_DSN = "postgresql://aether:aether@127.0.0.1:15433/aether"


def consensus_enabled(config: dict) -> bool:
    """Le a politica de concordancia entre TCN e indicador."""
    return bool(config.get("indicator_consensus", {}).get("enabled", False))


async def refresh_indicator_consensus(orch) -> None:
    """Busca um snapshot por ciclo sem reutilizar leituras de ciclos anteriores."""
    orch._indicator_signals = {}
    orch._indicator_epoch = int(time.time() // 300) * 300 - 300
    if not consensus_enabled(orch.config):
        return
    dsn = os.environ.get("AETHER_INDICATOR_DSN", DEFAULT_DSN)
    try:
        orch._indicator_signals = await read_approved_m5_signals(
            dsn, list(orch.symbols), expected_epoch=orch._indicator_epoch
        )
    except Exception as exc:
        logger.warning("INDICADOR: leitura indisponivel; nenhuma ordem neste ciclo: %s", exc)


def candidate_agrees_with_indicator(orch, symbol: str, direction: str, metrics: dict) -> bool:
    """Exige mesmo ativo, lado, modelo aprovado e vela fechada da inferencia."""
    if not consensus_enabled(orch.config):
        return True
    signals: dict[str, ApprovedSignal] = getattr(orch, "_indicator_signals", {})
    signal = signals.get(symbol)
    epoch = int(getattr(orch, "_indicator_epoch", 0))
    if not agrees_with_execution(signal, symbol, direction, epoch):
        return False
    if str(metrics.get("tcn_direction") or "").upper() != direction:
        return False
    if int(metrics.get("model_input_candle_epoch") or 0) != epoch:
        return False
    metrics["indicator_model_version"] = signal.model_version
    metrics["indicator_call_probability"] = signal.probability
    metrics["indicator_candle_epoch"] = signal.candle_epoch
    return True


def filter_indicator_candidates(orch, candidates: list[tuple]) -> list[tuple]:
    """Remove divergencias e contratos ja abertos no mesmo ativo."""
    if not consensus_enabled(orch.config):
        return candidates
    active_ids = set(getattr(orch.state, "active_contracts", {}))
    active_symbols = {
        symbol
        for contract_id, symbol in getattr(orch.risk_manager, "contract_to_symbol", {}).items()
        if contract_id in active_ids
    }
    accepted = []
    seen_symbols = set(active_symbols)
    for item in candidates:
        symbol, direction, metrics = item
        if symbol in seen_symbols or not candidate_agrees_with_indicator(orch, symbol, direction.name, metrics):
            continue
        accepted.append(item)
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
    if not consensus_enabled(config):
        return True
    expected_epoch = int(time.time() // 300) * 300 - 300
    return int(metrics.get("indicator_candle_epoch") or 0) == expected_epoch
