"""Isola falhas por simbolo na coleta paralela das decisoes DL."""

import asyncio
import logging

from src.application.services.deep_learning.dl_bridge_helpers import build_insufficient_data_entry


def collect_symbol_results(symbols: list[str], results: list, logger: logging.Logger) -> tuple[dict, list[str], str]:
    """Preserva decisoes saudaveis quando uma inferencia individual falha."""
    decisions: dict[str, dict] = {}
    trained: list[str] = []
    train_reason = ""
    for symbol, result in zip(symbols, results, strict=True):
        if isinstance(result, BaseException):
            if isinstance(result, asyncio.CancelledError):
                raise result
            logger.error("DL: simbolo %s indisponivel neste ciclo: %s", symbol, result)
            decisions[symbol] = build_insufficient_data_entry()
            continue
        entry, reason = result
        decisions[symbol] = entry
        if reason:
            trained.append(symbol)
            train_reason = reason
    return decisions, trained, train_reason
