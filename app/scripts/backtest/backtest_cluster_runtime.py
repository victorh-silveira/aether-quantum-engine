"""Estado do orquestrador replicado no backtest (pausa cluster, quarentena, config)."""

from __future__ import annotations

import logging
from typing import Any

from src.application.services.llm.cluster_post_loss import record_cluster_loss

from src.domain.models.trade import TradeDirection


class BacktestClusterRuntime:
    """Substituto minimo do Orchestrator para propagate_cluster_decisions."""

    def __init__(self, config: dict[str, Any], *, symbols: list[str], anchor: str) -> None:
        self.config = config
        self.symbols = list(symbols)
        self.anchor = str(anchor)
        self.logger = logging.getLogger("AETH")
        self._cluster_pause_cycles_remaining = 0
        self._cluster_pause_after_loss_active = False
        self._last_loss_symbol = ""
        self._last_loss_direction = ""
        self._invert_quarantine_cycles_remaining = 0
        self._invert_quarantine_active = False
        self._cluster_refresh_without_llm = False
        self._last_cluster_cycle_end = 0.0

    def begin_cycle(self) -> None:
        self._invert_quarantine_active = self._invert_quarantine_cycles_remaining > 0
        if self._invert_quarantine_active:
            self._invert_quarantine_cycles_remaining -= 1
        self._cluster_pause_after_loss_active = self._cluster_pause_cycles_remaining > 0
        if self._cluster_pause_after_loss_active:
            self._cluster_pause_cycles_remaining -= 1

    def end_cycle(self) -> None:
        self._invert_quarantine_active = False
        self._cluster_pause_after_loss_active = False
        self._cluster_refresh_without_llm = False

    def on_trade_loss(self, *, symbol: str, direction: TradeDirection) -> None:
        record_cluster_loss(self, symbol=symbol, direction=direction)
        self._invert_quarantine_cycles_remaining = 1
