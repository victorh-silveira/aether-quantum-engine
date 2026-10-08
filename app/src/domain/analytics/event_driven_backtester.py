"""Backtester orientado a eventos fiel as especificacoes contratuais da Deriv."""

from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class TradeSignal:
    """Evento de sinal direcional gerado pelo pipeline analitico."""

    index: int
    direction: str
    stake: float
    probability: float = 0.50


@dataclass(frozen=True)
class BacktestReport:
    """Relatorio consolidado de performance da simulacao de eventos."""

    total_trades: int
    wins: int
    losses: int
    win_rate: float
    total_pnl: float
    profit_factor: float
    expectancy_per_trade: float
    max_drawdown: float
    max_drawdown_pct: float
    sharpe_ratio: float


class EventDrivenBacktester:
    """Simulador de execucao e liquidacao com slippage temporal e regras de empate."""

    def __init__(
        self,
        *,
        horizon_steps: int = 300,
        execution_delay_steps: int = 1,
        payout_rate: float = 0.85,
    ) -> None:
        """Configura parametros contratuais de horizonte, atraso de rede e payout."""
        self._horizon = max(1, horizon_steps)
        self._delay = max(0, execution_delay_steps)
        self._payout_rate = max(0.01, float(payout_rate))

    def run(self, prices: np.ndarray, signals: list[TradeSignal]) -> BacktestReport:
        """Executa a simulacao cronologica sobre a serie de precos fornecida."""
        arr = np.asarray(prices, dtype=np.float64)
        total_len = len(arr)

        wins = 0
        losses = 0
        total_pnl = 0.0
        gross_profit = 0.0
        gross_loss = 0.0
        pnls: list[float] = []

        equity = 0.0
        peak = 0.0
        max_dd = 0.0
        max_dd_pct = 0.0

        for sig in signals:
            fill_idx = sig.index + self._delay
            expiry_idx = fill_idx + self._horizon

            if expiry_idx >= total_len or fill_idx >= total_len:
                continue

            fill_price = arr[fill_idx]
            expiry_price = arr[expiry_idx]

            is_win = False
            if sig.direction == "CALL":
                is_win = expiry_price > fill_price
            elif sig.direction == "PUT":
                is_win = expiry_price < fill_price

            if is_win:
                pnl = sig.stake * self._payout_rate
                wins += 1
                gross_profit += pnl
            else:
                pnl = -sig.stake
                losses += 1
                gross_loss += abs(pnl)

            total_pnl += pnl
            pnls.append(pnl)

            equity += pnl
            peak = max(peak, equity)
            dd = peak - equity
            max_dd = max(max_dd, dd)
            if peak > 0.0:
                dd_pct = (dd / peak) * 100.0
                max_dd_pct = max(max_dd_pct, dd_pct)

        total_trades = wins + losses
        win_rate = (wins / total_trades) if total_trades > 0 else 0.0
        profit_factor = (gross_profit / gross_loss) if gross_loss > 0.0 else (float("inf") if gross_profit > 0 else 0.0)
        expectancy = (total_pnl / total_trades) if total_trades > 0 else 0.0

        sharpe = 0.0
        if len(pnls) >= 2:
            pnl_arr = np.array(pnls, dtype=np.float64)
            std = float(np.std(pnl_arr))
            if std > 1e-12:
                sharpe = float(np.mean(pnl_arr) / std * math.sqrt(len(pnl_arr)))

        return BacktestReport(
            total_trades=total_trades,
            wins=wins,
            losses=losses,
            win_rate=win_rate,
            total_pnl=total_pnl,
            profit_factor=profit_factor,
            expectancy_per_trade=expectancy,
            max_drawdown=max_dd,
            max_drawdown_pct=max_dd_pct,
            sharpe_ratio=sharpe,
        )
