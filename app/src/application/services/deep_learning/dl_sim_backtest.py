"""Simulacao walk-forward Deep Learning para backtest offline."""

from dataclasses import dataclass
from typing import Any

import numpy as np

from src.application.services.deep_learning.dl_calibration import CalibratorState
from src.application.services.deep_learning.dl_features import FEATURE_DIM
from src.application.services.deep_learning.dl_predict import predict_symbol_decision
from src.application.services.deep_learning.dl_training import train_model_walkforward
from src.application.services.deep_learning.model import create_direction_model, fit_norm_stats
from src.domain.models.trade import TradeDirection


DEPLOY_MAX_BRIER = 0.24
DEPLOY_MIN_WIN_RATE = 0.52


@dataclass
class DlTrade:
    """Registro de um trade simulado no backtest walk-forward."""

    bar_index: int
    direction: TradeDirection
    won: bool
    trade_score: float


@dataclass
class DlBacktestResult:
    """Metricas agregadas do backtest walk-forward Deep Learning."""

    trades: list[DlTrade]
    win_rate: float
    profit_factor: float
    max_drawdown: float
    trades_per_day: float
    val_brier: float
    deploy_ok: bool


def direction_wins(direction: TradeDirection, prices: np.ndarray, index: int) -> bool:
    """Indica se a direcao prevista venceu na barra seguinte."""
    if index + 1 >= len(prices):
        return False
    up = prices[index + 1] > prices[index]
    return up if direction == TradeDirection.CALL else not up


def _apply_train_result(runtime: dict[str, Any], result) -> None:
    """Atualiza runtime com metricas do ultimo treino walk-forward."""
    runtime["val_accuracy"] = result.val_accuracy
    runtime["calibrator"] = result.calibrator or CalibratorState()
    runtime["val_brier"] = result.val_brier
    runtime["val_ece"] = result.val_ece


def run_dl_walkforward(prices: np.ndarray, params: dict[str, Any], *, retrain_every: int = 120) -> DlBacktestResult:
    """Simula ciclo DL barra a barra com retreino periodico e gating de precisao."""
    lookback = int(params["lookback"])
    min_len = lookback + int(params["validation_bars"]) + 20
    if len(prices) < min_len + 10:
        raise RuntimeError(f"Historico insuficiente: {len(prices)} velas, minimo {min_len + 10}")
    model = create_direction_model(arch=params["arch"])
    norm_stats = fit_norm_stats(np.zeros((1, lookback, FEATURE_DIM), dtype=np.float32))
    runtime: dict[str, Any] = {
        "val_accuracy": 0.0,
        "calibrator": CalibratorState(),
        "val_brier": 1.0,
        "val_ece": 1.0,
        "lookback": lookback,
    }
    orch = type("_DlBacktestOrch", (), {"config": {"deep_learning": {}}})()
    trades: list[DlTrade] = []
    equity = peak = max_dd = wins = gross_win = gross_loss = 0.0
    start = min_len
    end = len(prices) - 2
    step = max(30, retrain_every)
    for bar in range(start, end + 1):
        window = prices[: bar + 1]
        if bar == start or (bar - start) % step == 0:
            result = train_model_walkforward(
                model,
                window,
                lookback,
                min(8, int(params["epochs"])),
                params["lr"],
                params["validation_bars"],
                weight_decay=params["weight_decay"],
                label_smoothing=params["label_smoothing"],
                label_min_move_pct=params["label_min_move_pct"],
                early_stopping_patience=params["early_stopping_patience"],
                focal_gamma=params["focal_gamma"],
                calib_ratio=params["calib_ratio"],
            )
            if result is not None:
                norm_stats = result.norm_stats
                _apply_train_result(runtime, result)
        entry = predict_symbol_decision(
            orch,
            "BT",
            model,
            window,
            norm_stats,
            runtime,
            params,
            None,
            recovery_active=False,
        )
        if not entry["metrics"].get("execute") or entry["direction"] is None:
            continue
        direction = entry["direction"]
        won = direction_wins(direction, prices, bar)
        score = float(entry["metrics"].get("trade_score", entry["metrics"].get("conviction", 0.5)))
        trades.append(DlTrade(bar_index=bar, direction=direction, won=won, trade_score=score))
        pnl = 1.0 if won else -1.0
        equity += pnl
        peak = max(peak, equity)
        max_dd = max(max_dd, peak - equity)
        if won:
            wins += 1
            gross_win += 1.0
        else:
            gross_loss += 1.0
    total = len(trades)
    win_rate = (wins / total) if total else 0.0
    pf = (gross_win / gross_loss) if gross_loss > 0 else gross_win
    minutes = max(1, end - start + 1)
    trades_per_day = total / (minutes / (24.0 * 60.0))
    val_brier = float(runtime.get("val_brier", 1.0))
    deploy_ok = val_brier < DEPLOY_MAX_BRIER and win_rate > DEPLOY_MIN_WIN_RATE
    return DlBacktestResult(
        trades=trades,
        win_rate=win_rate,
        profit_factor=pf,
        max_drawdown=max_dd,
        trades_per_day=trades_per_day,
        val_brier=val_brier,
        deploy_ok=deploy_ok,
    )
