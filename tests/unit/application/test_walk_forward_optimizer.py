"""Testes unitarios para o otimizador walk-forward causal."""

import importlib
import sys

import numpy as np

import src.application.services.walk_forward_optimizer as wfo_mod
from src.application.services.walk_forward_optimizer import WalkForwardOptimizer
from src.domain.analytics.event_driven_backtester import TradeSignal


def test_walk_forward_optimizer_execution_and_result():
    prices = np.linspace(100.0, 150.0, 50, dtype=np.float64)
    signals = [TradeSignal(index=i, direction="CALL", stake=10.0, probability=0.60) for i in range(0, 45, 2)]

    optimizer = WalkForwardOptimizer(prices, signals)
    res = optimizer.optimize(n_trials=10)

    assert res.total_trials == 10
    assert 0.01 <= res.best_edge_threshold <= 0.15
    assert 0.20 <= res.best_compression_ceiling <= 0.60
    assert 0.05 <= res.best_conformal_alpha <= 0.20
    assert isinstance(res.best_sharpe_ratio, float)
    assert res.optimizer_engine in {"optuna_bayesian_tpe", "randomized_search_fallback"}


def test_walk_forward_optimizer_empty_signals():
    prices = np.array([100.0, 101.0, 102.0], dtype=np.float64)
    optimizer = WalkForwardOptimizer(prices, [])
    res = optimizer.optimize(n_trials=5)
    assert res.best_sharpe_ratio <= 0.0


def test_walk_forward_optimizer_few_trades():
    prices = np.linspace(100.0, 110.0, 10, dtype=np.float64)
    signals = [TradeSignal(index=0, direction="CALL", stake=10.0, probability=0.90)]
    optimizer = WalkForwardOptimizer(prices, signals)
    res = optimizer.optimize(n_trials=5)
    assert res.best_sharpe_ratio == -5.0


def test_walk_forward_optimizer_fallback_branch(monkeypatch):
    monkeypatch.setattr(wfo_mod, "_OPTUNA_AVAILABLE", False)
    monkeypatch.setattr(wfo_mod, "optuna", None)

    prices = np.linspace(100.0, 150.0, 50, dtype=np.float64)
    signals = [TradeSignal(index=i, direction="CALL", stake=10.0, probability=0.60) for i in range(0, 45, 2)]

    optimizer = WalkForwardOptimizer(prices, signals)
    res = optimizer.optimize(n_trials=5)
    assert res.optimizer_engine == "randomized_search_fallback"
    assert res.total_trials == 5


def test_walk_forward_optimizer_is_optuna_available():
    available = WalkForwardOptimizer.is_optuna_available()
    assert isinstance(available, bool)


def test_walk_forward_optimizer_reload_without_optuna(monkeypatch):
    monkeypatch.setitem(sys.modules, "optuna", None)
    importlib.reload(wfo_mod)
    assert wfo_mod.WalkForwardOptimizer.is_optuna_available() is False

    monkeypatch.delitem(sys.modules, "optuna", raising=False)
    importlib.reload(wfo_mod)
