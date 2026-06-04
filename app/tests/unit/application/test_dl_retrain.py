from types import SimpleNamespace

from src.application.services.deep_learning.dl_retrain import (
    clear_force_retrain,
    mark_force_retrain,
    reset_bars_since_train,
    should_retrain_symbol,
    tick_bars_since_train,
)


def test_force_retrain_and_clear():
    orch = SimpleNamespace()
    mark_force_retrain(orch, "RDBULL")
    ok, reason = should_retrain_symbol(orch, "RDBULL", {}, {"train_on_new_candle": False}, 1)
    assert ok and reason == "loss_retrain"
    clear_force_retrain(orch, "RDBULL")
    assert not (getattr(orch, "_dl_force_retrain", None) or {}).get("RDBULL")


def test_rolling_retrain_trigger():
    orch = SimpleNamespace(_dl_bars_since_train={"X": 12})
    runtime = {"last_candle_epoch": 5}
    params = {"train_on_new_candle": True, "rolling_retrain_bars": 12}
    ok, reason = should_retrain_symbol(orch, "X", runtime, params, 5)
    assert ok and reason == "rolling"


def test_tick_bars_initializes():
    orch = SimpleNamespace()
    tick_bars_since_train(orch, ["A", "B"])
    assert orch._dl_bars_since_train["A"] == 0
    tick_bars_since_train(orch, ["A"])
    assert orch._dl_bars_since_train["A"] == 1
    reset_bars_since_train(orch, "A")
    assert orch._dl_bars_since_train["A"] == 0
