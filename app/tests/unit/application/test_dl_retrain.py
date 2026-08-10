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
    mark_force_retrain(orch, "R_10")
    ok, reason = should_retrain_symbol(orch, "R_10", {}, {"train_on_new_candle": False}, 1)
    assert ok and reason == "trade_retrain"
    clear_force_retrain(orch, "R_10")
    assert not (getattr(orch, "_dl_force_retrain", None) or {}).get("R_10")


def test_online_training_disabled_skips_all_runtime_retrain():
    orch = SimpleNamespace(_dl_force_retrain={"R_10": True}, _dl_bars_since_train={"R_10": 99})
    runtime = {"last_candle_epoch": 0, "session_trained": False}
    params = {"online_training": False, "train_on_new_candle": True, "retrain_min_bars": 0}
    ok, reason = should_retrain_symbol(orch, "R_10", runtime, params, 1)
    assert not ok and reason == ""


def test_rolling_retrain_trigger():
    orch = SimpleNamespace(_dl_bars_since_train={"X": 12})
    runtime = {"last_candle_epoch": 5, "session_trained": True}
    params = {"train_on_new_candle": True, "rolling_retrain_bars": 12, "retrain_min_bars": 0}
    ok, reason = should_retrain_symbol(orch, "X", runtime, params, 5)
    assert ok and reason == "rolling"


def test_retrain_min_bars_defers_scheduled_retrain():
    orch = SimpleNamespace(_dl_bars_since_train={"X": 3})
    runtime = {"last_candle_epoch": 100, "session_trained": True}
    params = {"train_on_new_candle": True, "retrain_min_bars": 12, "rolling_retrain_bars": 48}
    ok, reason = should_retrain_symbol(orch, "X", runtime, params, 200)
    assert not ok and reason == ""


def test_new_candle_retrain_after_min_interval():
    orch = SimpleNamespace(_dl_bars_since_train={"X": 12})
    runtime = {"last_candle_epoch": 100, "session_trained": True}
    params = {"train_on_new_candle": True, "retrain_min_bars": 12, "rolling_retrain_bars": 48}
    ok, reason = should_retrain_symbol(orch, "X", runtime, params, 200)
    assert ok and reason == "new_candle"


def test_bootstrap_retrain_when_checkpoint_loaded_but_not_trained_in_session():
    orch = SimpleNamespace(_dl_bars_since_train={"X": 0})
    runtime = {"last_candle_epoch": 100, "session_trained": False}
    params = {"train_on_new_candle": True, "retrain_min_bars": 12, "rolling_retrain_bars": 48}
    ok, reason = should_retrain_symbol(orch, "X", runtime, params, 100)
    assert ok and reason == "bootstrap"


def test_deferred_train_pending_skips_retrain():
    pending = SimpleNamespace(done=lambda: False)
    orch = SimpleNamespace(_dl_deferred_tasks={"R_10": pending})
    ok, reason = should_retrain_symbol(
        orch,
        "R_10",
        {"last_candle_epoch": 0},
        {"retrain_min_bars": 0},
        5,
    )
    assert not ok and reason == ""


def test_bootstrap_retrain_when_never_trained():
    ok, reason = should_retrain_symbol(
        SimpleNamespace(),
        "X",
        {"last_candle_epoch": 0},
        {"retrain_min_bars": 12},
        5,
    )
    assert ok and reason == "bootstrap"


def test_bootstrap_retrain_when_no_checkpoint_epoch():
    ok, reason = should_retrain_symbol(
        SimpleNamespace(),
        "X",
        {"last_candle_epoch": 0, "session_trained": True},
        {"retrain_min_bars": 12},
        5,
    )
    assert ok and reason == "bootstrap"


def test_tick_bars_initializes():
    orch = SimpleNamespace()
    tick_bars_since_train(orch, ["A", "B"])
    assert orch._dl_bars_since_train["A"] == 0
    tick_bars_since_train(orch, ["A"])
    assert orch._dl_bars_since_train["A"] == 1
    reset_bars_since_train(orch, "A")
    assert orch._dl_bars_since_train["A"] == 0
