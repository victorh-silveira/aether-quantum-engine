import threading
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.application.services.deep_learning.dl_deploy_eval import evaluate_mini_deploy
from src.application.services.deep_learning.dl_features import FEATURE_DIM
from src.application.services.deep_learning.dl_symbol_runtime import guard_symbol_model
from src.application.services.deep_learning.model import create_direction_model, fit_norm_stats
from src.domain.models.trade import TradeDirection


def test_guard_symbol_model_without_lock():
    with guard_symbol_model({}):
        pass


def test_guard_symbol_model_reentrant_same_thread():
    runtime = {"model_lock": threading.RLock()}
    with guard_symbol_model(runtime), guard_symbol_model(runtime):
        pass


def test_mini_deploy_runs_inside_training_guard():
    prices = np.linspace(100.0, 130.0, 140)
    model = create_direction_model(input_dim=FEATURE_DIM)
    stats = fit_norm_stats(np.zeros((2, 20, FEATURE_DIM), dtype=np.float32))
    runtime = {
        "lookback": 20,
        "val_accuracy": 0.55,
        "val_brier": 0.2,
        "calibrator": None,
        "model_lock": threading.RLock(),
    }
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {"execute": True, "trade_score": 0.7},
    }
    orch = SimpleNamespace(config={"deep_learning": {}})
    with (
        guard_symbol_model(runtime),
        patch(
            "src.application.services.deep_learning.dl_deploy_eval.predict_symbol_decision",
            return_value=entry,
        ),
    ):
        ok, wr, _ = evaluate_mini_deploy(
            orch,
            "OTC_SPC",
            model,
            prices,
            stats,
            runtime,
            {"lookback": 20},
            gate_cfg={
                "enabled": True,
                "mini_bars": 40,
                "min_trades": 2,
                "max_brier": 0.5,
                "min_win_rate": 0.4,
                "max_eval_steps": 8,
            },
        )
    assert ok is True
    assert wr > 0.0


def test_guard_symbol_model_serializes_access():
    lock = threading.RLock()
    runtime = {"model_lock": lock}
    order = []

    def worker(tag: str):
        with guard_symbol_model(runtime):
            order.append(f"{tag}-in")
            order.append(f"{tag}-out")

    threads = [threading.Thread(target=worker, args=(str(i),)) for i in range(3)]
    for thread in threads:
        thread.start()
    for thread in threads:
        thread.join()
    for idx in range(0, len(order), 2):
        assert order[idx].endswith("-in")
        assert order[idx + 1].endswith("-out")
