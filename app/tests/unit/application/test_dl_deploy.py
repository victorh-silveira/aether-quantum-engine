import threading
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np

from src.application.services.deep_learning.dl_deploy import apply_deploy_to_runtime, direction_wins
from src.application.services.deep_learning.dl_deploy_eval import (
    _deploy_eval_bar_indices,
    evaluate_mini_deploy,
    resolve_settlement_horizon_bars,
)
from src.application.services.deep_learning.dl_features import FEATURE_DIM
from src.application.services.deep_learning.dl_gate_config import parse_deploy_gate_config
from src.application.services.deep_learning.model import create_direction_model, fit_norm_stats
from src.domain.models.trade import TradeDirection


def test_parse_deploy_gate_config_defaults():
    cfg = parse_deploy_gate_config({})
    assert cfg["enabled"] is True
    assert cfg["max_brier"] == 0.22
    assert cfg["max_eval_steps"] == 24


def test_deploy_eval_bar_indices_caps_steps():
    dense = _deploy_eval_bar_indices(0, 500, 120)
    assert len(dense) <= 120
    assert dense[0] == 0
    assert dense[-1] < 500
    small = _deploy_eval_bar_indices(10, 20, 160)
    assert small == list(range(10, 20))


def test_evaluate_mini_deploy_insufficient_history():
    orch = type("O", (), {"config": {"deep_learning": {}}})()
    model = create_direction_model(input_dim=FEATURE_DIM)
    stats = fit_norm_stats(np.zeros((1, 32, FEATURE_DIM), dtype=np.float32))
    runtime = {"val_accuracy": 0.5, "val_brier": 1.0, "lookback": 32, "calibrator": None}
    params = {"lookback": 32, "validation_bars": 40}
    ok, wr, brier = evaluate_mini_deploy(
        orch,
        "X",
        model,
        np.linspace(1.0, 2.0, 50),
        stats,
        runtime,
        params,
        gate_cfg={"enabled": True, "mini_bars": 80, "min_trades": 8, "max_brier": 0.24, "min_win_rate": 0.52},
    )
    assert ok is False
    assert wr == 0.0


def test_evaluate_mini_deploy_forces_local_predict():
    orch = type("O", (), {"config": {"infra": {"triton": {"enabled": True}}}})()
    model = create_direction_model(input_dim=FEATURE_DIM)
    stats = fit_norm_stats(np.zeros((1, 32, FEATURE_DIM), dtype=np.float32))
    runtime = {
        "val_accuracy": 0.55,
        "val_brier": 0.2,
        "lookback": 8,
        "calibrator": None,
        "model": model,
        "model_lock": threading.RLock(),
    }
    params = {
        "lookback": 8,
        "validation_bars": 8,
        "confidence_call_threshold": 0.55,
        "confidence_put_threshold": 0.45,
        "contract_duration": 60,
        "implied_vol_bars": 4,
    }
    prices = np.linspace(100.0, 110.0, 40)
    with patch(
        "src.application.services.deep_learning.dl_deploy_eval.predict_symbol_decision",
        return_value={
            "direction": TradeDirection.CALL,
            "metrics": {"execute": True, "raw_prob": 0.7},
        },
    ) as mock_predict:
        evaluate_mini_deploy(
            orch,
            "R_10",
            model,
            prices,
            stats,
            runtime,
            params,
            gate_cfg={
                "enabled": True,
                "mini_bars": 20,
                "min_trades": 1,
                "max_brier": 0.99,
                "min_win_rate": 0.0,
                "max_eval_steps": 4,
            },
        )
    assert mock_predict.called
    assert mock_predict.call_args.kwargs.get("force_local") is True


def test_direction_wins_boundary():
    prices = np.array([10.0, 11.0])
    assert direction_wins(TradeDirection.CALL, prices, 1) is False


def test_deploy_gate_disabled():
    runtime = {"val_accuracy": 0.55, "val_brier": 0.2, "lookback": 20}
    ok, wr, b = evaluate_mini_deploy(
        SimpleNamespace(),
        "X",
        None,
        np.linspace(1.0, 2.0, 100),
        None,
        runtime,
        {},
        gate_cfg={"enabled": False},
    )
    assert ok is True


def test_apply_deploy_to_runtime_updates_brier():
    runtime = {"val_brier": 0.5}
    apply_deploy_to_runtime(runtime, deploy_ok=True, deploy_win_rate=0.6, val_brier=0.18)
    assert runtime["deploy_ok"] is True
    assert runtime["val_brier"] == 0.18


def test_evaluate_mini_deploy_passes_with_mock_predict():
    prices = np.linspace(100.0, 130.0, 120)
    model = create_direction_model(input_dim=FEATURE_DIM)
    stats = fit_norm_stats(np.zeros((2, 20, FEATURE_DIM), dtype=np.float32))
    runtime = {"lookback": 20, "val_accuracy": 0.55, "val_brier": 0.2, "calibrator": None}
    params = {"lookback": 20}
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {"execute": True, "trade_score": 0.7},
    }

    def always_exec(*_a, **_k):
        return entry

    orch = SimpleNamespace(config={"deep_learning": {}})
    with patch(
        "src.application.services.deep_learning.dl_deploy_eval.predict_symbol_decision",
        side_effect=always_exec,
    ):
        ok, wr, brier = evaluate_mini_deploy(
            orch,
            "X",
            model,
            prices,
            stats,
            runtime,
            params,
            gate_cfg={
                "enabled": True,
                "mini_bars": 40,
                "min_trades": 5,
                "max_brier": 0.5,
                "min_win_rate": 0.4,
            },
        )
    assert ok is True
    assert wr > 0.4
    assert parse_deploy_gate_config({"deploy_gate": {"enabled": False}})["enabled"] is False


def test_evaluate_mini_deploy_skips_non_execute_and_put_label():
    prices = np.linspace(100.0, 130.0, 120)
    model = create_direction_model(input_dim=FEATURE_DIM)
    stats = fit_norm_stats(np.zeros((2, 20, FEATURE_DIM), dtype=np.float32))
    runtime = {"lookback": 20, "val_accuracy": 0.55, "val_brier": 0.2, "calibrator": None}
    params = {"lookback": 20}
    calls = {"n": 0}

    def alternating(*_a, **_k):
        calls["n"] += 1
        if calls["n"] % 2 == 0:
            return {"direction": None, "metrics": {"execute": False}}
        return {
            "direction": TradeDirection.PUT,
            "metrics": {"execute": True, "trade_score": 0.4},
        }

    orch = SimpleNamespace(config={"deep_learning": {}})
    with patch(
        "src.application.services.deep_learning.dl_deploy_eval.predict_symbol_decision",
        side_effect=alternating,
    ):
        ok, wr, _ = evaluate_mini_deploy(
            orch,
            "X",
            model,
            prices,
            stats,
            runtime,
            params,
            gate_cfg={
                "enabled": True,
                "mini_bars": 40,
                "min_trades": 100,
                "max_brier": 0.5,
                "min_win_rate": 0.0,
            },
        )
    assert ok is False
    assert wr == 0.0


def test_resolve_settlement_horizon_bars_uses_risk_params():
    assert resolve_settlement_horizon_bars({"risk_params": {"duration": 120, "duration_unit": "s"}}, 120) == 1
    assert resolve_settlement_horizon_bars({"contract_duration_seconds": 240}, 120) == 2
    assert resolve_settlement_horizon_bars({}, 120) == 1


def test_evaluate_mini_deploy_falls_back_to_label_metrics_when_settlement_fails():
    orch = SimpleNamespace(config={"deep_learning": {}})
    model = create_direction_model(input_dim=FEATURE_DIM)
    prices = np.linspace(100.0, 110.0, 80)
    stats = fit_norm_stats(np.zeros((1, 32, FEATURE_DIM), dtype=np.float32))
    runtime = {"val_accuracy": 0.7, "val_brier": 0.2, "lookback": 32, "calibrator": None}
    params = {
        "lookback": 32,
        "granularity": 600,
        "contract_duration_seconds": 120,
        "label_horizon_bars": 1,
        "label_mode": "ma_trend",
        "label_ma_window": 5,
        "label_smooth_bars": 1,
    }

    def always_execute(*_a, **_k):
        return {
            "direction": TradeDirection.CALL,
            "metrics": {"execute": True, "raw_prob": 0.9},
        }

    with (
        patch(
            "src.application.services.deep_learning.dl_deploy_eval.predict_symbol_decision",
            side_effect=always_execute,
        ),
        patch(
            "src.application.services.deep_learning.dl_deploy_eval.direction_wins",
            side_effect=lambda direction, prices, bar, label_spec=None: bool(
                label_spec is not None and str(getattr(label_spec, "label_mode", "")) != "spot_forward"
            ),
        ),
    ):
        ok, wr, brier = evaluate_mini_deploy(
            orch,
            "X",
            model,
            prices,
            stats,
            runtime,
            params,
            gate_cfg={
                "enabled": True,
                "mini_bars": 40,
                "min_trades": 2,
                "max_brier": 0.5,
                "min_win_rate": 0.5,
                "max_eval_steps": 8,
            },
            micro={"tick_count": np.ones(80, dtype=np.float32)},
        )
    assert isinstance(ok, bool)
    assert "deploy_settlement_win_rate" in runtime
    assert wr >= 0.0
    assert brier >= 0.0
