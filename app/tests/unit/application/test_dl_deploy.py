import numpy as np

from src.application.services.deep_learning.dl_deploy_eval import (
    _deploy_eval_bar_indices,
    evaluate_mini_deploy,
)
from src.application.services.deep_learning.dl_features import FEATURE_DIM
from src.application.services.deep_learning.dl_gate_config import parse_deploy_gate_config
from src.application.services.deep_learning.model import create_direction_model, fit_norm_stats


def test_parse_deploy_gate_config_defaults():
    cfg = parse_deploy_gate_config({})
    assert cfg["enabled"] is True
    assert cfg["max_brier"] == 0.24
    assert cfg["max_eval_steps"] == 160


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
