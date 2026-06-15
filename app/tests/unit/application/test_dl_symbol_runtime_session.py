from unittest.mock import MagicMock, patch

import numpy as np

from src.application.services.deep_learning.dl_symbol_runtime import get_symbol_runtime
from src.application.services.deep_learning.model import fit_norm_stats


def _loaded_checkpoint(*, deploy_ok: bool, val_brier: float = 0.22):
    lookback = 48
    stats = fit_norm_stats(np.zeros((1, lookback, 21), dtype=np.float32))
    model = MagicMock()
    return (
        model,
        stats,
        12345,
        MagicMock(),
        lookback,
        0.62,
        val_brier,
        0.1,
        deploy_ok,
        0.55,
    )


def test_get_symbol_runtime_marks_session_trained_when_deploy_ok_checkpoint():
    orch = MagicMock()
    orch._dl_runtime = {}
    dl_config = {"model_path_template": "data/dl/{symbol}.pth"}
    params = {"lookback": 48, "arch": "tcn"}
    with patch(
        "src.application.services.deep_learning.dl_symbol_runtime.load_model_checkpoint",
        return_value=_loaded_checkpoint(deploy_ok=True),
    ):
        runtime = get_symbol_runtime(orch, "R_100", dl_config, params)
    assert runtime["session_trained"] is True
    assert runtime["deploy_ok"] is True


def test_get_symbol_runtime_keeps_session_untrained_without_deploy_ok():
    orch = MagicMock()
    orch._dl_runtime = {}
    dl_config = {"model_path_template": "data/dl/{symbol}.pth"}
    params = {"lookback": 48, "arch": "tcn"}
    with patch(
        "src.application.services.deep_learning.dl_symbol_runtime.load_model_checkpoint",
        return_value=_loaded_checkpoint(deploy_ok=False, val_brier=0.9),
    ):
        runtime = get_symbol_runtime(orch, "R_100", dl_config, params)
    assert runtime["session_trained"] is False
