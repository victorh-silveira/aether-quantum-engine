from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.application.services.deep_learning.dl_model_artifacts import bootstrap_and_validate_models
from src.application.services.deep_learning.dl_model_types import FeatureNormStats
from src.application.services.deep_learning.dl_predict import predict_symbol_decision
from src.application.services.deep_learning.dl_predict_async import predict_symbol_decision_async
from src.application.services.deep_learning.dl_predict_metrics import attach_dynamic_metrics
from src.application.services.execution_direction_cross_corr import (
    adjust_dl_weight_with_correlation,
    cached_correlation_matrix,
)
from src.application.services.execution_entropy_fallback import pick_entropy_fallback_candidate
from src.domain.models.trade import TradeDirection
from src.domain.risk.kelly_f_star_adjustments import apply_kelly_fraction_scale


def test_entropy_fallback_returns_none_without_prob():
    entry = {"metrics": {"deploy_ok": True, "gate_reason": None}}
    assert pick_entropy_fallback_candidate(["R_10"], {"R_10": entry}) is None


def test_cross_corr_low_correlation_boosts_weight():
    weights = {"dl_raw_weight": 0.45}
    metrics = {"direction_margin": 0.01, "indicators": {"vol_ratio": 0.7}, "bb_squeeze": True}
    corr = {("R_10", "R_50"): 0.1, ("R_50", "R_10"): 0.1}
    out = adjust_dl_weight_with_correlation(weights, "R_10", metrics, corr)
    assert out["dl_raw_weight"] > weights["dl_raw_weight"]


def test_cached_correlation_matrix_empty():
    assert cached_correlation_matrix(object()) == {}


def test_apply_kelly_fraction_scale_attenuates():
    assert apply_kelly_fraction_scale(2.0, {"kelly_fraction_scale": 0.5}) == 1.0


def test_predict_symbol_decision_exception_path():
    orch = MagicMock()
    orch.config = {"orchestrator": {"execution": {}}, "deep_learning": {}}
    runtime = {
        "val_accuracy": 0.5,
        "lookback": 4,
        "norm_stats": FeatureNormStats(mean=np.zeros(34, dtype=np.float32), std=np.ones(34, dtype=np.float32)),
        "calibrator": None,
        "deploy_ok": True,
        "val_brier": 1.0,
        "val_ece": 1.0,
    }
    with patch(
        "src.application.services.deep_learning.dl_predict_build.precompute_price_series",
        side_effect=RuntimeError("boom"),
    ):
        entry = predict_symbol_decision(
            orch,
            "R_10",
            MagicMock(),
            np.linspace(1.0, 2.0, 20),
            runtime["norm_stats"],
            runtime,
            {"lookback": 4, "implied_vol_bars": 60},
            None,
        )
    assert entry["metrics"]["gate_reason"] == "predict_error"


@pytest.mark.asyncio
async def test_predict_symbol_decision_async_eager_path():
    orch = MagicMock()
    orch.config = {"infra": {}, "orchestrator": {"execution": {}}, "deep_learning": {}}
    runtime = {
        "model": MagicMock(),
        "norm_stats": FeatureNormStats(mean=np.zeros(34, dtype=np.float32), std=np.ones(34, dtype=np.float32)),
        "lookback": 4,
        "val_accuracy": 0.5,
        "val_brier": 1.0,
        "val_ece": 1.0,
        "deploy_ok": True,
        "calibrator": None,
        "model_lock": None,
    }
    with (
        patch(
            "src.application.services.deep_learning.dl_predict_build.predict_next_direction",
            return_value=(TradeDirection.CALL, 0.7, 0.7),
        ),
        patch("src.application.services.deep_learning.dl_predict_build.precompute_price_series") as mock_series,
    ):
        mock_series.return_value = {
            "bb_width": np.array([0.1]),
            "atr_norm": np.array([0.1]),
            "adx": np.array([0.2]),
            "vol_ratio_short_long": np.array([1.0]),
            "implied_vol_ratio": np.array([1.0]),
            "hurst": np.array([0.5]),
            "cmo": np.array([0.0]),
            "keltner_pct_b": np.array([0.5]),
            "rsi": np.array([0.5]),
            "macd": np.array([0.0]),
            "macd_signal": np.array([0.0]),
            "di_diff": np.array([0.0]),
        }
        entry = await predict_symbol_decision_async(
            orch,
            "R_10",
            runtime["model"],
            np.linspace(1.0, 2.0, 30),
            runtime["norm_stats"],
            runtime,
            {"lookback": 4, "implied_vol_bars": 60, "contract_duration": 60},
            None,
        )
    assert entry["direction"] == TradeDirection.CALL


def test_attach_dynamic_metrics_runtime_entropy():
    metrics: dict = {}
    attach_dynamic_metrics(
        metrics,
        dynamic=None,
        bb_width=0.1,
        vol_ratio=1.0,
        implied_vol_ratio=1.0,
        symbol="R_10",
        bb_history=[],
        scale_enabled=False,
        runtime={"calibrated_entropy": 0.4, "entropy_violation": True},
    )
    assert metrics["calibrated_entropy"] == 0.4
    assert metrics["entropy_violation"] is True


def test_entropy_fallback_skips_blocked_symbol():
    blocked = {"metrics": {"gate_reason": "predict_error", "deploy_ok": False}}
    viable = {"metrics": {"deploy_ok": True, "calibrated_prob": 0.62, "gate_reason": None}}
    picked = pick_entropy_fallback_candidate(["R_10", "R_50"], {"R_10": blocked, "R_50": viable})
    assert picked is not None
    assert picked[0] in {"R_10", "R_50"}


@pytest.mark.asyncio
async def test_bootstrap_validates_local_torchscript(tmp_path):
    orch = MagicMock()
    orch.symbols = ["R_10"]
    orch.config = {
        "deep_learning": {"enabled": True, "arch": "tcn", "lookback": 48},
        "data_handler": {},
        "risk_management": {"params": {}},
        "infra": {},
    }
    ckpt = tmp_path / "R_10.pth"
    ckpt.write_bytes(b"x")
    ts_path = tmp_path / "R_10_ts.pt"
    ts_path.write_bytes(b"ts")
    store = MagicMock()
    store.download_torchscript = AsyncMock(return_value=False)
    store.sanity_check_torchscript = AsyncMock()
    orch.model_store = store
    with (
        patch(
            "src.application.services.deep_learning.dl_model_artifacts.ensure_local_model_checkpoint",
            new=AsyncMock(return_value=ckpt),
        ),
        patch(
            "src.application.services.deep_learning.dl_model_artifacts._scripted_path",
            return_value=ts_path,
        ),
    ):
        await bootstrap_and_validate_models(orch)
    store.sanity_check_torchscript.assert_awaited_once()
