from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.application.services.deep_learning.dl_model_artifacts import bootstrap_and_validate_models
from src.application.services.deep_learning.dl_model_types import FeatureNormStats
from src.application.services.deep_learning.dl_predict import predict_symbol_decision
from src.application.services.deep_learning.dl_predict_async import predict_symbol_decision_async
from src.application.services.deep_learning.dl_predict_metrics import attach_dynamic_metrics
from src.application.services.deep_learning.dl_predict_triton import predict_raw_prob_async
from src.application.services.execution_direction_cross_corr import (
    adjust_dl_weight_with_correlation,
    cached_correlation_matrix,
)
from src.application.services.execution_entropy_fallback import pick_entropy_fallback_candidate
from src.domain.models.trade import TradeDirection
from src.domain.risk.kelly_f_star_adjustments import apply_kelly_fraction_scale
from src.infrastructure.inference.triton_tensor_builder import build_inference_tensor


@pytest.mark.asyncio
async def test_predict_raw_prob_async_triton_path():
    orch = MagicMock()
    orch.config = {"infra": {"triton": {"enabled": True}}}
    runtime = {
        "lookback": 4,
        "norm_stats": FeatureNormStats(mean=np.zeros(34, dtype=np.float32), std=np.ones(34, dtype=np.float32)),
        "calibrator": None,
    }
    prices = np.linspace(1.0, 2.0, 20)
    with patch(
        "src.application.services.deep_learning.dl_predict_triton.infer_symbol_async",
        new_callable=AsyncMock,
        return_value=0.62,
    ):
        direction, prob, raw = await predict_raw_prob_async(
            orch,
            "RDBEAR",
            prices,
            runtime,
            {"lookback": 4, "confidence_call_threshold": 0.6, "confidence_put_threshold": 0.4},
            granularity=60,
            call_threshold=0.6,
            put_threshold=0.4,
        )
    assert direction == TradeDirection.CALL
    assert prob == pytest.approx(0.62)


@pytest.mark.asyncio
async def test_predict_raw_prob_async_eager_without_model():
    orch = MagicMock()
    orch.config = {"infra": {"triton": {"enabled": False}}}
    runtime = {"lookback": 4, "norm_stats": None, "model": None}
    direction, prob, raw = await predict_raw_prob_async(
        orch,
        "RDBEAR",
        np.linspace(1.0, 2.0, 20),
        runtime,
        {"lookback": 4},
        granularity=60,
    )
    assert direction is None
    assert prob == 0.5


def test_build_inference_tensor_raises_on_short_history():
    stats = FeatureNormStats(mean=np.zeros(34, dtype=np.float32), std=np.ones(34, dtype=np.float32))
    with pytest.raises(ValueError):
        build_inference_tensor(np.array([1.0, 2.0]), 8, stats)


def test_entropy_fallback_returns_none_without_prob():
    entry = {"metrics": {"deploy_ok": True, "gate_reason": None}}
    assert pick_entropy_fallback_candidate(["RDBEAR"], {"RDBEAR": entry}) is None


def test_cross_corr_low_correlation_boosts_weight():
    weights = {"dl_raw_weight": 0.45}
    metrics = {"direction_margin": 0.01, "indicators": {"vol_ratio": 0.7}, "bb_squeeze": True}
    corr = {("RDBULL", "RDBEAR"): 0.1, ("RDBEAR", "RDBULL"): 0.1}
    out = adjust_dl_weight_with_correlation(weights, "RDBULL", metrics, corr)
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
            "RDBEAR",
            MagicMock(),
            np.linspace(1.0, 2.0, 20),
            runtime["norm_stats"],
            runtime,
            {"lookback": 4, "implied_vol_bars": 60},
            None,
            recovery_active=False,
        )
    assert entry["metrics"]["gate_reason"] == "predict_error"


@pytest.mark.asyncio
async def test_predict_symbol_decision_async_eager_path():
    orch = MagicMock()
    orch.config = {"infra": {"triton": {"enabled": False}}, "orchestrator": {"execution": {}}, "deep_learning": {}}
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
            "RDBEAR",
            runtime["model"],
            np.linspace(1.0, 2.0, 30),
            runtime["norm_stats"],
            runtime,
            {"lookback": 4, "implied_vol_bars": 60, "contract_duration": 60},
            None,
            recovery_active=False,
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
        symbol="RDBEAR",
        bb_history=[],
        scale_enabled=False,
        runtime={"calibrated_entropy": 0.4, "entropy_violation": True},
    )
    assert metrics["calibrated_entropy"] == 0.4
    assert metrics["entropy_violation"] is True


def test_entropy_fallback_skips_blocked_symbol():
    blocked = {"metrics": {"gate_reason": "predict_error", "deploy_ok": False}}
    viable = {"metrics": {"deploy_ok": True, "calibrated_prob": 0.62, "gate_reason": None}}
    picked = pick_entropy_fallback_candidate(["RDBEAR", "RDBULL"], {"RDBEAR": blocked, "RDBULL": viable})
    assert picked is not None
    assert picked[0] == "RDBULL"


@pytest.mark.asyncio
async def test_bootstrap_calls_triton_sync_and_schema_probe(tmp_path):
    orch = MagicMock()
    orch.symbols = ["RDBEAR"]
    orch.config = {
        "deep_learning": {"enabled": True, "arch": "tcn", "lookback": 48},
        "data_handler": {},
        "risk_management": {"params": {}},
        "infra": {"triton": {"enabled": True}},
    }
    ckpt = tmp_path / "RDBEAR.pth"
    ckpt.write_bytes(b"x")
    ts_path = tmp_path / "RDBEAR_ts.pt"
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
        patch(
            "src.application.services.deep_learning.dl_model_artifacts.sync_all_symbols_to_triton",
            new_callable=AsyncMock,
        ) as mock_sync,
        patch(
            "src.application.services.deep_learning.dl_model_artifacts.verify_triton_schema_alignment_async",
            new_callable=AsyncMock,
        ) as mock_schema,
        patch(
            "src.application.services.deep_learning.dl_model_artifacts.verify_triton_stressed_inference_async",
            new_callable=AsyncMock,
        ) as mock_stress,
    ):
        await bootstrap_and_validate_models(orch)
    mock_sync.assert_awaited_once()
    mock_schema.assert_awaited_once()
    mock_stress.assert_awaited_once()
