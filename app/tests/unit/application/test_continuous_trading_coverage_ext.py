from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.application.services.deep_learning.dl_model_types import FeatureNormStats
from src.application.services.deep_learning.dl_predict_async import predict_symbol_decision_async
from src.application.services.deep_learning.dl_predict_triton import predict_raw_prob_async
from src.application.services.execution_direction_cross_corr import (
    adjust_dl_weight_with_correlation,
    cached_correlation_matrix,
)
from src.application.services.execution_direction_resolver import resolve_execution_direction
from src.application.services.execution_direction_softmax_conflict import (
    _severe_exhaustion_conflict,
    _softmax_pair,
    resolve_softmax_exhaustion_conflict,
)
from src.application.services.execution_entropy_fallback import pick_entropy_fallback_candidate
from src.application.services.orchestrator.trading_cycle_entry import run_trading_cycle_if_ready
from src.domain.models.trade import TradeDirection


def test_severe_exhaustion_put_branch():
    metrics = {"exhaustion_conflict": True, "exhaustion_penalty": 0.2, "indicators": {"rsi": 0.2, "cmo": -0.5}}
    assert _severe_exhaustion_conflict(metrics, TradeDirection.PUT, cfg={"exhaustion_gate": {}})


@pytest.mark.asyncio
async def test_predict_raw_prob_async_put_and_neutral():
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
        return_value=0.35,
    ):
        direction, prob, raw = await predict_raw_prob_async(
            orch,
            "R_10",
            prices,
            runtime,
            {"lookback": 4},
            granularity=60,
            call_threshold=0.6,
            put_threshold=0.4,
        )
    assert direction == TradeDirection.PUT
    with patch(
        "src.application.services.deep_learning.dl_predict_triton.infer_symbol_async",
        new_callable=AsyncMock,
        return_value=0.5,
    ):
        direction, prob, raw = await predict_raw_prob_async(
            orch,
            "R_10",
            prices,
            runtime,
            {"lookback": 4},
            granularity=60,
            call_threshold=0.6,
            put_threshold=0.4,
        )
    assert direction is None


@pytest.mark.asyncio
async def test_predict_raw_prob_async_eager_branch():
    orch = MagicMock()
    orch.config = {"infra": {"triton": {"enabled": False}}}
    model = MagicMock()
    runtime = {
        "lookback": 4,
        "norm_stats": FeatureNormStats(mean=np.zeros(34, dtype=np.float32), std=np.ones(34, dtype=np.float32)),
        "model": model,
        "calibrator": None,
    }
    with patch(
        "src.application.services.deep_learning.dl_predict_triton.predict_next_direction",
        return_value=(TradeDirection.CALL, 0.7, 0.7),
    ):
        direction, prob, raw = await predict_raw_prob_async(
            orch,
            "R_10",
            np.linspace(1.0, 2.0, 20),
            runtime,
            {"lookback": 4},
            granularity=60,
        )
    assert direction == TradeDirection.CALL


def test_entropy_fallback_build_returns_none(monkeypatch):
    entry = {"metrics": {"deploy_ok": True, "calibrated_prob": 0.6, "gate_reason": None}}
    monkeypatch.setattr(
        "src.application.services.execution_entropy_fallback.build_execution_candidate",
        lambda *_a, **_k: None,
    )
    assert pick_entropy_fallback_candidate(["R_10"], {"R_10": entry}) is None


def test_cached_correlation_matrix_from_orch():
    orch = MagicMock()
    orch._corr_matrix_cache = {("R_10", "R_50"): 0.3}
    assert cached_correlation_matrix(orch)[("R_10", "R_50")] == 0.3


def test_cross_corr_no_adjustment_without_squeeze():
    weights = {"dl_raw_weight": 0.45}
    metrics = {"direction_margin": 0.2, "indicators": {"vol_ratio": 1.2}}
    assert adjust_dl_weight_with_correlation(weights, "R_50", metrics, {}) == weights


def test_resolve_execution_direction_with_corr_matrix():
    entry = {
        "direction": TradeDirection.CALL,
        "metrics": {
            "calibrated_prob": 0.62,
            "raw_prob": 0.6,
            "val_accuracy": 0.55,
            "deploy_ok": True,
            "direction_margin": 0.02,
            "indicators": {"vol_ratio": 0.7, "rsi": 0.5, "adx": 0.3, "hurst": 0.5, "cmo": 0.0},
            "bb_squeeze": True,
        },
    }
    corr = {("R_50", "R_10"): 0.8, ("R_10", "R_50"): 0.8}
    resolved = resolve_execution_direction(
        entry,
        exec_cfg={"quality_gate": {"min_direction_margin": 0.05}},
        symbol="R_50",
        corr_matrix=corr,
    )
    assert resolved is not None


@pytest.mark.asyncio
async def test_trading_cycle_refreshes_correlation_on_interval():
    orch = MagicMock()
    orch._reconciliation_pending = False
    orch.config = {
        "orchestrator": {"cycle_interval_seconds": 0},
        "infra": {"triton": {"correlation_refresh_cycles": 1}},
        "deep_learning": {"enabled": True},
    }
    orch.running = True
    orch.is_trading = False
    orch.state.active_contracts = []
    orch.ws.is_running = True
    orch.stream.is_synchronized = True
    orch._dl_fast_cycle = True
    orch._cycle_seq = 1
    orch.lock = AsyncMock()
    orch.lock.__aenter__ = AsyncMock(return_value=None)
    orch.lock.__aexit__ = AsyncMock(return_value=None)
    orch.executor.execute_cluster = AsyncMock()
    with (
        patch(
            "src.application.services.orchestrator.trading_cycle_entry.collect_deep_learning_decisions",
            new_callable=AsyncMock,
            return_value={},
        ),
        patch(
            "src.application.services.orchestrator.trading_cycle_entry.refresh_correlation_cache",
            new_callable=AsyncMock,
        ) as mock_refresh,
        patch(
            "src.application.services.orchestrator.trading_cycle_entry.resolve_decision_mode",
            return_value="deep_learning",
        ),
        patch("src.application.services.orchestrator.trading_cycle_entry.mark_bar_processed", new_callable=AsyncMock),
    ):
        await run_trading_cycle_if_ready(orch)
    mock_refresh.assert_awaited_once()


@pytest.mark.asyncio
async def test_predict_symbol_decision_async_triton_path():
    orch = MagicMock()
    orch.config = {"infra": {"triton": {"enabled": True}}, "orchestrator": {"execution": {}}, "deep_learning": {}}
    runtime = {
        "lookback": 4,
        "norm_stats": FeatureNormStats(mean=np.zeros(34, dtype=np.float32), std=np.ones(34, dtype=np.float32)),
        "val_accuracy": 0.5,
        "val_brier": 1.0,
        "val_ece": 1.0,
        "deploy_ok": True,
        "calibrator": None,
    }
    with (
        patch(
            "src.application.services.deep_learning.dl_predict_async.predict_raw_prob_async",
            new_callable=AsyncMock,
            return_value=(TradeDirection.PUT, 0.35, 0.35),
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
            MagicMock(),
            np.linspace(1.0, 2.0, 30),
            runtime["norm_stats"],
            runtime,
            {"lookback": 4, "implied_vol_bars": 60, "contract_duration": 60},
            None,
            recovery_active=False,
        )
    assert entry["direction"] == TradeDirection.PUT


def test_entropy_fallback_skips_symbol_in_set():
    entry = {"metrics": {"deploy_ok": True, "calibrated_prob": 0.62, "gate_reason": None}}
    assert (
        pick_entropy_fallback_candidate(["R_10", "R_50"], {"R_50": entry}, skip_symbols=frozenset({"R_10"})) is not None
    )


def test_entropy_fallback_infers_direction_when_missing():
    entry = {"metrics": {"deploy_ok": True, "calibrated_prob": 0.72, "gate_reason": None}}
    with patch("src.application.services.execution_entropy_fallback.infer_dl_direction", return_value=None):
        picked = pick_entropy_fallback_candidate(["R_10"], {"R_10": entry})
    assert picked is not None
    assert picked[1] == TradeDirection.CALL


def test_softmax_pair_zero_total():
    with patch("src.application.services.execution_direction_softmax_conflict.math.exp", return_value=0.0):
        a, b = _softmax_pair(1.0, 2.0, 0.1)
    assert a == 0.5 and b == 0.5


def test_severe_exhaustion_penalty_threshold():
    metrics = {
        "exhaustion_conflict": True,
        "exhaustion_penalty": 0.2,
        "indicators": {"rsi": 0.5, "cmo": 0.0},
    }
    assert _severe_exhaustion_conflict(
        metrics,
        TradeDirection.CALL,
        cfg={"exhaustion_gate": {"min_penalty_skip": 0.12}},
    )


def test_softmax_put_dl_branch_resolves():
    metrics = {
        "exhaustion_conflict": True,
        "exhaustion_penalty": 0.2,
        "indicators": {"rsi": 0.2, "cmo": -0.6, "implied_vol_ratio": 0.9},
        "kelly_fraction_scale": 1.0,
        "direction_hints": [],
    }
    resolved, out = resolve_softmax_exhaustion_conflict(
        TradeDirection.PUT,
        TradeDirection.PUT,
        metrics,
        call_score=0.3,
        put_score=0.7,
        exec_cfg={"exhaustion_gate": {"rsi_oversold": 0.27, "cmo_bear": -0.48}},
    )
    assert resolved in (TradeDirection.CALL, TradeDirection.PUT)
    assert out.get("execution_mode") == "EXEC_DIVERGENT"
