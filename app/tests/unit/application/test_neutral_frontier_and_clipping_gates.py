"""Homologacao dos gates de fronteira neutra e clipping OOD +-3.0 do meta 43D."""

from __future__ import annotations

import numpy as np
import pytest

from src.application.services.deep_learning.dl_feature_oscillators import rolling_zscore_1024_fast
from src.application.services.execution_direction_resolver import resolve_execution_direction
from src.application.services.meta_classifier_cross_symbol import META_FEATURE_DIM
from src.application.services.meta_classifier_features import (
    clip_feature_zscore,
    extract_meta_feature_vector,
)
from src.application.services.meta_payoff_regression import CALIBRATION_NEUTRAL_DRIFT
from src.domain.models.trade import TradeDirection
from src.domain.risk.consensus_stake_penalty import max_safe_stake_cap
from src.domain.risk.risk_recovery_state import (
    cointegration_pair_score,
    cointegration_redirect_armed,
    micro_tail_stake_cap,
    select_cointegration_redirect_candidate,
)


def test_clip_feature_zscore_saturates_at_three() -> None:
    assert clip_feature_zscore(4.16) == 3.0
    assert clip_feature_zscore(-5.0) == -3.0
    assert clip_feature_zscore(1.25) == pytest.approx(1.25)


def test_rolling_zscore_1024_fast_clips_ood_spikes() -> None:
    series = np.ones(1024, dtype=np.float64)
    series[-1] = 50.0
    z = rolling_zscore_1024_fast(series)
    assert float(np.max(z)) <= 3.0
    assert float(np.min(z)) >= -3.0
    assert z[-1] == pytest.approx(3.0)


def test_extract_meta_feature_vector_rigid_dim_and_micro_zscore_clip() -> None:
    metrics = {
        "feature_vector": [0.0] * 34,
        "flow_features": {
            "micro_bid_ask_spread_momentum": 0.12,
            "micro_bid_ask_spread_momentum_zscore": 4.8,
            "volatility_shadow_ratio": 0.31,
            "volatility_shadow_ratio_zscore": -3.7,
        },
        "cross_symbol_features": {
            "cross_symbol_prob_delta": 0.1,
            "cross_symbol_vol_ratio_diff": 0.0,
            "cross_symbol_rsi_spread": -0.2,
        },
    }
    vector = extract_meta_feature_vector(metrics)
    assert len(vector) == META_FEATURE_DIM == 43
    assert vector[35] == 3.0
    assert vector[37] == -3.0
    assert metrics["meta_feature_vector"] is vector


def test_extract_meta_feature_vector_reclips_cached_payload() -> None:
    cached = [0.0] * META_FEATURE_DIM
    cached[35] = 9.0
    cached[37] = -9.0
    metrics = {"meta_feature_vector": cached}
    vector = extract_meta_feature_vector(metrics)
    assert vector[35] == 3.0
    assert vector[37] == -3.0


@pytest.mark.parametrize(
    ("raw_prob", "calibrated_prob"),
    (
        (0.41, 0.56),
        (0.62, 0.44),
    ),
)
def test_resolve_execution_direction_calibration_neutral_drift_veto(
    raw_prob: float,
    calibrated_prob: float,
) -> None:
    entry = {
        "direction": TradeDirection.CALL if raw_prob > 0.5 else TradeDirection.PUT,
        "metrics": {
            "deploy_ok": True,
            "execute": True,
            "raw_prob": raw_prob,
            "calibrated_prob": calibrated_prob,
            "val_accuracy": 0.66,
        },
    }
    result = resolve_execution_direction(entry, symbol="OTC_SPC")
    assert result is not None
    assert entry["metrics"].get("resolved_direction") is not None
    assert entry["metrics"].get("gate_reason") != CALIBRATION_NEUTRAL_DRIFT


def test_resolve_execution_direction_allows_same_side_calibration() -> None:
    entry = {
        "direction": TradeDirection.PUT,
        "metrics": {
            "deploy_ok": True,
            "execute": True,
            "raw_prob": 0.38,
            "calibrated_prob": 0.41,
            "val_accuracy": 0.66,
            "call_votes": 1,
            "put_votes": 5,
        },
    }
    result = resolve_execution_direction(entry, symbol="OTC_SPC")
    assert result is not None
    assert entry["metrics"].get("gate_reason") != CALIBRATION_NEUTRAL_DRIFT


def test_cointegration_redirect_armed_at_fifteen_percent_of_live_capital() -> None:
    assert cointegration_redirect_armed(100.0, 15.0) is False
    assert cointegration_redirect_armed(100.0, 15.01) is True
    assert cointegration_redirect_armed(300.0, 50.0) is False


def test_select_cointegration_redirect_prefers_high_z_low_entropy() -> None:
    candidates = [
        ("R_50", TradeDirection.CALL, {"calibrated_prob": 0.9, "edge_zscore": 2.0}),
        ("OTC_SPC", TradeDirection.CALL, {"calibrated_prob": 0.55, "edge_zscore": 0.4}),
        ("R_75", TradeDirection.PUT, {"calibrated_prob": 0.80, "edge_zscore": 1.5}),
    ]
    selected = select_cointegration_redirect_candidate(candidates)
    assert len(selected) == 1
    assert selected[0][0] == "OTC_SPC"
    assert cointegration_pair_score(candidates[1][2]) < cointegration_pair_score(candidates[2][2])


def test_micro_tail_stake_cap_and_max_safe_flatten_at_linear_four() -> None:
    assert micro_tail_stake_cap(100.0) == pytest.approx(4.20)
    assert max_safe_stake_cap(100.0, consecutive_losses_linear=0) == pytest.approx(5.0)
    assert max_safe_stake_cap(100.0, consecutive_losses_linear=4) == pytest.approx(8.40)
    assert max_safe_stake_cap(100.0, consecutive_losses_linear=8) == pytest.approx(8.40)
    assert max_safe_stake_cap(10000.0, consecutive_losses_linear=4) == pytest.approx(500.0)
