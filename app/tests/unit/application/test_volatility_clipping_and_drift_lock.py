from unittest.mock import MagicMock

import numpy as np

from src.application.services.deep_learning.dl_feature_build import precompute_price_series
from src.application.services.meta_classifier_features import _base_feature_vector, extract_meta_feature_vector
from src.domain.models.trade import TradeDirection
from src.domain.risk.risk_recovery_state import evaluate_anti_trend_lock
from src.domain.risk.risk_stake_calc import calculate_stake_for_manager


def test_volatility_clipping_in_precompute_price_series():
    # Cria uma série de preços com 1024 elementos
    prices = np.ones(1024, dtype=np.float64) * 100.0
    # Adiciona um estouro de volatilidade no final
    prices[-1] = 150.0
    high = prices + 1.0
    low = prices - 1.0

    series = precompute_price_series(
        prices,
        granularity=15,
        symbol="R_10",
        high=high,
        low=low,
    )

    # Verifica se os z-scores de bb_width e atr_norm estão dentro do range [-3.0, 3.0]
    assert np.all(series["bb_width"] >= -3.0)
    assert np.all(series["bb_width"] <= 3.0)
    assert np.all(series["atr_norm"] >= -3.0)
    assert np.all(series["atr_norm"] <= 3.0)


def test_meta_classifier_feature_clipping():
    # Caso 1: A partir do cached feature_vector
    feature_vector = [0.0] * 34
    feature_vector[8] = 5.5  # bb_width
    feature_vector[9] = -4.2  # atr_norm
    metrics = {"feature_vector": feature_vector}

    v = _base_feature_vector(metrics)
    assert v[8] == 3.0
    assert v[9] == -3.0

    # Caso 2: A partir dos indicators
    metrics_ind = {
        "indicators": {
            "bb_width": 4.16,
            "atr_norm": -5.0,
        },
        "raw_prob": 0.6,
    }
    v_ind = _base_feature_vector(metrics_ind)
    assert v_ind[4] == 3.0
    assert v_ind[5] == -3.0

    # Testar também extract_meta_feature_vector
    meta_v = extract_meta_feature_vector(metrics_ind)
    assert meta_v[4] == 3.0
    assert meta_v[5] == -3.0


def test_evaluate_anti_trend_lock_drift_bias_lock():
    resolved, action = evaluate_anti_trend_lock(
        symbol="R_10",
        proposed_direction=TradeDirection.PUT,
        consecutive_losses=0,
        bull_call_prob=0.5,
        bear_put_prob=0.5,
        probability_delta=0.0,
        predicted_payoff_edge=0.1,
        cross_symbol_prob_delta_mean=0.0,
        vol_ratio=2.5,
        bb_width_zscore=0.0,
    )
    assert resolved == TradeDirection.PUT
    assert action == "KEEP"

    resolved, action = evaluate_anti_trend_lock(
        symbol="R_10",
        proposed_direction=TradeDirection.CALL,
        consecutive_losses=0,
        bull_call_prob=0.5,
        bear_put_prob=0.5,
        probability_delta=0.0,
        predicted_payoff_edge=0.1,
        cross_symbol_prob_delta_mean=0.0,
        vol_ratio=0.0,
        bb_width_zscore=2.1,
    )
    assert resolved == TradeDirection.CALL
    assert action == "KEEP"

    resolved, action = evaluate_anti_trend_lock(
        symbol="R_10",
        proposed_direction=TradeDirection.CALL,
        consecutive_losses=0,
        bull_call_prob=0.5,
        bear_put_prob=0.5,
        probability_delta=0.0,
        predicted_payoff_edge=0.1,
        cross_symbol_prob_delta_mean=0.0,
        vol_ratio=1.0,
        bb_width_zscore=1.0,
    )
    assert resolved == TradeDirection.CALL
    assert action == "KEEP"


def test_risk_stake_calc_drift_bias_lock():
    rm = MagicMock()
    rm.kelly_config = {"consensus_penalty_enabled": False, "fraction": 0.001, "max_stake_pct": 1.0}
    rm.risk_params = {"payout_estimate": 0.95, "stake_min": 1.0}
    rm.effective_win_rate = MagicMock(return_value=0.6)
    rm._recovery_allowed = MagicMock(return_value=False)
    rm.dlambert_config = {}
    rm.consecutive_losses_linear = 0
    rm.pending_loss = {}
    rm.logger = MagicMock()

    stake = calculate_stake_for_manager(
        rm,
        bankroll=1000.0,
        symbol="R_10",
        conviction=0.7,
        silent=True,
        apply_stop_win=False,
        kwargs={
            "dl_metrics": {"execute": True, "vol_ratio": 2.2, "bb_width": 0.5},
            "order_direction": TradeDirection.PUT,
        },
    )
    assert stake > 0.0

    stake = calculate_stake_for_manager(
        rm,
        bankroll=1000.0,
        symbol="R_10",
        conviction=0.7,
        silent=True,
        apply_stop_win=False,
        kwargs={
            "dl_metrics": {"execute": True, "vol_ratio": 0.5, "bb_width": 2.5},
            "order_direction": TradeDirection.CALL,
        },
    )
    assert stake > 0.0
