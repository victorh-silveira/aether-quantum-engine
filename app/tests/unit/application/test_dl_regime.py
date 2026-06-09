from unittest.mock import patch

import numpy as np

from src.application.services.deep_learning.dl_bridge_helpers import apply_symbol_loss_cooldown
from src.application.services.deep_learning.dl_gating import gating_block_reason
from src.application.services.deep_learning.dl_outcomes import (
    blended_val_accuracy,
    is_symbol_session_paused,
    live_win_rate,
    maybe_pause_symbol_session,
    recent_loss_count,
    record_symbol_outcome,
    tick_dl_session_pauses,
)
from src.application.services.deep_learning.dl_params import parse_dl_params
from src.application.services.deep_learning.dl_predict import predict_symbol_decision
from src.application.services.deep_learning.dl_regime import (
    direction_aligns_with_regime,
    latest_momentum,
    regime_strength,
)
from src.application.services.deep_learning.dl_tcn import TemporalDirectionClassifier
from src.application.services.deep_learning.model import (
    INPUT_DIM,
    create_direction_model,
    fit_norm_stats,
    predict_next_direction,
)
from src.domain.models.trade import TradeDirection


def test_direction_margin_blocks_ambiguous_raw():
    model = create_direction_model(arch="tcn")
    prices = np.linspace(100.0, 101.0, 80)
    direction, _, _, raw = predict_next_direction(
        model,
        prices,
        lookback=20,
        min_direction_margin=0.15,
    )
    assert direction is None or abs(raw - 0.5) >= 0.15


def test_saturation_blocks_extreme_raw():
    assert (
        gating_block_reason(
            0.85,
            0.70,
            0.60,
            0.07,
            0.52,
            raw_prob=0.98,
            max_raw_saturation=0.90,
            saturation_min_trade_score=0.58,
        )
        == "saturation"
    )


def test_regime_call_requires_positive_momentum():
    up = np.linspace(100.0, 110.0, 40)
    down = np.linspace(110.0, 100.0, 40)
    assert direction_aligns_with_regime(TradeDirection.CALL, up) is True
    assert direction_aligns_with_regime(TradeDirection.CALL, down) is False


def test_recent_loss_count_empty_history():
    orch = type("O", (), {})()
    assert recent_loss_count(orch, "R_50") == 0


def test_blended_val_accuracy_drops_after_losses():
    orch = type("O", (), {"config": {"deep_learning": {}}})()
    for _ in range(6):
        record_symbol_outcome(orch, "R_75", won=False)
    blended = blended_val_accuracy(orch, "R_75", 0.80, live_weight=0.55)
    assert blended < 0.55
    assert live_win_rate(orch, "R_75") == 0.0


def test_latest_momentum_and_regime_strength_short_history():
    assert latest_momentum(np.array([1.0, 2.0, 3.0])) == (0.0, 0.0)
    assert direction_aligns_with_regime(TradeDirection.CALL, np.array([1.0, 2.0, 3.0])) is False
    assert regime_strength(np.linspace(100.0, 110.0, 40)) > 0.0


def test_put_aligns_on_downtrend():
    down = np.linspace(110.0, 100.0, 40)
    assert direction_aligns_with_regime(TradeDirection.PUT, down) is True


def test_maybe_pause_disabled_when_cycles_zero():
    orch = type("O", (), {"config": {"deep_learning": {}}})()
    record_symbol_outcome(orch, "R_75", won=False)
    maybe_pause_symbol_session(orch, "R_75", max_losses_in_window=1, window_trades=3, pause_cycles=0)
    assert not hasattr(orch, "_dl_session_pause")


def test_apply_symbol_session_pause_gate():
    orch = type("O", (), {"_dl_session_pause": {"R_75": 2}, "risk_manager": None})()
    entry = {"metrics": {"execute": True}}
    out = apply_symbol_loss_cooldown(orch, "R_75", entry)
    assert out["metrics"]["execute"] is False
    assert out["metrics"]["gate_reason"] == "session_pause"


def test_predict_blocks_regime_and_live_wr():
    params = parse_dl_params(
        {
            "min_conviction_execute": 0.50,
            "min_edge_margin": 0.01,
            "min_val_accuracy": 0.0,
            "require_regime_alignment": True,
            "min_direction_margin": 0.01,
            "min_live_win_rate": 0.50,
            "binary_signal": {
                "min_rel_vol_execute": 0.0,
                "sma_z_block_call": 99.0,
                "sma_z_block_put": -99.0,
                "variance_ratio_mean_rev_max": 0.0,
            },
        }
    )
    orch = type("O", (), {"config": {"deep_learning": {}}, "_dl_outcome_flags": {"R_75": [False] * 6}})()
    runtime = {"val_accuracy": 0.7, "calibrator": None, "val_brier": 0.2, "val_ece": 0.1, "lookback": 15}
    down = np.linspace(110.0, 100.0, 80)
    with patch(
        "src.application.services.deep_learning.dl_predict.predict_next_direction",
        return_value=(TradeDirection.CALL, 0.72, 0.72, 0.72),
    ):
        entry = predict_symbol_decision(
            orch,
            "R_75",
            TemporalDirectionClassifier(input_dim=INPUT_DIM),
            down,
            fit_norm_stats(np.zeros((2, 15, INPUT_DIM), dtype=np.float32)),
            runtime,
            params,
            None,
            recovery_active=False,
        )
    assert entry["metrics"]["execute"] is False
    assert entry["metrics"]["gate_reason"] == "regime"
    up = np.linspace(100.0, 110.0, 80)
    with patch(
        "src.application.services.deep_learning.dl_predict.predict_next_direction",
        return_value=(TradeDirection.CALL, 0.72, 0.72, 0.72),
    ):
        entry_wr = predict_symbol_decision(
            orch,
            "R_75",
            TemporalDirectionClassifier(input_dim=INPUT_DIM),
            up,
            fit_norm_stats(np.zeros((2, 15, INPUT_DIM), dtype=np.float32)),
            runtime,
            params,
            None,
            recovery_active=False,
        )
    assert entry_wr["metrics"]["execute"] is False
    assert entry_wr["metrics"]["gate_reason"] == "live_wr"


def test_predict_handles_model_failure():
    params = parse_dl_params({"require_regime_alignment": False, "min_direction_margin": 0.01})
    orch = type("O", (), {"config": {"deep_learning": {}}})()
    runtime = {"val_accuracy": 0.5, "calibrator": None, "val_brier": 0.2, "val_ece": 0.1, "lookback": 15}
    with patch(
        "src.application.services.deep_learning.dl_predict.predict_next_direction",
        side_effect=RuntimeError("boom"),
    ):
        entry = predict_symbol_decision(
            orch,
            "R_75",
            TemporalDirectionClassifier(input_dim=INPUT_DIM),
            np.linspace(100.0, 110.0, 80),
            fit_norm_stats(np.zeros((2, 15, INPUT_DIM), dtype=np.float32)),
            runtime,
            params,
            None,
            recovery_active=False,
        )
    assert entry["metrics"]["execute"] is False


def test_predict_moderate_bypass_flag():
    params = parse_dl_params(
        {
            "min_conviction_execute": 0.70,
            "min_edge_margin": 0.06,
            "min_val_accuracy": 0.55,
            "moderate_signal_bypass": {
                "min_conviction_execute": 0.55,
                "min_edge_margin": 0.06,
                "min_val_accuracy": 0.40,
            },
            "require_regime_alignment": False,
            "min_direction_margin": 0.01,
            "binary_signal": {
                "min_rel_vol_execute": 0.0,
                "sma_z_block_call": 99.0,
                "sma_z_block_put": -99.0,
                "variance_ratio_mean_rev_max": 0.0,
            },
        }
    )
    orch = type("O", (), {"config": {"deep_learning": {}}})()
    runtime = {"val_accuracy": 0.50, "calibrator": None, "val_brier": 0.2, "val_ece": 0.1, "lookback": 15}
    with patch(
        "src.application.services.deep_learning.dl_predict.predict_next_direction",
        return_value=(TradeDirection.CALL, 0.72, 0.72, 0.62),
    ):
        entry = predict_symbol_decision(
            orch,
            "R_75",
            TemporalDirectionClassifier(input_dim=INPUT_DIM),
            np.linspace(100.0, 110.0, 80),
            fit_norm_stats(np.zeros((2, 15, INPUT_DIM), dtype=np.float32)),
            runtime,
            params,
            None,
            recovery_active=False,
        )
    assert entry["metrics"]["execute"] is True
    assert entry["metrics"]["bypass_val_acc"] is True


def test_predict_direction_margin_early_exit():
    params = parse_dl_params({"min_direction_margin": 0.20, "require_regime_alignment": False})
    orch = type("O", (), {"config": {"deep_learning": {}}})()
    runtime = {"val_accuracy": 0.5, "calibrator": None, "val_brier": 0.2, "val_ece": 0.1, "lookback": 15}
    with patch(
        "src.application.services.deep_learning.dl_predict.predict_next_direction",
        return_value=(None, 0.5, 0.5, 0.52),
    ):
        entry = predict_symbol_decision(
            orch,
            "R_75",
            TemporalDirectionClassifier(input_dim=INPUT_DIM),
            np.linspace(100.0, 110.0, 80),
            fit_norm_stats(np.zeros((2, 15, INPUT_DIM), dtype=np.float32)),
            runtime,
            params,
            None,
            recovery_active=False,
        )
    assert entry["metrics"]["gate_reason"] == "direction_margin"


def test_session_pause_after_loss_streak():
    orch = type(
        "O",
        (),
        {
            "config": {
                "deep_learning": {
                    "session_max_losses_in_window": 3,
                    "session_window_trades": 5,
                    "session_pause_cycles": 4,
                }
            }
        },
    )()
    for _ in range(3):
        record_symbol_outcome(orch, "R_75", won=False)
    assert is_symbol_session_paused(orch, "R_75") is True
    tick_dl_session_pauses(orch)
    tick_dl_session_pauses(orch)
    tick_dl_session_pauses(orch)
    tick_dl_session_pauses(orch)
    assert is_symbol_session_paused(orch, "R_75") is False


def test_direction_aligns_with_regime_rsi_exhaustion():
    up_extreme = np.linspace(100.0, 200.0, 50)
    assert direction_aligns_with_regime(TradeDirection.CALL, up_extreme, rsi_overbought=0.78) is False
    down_extreme = np.linspace(200.0, 100.0, 50)
    assert direction_aligns_with_regime(TradeDirection.PUT, down_extreme, rsi_oversold=0.22) is False
