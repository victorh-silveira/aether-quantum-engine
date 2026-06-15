from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.application.services.deep_learning.dl_bridge_helpers import apply_symbol_loss_cooldown
from src.application.services.deep_learning.dl_deploy_eval import evaluate_mini_deploy
from src.application.services.deep_learning.dl_feature_build import attach_microstructure
from src.application.services.deep_learning.dl_hurst import hurst_exponent
from src.application.services.deep_learning.dl_outcomes import (
    live_win_rate,
    maybe_pause_symbol_session,
    tick_dl_session_pauses,
)
from src.application.services.deep_learning.dl_params import optional_float, parse_dl_params
from src.application.services.deep_learning.dl_tcn import TemporalDirectionClassifier
from src.application.services.deep_learning.model import (
    INPUT_DIM,
    FeatureNormStats,
    create_direction_model,
    load_model_checkpoint,
    save_model_checkpoint,
)
from src.application.services.execution_mandatory_pick import _recovery_hedge_pick
from src.application.services.execution_market_rank import mandatory_pool_eligible, market_decision_score
from src.application.services.execution_symbols_recovery import (
    inject_recovery_hedge_candidates,
    recovery_rank_score,
)
from src.domain.models.trade import TradeDirection


def test_mandatory_pool_deploy_false():
    entry = {"direction": TradeDirection.CALL, "metrics": {"deploy_ok": False}}
    assert mandatory_pool_eligible(entry) is False


def test_market_decision_score_high_brier_penalty():
    score = market_decision_score(
        {"trade_score": 0.8, "raw_prob": 0.8, "val_accuracy": 0.55, "val_brier": 0.35, "execute": True},
        exec_direction=TradeDirection.CALL,
    )
    base = market_decision_score(
        {"trade_score": 0.8, "raw_prob": 0.8, "val_accuracy": 0.55, "val_brier": 0.20, "execute": True},
        exec_direction=TradeDirection.CALL,
    )
    assert score < base


def test_recovery_rank_score_same_symbol_penalty():
    item = ("R_50", TradeDirection.CALL, {"raw_prob": 0.8, "execute": True})
    penalized = recovery_rank_score(
        item,
        last_loss_symbol="R_50",
        last_loss_direction="CALL",
        base_score=0.7,
    )
    diversified = recovery_rank_score(
        item,
        last_loss_symbol="R_75",
        last_loss_direction="CALL",
        base_score=0.7,
    )
    assert penalized < diversified


def test_inject_recovery_hedge_missing_peer_entry():
    candidates = [("R_50", TradeDirection.PUT, {"trade_score": 0.8})]
    out = inject_recovery_hedge_candidates(
        candidates,
        {},
        last_loss_symbol="R_100",
        last_loss_direction="PUT",
    )
    assert out == candidates


def test_recovery_hedge_pick_forced_recovery_path():
    decisions = {
        "R_10": {
            "direction": TradeDirection.PUT,
            "metrics": {"deploy_ok": True, "raw_prob": 0.42},
        },
    }
    with patch(
        "src.application.services.execution_mandatory_pick.build_forced_direction_candidate",
        return_value=None,
    ):
        picked = _recovery_hedge_pick(
            decisions,
            last_loss_symbol="R_100",
            last_loss_direction="PUT",
            skip_symbols=frozenset(),
        )
    assert picked is not None
    assert picked[0] == "R_10"


def test_load_checkpoint_invalid_torchscript(tmp_path):
    model = create_direction_model(arch="tcn", input_dim=INPUT_DIM)
    path = tmp_path / "m.pth"
    save_model_checkpoint(
        path,
        model,
        FeatureNormStats(
            mean=np.zeros(INPUT_DIM, dtype=np.float32),
            std=np.ones(INPUT_DIM, dtype=np.float32),
        ),
        1,
        lookback=8,
        val_accuracy=0.55,
        val_brier=0.2,
        val_ece=0.1,
        deploy_ok=True,
        deploy_win_rate=0.6,
    )
    bad_ts = path.with_name(path.stem + "_ts.pt")
    bad_ts.write_bytes(b"not-script")
    loaded = load_model_checkpoint(path, params={"use_torchscript": True})
    assert loaded is not None


def test_load_checkpoint_uses_torchscript_when_enabled(tmp_path):
    model = create_direction_model(arch="tcn", input_dim=INPUT_DIM)
    path = tmp_path / "m.pth"
    save_model_checkpoint(
        path,
        model,
        FeatureNormStats(
            mean=np.zeros(INPUT_DIM, dtype=np.float32),
            std=np.ones(INPUT_DIM, dtype=np.float32),
        ),
        1,
        lookback=8,
        val_accuracy=0.55,
        val_brier=0.2,
        val_ece=0.1,
        deploy_ok=True,
        deploy_win_rate=0.6,
    )
    loaded = load_model_checkpoint(path, params={"use_torchscript": True})
    assert loaded is not None


def test_attach_microstructure_valid_arrays():
    series = {"log_return": np.zeros(3)}
    micro = {
        k: np.arange(3, dtype=np.float64)
        for k in (
            "tick_count",
            "mean_inter_tick_ms",
            "price_velocity",
            "price_acceleration",
            "consecutive_diff_std",
        )
    }
    attach_microstructure(series, micro)
    assert series["tick_count"].tolist() == [0.0, 1.0, 2.0]


def test_hurst_skips_tiny_rs():
    prices = np.concatenate([np.linspace(100.0, 100.001, 20), np.linspace(100.001, 100.002, 60)])
    out = hurst_exponent(prices, window=12)
    assert np.isfinite(out).all()


def test_live_win_rate_insufficient_samples():
    orch = SimpleNamespace(_dl_outcome_flags={"R_50": [True, False, True]})
    assert live_win_rate(orch, "R_50") is None


def test_tick_dl_session_pauses_clears_zero():
    orch = SimpleNamespace(_dl_session_pause={"R_50": 1})
    tick_dl_session_pauses(orch)
    assert "R_50" not in orch._dl_session_pause


def test_maybe_pause_noop_when_disabled():
    orch = SimpleNamespace(_dl_outcome_flags={"R_50": [False, False, False]})
    maybe_pause_symbol_session(orch, "R_50", max_losses_in_window=2, window_trades=3, pause_cycles=0)
    assert not hasattr(orch, "_dl_session_pause")


def test_optional_float_present():
    assert optional_float({"x": "0.5"}, "x") == pytest.approx(0.5)


def test_create_direction_model_unknown_arch():
    assert isinstance(create_direction_model(arch="unknown"), TemporalDirectionClassifier)


def test_save_torchscript_failure(tmp_path):
    model = create_direction_model(arch="tcn", input_dim=INPUT_DIM)
    path = tmp_path / "m.pth"
    with patch(
        "src.application.services.deep_learning.dl_model_checkpoint.torch.jit.trace",
        side_effect=RuntimeError("trace fail"),
    ):
        save_model_checkpoint(
            path,
            model,
            FeatureNormStats(
                mean=np.zeros(INPUT_DIM, dtype=np.float32),
                std=np.ones(INPUT_DIM, dtype=np.float32),
            ),
            1,
            lookback=8,
        )
    assert not path.with_name(path.stem + "_ts.pt").exists()


def test_evaluate_mini_deploy_micro_slice_runs():
    orch = MagicMock()
    model = create_direction_model(arch="tcn", input_dim=INPUT_DIM)
    prices = np.linspace(100.0, 110.0, 200)
    n = len(prices)
    micro = {
        k: np.linspace(0.0, 1.0, n)
        for k in (
            "tick_count",
            "mean_inter_tick_ms",
            "price_velocity",
            "price_acceleration",
            "consecutive_diff_std",
        )
    }
    stats = FeatureNormStats(
        mean=np.zeros(INPUT_DIM, dtype=np.float32),
        std=np.ones(INPUT_DIM, dtype=np.float32),
    )
    runtime = {"lookback": 48, "val_accuracy": 0.55, "val_brier": 0.2, "deploy_ok": True}
    params = parse_dl_params(
        {
            "lookback": 48,
            "confidence_call_threshold": 0.75,
            "confidence_put_threshold": 0.25,
            "deploy_gate": {"enabled": True, "mini_bars": 120, "min_trades": 1, "max_brier": 0.99, "min_win_rate": 0.0},
        },
        {},
    )
    with patch(
        "src.application.services.deep_learning.dl_deploy_eval.predict_symbol_decision",
        return_value={"direction": TradeDirection.CALL, "metrics": {"execute": True, "raw_prob": 0.9}},
    ) as mock_predict:
        ok, wr, brier = evaluate_mini_deploy(
            orch,
            "R_50",
            model,
            prices,
            stats,
            runtime,
            params,
            micro=micro,
        )
    assert mock_predict.called
    assert ok is True


def test_apply_symbol_loss_cooldown_session_pause():
    orch = SimpleNamespace(risk_manager=MagicMock(is_symbol_on_loss_cooldown=MagicMock(return_value=False)))
    orch._dl_session_pause = {"R_50": 3}
    entry = {"metrics": {"execute": True}}
    out = apply_symbol_loss_cooldown(orch, "R_50", entry)
    assert out["metrics"]["gate_reason"] == "session_pause"
    assert out["metrics"]["execute"] is False
