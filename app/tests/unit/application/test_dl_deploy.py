import threading
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
import pytest

from src.application.services.deep_learning.dl_deploy import apply_deploy_to_runtime, direction_wins
from src.application.services.deep_learning.dl_deploy_eval import (
    _deploy_eval_bar_indices,
    _load_active_meta_model,
    _score_deploy_bar,
    evaluate_mini_deploy,
    resolve_settlement_horizon_bars,
)
from src.application.services.deep_learning.dl_features import FEATURE_DIM
from src.application.services.deep_learning.dl_gate_config import parse_deploy_gate_config
from src.application.services.deep_learning.dl_labels import LabelSpec
from src.application.services.deep_learning.dl_statistical_gate import wilson_lower_bound
from src.application.services.deep_learning.model import create_direction_model, fit_norm_stats
from src.domain.models.trade import TradeDirection


def test_parse_deploy_gate_config_defaults():
    cfg = parse_deploy_gate_config({})
    assert cfg["enabled"] is True
    assert cfg["force_ok"] is True
    assert cfg["max_brier"] == 0.26
    assert cfg["max_eval_steps"] == 0
    assert cfg["mini_bars"] == 0
    assert float(cfg["soft_min_val_accuracy"]) == pytest.approx(0.50)
    assert float(cfg["settlement_confidence"]) == pytest.approx(0.90)


def test_wilson_lower_bound_exige_evidencia_acima_do_breakeven():
    assert wilson_lower_bound(wins=23, trials=48, confidence=0.90) < 0.540541
    assert wilson_lower_bound(wins=32, trials=48, confidence=0.90) > 0.540541


def test_deploy_eval_bar_indices_caps_steps():
    dense = _deploy_eval_bar_indices(0, 500, 120)
    assert len(dense) <= 120
    assert dense[0] == 0
    assert dense[-1] < 500
    small = _deploy_eval_bar_indices(10, 20, 160)
    assert small == list(range(10, 20))


def test_deploy_eval_uses_same_ohlc_window_as_live_inference():
    prices = np.arange(100.0, 200.0)
    open_ = prices - 0.1
    high = prices + 0.2
    low = prices - 0.3
    micro = {"tick_count": np.arange(len(prices))}
    with patch(
        "src.application.services.deep_learning.dl_deploy_eval.predict_symbol_decision",
        return_value={"direction": None, "metrics": {"execute": False}},
    ) as predict:
        assert (
            _score_deploy_bar(
                orch=SimpleNamespace(),
                symbol="1HZ75V",
                model=None,
                prices=prices,
                norm_stats=None,
                sim_runtime={},
                eval_params={"inference_history_bars": 32, "granularity": 300},
                bar=80,
                open_=open_,
                high=high,
                low=low,
                micro=micro,
                settlement_spec=LabelSpec(),
            )
            is None
        )
    args = predict.call_args
    assert args.kwargs["granularity"] == 300
    np.testing.assert_array_equal(args.args[3], prices[49:81])
    np.testing.assert_array_equal(args.kwargs["open_"], open_[49:81])
    np.testing.assert_array_equal(args.kwargs["high"], high[49:81])
    np.testing.assert_array_equal(args.kwargs["low"], low[49:81])
    np.testing.assert_array_equal(args.kwargs["micro"]["tick_count"], micro["tick_count"][49:81])


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
    orch = type("O", (), {"config": {"infra": {}}})()
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


def test_m5_proxy_cannot_qualify_when_broker_audit_required():
    prices = np.linspace(100.0, 110.0, 40)
    model = create_direction_model(input_dim=FEATURE_DIM)
    stats = fit_norm_stats(np.zeros((1, 8, FEATURE_DIM), dtype=np.float32))
    runtime = {"val_brier": 0.2, "lookback": 8, "calibrator": None}
    with patch(
        "src.application.services.deep_learning.dl_deploy_eval.predict_symbol_decision",
        return_value={"direction": TradeDirection.CALL, "metrics": {"execute": True, "raw_prob": 0.9}},
    ):
        ok, wr, _ = evaluate_mini_deploy(
            SimpleNamespace(config={}),
            "1HZ75V",
            model,
            prices,
            stats,
            runtime,
            {"lookback": 8, "contract_duration_seconds": 300},
            gate_cfg={
                "enabled": True,
                "mini_bars": 20,
                "min_trades": 1,
                "max_brier": 1.0,
                "min_win_rate": 0.0,
                "max_eval_steps": 4,
                "require_broker_settlement": True,
                "provisional_min_trades": 1,
                "provisional_max_brier": 1.0,
                "provisional_min_win_rate": 0.0,
            },
        )
    assert wr == 1.0
    assert ok is False
    assert runtime["deploy_settlement_source"] == "m5_close_proxy"
    assert runtime["deploy_provisional_ok"] is False


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


def test_apply_deploy_to_runtime_preserva_provisorio_sem_promocao_plena():
    runtime = {"val_brier": 0.5}
    apply_deploy_to_runtime(runtime, deploy_ok=False, deploy_win_rate=0.58, val_brier=0.24, provisional_ok=True)
    assert runtime["deploy_ok"] is False
    assert runtime["deploy_provisional_ok"] is True


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
                "max_eval_steps": 10,
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
                "max_eval_steps": 10,
            },
        )
    assert ok is False
    assert wr == 0.0


def test_resolve_settlement_horizon_bars_uses_risk_params():
    assert resolve_settlement_horizon_bars({"risk_params": {"duration": 120, "duration_unit": "s"}}, 120) == 1
    assert resolve_settlement_horizon_bars({"contract_duration_seconds": 240}, 120) == 2
    assert resolve_settlement_horizon_bars({}, 120) == 1


def test_evaluate_mini_deploy_rejeita_label_quando_settlement_falha():
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
    assert ok is False
    assert "deploy_settlement_win_rate" in runtime
    assert wr >= 0.0
    assert brier >= 0.0


def test_wilson_rejeita_confianca_invalida_e_zero_amostras():
    assert wilson_lower_bound(wins=0, trials=0, confidence=0.95) == 0.0
    with pytest.raises(ValueError, match="settlement_confidence"):
        wilson_lower_bound(wins=1, trials=1, confidence=0.8)


def test_load_active_meta_model_paths_and_exceptions(tmp_path):
    import joblib

    assert _load_active_meta_model({"meta_model_path": tmp_path / "nonexistent.pkl"}) is None

    p_dict = tmp_path / "model_dict.pkl"
    dummy_model = {"key": "val"}
    joblib.dump({"model": dummy_model}, p_dict)
    assert _load_active_meta_model({"meta_model_path": p_dict}) == dummy_model

    p_non_dict = tmp_path / "model_list.pkl"
    joblib.dump(["a", "b"], p_non_dict)
    assert _load_active_meta_model({"meta_model_path": p_non_dict}) == ["a", "b"]

    p_corrupt = tmp_path / "corrupt.pkl"
    p_corrupt.write_bytes(b"invalid data")
    assert _load_active_meta_model({"meta_model_path": p_corrupt}) is None


def test_score_deploy_bar_meta_filter():
    class DummyMeta:
        def __init__(self, edge: float, *, fail: bool = False):
            self.edge = edge
            self.fail = fail

        def predict(self, _x):
            if self.fail:
                raise RuntimeError("predict boom")
            return [self.edge]

    prices = np.linspace(100.0, 110.0, 50)
    open_ = prices.copy()
    high = prices + 0.1
    low = prices - 0.1
    micro = {"tick_count": np.ones(50, dtype=np.float32)}

    with patch(
        "src.application.services.deep_learning.dl_deploy_eval.predict_symbol_decision",
        return_value={"direction": TradeDirection.CALL, "metrics": {"execute": True, "raw_prob": 0.8}},
    ):
        res_veto = _score_deploy_bar(
            orch=SimpleNamespace(),
            symbol="1HZ75V",
            model=None,
            prices=prices,
            norm_stats=None,
            sim_runtime={},
            eval_params={"meta_min_edge": 0.02},
            bar=35,
            settlement_spec=LabelSpec(label_mode="spot_forward", horizon_bars=1),
            meta_model=DummyMeta(edge=0.01),
            open_=open_,
            high=high,
            low=low,
            micro=micro,
        )
        assert res_veto is None

        res_fail = _score_deploy_bar(
            orch=SimpleNamespace(),
            symbol="1HZ75V",
            model=None,
            prices=prices,
            norm_stats=None,
            sim_runtime={},
            eval_params={"meta_min_edge": 0.02},
            bar=35,
            settlement_spec=LabelSpec(label_mode="spot_forward", horizon_bars=1),
            meta_model=DummyMeta(edge=0.0, fail=True),
            open_=open_,
            high=high,
            low=low,
            micro=micro,
        )
        assert res_fail is not None
        assert res_fail[0] in (True, False)


def test_evaluate_mini_deploy_logs_settlement_progress():
    prices = np.linspace(100.0, 110.0, 60)
    open_ = prices.copy()
    high = prices + 0.1
    low = prices - 0.1
    micro = {"tick_count": np.ones(60, dtype=np.float32)}

    with (
        patch(
            "src.application.services.deep_learning.dl_deploy_eval.predict_symbol_decision",
            return_value={"direction": TradeDirection.CALL, "metrics": {"execute": True, "raw_prob": 0.8}},
        ),
        patch("src.application.services.deep_learning.dl_deploy_eval.logger.info") as log_mock,
    ):
        evaluate_mini_deploy(
            SimpleNamespace(),
            "1HZ75V",
            model=None,
            prices=prices,
            norm_stats=None,
            runtime={"lookback": 10},
            params={"lookback": 10},
            gate_cfg={
                "enabled": True,
                "mini_bars": 20,
                "max_eval_steps": 5,
                "min_trades": 1,
                "min_win_rate": 0.5,
                "max_brier": 0.5,
            },
            open_=open_,
            high=high,
            low=low,
            micro=micro,
        )
        assert log_mock.call_count >= 2
        messages = [call[0][0] for call in log_mock.call_args_list]
        assert any("SETTLE | Iniciando avaliacao OOS" in m for m in messages)
        assert any("SETTLE | progresso=" in m for m in messages)


def test_evaluate_mini_deploy_skips_when_max_eval_steps_zero():
    prices = np.linspace(100.0, 110.0, 60)
    runtime = {"lookback": 10, "val_accuracy": 0.55, "val_brier": 0.22}
    gate_cfg = {
        "enabled": True,
        "enforce_settle_gate": False,
        "max_eval_steps": 0,
        "mini_bars": 20,
        "min_trades": 5,
        "min_win_rate": 0.55,
        "max_brier": 0.25,
        "force_ok": False,
    }
    ok, wr, brier = evaluate_mini_deploy(
        SimpleNamespace(),
        "1HZ75V",
        model=None,
        prices=prices,
        norm_stats=None,
        runtime=runtime,
        params={"lookback": 10},
        gate_cfg=gate_cfg,
    )
    assert ok is True
    assert wr == 0.55
    assert brier == 0.22
    assert runtime["deploy_settlement_n"] == 0
    assert runtime["deploy_provisional_ok"] is True


def test_evaluate_mini_deploy_waives_settle_gate_when_enforce_false():
    prices = np.linspace(100.0, 110.0, 60)
    runtime = {"lookback": 10, "val_accuracy": 0.56, "val_brier": 0.23}
    gate_cfg = {
        "enabled": True,
        "enforce_settle_gate": False,
        "max_eval_steps": 2,
        "mini_bars": 20,
        "min_trades": 10,
        "min_win_rate": 0.60,
        "max_brier": 0.20,
        "require_broker_settlement": True,
    }
    with patch(
        "src.application.services.deep_learning.dl_deploy_eval.predict_symbol_decision",
        return_value={"direction": None, "metrics": {"execute": False}},
    ):
        ok, wr, brier = evaluate_mini_deploy(
            SimpleNamespace(),
            "1HZ75V",
            model=None,
            prices=prices,
            norm_stats=None,
            runtime=runtime,
            params={"lookback": 10},
            gate_cfg=gate_cfg,
        )
    assert ok is True
    assert runtime["deploy_provisional_ok"] is True


def test_evaluate_mini_deploy_reaches_final_settle_when_enforce_false():
    prices = np.linspace(100.0, 110.0, 60)
    runtime = {"lookback": 10, "val_accuracy": 0.56, "val_brier": 0.23}
    gate_cfg = {
        "enabled": True,
        "enforce_settle_gate": False,
        "max_eval_steps": 2,
        "mini_bars": 20,
        "min_trades": 1,
        "min_win_rate": 0.99,
        "max_brier": 0.01,
        "require_broker_settlement": True,
    }
    with patch(
        "src.application.services.deep_learning.dl_deploy_eval.predict_symbol_decision",
        return_value={"direction": TradeDirection.CALL, "metrics": {"execute": True, "raw_prob": 0.8}},
    ):
        ok, wr, brier = evaluate_mini_deploy(
            SimpleNamespace(),
            "1HZ75V",
            model=None,
            prices=prices,
            norm_stats=None,
            runtime=runtime,
            params={"lookback": 10},
            gate_cfg=gate_cfg,
        )
    assert ok is True
    assert runtime["deploy_provisional_ok"] is True
