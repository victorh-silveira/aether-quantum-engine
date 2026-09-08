"""Cobertura residual pos-purge loss_classifier e modulos adjacentes."""

from __future__ import annotations

import copy
import logging
from types import SimpleNamespace
from unittest.mock import AsyncMock, MagicMock, patch

import numpy as np
import pytest

from src.application.services.deep_learning.dl_calibration_fit import fit_calibrator
from src.application.services.deep_learning.dl_cycle_log import _log_scale_lines
from src.application.services.deep_learning.dl_predict_telemetry import stamp_macro_frame_telemetry
from src.application.services.deep_learning.dl_startup import prepare_inference_run_loop
from src.application.services.deep_learning.dl_training_epochs import _shuffled_batch_indices
from src.application.services.deep_learning.horizon_sweep import build_horizon_candidates
from src.application.services.doctrine_invariants import (
    assert_production_doctrine,
    load_doctrine_invariants,
    reset_doctrine_invariants_cache,
)
from src.application.services.execution_direction_checks import infer_dl_direction
from src.application.services.execution_scale_tape import (
    compute_tape_strong,
    mili_direction_from_flow,
    mini_pair_opposes_tcn,
)
from src.application.services.execution_scale_vision import compute_scale_directions
from src.application.services.loss_classifier_features import (
    LOSS_FEATURE_DIM,
    _f,
    _tick_accel,
    build_loss_feature_vector,
)
from src.application.services.loss_classifier_gate import apply_loss_classifier_gate
from src.application.services.loss_classifier_gate_support import (
    clear_stale_loss_clf_metrics,
    resolve_tcn_ref,
)
from src.application.services.loss_classifier_vectors import (
    bind_loss_feature_vector_to_contract,
    pop_loss_feature_vector,
)
from src.application.services.market_audit_ops_window import (
    closed_micro_candles,
    ops_window_candle_body,
    ops_window_candle_side,
    ops_window_from_candles,
    ops_window_from_stream,
    ops_window_stamped,
)
from src.application.services.orchestrator.execution_collect_helpers import log_execution_decision
from src.application.services.orchestrator.post_settlement_loss_cooldown import (
    log_trading_cycle_cooldown_skip,
    post_loss_cooldown_blocks_trading_cycle,
)
from src.application.services.orchestrator.settlement_outcome import _feed_loss_classifier_learn
from src.domain.config_knobs import load_settings_json
from src.domain.models.market_data import Candle
from src.domain.models.trade import TradeDirection
from src.domain.risk.kelly_p_align import _read_call_prob, apply_kelly_side_p
from src.domain.risk.risk_stake_calc_helpers import apply_named_soft_stake_cap, apply_scale_stake_cap
from src.domain.risk.soft_recovery_config import pending_waives_scale_explore


@pytest.fixture(autouse=True)
def _reset_doctrine_cache():
    reset_doctrine_invariants_cache()
    yield
    reset_doctrine_invariants_cache()


def test_fit_calibrator_identity_explicit_large_sample():
    probs = [0.7 + 0.01 * i for i in range(40)]
    labels = [1.0 if p >= 0.75 else 0.0 for p in probs]
    cal = fit_calibrator(probs, labels, calibration_cfg={"method": "identity", "small_sample_identity": False})
    assert cal.method == "identity"


def test_log_dl_cycle_scale_line_branches(caplog):
    logger = logging.getLogger("cov.dl_cycle_scale")
    logger.setLevel(logging.DEBUG)
    decisions = {
        "bad": "x",
        "nom": {"metrics": "bad"},
        "nos": {"metrics": {"scale_reason": "disabled"}},
        "nof": {"metrics": {"scale_audit": "SCALE || no macro here"}},
        "fmt": {
            "metrics": {"scale_micro_dir": "CALL", "scale_macro_dir": "CALL", "scale_reason": "ok"},
        },
    }
    with caplog.at_level(logging.DEBUG):
        _log_scale_lines(logger, decisions, orch=None, cycle_id=3)
    assert any("MACRO=CALL" in r.message for r in caplog.records if r.levelname == "DEBUG")


def test_stamp_macro_frame_telemetry_with_stream():
    closes = np.linspace(1.0, 1.5, 20)
    stream = SimpleNamespace(get_numpy_series=lambda _s, _f="close": closes, macro_granularity=3600)
    orch = SimpleNamespace(stream=stream)
    metrics: dict = {}
    stamp_macro_frame_telemetry(orch, "1HZ75V", metrics, {"granularity": 3600})
    assert "macro_indicators" in metrics
    assert "rsi" in metrics["macro_indicators"]


def test_prepare_inference_run_loop_train_mode_without_inference_startup():
    orch = SimpleNamespace(
        symbols=["R_10"],
        config={
            "deep_learning": {"online_training": True},
            "data_handler": {},
            "orchestrator": {"engine_mode": "train"},
        },
    )
    with patch(
        "src.application.services.deep_learning.dl_startup.all_symbols_have_checkpoints",
        return_value=True,
    ):
        assert prepare_inference_run_loop(orch) is False


def test_shuffled_batch_indices_single_batch_when_large_batch_size():
    batches = _shuffled_batch_indices(8, batch_size=16)
    assert len(batches) == 1
    assert len(batches[0]) == 8


def test_build_horizon_candidates_from_n_bars_list():
    rows = build_horizon_candidates({"deep_learning": {"horizon_sweep": {"n_bars": [2, 4]}}})
    assert [r["label_horizon_bars"] for r in rows] == [2, 4]


def test_load_doctrine_invariants_cache_and_validation_errors():
    first = load_doctrine_invariants()
    second = load_doctrine_invariants()
    assert first == second
    settings = copy.deepcopy(load_settings_json())
    settings["orchestrator"]["execution"] = "bad"
    with pytest.raises(ValueError, match="execution"):
        load_doctrine_invariants(settings)
    settings = copy.deepcopy(load_settings_json())
    del settings["orchestrator"]["execution"]["sample_size_policy"]["explore_stake_scale_floor"]
    with pytest.raises(ValueError, match="explore_stake_scale_floor"):
        load_doctrine_invariants(settings)
    settings = copy.deepcopy(load_settings_json())
    settings["risk_management"]["soft_recovery"] = "bad"
    with pytest.raises(ValueError, match="soft_recovery"):
        load_doctrine_invariants(settings)
    settings = copy.deepcopy(load_settings_json())
    settings["infra"] = {}
    with pytest.raises(ValueError, match="loss_classifier"):
        load_doctrine_invariants(settings)
    settings = copy.deepcopy(load_settings_json())
    del settings["risk_management"]["kelly"]
    with pytest.raises(ValueError, match="kelly"):
        load_doctrine_invariants(settings)
    settings = copy.deepcopy(load_settings_json())
    settings["orchestrator"]["execution"]["invert_exec_side"] = True
    with pytest.raises(ValueError, match="invert_exec_side"):
        load_doctrine_invariants(settings)
    settings = copy.deepcopy(load_settings_json())
    del settings["risk_management"]["large_account_stop_win_pct"]
    with pytest.raises(ValueError, match="large_account_stop_win_pct"):
        load_doctrine_invariants(settings)
    settings = copy.deepcopy(load_settings_json())
    del settings["deep_learning"]["online_training"]
    with pytest.raises(ValueError, match="online_training"):
        load_doctrine_invariants(settings)


def _mutate_inv(key: str, *, value: object):
    def _apply(inv: dict) -> dict:
        inv[key] = value
        return inv

    return _apply


@pytest.mark.parametrize(
    "mutator,match",
    [
        (_mutate_inv("mandatory_trade_each_cycle", value=True), "mandatory_trade"),
        (_mutate_inv("loss_clf_hard_p_loss_floor", value=0.8), "hard_p_loss_floor"),
        (_mutate_inv("loss_clf_enabled", value=False), "enabled"),
        (_mutate_inv("watchdog_stale_tick_seconds", value=120), "watchdog"),
        (_mutate_inv("settlement_tolerance_window_seconds", value=120), "settlement_tolerance"),
        (_mutate_inv("post_settlement_is_trading_wait_seconds", value=30), "post_settlement"),
        (_mutate_inv("amort_cycles_min", value=0), "amort_cycles"),
        (_mutate_inv("cover_multiple", value=3.0), "cover_multiple"),
        (_mutate_inv("cover_enabled", value=True), "cover_enabled"),
        (_mutate_inv("neutral_bankroll_pct", value=0.02), "neutral_bankroll_pct"),
        (_mutate_inv("min_stake_pct", value=0.02), "min_stake_pct"),
        (_mutate_inv("max_safe_stake_pct_linear3", value=0.05), "max_safe_stake_pct_linear3"),
        (_mutate_inv("large_account_stop_win_pct", value=0.0), "large_account_stop_win_pct"),
        (_mutate_inv("explore_stake_scale_floor", value=0.30), "explore_stake_scale_floor"),
        (_mutate_inv("max_safe_stake_cap", value=0.0), "max_safe_stake_cap"),
    ],
)
def test_assert_production_doctrine_rejects_bad_invariants(mutator, match):
    inv = load_doctrine_invariants()
    mutator(inv)
    with pytest.raises(ValueError, match=match):
        assert_production_doctrine(_settings_from_invariants(inv))


def test_invalidate_ema_cache_clears_state():
    from src.application.services import execution_ema_cache as ema

    ema._STATE["cache"] = {"x": 1}
    ema.invalidate_ema_cache(7)
    assert ema._STATE["cache"] == {}
    assert ema._STATE["cycle"] == 7
    ema.invalidate_ema_cache()
    assert ema._STATE["cycle"] is None
    ema._STATE["cache"] = "bad"
    ema.invalidate_ema_cache(1)
    assert ema._STATE["cycle"] == 1


def _settings_from_invariants(inv: dict) -> dict:
    settings = copy.deepcopy(load_settings_json())
    settings["orchestrator"]["execution"]["force_trade_every_cycle"] = inv["force_trade_every_cycle"]
    settings["orchestrator"]["execution"]["mandatory_trade_each_cycle"] = inv["mandatory_trade_each_cycle"]
    settings["deep_learning"]["online_training"] = inv["online_training"]
    settings["infra"]["loss_classifier"]["veto_mode"] = inv["loss_clf_veto_mode"]
    settings["infra"]["loss_classifier"]["hard_p_loss_floor"] = inv["loss_clf_hard_p_loss_floor"]
    settings["infra"]["loss_classifier"]["enabled"] = inv["loss_clf_enabled"]
    settings["orchestrator"]["watchdog_stale_tick_seconds"] = inv["watchdog_stale_tick_seconds"]
    settings["orchestrator"]["settlement_tolerance_window_seconds"] = inv["settlement_tolerance_window_seconds"]
    settings["orchestrator"]["post_settlement_is_trading_wait_seconds"] = inv["post_settlement_is_trading_wait_seconds"]
    settings["risk_management"]["soft_recovery"]["amort_cycles_min"] = inv["amort_cycles_min"]
    settings["risk_management"]["soft_recovery"]["amort_cycles_max"] = inv["amort_cycles_max"]
    settings["risk_management"]["soft_recovery"]["cover_multiple"] = inv["cover_multiple"]
    settings["risk_management"]["soft_recovery"]["cover_enabled"] = inv["cover_enabled"]
    settings["risk_management"]["kelly"]["neutral_bankroll_pct"] = inv["neutral_bankroll_pct"]
    settings["risk_management"]["kelly"]["min_stake_pct"] = inv["min_stake_pct"]
    settings["risk_management"]["soft_recovery"]["max_safe_stake_pct_linear3"] = inv["max_safe_stake_pct_linear3"]
    settings["risk_management"]["large_account_stop_win_pct"] = inv["large_account_stop_win_pct"]
    settings["orchestrator"]["execution"]["sample_size_policy"]["explore_stake_scale_floor"] = inv[
        "explore_stake_scale_floor"
    ]
    settings["risk_management"]["soft_recovery"]["max_safe_stake_cap"] = inv["max_safe_stake_cap"]
    settings["risk_management"]["soft_recovery"]["max_safe_stake_pct"] = inv["max_safe_stake_pct"]
    settings["risk_management"]["min_validation_accuracy_gate"] = inv["min_validation_accuracy_gate"]
    return settings


def test_infer_dl_direction_neutral_zone():
    assert infer_dl_direction({"metrics": {"calibration_mode": "neutral_zone", "raw_prob": 0.6}}) is None


def test_mili_direction_from_flow_bad_types():
    assert mili_direction_from_flow({"price_velocity": "bad", "micro_tick_acceleration": "bad"}, None, "R_10") is None


def test_compute_tape_strong_micro_reinforcement():
    metrics = {
        "scale_mini_prev_bar_dir": "CALL",
        "scale_mini_bar_dir": "CALL",
        "scale_mili_dir": "PUT",
        "scale_micro_prev_bar_dir": "CALL",
        "scale_micro_bar_dir": "CALL",
    }
    assert compute_tape_strong(metrics, "CALL", mini_pair_sufficient=False) is True
    assert mini_pair_opposes_tcn(metrics, None) is False


def test_compute_scale_directions_invalid_tcn_and_missing_getter():
    metrics: dict = {}
    compute_scale_directions(None, "R_10", "INVALID", metrics)
    assert metrics["scale_micro_dir"] is None

    class Stream:
        tick_buffer = None

    compute_scale_directions(SimpleNamespace(stream=Stream()), "R_10", TradeDirection.CALL, metrics)
    assert metrics["scale_macro_dir"] is None


def test_loss_classifier_feature_helpers():
    assert _f({"x": "bad"}, "x", 0.5) == pytest.approx(0.5)
    assert _tick_accel({"flow_features": {"micro_tick_acceleration": "bad"}}) == 0.0
    metrics = {"edge_zscore": 0.2, "tcn_direction": "CALL"}
    vec = build_loss_feature_vector(metrics, TradeDirection.CALL)
    assert len(vec) == LOSS_FEATURE_DIM
    metrics2 = {"predicted_payoff_edge": "bad", "edge_zscore": 0.3, "tcn_direction": "PUT"}
    vec2 = build_loss_feature_vector(metrics2, TradeDirection.PUT)
    assert vec2[10] == pytest.approx(0.3)
    with (
        patch(
            "src.application.services.loss_classifier_features.LOSS_FEATURE_DIM",
            len(vec) + 1,
        ),
        pytest.raises(ValueError, match="loss feature dim"),
    ):
        build_loss_feature_vector(metrics, TradeDirection.CALL)


def test_loss_classifier_gate_support_and_vectors():
    metrics = {"gate_reason": "loss_clf", "signal_status": "SKIP:LOSS_CLF"}
    clear_stale_loss_clf_metrics(metrics)
    assert "gate_reason" not in metrics
    assert resolve_tcn_ref({}, TradeDirection.PUT) == TradeDirection.PUT
    orch = SimpleNamespace()
    orch._loss_clf_vectors = {"R_10": [0.1] * 24}
    bind_loss_feature_vector_to_contract(orch, "R_10", 9)
    assert pop_loss_feature_vector(orch, "R_10", 9) == [0.1] * 24
    orch._loss_clf_vectors = {"R_10": [0.1] * 24}
    assert pop_loss_feature_vector(orch, "R_10", 0) == [0.1] * 24
    empty = SimpleNamespace(_loss_clf_vectors={})
    assert pop_loss_feature_vector(empty, "X", 1) is None


def test_doctrine_invariants_infra_and_risk_gate():
    settings = copy.deepcopy(load_settings_json())
    settings["infra"] = "bad"
    with pytest.raises(ValueError, match="infra obrigatorio"):
        load_doctrine_invariants(settings)
    settings = copy.deepcopy(load_settings_json())
    del settings["risk_management"]["min_validation_accuracy_gate"]
    with pytest.raises(ValueError, match="min_validation_accuracy_gate"):
        load_doctrine_invariants(settings)


def test_execution_scale_tape_remaining_branches():
    metrics = {
        "scale_mini_prev_bar_dir": "CALL",
        "scale_mini_bar_dir": "CALL",
        "scale_mili_dir": "CALL",
        "scale_micro_prev_bar_dir": "PUT",
        "scale_micro_bar_dir": "PUT",
    }
    assert compute_tape_strong(metrics, "CALL", mini_pair_sufficient=False) is True
    oppose = {
        "scale_mini_prev_bar_dir": "PUT",
        "scale_mini_bar_dir": "PUT",
    }
    assert mini_pair_opposes_tcn(oppose, "CALL") is True
    split = {"scale_mini_prev_bar_dir": "CALL", "scale_mini_bar_dir": "PUT"}
    assert mini_pair_opposes_tcn(split, "PUT") is False


def test_compute_scale_field_from_stream_none_array():
    class Stream:
        def get_numpy_series(self, _symbol, _field="close"):
            return None

        tick_buffer = None

    metrics: dict = {}
    compute_scale_directions(SimpleNamespace(stream=Stream()), "R_10", TradeDirection.CALL, metrics)
    assert metrics["scale_macro_dir"] is None


def test_market_audit_ops_window_edge_cases():
    assert ops_window_from_candles([], bars=3) == (None, None)
    epoch = 1_700_000_000
    inf_candle = Candle("R_10", float("inf"), 1.0, 0.9, 1.0, None, epoch)
    assert ops_window_from_candles([inf_candle], bars=1) == (None, None)
    assert ops_window_candle_body({"ops_window_candle_body": -1.0}) is None


@pytest.mark.asyncio
async def test_await_post_loss_cooldown_sleeps():
    from src.application.services.orchestrator.post_settlement_loss_cooldown import await_post_loss_cooldown

    orch = SimpleNamespace(_cooldown_until=9999999999.0)
    with patch(
        "src.application.services.orchestrator.post_settlement_loss_cooldown.asyncio.sleep",
        new=AsyncMock(),
    ) as sleep_mock:
        rem = await await_post_loss_cooldown(orch)
    assert rem > 0.0
    sleep_mock.assert_awaited_once()


def test_stake_and_soft_recovery_remaining_branches():
    soft_off = {"material_pending_min": 0.25, "pending_waives_scale_explore": False}
    assert pending_waives_scale_explore(1.0, soft_off) is False
    assert (
        apply_scale_stake_cap(
            10.0,
            100.0,
            {"scale_discordance": True},
            pending_total=0.0,
            soft_recovery=soft_off,
        )
        == 10.0
    )
    assert (
        apply_named_soft_stake_cap(
            10.0,
            100.0,
            {"loss_clf_soft": True},
            flag="loss_clf_soft",
            pct_key="loss_clf_soft_max_stake_pct",
        )
        == 10.0
    )


@pytest.mark.asyncio
async def test_meta_classifier_pool_learn_with_running_loop():
    from src.infrastructure.inference.meta_classifier_pool import (
        close_meta_classifier_client,
        learn_meta_via_config_sync,
    )

    await close_meta_classifier_client()
    cfg = {"infra": {"meta_classifier": {"enabled": False}}}

    async def _inside():
        return learn_meta_via_config_sync(
            cfg,
            feature_vector=[0.1] * 23,
            target=0.1,
            contract_id="1",
            symbol="R_10",
        )

    result = await _inside()
    assert result["skipped"] is True
    await close_meta_classifier_client()


def test_loss_classifier_gate_branches(monkeypatch):
    metrics = {"gate_reason": "data"}
    assert apply_loss_classifier_gate(metrics, TradeDirection.CALL, orch=_orch()) is False
    metrics2 = {"tcn_direction": "CALL"}
    orch = _orch()
    orch.config = {"infra": {"loss_classifier": {"enabled": False}}}
    assert apply_loss_classifier_gate(metrics2, TradeDirection.CALL, orch=orch) is False
    metrics3 = {"tcn_direction": "CALL"}
    bad_orch = _orch()
    bad_orch.state = SimpleNamespace(balance="bad")
    bad_orch.risk_manager = SimpleNamespace(pending_loss_total=lambda: 0.0, bankroll="bad", initial_bankroll="bad")
    monkeypatch.setattr(
        "src.application.services.loss_classifier_gate.predict_loss_via_config_sync",
        lambda *_a, **_k: {
            "p_loss": 0.5,
            "model_version": "t",
            "n_train": 1,
            "auto_learn_applied": False,
            "veto_ready": False,
        },
    )
    monkeypatch.setattr(
        "src.application.services.loss_classifier_gate.build_loss_feature_vector",
        lambda *_a, **_k: [0.0] * 24,
    )
    assert apply_loss_classifier_gate(metrics3, TradeDirection.CALL, orch=bad_orch) is False


def _orch():
    return MagicMock(
        config={"infra": {"loss_classifier": {"enabled": True}}},
        _active_cycle_id=1,
        risk_manager=MagicMock(pending_loss_total=lambda: 0.0, bankroll=1000.0),
        state=MagicMock(balance=1000.0),
    )


def test_market_audit_ops_window_branches():
    assert closed_micro_candles(None, "R_10") == []
    assert ops_window_candle_side(None) is None
    assert ops_window_candle_body(None) is None
    assert ops_window_stamped(None) is False
    epoch = 1_700_000_000
    candles = [
        Candle("R_10", 1.0, 1.1, 0.9, 1.05, None, epoch),
        Candle("R_10", 1.05, 1.2, 1.0, 1.15, None, epoch + 60),
    ]
    side, body = ops_window_from_candles(candles, bars=2)
    assert side in {"CALL", "PUT"}
    assert body is not None
    bad = [Candle("R_10", "bad", 1.0, 0.9, 1.0, None, epoch)]
    assert ops_window_from_candles(bad, bars=1) == (None, None)
    stream = SimpleNamespace(micro_candles={"R_10": candles + [candles[-1]]})
    ops_side, ops_body, stamped = ops_window_from_stream(stream, "R_10", bars=2)
    assert stamped is True
    assert ops_window_candle_body({"ops_window_candle_body": "bad"}) is None


def test_log_execution_decision_bad_cycle_id():
    exec_mgr = SimpleNamespace(logger=MagicMock(), orch=SimpleNamespace(_active_cycle_id=7))
    best = ("R_10", TradeDirection.CALL, {"val_accuracy": 0.55})
    log_execution_decision(exec_mgr, "bad", best, [best], 0.55)
    assert exec_mgr.logger.info.call_count >= 1


def test_post_settlement_cooldown_and_learn_success(caplog):
    orch = SimpleNamespace(logger=None, _cooldown_until=9999999999.0)
    log_trading_cycle_cooldown_skip(orch)
    assert post_loss_cooldown_blocks_trading_cycle(orch) is True
    orch2 = MagicMock()
    orch2._loss_clf_vectors = {"cid:3": [0.1] * 24}
    orch2.config = {"infra": {"loss_classifier": {"enabled": True}}}
    with patch(
        "src.application.services.orchestrator.settlement_outcome.learn_loss_via_config_sync",
        return_value={"ok": True, "buffer_n": 4, "retrained": True, "n_train": 5},
    ):
        _feed_loss_classifier_learn(orch2, "R_10", won=True, contract_id=3)
    assert "label=WIN" in orch2._last_loss_clf_learn
    orch3 = MagicMock()
    orch3._loss_clf_vectors = {"cid:4": [0.1] * 24}
    orch3.config = None
    _feed_loss_classifier_learn(orch3, "R_10", won=False, contract_id=4)


def test_kelly_and_stake_helper_branches():
    assert _read_call_prob({"fusion_applied": True, "fusion_p_call": "bad", "raw_prob": 0.62}) == pytest.approx(0.62)
    assert _read_call_prob({"calibrated_prob": "bad", "raw_prob": 0.58}) == pytest.approx(0.58)
    metrics = {
        "calibrated_prob": 0.62,
        "scale_adapted": True,
        "tcn_direction": "PUT",
        "exec_direction": "CALL",
    }
    p = apply_kelly_side_p(
        metrics,
        order_direction="CALL",
        kelly_config={"kelly_p_floor": 0.55},
        conviction=0.55,
        payout=0.85,
    )
    assert metrics["scale_kelly_side_synced"] is True
    assert p >= 0.55
    soft = {"material_pending_min": 0.25, "pending_waives_scale_explore": True}
    assert pending_waives_scale_explore(0.30, soft) is True
    assert pending_waives_scale_explore(0.10, soft) is False
    assert (
        apply_scale_stake_cap(
            10.0,
            100.0,
            {"scale_discordance": True, "scale_max_stake_pct": "bad"},
            pending_total=0.0,
            soft_recovery=soft,
        )
        == 10.0
    )
    assert (
        apply_named_soft_stake_cap(
            10.0,
            100.0,
            {"loss_clf_soft": True, "loss_clf_soft_max_stake_pct": "bad"},
            flag="loss_clf_soft",
            pct_key="loss_clf_soft_max_stake_pct",
            pending_total=0.30,
            soft_recovery=soft,
        )
        == 10.0
    )


@pytest.mark.asyncio
async def test_stream_handler_apply_mini_unknown_symbol():
    from src.domain.models.market_data import Candle as CandleModel
    from src.infrastructure.handlers.stream_handler import StreamHandler

    ws = AsyncMock()
    ws.is_running = True
    sh = StreamHandler(ws, ["R_10"], {"granularity": 300, "micro_granularity": 60, "mini_granularity": 180})
    candle = CandleModel("R_10", 1.0, 1.1, 0.9, 1.05, None, 1000)
    await sh._apply_mini_candle("UNKNOWN", candle)


def test_stream_timeframe_mini_fallback_paths():
    from src.infrastructure.handlers.stream_timeframe import (
        ohlc_payload_granularity,
        resolve_mini_fetch_count,
    )

    assert ohlc_payload_granularity({"open_time": 90}, 300, 60, 180) == 180
    assert ohlc_payload_granularity({"open_time": 150}, 300, 60, 180) == 180
    assert ohlc_payload_granularity({"open_time": 77}, 300, 60, 180) == 180
    assert resolve_mini_fetch_count({"mini_history_bars": 128}) == 128
