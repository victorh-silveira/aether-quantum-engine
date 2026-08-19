"""Janela ops N M1 fechadas: helper puro, SCALE stamp e replay de gates."""

from datetime import UTC, datetime
from types import SimpleNamespace

import pytest

from src.application.services.execution_anti_loss import apply_anti_loss_seed_discord
from src.application.services.execution_direction_fusion import (
    apply_direction_fusion,
    parse_direction_fusion_config,
)
from src.application.services.execution_signal_skip import parse_signal_skip_config
from src.application.services.loss_classifier_flip import (
    seed_candle_blocks_flip,
    tcn_pos_edge_blocks_flip,
)
from src.application.services.market_audit_candle import closed_micro_candle_dir_from_stream
from src.application.services.market_audit_ops_window import (
    closed_micro_candles,
    ops_window_candle_body,
    ops_window_candle_side,
    ops_window_from_candles,
    ops_window_from_stream,
    ops_window_stamped,
    stamp_ops_window_metrics,
)
from src.domain.models.market_data import Candle
from src.domain.models.trade import TradeDirection


def _c(epoch: int, open_: float, close: float) -> Candle:
    return Candle(
        symbol="R_10",
        open=open_,
        high=max(open_, close),
        low=min(open_, close),
        close=close,
        time=datetime.fromtimestamp(epoch, tz=UTC),
        epoch=epoch,
    )


def _forming() -> Candle:
    return _c(360, 100.0, 100.0)


def test_closed_micro_candles_skips_forming_and_junk():
    forming = _forming()
    closed = _c(60, 1.0, 1.1)
    stream = SimpleNamespace(micro_candles={"R_10": [closed, "x", forming]})
    assert closed_micro_candles(stream, "R_10") == [closed]
    assert closed_micro_candles(None, "R_10") == []
    assert closed_micro_candles(SimpleNamespace(micro_candles="bad"), "R_10") == []
    assert closed_micro_candles(SimpleNamespace(micro_candles={"R_10": [forming]}), "R_10") == []
    assert closed_micro_candles(SimpleNamespace(micro_candles={"R_10": "bad"}), "R_10") == []


def test_ops_window_from_candles_n_doji_and_short():
    bars = [_c(60 * i, 10.0 + i, 10.1 + i) for i in range(1, 6)]
    side, body = ops_window_from_candles(bars, bars=5)
    assert side == "CALL"
    assert body == pytest.approx(4.1)
    put_bars = [_c(60, 10.0, 9.5), _c(120, 9.5, 9.0)]
    side_put, body_put = ops_window_from_candles(put_bars, bars=2)
    assert side_put == "PUT"
    assert body_put == pytest.approx(1.0)
    doji = [_c(60, 1.0, 1.0), _c(120, 1.0, 1.0)]
    assert ops_window_from_candles(doji, bars=2) == (None, None)
    assert ops_window_from_candles(bars[:3], bars=5) == (None, None)
    bad = [_c(60, float("nan"), 1.0), _c(120, 1.0, 1.1)]
    assert ops_window_from_candles(bad, bars=2) == (None, None)
    junk = SimpleNamespace(symbol="R_10", open="x", close=1.0, time=datetime.now(UTC), epoch=1)
    assert ops_window_from_candles([junk, junk], bars=2) == (None, None)


def test_ops_window_from_stream_fail_closed_when_short():
    forming = _forming()
    short = SimpleNamespace(micro_candles={"R_10": [_c(60, 1.0, 1.2), forming]})
    assert ops_window_from_stream(short, "R_10", bars=5) == (None, None, False)
    metrics: dict = {}
    stamp_ops_window_metrics(metrics, short, "R_10", bars=5)
    assert metrics["ops_window_stamped"] is False
    assert metrics["ops_window_candle_dir"] is None
    closed = [_c(60 * i, 100.0, 100.1) for i in range(1, 6)]
    full = SimpleNamespace(micro_candles={"R_10": [*closed, forming]})
    side, body, stamped = ops_window_from_stream(full, "R_10", bars=5)
    assert stamped is True
    assert side == "CALL"


def test_ops_window_readers_ignore_m1_keys():
    assert ops_window_candle_side({"closed_micro_candle_dir": "PUT"}) is None
    assert ops_window_candle_side({"ops_window_candle_dir": "DOJI"}) is None
    assert ops_window_candle_body(None) is None
    assert ops_window_candle_body({}) is None
    assert ops_window_candle_body({"ops_window_candle_body": float("inf")}) is None
    assert ops_window_candle_side(None) is None
    assert ops_window_candle_body({"ops_window_candle_body": 0.5}) == pytest.approx(0.5)
    assert ops_window_candle_body({"ops_window_candle_body": "x"}) is None
    assert ops_window_candle_body({"ops_window_candle_body": -0.1}) is None
    assert ops_window_stamped({"ops_window_stamped": True}) is True
    assert ops_window_stamped(None) is False


def test_stamp_ops_window_net_call_last_m1_put():
    closed = [
        _c(60, 100.000, 100.200),
        _c(120, 100.200, 100.400),
        _c(180, 100.400, 100.600),
        _c(240, 100.600, 101.000),
        _c(300, 101.000, 100.383),
    ]
    stream = SimpleNamespace(micro_candles={"R_10": [*closed, _forming()]})
    metrics: dict = {}
    stamp_ops_window_metrics(metrics, stream, "R_10", bars=5)
    assert metrics["ops_window_stamped"] is True
    assert metrics["ops_window_bars"] == 5
    assert metrics["ops_window_candle_dir"] == "CALL"
    assert metrics["ops_window_candle_body"] == pytest.approx(0.383)
    assert closed_micro_candle_dir_from_stream(stream, "R_10") == "PUT"


def test_anti_loss_live_follows_window_not_last_m1():
    metrics = {
        "execution_candidate_ready": True,
        "tcn_direction": "PUT",
        "resolved_direction": "PUT",
        "exec_direction": "CALL",
        "loss_clf_p_loss": 0.34,
        "loss_clf_auto_learn": True,
        "closed_micro_candle_dir": "PUT",
        "closed_micro_candle_body": 0.617,
        "closed_micro_candle_stamped": True,
        "ops_window_candle_dir": "CALL",
        "ops_window_candle_body": 0.035,
        "ops_window_stamped": True,
        "fusion_blocked_tcn_pos_edge": True,
        "kelly_fraction_scale": 1.0,
    }
    cfg = parse_signal_skip_config({"anti_loss_live_confirm_enabled": True})
    assert apply_anti_loss_seed_discord(metrics, cfg=cfg) is True
    assert metrics["anti_loss_why"] == "live_confirm_weak"


def test_anti_loss_live_agree_strong_on_window_put():
    metrics = {
        "execution_candidate_ready": True,
        "tcn_direction": "PUT",
        "resolved_direction": "PUT",
        "exec_direction": "PUT",
        "loss_clf_auto_learn": True,
        "closed_micro_candle_dir": "CALL",
        "closed_micro_candle_body": 0.05,
        "closed_micro_candle_stamped": True,
        "ops_window_candle_dir": "PUT",
        "ops_window_candle_body": 0.80,
        "ops_window_stamped": True,
        "fusion_blocked_tcn_pos_edge": True,
        "kelly_fraction_scale": 1.0,
    }
    cfg = parse_signal_skip_config({})
    assert apply_anti_loss_seed_discord(metrics, cfg=cfg) is False
    assert metrics.get("gate_reason") is None


def test_anti_loss_incomplete_window_does_not_use_m1():
    metrics = {
        "execution_candidate_ready": True,
        "tcn_direction": "PUT",
        "resolved_direction": "PUT",
        "exec_direction": "PUT",
        "loss_clf_p_loss": 0.34,
        "loss_clf_auto_learn": True,
        "closed_micro_candle_dir": "PUT",
        "closed_micro_candle_body": 0.617,
        "closed_micro_candle_stamped": True,
        "ops_window_candle_dir": None,
        "ops_window_candle_body": None,
        "ops_window_stamped": False,
        "fusion_blocked_tcn_pos_edge": True,
        "kelly_fraction_scale": 1.0,
    }
    cfg = parse_signal_skip_config({})
    assert apply_anti_loss_seed_discord(metrics, cfg=cfg) is False
    assert metrics.get("anti_loss_why") is None


def test_fusion_does_not_lock_tcn_when_window_discords():
    metrics = {
        "calibrated_prob": 0.53,
        "tcn_direction": "PUT",
        "scale_micro_dir": "PUT",
        "scale_macro_dir": "CALL",
        "scale_mini_dir": "CALL",
        "scale_mili_dir": "CALL",
        "scale_tape_consensus": "CALL",
        "closed_micro_candle_dir": "PUT",
        "ops_window_candle_dir": "CALL",
        "loss_clf_auto_learn": False,
        "execution_candidate_ready": True,
        "exec_direction": "PUT",
    }
    cfg = parse_direction_fusion_config({})
    apply_direction_fusion(metrics, TradeDirection.PUT, cfg=cfg)
    assert metrics.get("fusion_reason") != "tcn_candle_agree"
    assert metrics.get("fusion_blocked_tcn_candle") is not True


def test_flip_waiver_and_block_follow_window():
    cfg = {
        "flip_block_when_tcn_pos_edge": True,
        "flip_min_edge_execute": 0.04,
        "flip_waive_tcn_pos_edge_on_discord": True,
        "flip_seed_block_against_closed_candle": True,
    }
    waive = {
        "calibrated_prob": 0.36,
        "raw_prob": 0.36,
        "scale_tape_consensus": "CALL",
        "ops_window_candle_dir": "CALL",
        "closed_micro_candle_dir": "PUT",
    }
    assert tcn_pos_edge_blocks_flip(waive, TradeDirection.PUT, cfg=cfg) is False
    assert waive.get("loss_clf_flip_tcn_edge_waive_discord") is True
    keep = {
        "calibrated_prob": 0.36,
        "raw_prob": 0.36,
        "scale_tape_consensus": "PUT",
        "ops_window_candle_dir": "PUT",
        "closed_micro_candle_dir": "CALL",
    }
    assert tcn_pos_edge_blocks_flip(keep, TradeDirection.PUT, cfg=cfg) is True
    seed = {"ops_window_candle_dir": "PUT", "closed_micro_candle_dir": "CALL"}
    assert seed_candle_blocks_flip(seed, {"auto_learn_applied": False}, TradeDirection.PUT, cfg=cfg) is True
