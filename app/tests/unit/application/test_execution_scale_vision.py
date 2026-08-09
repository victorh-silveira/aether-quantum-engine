"""Testes de last-bar, consenso de fita e auditoria SCALE."""

import numpy as np
import pytest

from src.application.services.execution_scale_vision import (
    bar_direction_at,
    compute_scale_directions,
    format_scale_audit_line,
    format_scale_ind_token,
    last_bar_direction,
    mili_direction_from_flow,
    parse_scale_vision_config,
    prev_bar_direction,
    slope_direction,
    tape_consensus,
)
from src.domain.models.trade import TradeDirection


def test_slope_direction_call_and_put():
    assert slope_direction([1.0, 1.1, 1.2, 1.3, 1.4], bars=5) == "CALL"
    assert slope_direction([1.4, 1.3, 1.2, 1.1, 1.0], bars=5) == "PUT"
    assert slope_direction([1.0, 1.0], bars=5) is None


def test_last_and_prev_bar_direction():
    opens = [1.0, 1.0, 1.2]
    closes = [0.9, 1.1, 1.0]
    assert last_bar_direction(opens, closes) == "PUT"
    assert prev_bar_direction(opens, closes) == "CALL"
    assert bar_direction_at(opens, closes, offset=-1) == "PUT"
    assert bar_direction_at(opens, closes, offset=-2) == "CALL"
    assert prev_bar_direction([1.0], [1.1]) is None
    assert last_bar_direction([1.0], [1.0]) is None
    assert last_bar_direction([], [1.0]) is None


def test_tape_consensus():
    assert tape_consensus(["CALL", "CALL", "PUT"], min_votes=2) == "CALL"
    assert tape_consensus(["PUT", "PUT"], min_votes=2) == "PUT"
    assert tape_consensus(["CALL", "PUT"], min_votes=2) is None
    assert tape_consensus(["CALL", None], min_votes=2) is None


def test_mili_direction_from_flow():
    assert mili_direction_from_flow({"price_velocity": 0.5}, None, "R_10") == "CALL"
    assert mili_direction_from_flow({"micro_tick_acceleration": -1.0}, None, "R_10") == "PUT"
    assert mili_direction_from_flow({}, None, "R_10") is None


def test_parse_scale_vision_from_ssot():
    cfg = parse_scale_vision_config({})
    assert cfg["enabled"] is True
    assert cfg["kelly_mult_discord"] <= 1.0
    assert cfg["adapt_direction_enabled"] is True
    assert cfg["use_last_bar"] is True
    assert cfg["adapt_require_bar_pair_agree"] is True
    assert cfg["adapt_allow_strong_tape"] is False
    assert "adapt_max_cal_margin" not in cfg
    assert "adapt_min_cal_margin" not in cfg
    assert cfg["adapt_strong_mini_pair"] is True
    assert cfg["adapt_kelly_p_floor"] >= 0.51
    assert cfg["adapt_min_votes"] >= 1
    assert cfg["adapt_on_retraction"] is True
    assert cfg["adapt_on_explosion"] is True
    assert cfg["adapt_on_mili_tape"] is True
    assert cfg["adapt_mili_tape_skip_chop"] is True
    assert cfg["adapt_skip_chop"] is True
    assert cfg["adapt_require_cal_agree"] is True
    assert cfg["adapt_on_majority_votes"] is True
    assert cfg["adapt_majority_min_lead"] == 2
    assert cfg["adapt_majority_min_votes"] == 3
    assert cfg["adapt_majority_include_rsi"] is True
    assert cfg["adapt_majority_include_micro_bar"] is False
    assert cfg["adapt_majority_rsi_neutral"] == pytest.approx(0.5)
    assert cfg["retraction_require_mili"] is True
    assert cfg["retraction_use_tick_accel"] is True
    assert cfg["max_stake_pct_discord"] > 0.0


def test_compute_scale_discordance():
    class Stream:
        def get_numpy_series(self, _symbol, _field="close"):
            return np.array([1.0, 1.05, 1.1, 1.15, 1.2])

        def get_mini_numpy_series(self, _symbol, field="close"):
            if field == "open":
                return np.array([1.0, 1.0, 1.0, 1.0, 1.0])
            return np.array([1.0, 1.05, 1.1, 1.15, 1.2])

        def get_micro_numpy_series(self, _symbol, field="close"):
            if field == "open":
                return np.array([1.0, 1.0, 1.0])
            return np.array([0.95, 1.05, 1.1])

        tick_buffer = None

    orch = type("O", (), {"stream": Stream()})()
    metrics = {"flow_features": {"price_velocity": 1.0}}
    compute_scale_directions(orch, "R_10", TradeDirection.PUT, metrics)
    assert metrics["scale_micro_dir"] == "PUT"
    assert metrics["scale_macro_dir"] == "CALL"
    assert metrics["scale_mini_prev_bar_dir"] == "CALL"
    assert metrics["scale_mini_bar_dir"] == "CALL"
    assert metrics["scale_micro_prev_bar_dir"] == "CALL"
    assert metrics["scale_micro_bar_dir"] == "CALL"
    assert metrics["scale_mili_dir"] == "CALL"
    assert metrics["scale_discordance"] is True
    assert metrics["scale_tape_consensus"] == "CALL"
    assert metrics["scale_tape_strong"] is True
    assert metrics["scale_mini_pair_oppose"] is True
    line = format_scale_audit_line(metrics)
    assert "MACRO=CALL" in line
    assert "mi_prev=CALL" in line
    assert "mi_cur=CALL" in line
    assert "tape=CALL" in line
    assert "micro=" in line
    assert "mi_p=CALL" in format_scale_ind_token(metrics)
    assert "micro=" in format_scale_ind_token(metrics)
    metrics["scale_vote_call_n"] = 1
    metrics["scale_vote_put_n"] = 3
    assert "votes=C1/P3" in format_scale_ind_token(metrics)
    assert format_scale_audit_line(None).startswith("SCALE ||")
    assert "votes=" not in format_scale_ind_token(None)


def test_compute_scale_disabled():
    metrics = {}
    compute_scale_directions(None, "R_10", TradeDirection.CALL, metrics, cfg={"enabled": False})
    assert metrics["scale_reason"] == "disabled"


def test_compute_scale_without_last_bar():
    class Stream:
        def get_numpy_series(self, _symbol, _field="close"):
            return np.array([1.0, 0.9, 0.8, 0.7, 0.6])

        def get_mini_numpy_series(self, _symbol, _field="close"):
            return np.array([1.0, 0.9, 0.8, 0.7, 0.6])

        tick_buffer = None

    metrics = {"flow_features": {"price_velocity": -1.0}}
    compute_scale_directions(
        type("O", (), {"stream": Stream()})(),
        "R_10",
        TradeDirection.CALL,
        metrics,
        cfg={
            "enabled": True,
            "slope_bars": 5,
            "min_disagree_to_dampen": 2,
            "use_last_bar": False,
            "adapt_min_votes": 2,
        },
    )
    assert metrics["scale_mini_bar_dir"] is None
    assert metrics["scale_discordance"] is True
    assert "adapted=1" in format_scale_audit_line({**metrics, "scale_adapted": True})


def test_compute_scale_uses_patched_ohlc_snapshot():
    from src.application.services.deep_learning.dl_live_bar_patch import store_patched_ohlc_snapshot

    class Stream:
        def get_numpy_series(self, _symbol, _field="close"):
            return np.array([1.0, 1.0, 1.0, 1.0, 1.0])

        def get_mini_numpy_series(self, _symbol, field="close"):
            if field == "open":
                return np.array([1.0, 1.0, 1.0])
            return np.array([1.0, 1.0, 1.0])

        def get_micro_numpy_series(self, _symbol, field="close"):
            if field == "open":
                return np.array([1.0, 1.0, 1.0])
            return np.array([1.0, 1.0, 1.0])

        tick_buffer = None

    orch = type("O", (), {"stream": Stream()})()
    store_patched_ohlc_snapshot(
        orch,
        "R_10",
        np.array([1.0, 1.0, 0.5]),
        np.array([1.0, 1.0, 1.0]),
        None,
        None,
    )
    metrics = {"flow_features": {}}
    compute_scale_directions(
        orch,
        "R_10",
        TradeDirection.CALL,
        metrics,
        cfg={
            "enabled": True,
            "slope_bars": 5,
            "min_disagree_to_dampen": 2,
            "use_last_bar": True,
            "adapt_min_votes": 2,
        },
    )
    assert metrics["scale_micro_bar_dir"] == "PUT"
