"""Cobertura residual da visao multi-escala e telemetria MACRO."""

from types import SimpleNamespace
from unittest.mock import MagicMock, patch

import numpy as np
import pytest

from src.application.services.deep_learning.dl_cycle_log import _log_scale_lines, log_dl_cycle_summary
from src.application.services.deep_learning.dl_predict_telemetry import stamp_macro_frame_telemetry
from src.application.services.execution_scale_sizing import apply_scale_kelly_sizing
from src.application.services.execution_scale_vision import (
    compute_scale_directions,
    mili_direction_from_flow,
    slope_direction,
)
from src.domain.models.market_data import Candle
from src.domain.models.trade import TradeDirection
from src.domain.risk.risk_stake_calc import calculate_stake_for_manager
from src.infrastructure.handlers.stream_handler import StreamHandler
from src.infrastructure.handlers.stream_timeframe import resolve_mini_fetch_count


def test_log_scale_lines_branches():
    logger = MagicMock()
    _log_scale_lines(logger, {"R_10": "bad"}, orch=None, cycle_id=1)
    _log_scale_lines(logger, {"R_10": {"metrics": "x"}}, orch=None, cycle_id=1)
    _log_scale_lines(logger, {"R_10": {"metrics": {"scale_audit": "nope"}}}, orch=None, cycle_id=1)
    _log_scale_lines(
        logger,
        {"R_10": {"metrics": {"scale_audit": "SCALE || MACRO=CALL MICRO=PUT MINI=- MILI=- agree=1/4 discord=False"}}},
        orch=None,
        cycle_id=2,
    )
    assert logger.info.call_count == 1


def test_stamp_macro_frame_telemetry_paths():
    stamp_macro_frame_telemetry(SimpleNamespace(stream=None), "R_10", {}, {})
    stream = SimpleNamespace(get_numpy_series=lambda *a, **k: np.array([]), macro_granularity=600)
    stamp_macro_frame_telemetry(SimpleNamespace(stream=stream), "R_10", {}, {})
    stream2 = SimpleNamespace(
        get_numpy_series=lambda *a, **k: np.linspace(1.0, 1.2, 20),
        macro_granularity=600,
    )
    metrics = {}
    with patch(
        "src.application.services.deep_learning.dl_predict_telemetry.precompute_price_series",
        return_value={
            "rsi": np.array([0.5]),
            "vol_ratio_short_long": np.array([1.0]),
            "adx": np.array([0.2]),
            "hurst": np.array([0.5]),
        },
    ):
        stamp_macro_frame_telemetry(SimpleNamespace(stream=stream2), "R_10", metrics, {"granularity": 600})
    assert "macro_indicators" in metrics


def test_scale_sizing_disabled_and_dampen_only():
    metrics = {"kelly_fraction_scale": 1.0, "scale_discordance": True}
    with patch(
        "src.application.services.execution_scale_sizing.parse_scale_vision_config",
        return_value={"enabled": False, "kelly_mult_discord": 0.35, "block_recover_on_discord": True},
    ):
        apply_scale_kelly_sizing(None, "R_10", TradeDirection.PUT, metrics)
    assert metrics["scale_sizing_reason"] == "disabled"
    metrics2 = {"kelly_fraction_scale": 1.0, "scale_discordance": True}
    with patch(
        "src.application.services.execution_scale_sizing.parse_scale_vision_config",
        return_value={"enabled": True, "kelly_mult_discord": 0.35, "block_recover_on_discord": False},
    ):
        apply_scale_kelly_sizing(None, "R_10", TradeDirection.PUT, metrics2)
    assert metrics2["scale_force_explore"] is False
    assert metrics2["scale_sizing_reason"] == "discord_dampen"


def test_mili_and_slope_edge_branches():
    assert slope_direction([1.0] * 5, bars=5) is None

    class TB:
        def live_tick_acceleration(self, _symbol):
            raise RuntimeError("x")

    assert mili_direction_from_flow({"price_velocity": "bad"}, TB(), "R_10") is None
    assert mili_direction_from_flow({"micro_tick_acceleration": "bad"}, None, "R_10") is None


def test_compute_scale_no_micro_dir():
    metrics = {}
    compute_scale_directions(
        SimpleNamespace(stream=None),
        "R_10",
        None,
        metrics,
        cfg={"enabled": True, "slope_bars": 5, "min_disagree_to_dampen": 2},
    )
    assert metrics["scale_micro_dir"] is None


def test_compute_scale_closes_getter_missing():
    metrics = {}
    compute_scale_directions(
        SimpleNamespace(stream=SimpleNamespace()),
        "R_10",
        TradeDirection.CALL,
        metrics,
        cfg={"enabled": True, "slope_bars": 5, "min_disagree_to_dampen": 2},
    )
    assert metrics["scale_macro_dir"] is None


def test_stake_force_explore_on_scale_flag():
    kelly = {
        "kelly": {
            "kelly_fraction": 0.25,
            "max_bet_pct": 0.05,
            "min_bet": 0.35,
            "payout_rate": 0.87,
            "explore_stake_scale_floor": 0.25,
            "mandatory_weak_conviction_cap": 0.55,
            "recovery_min_conviction": 0.50,
        },
        "params": {"duration": 120, "compounding_enabled": False},
        "soft_recovery": {"max_safe_stake_cap": 300.0, "max_safe_stake_pct": 0.03},
        "dlambert": {},
    }
    rm = MagicMock()
    rm.config = kelly
    rm.kelly_config = kelly["kelly"]
    rm.soft_recovery_config = kelly["soft_recovery"]
    rm.dlambert_config = {}
    rm.risk_params = kelly["params"]
    rm.initial_bankroll = 10000.0
    rm.total_session_profit = 0.0
    rm.pending_loss = {"R_10": 50.0}
    rm.active_contract_ids = []
    rm.consecutive_losses_linear = 2
    rm.dlambert_unit = 0.0
    rm.last_loss_stake = 0.0
    rm.logger = MagicMock()
    rm.effective_win_rate = MagicMock(return_value=0.55)
    rm._recovery_allowed = MagicMock(return_value=True)
    stake = calculate_stake_for_manager(
        rm,
        5000.0,
        "R_10",
        0.6,
        silent=True,
        apply_stop_win=False,
        kwargs={
            "dl_metrics": {
                "execute": True,
                "scale_force_explore": True,
                "scale_adapted": True,
                "scale_discordance": True,
                "scale_max_stake_pct": 0.005,
                "kelly_fraction_scale": 1.0,
            }
        },
    )
    assert stake > 5000.0 * 0.005
    assert stake <= 5000.0 * 0.05 + 1e-9


@pytest.mark.asyncio
async def test_stream_mini_candle_and_series():
    from datetime import UTC, datetime

    ws = MagicMock()
    ws.is_running = True
    sh = StreamHandler(
        ws,
        ["R_10"],
        {"granularity": 600, "micro_granularity": 120, "mini_granularity": 60, "buffer_limit": 100},
    )
    candle = Candle(
        symbol="R_10",
        epoch=60,
        open=1.0,
        high=1.1,
        low=0.9,
        close=1.05,
        time=datetime.fromtimestamp(60, tz=UTC),
    )
    await sh._apply_mini_candle(candle.symbol, candle)
    assert len(sh.mini_candles["R_10"]) == 1
    assert sh.get_mini_numpy_series("R_10").tolist() == [1.05]
    with patch(
        "src.infrastructure.handlers.stream_handler.candle_from_ohlc",
        return_value=Candle(
            symbol="R_10",
            epoch=120,
            open=1.0,
            high=1.0,
            low=1.0,
            close=1.02,
            time=datetime.fromtimestamp(120, tz=UTC),
        ),
    ):
        await sh._on_candle(
            {
                "ohlc": {
                    "symbol": "R_10",
                    "granularity": 60,
                    "open_time": 120,
                    "open": 1,
                    "high": 1,
                    "low": 1,
                    "close": 1.02,
                    "epoch": 120,
                }
            }
        )
    assert len(sh.mini_candles["R_10"]) >= 1


def test_log_scale_from_scale_micro_dir_without_audit():
    logger = MagicMock()
    _log_scale_lines(
        logger,
        {"R_10": {"metrics": {"scale_micro_dir": "CALL"}}},
        orch=None,
        cycle_id=3,
    )
    assert logger.info.call_count == 1


def test_closes_none_and_agree_peer():
    class Stream:
        def get_numpy_series(self, _symbol, _field="close"):
            return None

        def get_mini_numpy_series(self, _symbol, _field="close"):
            return np.array([1.0, 1.0, 1.0, 1.0, 1.1])

        tick_buffer = None

    metrics = {"flow_features": {"price_velocity": 0.0}}
    compute_scale_directions(
        SimpleNamespace(stream=Stream()),
        "R_10",
        TradeDirection.CALL,
        metrics,
        cfg={"enabled": True, "slope_bars": 5, "min_disagree_to_dampen": 2},
    )
    assert metrics["scale_mini_dir"] == "CALL"
    assert metrics["scale_agree_n"] >= 2


@pytest.mark.asyncio
async def test_apply_mini_unknown_symbol():
    from datetime import UTC, datetime

    ws = MagicMock()
    sh = StreamHandler(ws, ["R_10"], {"granularity": 600, "micro_granularity": 120, "mini_granularity": 60})
    candle = Candle(
        symbol="R_50",
        epoch=60,
        open=1.0,
        high=1.0,
        low=1.0,
        close=1.0,
        time=datetime.fromtimestamp(60, tz=UTC),
    )
    await sh._apply_mini_candle("R_50", candle)
    assert "R_50" not in sh.mini_candles


def test_ohlc_payload_without_mini_falls_back_micro():
    from src.infrastructure.handlers.stream_timeframe import ohlc_payload_granularity

    assert ohlc_payload_granularity({"open_time": 61}, 600, 120, None) == 120


def test_mini_fetch_history_bars_path():
    assert resolve_mini_fetch_count({"mini_history_bars": 77}) == 77
    assert resolve_mini_fetch_count({"mini_fetch_count": 12}) == 12
    assert resolve_mini_fetch_count({"startup_fetch_bars": 100}) == 100


def test_log_dl_cycle_with_scale_audit_dedupes():
    logger = MagicMock()
    orch = SimpleNamespace(_active_cycle_id=1, config={"data_handler": {"micro_granularity": 120}})
    decisions = {
        "R_10": {
            "direction": TradeDirection.CALL,
            "metrics": {
                "conviction": 0.70,
                "execute": True,
                "val_accuracy": 0.55,
                "cal_margin": 0.05,
                "scale_audit": "SCALE || MACRO=CALL MICRO=CALL MINI=CALL MILI=CALL agree=4/4 discord=False",
            },
        },
    }
    log_dl_cycle_summary(logger, decisions, recovery_active=False, pending_loss_total=0.0, orch=orch)
    log_dl_cycle_summary(logger, decisions, recovery_active=False, pending_loss_total=0.0, orch=orch)
    assert logger.info.call_count == 2
