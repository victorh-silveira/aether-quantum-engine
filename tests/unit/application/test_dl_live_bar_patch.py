"""Testes do patch de tick live na vela OHLC e microestrutura."""

from __future__ import annotations

from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from src.application.services.deep_learning.dl_live_bar_patch import (
    get_patched_ohlc_snapshot,
    patch_forming_bar_microstructure,
    patch_forming_bar_with_live_tick,
    store_patched_ohlc_snapshot,
)
from src.application.services.execution_scale_tape import last_bar_direction
from src.domain.models.trade import TradeDirection
from src.infrastructure.handlers.tick_buffer import TickBuffer


def test_patch_forming_bar_updates_close_high_low():
    buf = TickBuffer(["R_10"])
    buf.record_tick("R_10", 1_000, 105.0)
    orch = SimpleNamespace(stream=SimpleNamespace(tick_buffer=buf))
    close = np.array([100.0, 101.0, 102.0])
    open_ = np.array([99.0, 100.0, 101.0])
    high = np.array([100.5, 101.5, 102.5])
    low = np.array([98.0, 99.0, 100.0])
    c2, o2, h2, l2 = patch_forming_bar_with_live_tick(orch, "R_10", close, open_, high, low)
    assert c2[-1] == pytest.approx(105.0)
    assert h2[-1] == pytest.approx(105.0)
    assert l2[-1] == pytest.approx(100.0)
    assert o2[-1] == pytest.approx(101.0)
    assert close[-1] == pytest.approx(102.0)


def test_patch_forming_bar_fail_open_without_tick():
    orch = SimpleNamespace(stream=SimpleNamespace(tick_buffer=TickBuffer(["R_10"])))
    close = np.array([1.0, 2.0])
    open_ = np.array([1.0, 2.0])
    high = np.array([1.0, 2.0])
    low = np.array([1.0, 2.0])
    c2, o2, h2, l2 = patch_forming_bar_with_live_tick(orch, "R_10", close, open_, high, low)
    assert c2 is close
    assert o2 is open_
    assert h2 is high
    assert l2 is low
    empty = MagicMock()
    empty.stream = None
    assert patch_forming_bar_with_live_tick(empty, "R_10", close, open_, high, low)[0] is close


def test_tick_buffer_latest_price_and_forming_stats():
    buf = TickBuffer(["R_10"])
    assert buf.latest_price("R_10") is None
    assert buf.live_tick_count("R_10") == 0
    buf.record_tick("R_10", 1000, 12.5)
    assert buf.latest_price("R_10") == 12.5
    assert buf.live_tick_count("R_10") == 1
    buf.record_tick("R_10", 1100, 12.7)
    stats = buf.forming_bar_micro_stats("R_10")
    assert stats.tick_count == 2.0
    buf._live["R_10"].append((2000, object()))
    assert buf.latest_price("R_10") is None


def test_patch_forming_bar_empty_or_nonfinite():
    buf = TickBuffer(["R_10"])
    buf.record_tick("R_10", 1, float("nan"))
    orch = SimpleNamespace(stream=SimpleNamespace(tick_buffer=buf))
    empty = np.array([])
    assert patch_forming_bar_with_live_tick(orch, "R_10", empty, None, None, None)[0] is empty
    close = np.array([1.0, 2.0])
    c2, _, _, _ = patch_forming_bar_with_live_tick(orch, "R_10", close, None, None, None)
    assert c2 is close


def test_patch_forming_bar_microstructure_last_row():
    buf = TickBuffer(["R_10"])
    buf.record_tick("R_10", 1000, 100.0)
    buf.record_tick("R_10", 1100, 100.5)
    buf.record_tick("R_10", 1200, 101.0)
    orch = SimpleNamespace(stream=SimpleNamespace(tick_buffer=buf))
    micro = {
        "tick_count": np.array([0.0, 0.0, 0.0]),
        "price_velocity": np.array([0.0, 0.0, 0.0]),
        "price_acceleration": np.array([0.0, 0.0, 0.0]),
        "mean_inter_tick_ms": np.array([0.0, 0.0, 0.0]),
        "consecutive_diff_std": np.array([0.0, 0.0, 0.0]),
        "micro_bid_ask_spread_momentum": np.array([0.0, 0.0, 0.0]),
        "volatility_shadow_ratio": np.array([0.0, 0.0, 0.0]),
    }
    out = patch_forming_bar_microstructure(orch, "R_10", micro)
    assert out["tick_count"][-1] == pytest.approx(3.0)
    assert out["tick_count"][0] == pytest.approx(0.0)
    assert out["price_velocity"][-1] != pytest.approx(0.0)
    assert patch_forming_bar_microstructure(orch, "R_10", None) is None
    sparse = TickBuffer(["R_10"])
    sparse.record_tick("R_10", 1, 1.0)
    orch2 = SimpleNamespace(stream=SimpleNamespace(tick_buffer=sparse))
    assert patch_forming_bar_microstructure(orch2, "R_10", micro) is micro
    plain = SimpleNamespace(stream=SimpleNamespace(tick_buffer=object()))
    assert patch_forming_bar_microstructure(plain, "R_10", micro) is micro
    micro_extra = {
        "tick_count": np.array([]),
        "price_velocity": np.array([0.0, 0.0]),
        "extra_field": np.array([1.0, 2.0]),
    }
    buf.record_tick("R_10", 1300, 101.2)
    out2 = patch_forming_bar_microstructure(orch, "R_10", micro_extra)
    assert "extra_field" in out2
    assert out2["extra_field"] is micro_extra["extra_field"]
    assert "price_velocity" in out2


def test_store_and_get_patched_ohlc_snapshot_drives_last_bar_dir():
    orch = SimpleNamespace()
    close = np.array([10.0, 11.0, 12.5])
    open_ = np.array([10.0, 11.0, 11.0])
    store_patched_ohlc_snapshot(orch, "R_10", close, open_, None, None)
    snap = get_patched_ohlc_snapshot(orch, "R_10")
    assert snap is not None
    direction = last_bar_direction(snap["open"], snap["close"])
    assert direction == TradeDirection.CALL.name
    assert get_patched_ohlc_snapshot(SimpleNamespace(), "R_10") is None
