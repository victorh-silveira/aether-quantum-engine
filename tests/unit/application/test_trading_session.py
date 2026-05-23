import time

from src.application.services.orchestrator.trading_session import (
    _hour_in_window,
    resolve_trading_session,
    trading_session_allows_entry,
)


def test_trading_session_warmup_blocks():
    cfg = {"trading": {"session": {"enabled": True, "warmup_seconds_after_stream_sync": 60}}}
    epoch = 1_700_000_000
    ok, note = trading_session_allows_entry(
        epoch_utc=epoch,
        stream_ready_at=time.time(),
        now_mono=time.time(),
        config=cfg,
    )
    assert ok is False
    assert "SESSION_WARMUP" in note


def test_trading_session_outside_hours():
    cfg = {
        "trading": {
            "session": {
                "enabled": True,
                "start_hour_utc": 8,
                "end_hour_utc": 18,
                "warmup_seconds_after_stream_sync": 0,
            }
        }
    }
    epoch = (1_700_000_000 // 86400) * 86400 + 2 * 3600
    ok, note = trading_session_allows_entry(
        epoch_utc=epoch,
        stream_ready_at=None,
        now_mono=time.time(),
        config=cfg,
    )
    assert ok is False
    assert "SESSION_CLOSED" in note


def test_trading_session_near_close():
    cfg = {
        "trading": {
            "session": {
                "enabled": True,
                "start_hour_utc": 7,
                "end_hour_utc": 22,
                "minutes_before_close_no_entry": 20,
                "warmup_seconds_after_stream_sync": 0,
            }
        }
    }
    epoch = 1_698_000_000
    hour = 21
    minute = 50
    epoch = epoch - (epoch % 86400) + hour * 3600 + minute * 60
    ok, note = trading_session_allows_entry(
        epoch_utc=epoch,
        stream_ready_at=None,
        now_mono=time.time(),
        config=cfg,
    )
    assert ok is False
    assert "SESSION_NEAR_CLOSE" in note


def test_hour_in_window_helpers():
    assert _hour_in_window(12, 7, 22) is True
    assert _hour_in_window(3, 7, 22) is False
    assert _hour_in_window(12, 0, 0) is True
    assert _hour_in_window(23, 22, 6) is True
    assert _hour_in_window(12, 22, 6) is False


def test_resolve_trading_session_defaults():
    sess = resolve_trading_session({})
    assert sess["enabled"] is True
    assert sess["start_hour_utc"] == 7


def test_trading_session_open_24h_when_start_equals_end():
    cfg = {
        "trading": {
            "session": {
                "enabled": True,
                "start_hour_utc": 0,
                "end_hour_utc": 0,
                "warmup_seconds_after_stream_sync": 0,
                "minutes_before_close_no_entry": 0,
            }
        }
    }
    epoch = (1_700_000_000 // 86400) * 86400 + 12 * 3600
    ok, _ = trading_session_allows_entry(
        epoch_utc=epoch,
        stream_ready_at=None,
        now_mono=time.time(),
        config=cfg,
    )
    assert ok is True


def test_trading_session_disabled():
    cfg = {"trading": {"session": {"enabled": False}}}
    ok, note = trading_session_allows_entry(
        epoch_utc=0,
        stream_ready_at=time.time(),
        now_mono=time.time(),
        config=cfg,
    )
    assert ok is True
    assert note == ""


def test_trading_session_wraparound_hours():
    cfg = {
        "trading": {
            "session": {
                "enabled": True,
                "start_hour_utc": 22,
                "end_hour_utc": 6,
                "warmup_seconds_after_stream_sync": 0,
                "minutes_before_close_no_entry": 0,
            }
        }
    }
    epoch_night = (1_700_000_000 // 86400) * 86400 + 23 * 3600
    ok, _ = trading_session_allows_entry(
        epoch_utc=epoch_night,
        stream_ready_at=None,
        now_mono=time.time(),
        config=cfg,
    )
    assert ok is True
    epoch_midday = (1_700_000_000 // 86400) * 86400 + 12 * 3600
    ok2, note2 = trading_session_allows_entry(
        epoch_utc=epoch_midday,
        stream_ready_at=None,
        now_mono=time.time(),
        config=cfg,
    )
    assert ok2 is False
    assert "SESSION_CLOSED" in note2
