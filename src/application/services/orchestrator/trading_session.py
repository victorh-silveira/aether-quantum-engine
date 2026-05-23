"""Janela operacional UTC e warm-up apos sincronia de velas."""

from __future__ import annotations

from typing import Any


def resolve_trading_session(config: dict[str, Any]) -> dict[str, Any]:
    """Normaliza trading.session com defaults conservadores."""
    trading = config.get("trading", {}) if isinstance(config.get("trading"), dict) else {}
    raw = trading.get("session", {}) if isinstance(trading.get("session"), dict) else {}
    return {
        "enabled": bool(raw.get("enabled", True)),
        "start_hour_utc": max(0, min(23, int(raw.get("start_hour_utc", 7)))),
        "end_hour_utc": max(0, min(24, int(raw.get("end_hour_utc", 22)))),
        "minutes_before_close_no_entry": max(0, int(raw.get("minutes_before_close_no_entry", 20))),
        "warmup_seconds_after_stream_sync": max(0, int(raw.get("warmup_seconds_after_stream_sync", 90))),
    }


def _hour_in_window(hour: int, start: int, end: int) -> bool:
    """True se a hora UTC estiver dentro da janela (suporta virada de meia-noite)."""
    if start == end:
        return True
    if start < end:
        return start <= hour < end
    return hour >= start or hour < end


def trading_session_allows_entry(
    *,
    epoch_utc: int,
    stream_ready_at: float | None,
    now_mono: float,
    config: dict[str, Any],
) -> tuple[bool, str]:
    """False fora da janela UTC, em warm-up ou nos minutos finais antes do fechamento."""
    sess = resolve_trading_session(config)
    if not sess["enabled"]:
        return True, ""

    warm = int(sess["warmup_seconds_after_stream_sync"])
    if warm > 0 and stream_ready_at is not None and (now_mono - stream_ready_at) < float(warm):
        return False, f"SESSION_WARMUP {int(warm - (now_mono - stream_ready_at))}s"

    hour = (int(epoch_utc) % 86400) // 3600
    minute = (int(epoch_utc) % 3600) // 60
    start_h = int(sess["start_hour_utc"])
    end_h = int(sess["end_hour_utc"])
    if not _hour_in_window(hour, start_h, end_h):
        return False, f"SESSION_CLOSED utc_hour={hour}"

    buffer_min = int(sess["minutes_before_close_no_entry"])
    if buffer_min > 0 and start_h < end_h:
        current_minute = hour * 60 + minute
        close_minute = end_h * 60
        remaining = close_minute - current_minute
        if 0 < remaining <= buffer_min:
            return False, f"SESSION_NEAR_CLOSE {buffer_min}m"

    return True, ""
