"""Helpers de telemetria SETTLE com canais estaveis."""

from __future__ import annotations

import logging
from typing import Any


SETTLE_ENQUEUE = "settle_enqueue"
SETTLE_ENQUEUE_ERR = "settle_enqueue_err"
SETTLE_PROCESS = "settle_process"
SETTLE_READ = "settle_read"
SETTLE_CONFIRM = "settle_confirm"
SETTLE_ORPHAN = "settle_orphan"
SETTLE_SUBSCRIBE = "settle_subscribe"
SETTLE_BROKER = "settle_broker"
SETTLE_RECONCILE = "settle_reconcile"
SETTLE_ALIGN = "settle_align"
SETTLE_TOLERANCE = "settle_tolerance"
SETTLE_TRACK = "settle_track"

KNOWN_SETTLE_CHANNELS = frozenset(
    {
        SETTLE_ENQUEUE,
        SETTLE_ENQUEUE_ERR,
        SETTLE_PROCESS,
        SETTLE_READ,
        SETTLE_CONFIRM,
        SETTLE_ORPHAN,
        SETTLE_SUBSCRIBE,
        SETTLE_BROKER,
        SETTLE_RECONCILE,
        SETTLE_ALIGN,
        SETTLE_TOLERANCE,
        SETTLE_TRACK,
    }
)


def settle_line(channel: str, message: str) -> str:
    """Formata `SETTLE.{canal}: mensagem`."""
    return f"SETTLE.{str(channel).strip()}: {message}"


def extract_settle_channel(message: str) -> str | None:
    """Extrai canal de `SETTLE.{canal}:` ou None."""
    text = str(message or "")
    if not text.startswith("SETTLE."):
        return None
    rest = text[7:]
    if ":" not in rest:
        return None
    channel = rest.split(":", 1)[0].strip()
    return channel or None


def quiet_channels_of(logger: logging.Logger) -> set[str]:
    """Conjunto de canais quietos anexados ao logger."""
    return set(getattr(logger, "quiet_channels", ()) or ())


def log_settle(
    logger: logging.Logger,
    channel: str,
    message: str,
    *args: Any,
    level: int = logging.INFO,
    quiet_channels: tuple[str, ...] | list[str] | None = None,
) -> None:
    """Emite SETTLE com canal; quiet_channels (arg ou logger) forcam DEBUG."""
    quiet = set(quiet_channels) if quiet_channels is not None else quiet_channels_of(logger)
    effective = logging.DEBUG if channel in quiet else level
    logger.log(effective, settle_line(channel, message), *args)
