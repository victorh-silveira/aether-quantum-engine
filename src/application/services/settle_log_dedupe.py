"""Dedupe SETTLE respeitando quiet_channels do logger."""

from __future__ import annotations

import logging
from typing import Any

from src.application.services.log_dedupe import log_info_if_changed, log_warning_if_changed
from src.presentation.terminal.settle_log import quiet_channels_of, settle_line


def log_settle_info_if_changed(
    owner: Any,
    logger: logging.Logger,
    settle_channel: str,
    dedupe_key: str,
    content: str,
    message: str,
    *args: Any,
) -> None:
    """INFO dedupe para SETTLE; canais quietos emitem so DEBUG."""
    line = settle_line(settle_channel, message)
    if settle_channel in quiet_channels_of(logger):
        logger.debug(line, *args)
        return
    log_info_if_changed(owner, logger, dedupe_key, content, line, *args)


def log_settle_warning_if_changed(
    owner: Any,
    logger: logging.Logger,
    settle_channel: str,
    dedupe_key: str,
    content: str,
    message: str,
    *args: Any,
) -> None:
    """WARNING dedupe para SETTLE; canais quietos emitem so DEBUG."""
    line = settle_line(settle_channel, message)
    if settle_channel in quiet_channels_of(logger):
        logger.debug(line, *args)
        return
    log_warning_if_changed(owner, logger, dedupe_key, content, line, *args)
