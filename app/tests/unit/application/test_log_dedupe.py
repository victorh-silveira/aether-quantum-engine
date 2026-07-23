from unittest.mock import MagicMock, patch

import pytest

from src.application.services.deep_learning.dl_cycle_log import log_dl_cycle_summary
from src.application.services.log_dedupe import (
    LogDeduper,
    clear_log_channel,
    log_info_if_changed,
    log_warning_if_changed,
)
from src.domain.models.trade import TradeDirection


class Owner:
    pass


def test_log_info_if_changed_dedupes_repeats():
    owner = Owner()
    logger = MagicMock()
    log_info_if_changed(owner, logger, "ch", "a", "%s", "a")
    log_info_if_changed(owner, logger, "ch", "a", "%s", "a")
    log_info_if_changed(owner, logger, "ch", "b", "%s", "b")
    assert logger.info.call_count == 2
    assert logger.debug.call_count == 1


def test_log_warning_if_changed_dedupes_repeats():
    owner = Owner()
    logger = MagicMock()
    log_warning_if_changed(owner, logger, "ch", "pend=1", "EXEC: pend=%s", "1")
    log_warning_if_changed(owner, logger, "ch", "pend=1", "EXEC: pend=%s", "1")
    log_warning_if_changed(owner, logger, "ch", "pend=2", "EXEC: pend=%s", "2")
    assert logger.warning.call_count == 2
    assert logger.debug.call_count == 1


def test_clear_log_channel_returns_last_content():
    owner = Owner()
    assert clear_log_channel(owner, "ch") is None
    log_info_if_changed(owner, MagicMock(), "ch", "a", "%s", "a")
    assert clear_log_channel(owner, "ch") == "a"
    assert clear_log_channel(owner, "ch") is None


def test_log_dl_cycle_summary_dedupes_with_orch():
    logger = MagicMock()
    orch = Owner()
    decisions = {
        "R_10": {
            "direction": TradeDirection.CALL,
            "metrics": {"conviction": 0.70, "execute": True, "val_accuracy": 0.55, "cal_margin": 0.05},
        },
    }
    log_dl_cycle_summary(logger, decisions, recovery_active=False, pending_loss_total=0.0, orch=orch)
    log_dl_cycle_summary(logger, decisions, recovery_active=False, pending_loss_total=0.0, orch=orch)
    assert logger.info.call_count == 1


def test_log_cooldown_cooling_down_dedupes():
    owner = Owner()
    logger = MagicMock()
    deduper = LogDeduper(owner)

    with patch("src.application.services.log_dedupe.time.time", return_value=1000.0):
        deduper.log_cooldown_cooling_down(logger, "msg", 15.0, 2)
        deduper.log_cooldown_cooling_down(logger, "msg", 15.0, 2)

    assert logger.info.call_count == 1
    assert logger.debug.call_count == 1


@pytest.mark.asyncio
async def test_log_cooldown_cooling_down_inside_loop():
    owner = Owner()
    logger = MagicMock()
    deduper = LogDeduper(owner)

    with patch("src.application.services.log_dedupe.asyncio.get_running_loop") as get_loop:
        loop = MagicMock()
        loop.time.return_value = 1000.0
        get_loop.return_value = loop
        deduper.log_cooldown_cooling_down(logger, "msg", 15.0, 2)
        deduper.log_cooldown_cooling_down(logger, "msg", 15.0, 2)

    assert logger.info.call_count == 1
    assert logger.debug.call_count == 1


def test_log_cooldown_skip_dedupes():
    owner = Owner()
    logger = MagicMock()
    deduper = LogDeduper(owner)

    with patch("src.application.services.log_dedupe.time.time", return_value=1000.0):
        deduper.log_cooldown_skip(logger, "skip")
        deduper.log_cooldown_skip(logger, "skip")

    assert logger.info.call_count == 1
    assert logger.debug.call_count == 1


@pytest.mark.asyncio
async def test_log_cooldown_skip_inside_loop():
    owner = Owner()
    logger = MagicMock()
    deduper = LogDeduper(owner)

    with patch("src.application.services.log_dedupe.asyncio.get_running_loop") as get_loop:
        loop = MagicMock()
        loop.time.return_value = 1000.0
        get_loop.return_value = loop
        deduper.log_cooldown_skip(logger, "skip")
        deduper.log_cooldown_skip(logger, "skip")

    assert logger.info.call_count == 1
    assert logger.debug.call_count == 1
