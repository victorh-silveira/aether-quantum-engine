"""Cobertura de settle_log_dedupe e quiet channels."""

from __future__ import annotations

import logging
from types import SimpleNamespace

from src.application.services.settle_log_dedupe import log_settle_info_if_changed, log_settle_warning_if_changed
from src.presentation.terminal.settle_log import SETTLE_ENQUEUE, SETTLE_ENQUEUE_ERR


def test_log_settle_info_if_changed_quiet_skips_info(caplog):
    owner = SimpleNamespace()
    logger = logging.getLogger("AETH_settle_quiet")
    logger.handlers.clear()
    logger.quiet_channels = (SETTLE_ENQUEUE,)
    with caplog.at_level(logging.DEBUG, logger=logger.name):
        log_settle_info_if_changed(
            owner,
            logger,
            SETTLE_ENQUEUE,
            "k1",
            "queued",
            "Contrato %s enfileirado",
            9,
        )
    assert any("SETTLE.settle_enqueue:" in r.message for r in caplog.records)
    assert all(r.levelno == logging.DEBUG for r in caplog.records if "SETTLE.settle_enqueue:" in r.message)


def test_log_settle_warning_if_changed_quiet_skips_warning(caplog):
    owner = SimpleNamespace()
    logger = logging.getLogger("AETH_settle_warn_quiet")
    logger.handlers.clear()
    logger.quiet_channels = (SETTLE_ENQUEUE_ERR,)
    with caplog.at_level(logging.DEBUG, logger=logger.name):
        log_settle_warning_if_changed(
            owner,
            logger,
            SETTLE_ENQUEUE_ERR,
            "err",
            "boom",
            "Falha: %s",
            "boom",
        )
    assert any("SETTLE.settle_enqueue_err:" in r.message for r in caplog.records)
    assert all(r.levelno == logging.DEBUG for r in caplog.records if "SETTLE.settle_enqueue_err:" in r.message)


def test_log_settle_warning_if_changed_emits_warning(caplog):
    owner = SimpleNamespace()
    logger = logging.getLogger("AETH_settle_warn")
    logger.handlers.clear()
    logger.quiet_channels = ()
    with caplog.at_level(logging.WARNING, logger=logger.name):
        log_settle_warning_if_changed(
            owner,
            logger,
            SETTLE_ENQUEUE_ERR,
            "err",
            "boom",
            "Falha: %s",
            "boom",
        )
    assert any("SETTLE.settle_enqueue_err:" in r.message for r in caplog.records)
