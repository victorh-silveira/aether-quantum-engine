import io
import logging
import uuid
from unittest.mock import patch

import pytest

from src.presentation.terminal.logger import (
    AetherFormatter,
    BlankLineSquasher,
    CooldownDeduplicationFilter,
    _FlushStreamHandler,
    setup_logger,
)


def test_aether_formatter_maps_levels_and_blank_messages():
    formatter = AetherFormatter("%(levelname)s | %(message)s")
    record = logging.LogRecord("AETH", logging.CRITICAL, "", 0, "critico", (), None)
    assert "CRIT" in formatter.format(record)
    record.levelname = "UNKNOWN"
    assert formatter.format(record).startswith("UNKN")
    blank = logging.LogRecord("AETH", logging.INFO, "", 0, "   ", (), None)
    assert formatter.format(blank) == ""


def test_blank_line_squasher_allows_blank_after_content():
    squasher = BlankLineSquasher()
    first = logging.LogRecord("AETH", logging.INFO, "", 0, "ok", (), None)
    blank = logging.LogRecord("AETH", logging.INFO, "", 0, "", (), None)
    assert squasher.filter(first) is True
    assert squasher.filter(blank) is True
    assert squasher.filter(blank) is False


def test_flush_stream_handler_flushes_after_emit():
    handler = _FlushStreamHandler(io.StringIO())
    record = logging.LogRecord("AETH", logging.INFO, "", 0, "flush", (), None)
    with patch.object(handler.stream, "flush") as flush_mock:
        handler.emit(record)
    assert flush_mock.call_count >= 1


def test_setup_logger_writes_to_stdout_with_flush_handler():
    buf = io.StringIO()
    name = f"AETH_utest_{uuid.uuid4().hex}"
    with patch("src.presentation.terminal.logger.sys.stdout", buf):
        log = setup_logger(name, log_file=None)
        log.info("linha_probe")
    out = buf.getvalue()
    assert "linha_probe" in out


def test_setup_logger_attaches_file_handler_when_log_file_given(tmp_path):
    buf = io.StringIO()
    name = f"AETH_utest_{uuid.uuid4().hex}"
    log_path = tmp_path / "engine.log"
    with patch("src.presentation.terminal.logger.sys.stdout", buf):
        log = setup_logger(name, log_file=str(log_path))
        log.info("persistir")
    assert "persistir" in buf.getvalue()
    texto = log_path.read_text(encoding="utf-8")
    assert "persistir" in texto


def test_setup_logger_keeps_real_blank_line_for_empty_message():
    buf = io.StringIO()
    name = f"AETH_utest_{uuid.uuid4().hex}"
    with patch("src.presentation.terminal.logger.sys.stdout", buf):
        log = setup_logger(name, log_file=None)
        log.info("antes")
        log.info("")
        log.info("depois")
    out = buf.getvalue()
    assert "antes" in out and "depois" in out
    assert "\n\n" in out


def test_setup_logger_squashes_consecutive_and_leading_blank_lines():
    buf = io.StringIO()
    name = f"AETH_utest_{uuid.uuid4().hex}"
    with patch("src.presentation.terminal.logger.sys.stdout", buf):
        log = setup_logger(name, log_file=None)
        log.info("")
        log.info("primeira")
        log.info("")
        log.info("")
        log.info("")
        log.info("segunda")
    out = buf.getvalue()
    lines = out.splitlines()
    assert lines[0].endswith("primeira")
    assert lines[1] == ""
    assert lines[2].endswith("segunda")
    assert len(lines) == 3


def test_cooldown_deduplication_filter_under_loop():
    filt = CooldownDeduplicationFilter()
    rec1 = logging.LogRecord("AETH", logging.INFO, "", 0, "CICLO: cooling-down 15.0s pos-LOSS linear=2", (), None)
    rec2 = logging.LogRecord("AETH", logging.INFO, "", 0, "CICLO: cooling-down 15.0s pos-LOSS linear=2", (), None)
    rec3 = logging.LogRecord(
        "AETH", logging.INFO, "", 0, "CICLO: resfriamento pos-LOSS ativo (14.9s restantes); ciclo suspenso", (), None
    )
    rec4 = logging.LogRecord(
        "AETH", logging.INFO, "", 0, "CICLO: resfriamento pos-LOSS ativo (14.9s restantes); ciclo suspenso", (), None
    )

    assert filt.filter(rec1) is True
    assert filt.filter(rec2) is False
    assert filt.filter(rec3) is True
    assert filt.filter(rec4) is False


@pytest.mark.asyncio
async def test_cooldown_deduplication_filter_with_running_loop():
    filt = CooldownDeduplicationFilter()
    rec1 = logging.LogRecord("AETH", logging.INFO, "", 0, "CICLO: cooling-down 15.0s pos-LOSS linear=2", (), None)
    rec2 = logging.LogRecord("AETH", logging.INFO, "", 0, "CICLO: cooling-down 15.0s pos-LOSS linear=2", (), None)

    assert filt.filter(rec1) is True
    assert filt.filter(rec2) is False
