import io
import uuid
from unittest.mock import patch

from src.presentation.terminal.logger import setup_logger


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
