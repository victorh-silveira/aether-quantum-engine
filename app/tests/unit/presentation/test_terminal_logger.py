import io
import logging
import uuid
from unittest.mock import patch

import pytest

from src.presentation.terminal.log_context import bind_log_context, clear_log_context
from src.presentation.terminal.logger import (
    AetherFormatter,
    BlankLineSquasher,
    CooldownDeduplicationFilter,
    SettlementSpamFilter,
    _FlushStreamHandler,
    get_logger,
    setup_logger,
)
from src.presentation.terminal.logging_config import resolve_logging_config
from src.presentation.terminal.settle_log import extract_settle_channel, log_settle, settle_line


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


def test_settlement_spam_filter_dedupes_identical_settle_lines():
    filt = SettlementSpamFilter()
    rec1 = logging.LogRecord(
        "AETH",
        logging.WARNING,
        "",
        0,
        "SETTLE.settle_broker: Broker offline. Enfileirando contrato 1 no Redis.",
        (),
        None,
    )
    rec2 = logging.LogRecord(
        "AETH",
        logging.WARNING,
        "",
        0,
        "SETTLE.settle_broker: Broker offline. Enfileirando contrato 2 no Redis.",
        (),
        None,
    )
    rec3 = logging.LogRecord("AETH", logging.INFO, "", 0, "[CLUSTER] M10 || OTC_SPC: PUT", (), None)
    assert filt.filter(rec1) is True
    assert filt.filter(rec2) is False
    assert filt.filter(rec3) is True


@pytest.mark.asyncio
async def test_settlement_spam_filter_with_running_loop():
    filt = SettlementSpamFilter()
    rec1 = logging.LogRecord("AETH", logging.INFO, "", 0, "[AETHER] WARMUP | aguardando ticks vivos da Deriv", (), None)
    rec2 = logging.LogRecord("AETH", logging.INFO, "", 0, "[AETHER] WARMUP | aguardando ticks vivos da Deriv", (), None)
    assert filt.filter(rec1) is True
    assert filt.filter(rec2) is False


def test_setup_logger_idempotent_handlers():
    name = f"AETH_utest_{uuid.uuid4().hex}"
    with patch("src.presentation.terminal.logger.sys.stdout", io.StringIO()):
        first = setup_logger(name, log_file=None)
        count = len(first.handlers)
        second = setup_logger(name, log_file=None)
    assert first is second
    assert len(second.handlers) == count


def test_get_logger_default_name():
    assert get_logger().name == "AETH"


def test_resolve_logging_config_defaults_and_quiet():
    cfg = resolve_logging_config({})
    assert cfg["level"] == logging.INFO
    assert cfg["log_file"] == "logs/engine.log"
    assert cfg["quiet_channels"] == ()
    armed = resolve_logging_config(
        {"logging": {"level": "DEBUG", "quiet_channels": ["settle_enqueue", "unknown_drop"]}}
    )
    assert armed["level"] == logging.DEBUG
    assert armed["quiet_channels"] == ("settle_enqueue",)


def test_log_settle_quiet_channel_uses_debug():
    name = f"AETH_utest_{uuid.uuid4().hex}"
    log = logging.getLogger(name)
    log.handlers.clear()
    log.setLevel(logging.DEBUG)
    log.quiet_channels = ("settle_enqueue",)
    buf = io.StringIO()
    handler = logging.StreamHandler(buf)
    handler.setLevel(logging.DEBUG)
    log.addHandler(handler)
    log_settle(log, "settle_enqueue", "Contrato %s", 7)
    assert "SETTLE.settle_enqueue:" in buf.getvalue()
    assert extract_settle_channel(settle_line("settle_enqueue", "x")) == "settle_enqueue"


def test_aether_formatter_prefixes_context_without_breaking_cluster():
    clear_log_context()
    bind_log_context(cycle_id=3, symbol="OTC_SPC")
    formatter = AetherFormatter("%(levelname)s | %(message)s")
    record = logging.LogRecord("AETH", logging.INFO, "", 0, "[CLUSTER] M10 || OTC_SPC: PUT", (), None)
    out = formatter.format(record)
    clear_log_context()
    assert "[c3|OTC_SPC]" in out
    assert "[CLUSTER]" in out


def test_format_log_context_prefix_partial_fields():
    from src.presentation.terminal.log_context import format_log_context_prefix

    clear_log_context()
    bind_log_context(cycle_id=9)
    assert format_log_context_prefix() == "[c9] "
    clear_log_context()
    bind_log_context(symbol="OTC_SPC")
    assert format_log_context_prefix() == "[OTC_SPC] "
    clear_log_context()
    assert format_log_context_prefix() == ""


def test_resolve_logging_config_level_int_and_invalid():
    assert resolve_logging_config({"logging": {"level": 10}})["level"] == 10
    assert resolve_logging_config({"logging": {"level": "NOT_A_LEVEL"}})["level"] == logging.INFO
    assert resolve_logging_config({"logging": "bad"})["log_file"] == "logs/engine.log"
    assert resolve_logging_config(None)["level"] == logging.INFO


def test_extract_settle_channel_edge_cases():
    assert extract_settle_channel("nope") is None
    assert extract_settle_channel("SETTLE.only") is None
    assert extract_settle_channel("SETTLE.: msg") is None
    assert extract_settle_channel("SETTLE.settle_read: x") == "settle_read"


def test_live_monitor_cluster_regex_still_matches_prefixed_line():
    import re

    cluster_re = re.compile(r"\[CLUSTER\]\s+(?P<tf>\S+)\s+\|\|\s+(?P<body>.+)$", re.IGNORECASE)
    line = "12:00:00 | INFO | [c3|OTC_SPC] [CLUSTER] M10 || OTC_SPC: PUT (Prob: 0.55 Cal: 0.60)"
    match = cluster_re.search(line)
    assert match is not None
    assert match.group("tf") == "M10"
