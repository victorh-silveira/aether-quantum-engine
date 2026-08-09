"""SSOT fail-closed para knobs de logging."""

from __future__ import annotations

import logging
from typing import Any

from src.presentation.terminal.settle_log import KNOWN_SETTLE_CHANNELS


_DEFAULT_LOG_FILE = "logs/engine.log"
_DEFAULT_LEVEL = "INFO"
_KNOWN_QUIET = (
    frozenset(
        {
            "settle_enqueue",
            "settle_process",
            "settle_read",
            "settle_tolerance",
            "ws_ping",
            "warmup_poll",
            "execution_flow",
        }
    )
    | KNOWN_SETTLE_CHANNELS
)


def _parse_level(raw: Any) -> int:
    """Converte nome ou int de nivel logging; invalido cai em INFO."""
    if isinstance(raw, int):
        return int(raw)
    name = str(raw or _DEFAULT_LEVEL).upper().strip()
    level = logging.getLevelName(name)
    if isinstance(level, int):
        return level
    return logging.INFO


def resolve_logging_config(config: dict[str, Any] | None = None) -> dict[str, Any]:
    """Resolve logging.level, log_file e quiet_channels a partir do settings raiz."""
    root = config if isinstance(config, dict) else {}
    raw_logging = root.get("logging")
    block = raw_logging if isinstance(raw_logging, dict) else {}
    log_file = block.get("log_file", _DEFAULT_LOG_FILE)
    quiet_raw = block.get("quiet_channels", [])
    quiet: list[str] = []
    if isinstance(quiet_raw, list):
        for item in quiet_raw:
            name = str(item).strip()
            if name in _KNOWN_QUIET:
                quiet.append(name)
    return {
        "level": _parse_level(block.get("level", _DEFAULT_LEVEL)),
        "level_name": logging.getLevelName(_parse_level(block.get("level", _DEFAULT_LEVEL))),
        "log_file": str(log_file) if log_file else None,
        "quiet_channels": tuple(quiet),
        "known_quiet_channels": tuple(sorted(_KNOWN_QUIET)),
    }
