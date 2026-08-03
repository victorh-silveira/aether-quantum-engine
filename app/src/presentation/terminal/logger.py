"""Configuracao de log de terminal para o Aether Engine."""

from __future__ import annotations

import asyncio
import logging
import sys
import time
from pathlib import Path

from src.presentation.terminal.log_context import format_log_context_prefix
from src.presentation.terminal.settle_log import extract_settle_channel


class _FlushStreamHandler(logging.StreamHandler):
    """Handler de stream que faz flush apos cada registro (melhor para terminal)."""

    def emit(self, record):
        """Envia o registro ao stream e chama flush."""
        super().emit(record)
        self.flush()


class BlankLineSquasher(logging.Filter):
    """Filtro que descarta linhas em branco consecutivas ou no inicio da sessao."""

    def __init__(self) -> None:
        """Inicializa o filtro considerando o inicio da sessao como linha em branco."""
        super().__init__()
        self._last_blank = True

    def filter(self, record: logging.LogRecord) -> bool:
        """Permite linha em branco apenas apos uma linha com conteudo."""
        blank = not str(record.getMessage() or "").strip()
        if blank and self._last_blank:
            return False
        self._last_blank = blank
        return True


class CooldownDeduplicationFilter(logging.Filter):
    """Filtro de log que suprime mensagens de cooldown identicas no mesmo tick."""

    def __init__(self) -> None:
        """Inicializa o filtro com um dicionario de ultimas mensagens vistas."""
        super().__init__()
        self._last_seen: dict[str, int] = {}

    def filter(self, record: logging.LogRecord) -> bool:
        """Suprime mensagens de cooldown repetidas no mesmo segundo."""
        msg = record.getMessage()
        if "CICLO: cooling-down" in msg or "CICLO: resfriamento pos-LOSS" in msg:
            try:
                loop = asyncio.get_running_loop()
                current_time = loop.time()
            except RuntimeError:
                current_time = time.time()
            tick = int(current_time)
            key = "cooling-down" if "cooling-down" in msg else "resfriamento"
            if self._last_seen.get(key) == tick:
                return False
            self._last_seen[key] = tick
        return True


class SettlementSpamFilter(logging.Filter):
    """Suprime repeticoes SETTLE/WARMUP/EXECUTION_FLOW no mesmo segundo por canal."""

    def __init__(self) -> None:
        """Inicializa o filtro com cache de ultima chave por tick."""
        super().__init__()
        self._last_seen: dict[str, int] = {}

    def _channel(self, msg: str) -> str | None:
        """Resolve canal estavel a partir da mensagem SETTLE/WARMUP/FLOW."""
        settle = extract_settle_channel(msg)
        if settle:
            return f"settle:{settle}"
        markers = (
            ("enfileirado no Redis", "settle_enqueue"),
            ("Enfileirando contrato", "settle_broker"),
            ("Erro ao ler fila de prioridade do Redis", "settle_read"),
            ("Falha ao enfileirar no Redis", "settle_enqueue_err"),
            ("[AETHER] EXECUTION_FLOW |", "execution_flow"),
            ("[AETHER] WARMUP |", "warmup_poll"),
        )
        for needle, channel in markers:
            if needle in msg:
                return channel
        return None

    def filter(self, record: logging.LogRecord) -> bool:
        """Descarta spam do mesmo canal no mesmo segundo."""
        msg = str(record.getMessage() or "")
        channel = self._channel(msg)
        if channel is None:
            return True
        try:
            loop = asyncio.get_running_loop()
            tick = int(loop.time())
        except RuntimeError:
            tick = int(time.time())
        if self._last_seen.get(channel) == tick:
            return False
        self._last_seen[channel] = tick
        return True


class AetherFormatter(logging.Formatter):
    """Formatador personalizado para impor niveis de 4 letras e contexto de ciclo."""

    LEVEL_MAP = {"DEBUG": "DEBG ", "INFO": "INFO ", "WARNING": "AVISO", "ERROR": "ERRO ", "CRITICAL": "CRIT "}

    def format(self, record):
        """Formata o registro de log com nomes de niveis truncados e prefixo de contexto."""
        if not str(record.getMessage() or "").strip():
            return ""
        record.levelname = self.LEVEL_MAP.get(record.levelname, record.levelname[:4])
        prefix = format_log_context_prefix()
        if not prefix:
            return super().format(record)
        original_msg, original_args = record.msg, record.args
        try:
            record.msg = f"{prefix}{record.getMessage()}"
            record.args = ()
            return super().format(record)
        finally:
            record.msg = original_msg
            record.args = original_args


def get_logger(name: str = "AETH") -> logging.Logger:
    """Retorna logger nomeado (padrao AETH)."""
    return logging.getLogger(name)


def setup_logger(
    name: str,
    log_file: str | None = None,
    *,
    level: int = logging.INFO,
    quiet_channels: tuple[str, ...] | list[str] | None = None,
):
    """Configura logger Aether de forma idempotente (handlers nao duplicam)."""
    logger = logging.getLogger(name)
    logger.setLevel(level)
    logger.quiet_channels = tuple(quiet_channels or ())
    if getattr(logger, "_aether_configured", False):
        return logger

    logger.handlers.clear()
    logger.filters.clear()
    logger.addFilter(BlankLineSquasher())
    logger.addFilter(CooldownDeduplicationFilter())
    logger.addFilter(SettlementSpamFilter())
    logger.propagate = False

    formatter = AetherFormatter("%(asctime)s | %(levelname)s | %(message)s", datefmt="%H:%M:%S")
    stdout_handler = _FlushStreamHandler(sys.stdout)
    stdout_handler.setFormatter(formatter)
    logger.addHandler(stdout_handler)

    if log_file:
        log_path = Path(log_file)
        log_path.parent.mkdir(parents=True, exist_ok=True)
        file_handler = logging.FileHandler(str(log_path), encoding="utf-8")
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)

    logger._aether_configured = True
    return logger
