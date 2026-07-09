"""Deduplicacao de linhas de log repetidas entre ciclos."""

import logging


def _log_if_changed(
    owner,
    logger: logging.Logger,
    level: str,
    channel: str,
    content: str,
    message: str,
    *args,
) -> None:
    """Emite log no nivel indicado quando o conteudo do canal muda."""
    cache = getattr(owner, "_log_dedupe", None)
    if cache is None:
        cache = {}
        owner._log_dedupe = cache
    if cache.get(channel) == content:
        logger.debug(message, *args)
        return
    cache[channel] = content
    getattr(logger, level)(message, *args)


def log_info_if_changed(owner, logger: logging.Logger, channel: str, content: str, message: str, *args) -> None:
    """Loga INFO quando o conteudo do canal muda; repeticoes identicas vao para DEBUG."""
    _log_if_changed(owner, logger, "info", channel, content, message, *args)


def log_warning_if_changed(owner, logger: logging.Logger, channel: str, content: str, message: str, *args) -> None:
    """Loga WARNING quando o conteudo do canal muda; repeticoes identicas vao para DEBUG."""
    _log_if_changed(owner, logger, "warning", channel, content, message, *args)


def clear_log_channel(owner, channel: str) -> str | None:
    """Remove e retorna o ultimo conteudo registrado do canal."""
    cache = getattr(owner, "_log_dedupe", None)
    if not cache:
        return None
    return cache.pop(channel, None)


class LogDeduper:
    """Deduplicacao de logs de quality guard e inanição por canal temporal."""

    def __init__(self, owner) -> None:
        self._owner = owner

    def log_quality_guard_cycle_minute(
        self,
        logger: logging.Logger,
        *,
        cycle_id: int,
        minute_bucket: str,
        message: str,
    ) -> None:
        """Emite log de suspensao do quality guard uma vez por bloco de minuto."""
        channel = f"quality_guard:{int(cycle_id)}:{minute_bucket}"
        log_info_if_changed(self._owner, logger, channel, "seen", "%s", message)

    def log_quality_starvation_escape(
        self,
        logger: logging.Logger,
        *,
        skipped_cycles: int,
        min_margin: float,
    ) -> None:
        """Emite log deduplicado quando a valvula de inanicao reduz o piso de margem."""
        message = (
            "[AETHER] EXECUTION_FLOW | Válvula de inanição ativa. "
            f"Limite mitigado por decaimento temporal para min {float(min_margin):.4f} | "
            f"skipped_cycles={int(skipped_cycles)}"
        )
        channel = f"starvation:{int(skipped_cycles)}:{float(min_margin):.4f}"
        log_info_if_changed(self._owner, logger, channel, message, "%s", message)
