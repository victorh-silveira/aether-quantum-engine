"""Deduplicacao de linhas de log repetidas entre ciclos."""

import logging


def log_info_if_changed(owner, logger: logging.Logger, channel: str, content: str, message: str, *args) -> None:
    """Loga INFO quando o conteudo do canal muda; repeticoes identicas vao para DEBUG."""
    cache = getattr(owner, "_log_dedupe", None)
    if cache is None:
        cache = {}
        owner._log_dedupe = cache
    if cache.get(channel) == content:
        logger.debug(message, *args)
        return
    cache[channel] = content
    logger.info(message, *args)


def clear_log_channel(owner, channel: str) -> str | None:
    """Remove e retorna o ultimo conteudo registrado do canal."""
    cache = getattr(owner, "_log_dedupe", None)
    if not cache:
        return None
    return cache.pop(channel, None)
