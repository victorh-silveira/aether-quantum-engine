"""Bootstrap de threads Polars consciente do event loop asyncio."""

from __future__ import annotations

import os


DEFAULT_POLARS_MAX_THREADS = "2"


def ensure_polars_max_threads(default: str = DEFAULT_POLARS_MAX_THREADS) -> str:
    """Define POLARS_MAX_THREADS se ausente; retorna o valor efetivo."""
    os.environ.setdefault("POLARS_MAX_THREADS", str(default))
    return str(os.environ["POLARS_MAX_THREADS"])
