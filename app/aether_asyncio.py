from __future__ import annotations

import asyncio
import logging
import os


def silence_asyncio_debug() -> None:
    os.environ.pop("PYTHONASYNCIODEBUG", None)
    os.environ.pop("PYTHONDEVMODE", None)
    logging.getLogger("asyncio").setLevel(logging.CRITICAL)


def run_async(coro, *, debug: bool = False):
    silence_asyncio_debug()
    return asyncio.run(coro, debug=debug)
