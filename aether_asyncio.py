import asyncio
import contextlib
import logging
import os


def silence_asyncio_debug() -> None:
    os.environ.pop("PYTHONASYNCIODEBUG", None)
    os.environ.pop("PYTHONDEVMODE", None)
    logging.getLogger("asyncio").setLevel(logging.CRITICAL)
    if os.name == "nt":
        with contextlib.suppress(Exception):
            asyncio.set_event_loop_policy(asyncio.WindowsSelectorEventLoopPolicy())


def run_async(coro, *, debug: bool = False):
    silence_asyncio_debug()
    return asyncio.run(coro, debug=debug)
