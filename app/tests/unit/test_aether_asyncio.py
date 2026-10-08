import logging
import os

from aether_asyncio import run_async, silence_asyncio_debug


def test_silence_asyncio_debug_clears_env_and_logger():
    os.environ["PYTHONASYNCIODEBUG"] = "0"
    os.environ["PYTHONDEVMODE"] = "1"
    silence_asyncio_debug()
    assert "PYTHONASYNCIODEBUG" not in os.environ
    assert "PYTHONDEVMODE" not in os.environ
    assert logging.getLogger("asyncio").level == logging.CRITICAL


async def _noop() -> str:
    return "ok"


def test_run_async_forces_debug_false():
    os.environ["PYTHONASYNCIODEBUG"] = "0"
    assert run_async(_noop()) == "ok"
    assert "PYTHONASYNCIODEBUG" not in os.environ
