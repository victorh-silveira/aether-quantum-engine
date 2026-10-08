from __future__ import annotations

import asyncio
from collections.abc import Callable
from typing import ParamSpec, TypeVar

P = ParamSpec("P")
T = TypeVar("T")


async def run_in_thread(fn: Callable[P, T], *args: P.args, **kwargs: P.kwargs) -> T:
    return await asyncio.to_thread(fn, *args, **kwargs)
