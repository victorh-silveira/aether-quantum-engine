"""Chamadas Gemini e cache no backtest."""

from __future__ import annotations

import logging
import time
from contextlib import asynccontextmanager
from pathlib import Path
from typing import Any
from unittest.mock import patch

from scripts.backtest.gemini_cache import append_cache_entry
from scripts.backtest.historical_stream import HistoricalStream
from scripts.backtest.snapshot_engine import build_snapshot_at_bar
from src.application.services.llm import symbol_decision_utils as symbol_decision_utils_mod
from src.application.services.llm.llm_symbol_io import request_llm_payload
from src.application.services.llm.symbol_decision_utils import build_symbol_prompt


_logger = logging.getLogger("AETH")


class BacktestOrchestrator:
    """Orquestrador minimo para reutilizar build_symbol_prompt no backtest."""

    def __init__(
        self,
        config: dict[str, Any],
        *,
        stream: HistoricalStream,
        anchor: str,
        symbols: list[str],
    ):
        self.config = config
        self.stream = stream
        self.anchor = anchor
        self.symbols = symbols
        self.logger = _logger
        self.risk_manager = None
        self._active_cycle_id = 1
        self._bt_snapshot = None


async def _macro_snapshot_bt(orch: BacktestOrchestrator, runtime: dict[str, Any]):
    snap = getattr(orch, "_bt_snapshot", None)
    if snap is not None:
        return snap
    return await symbol_decision_utils_mod.fetch_macro_snapshot(orch, runtime)


@asynccontextmanager
async def _patch_macro_snapshot():
    with patch.object(symbol_decision_utils_mod, "fetch_macro_snapshot", new=_macro_snapshot_bt):
        yield


async def gemini_payload_for_bar(
    orch: BacktestOrchestrator,
    *,
    anchor: str,
    runtime: dict[str, Any],
    cache: dict[str, dict[str, Any]],
    bar_index: int,
    cache_file: Path | None,
) -> dict[str, Any]:
    """Consulta Gemini para um ponto agendado e persiste no cache."""
    key = str(bar_index)
    if key in cache:
        return cache[key]
    if not str(runtime.get("gemini_api_key") or "").strip():
        raise RuntimeError("GEMINI_API_KEY ausente no .env para backtest --mode gemini")
    async with _patch_macro_snapshot():
        prompt = (await build_symbol_prompt(orch, anchor, runtime))[0]
        payload = await request_llm_payload(
            orch,
            anchor,
            runtime,
            prompt,
            system=str(runtime.get("llm_system") or ""),
            cycle_id=bar_index,
        )
    cache[key] = payload
    if cache_file is not None:
        append_cache_entry(cache_file, bar_index, payload)
    return payload


async def warm_gemini_cache(
    *,
    orch: BacktestOrchestrator,
    stream: HistoricalStream,
    anchor: str,
    runtime: dict[str, Any],
    m15: dict[str, list[float]],
    m5: dict[str, list[float]],
    us_syms: list[str],
    eu_syms: list[str],
    macro_cfg: dict[str, Any] | None,
    targets: list[int],
    cache: dict[str, dict[str, Any]],
    cache_file: Path | None,
) -> tuple[int, int]:
    """Fase 1: consulta API apenas nos pontos agendados."""
    total = len(targets)
    calls = 0
    failures = 0
    t0 = time.perf_counter()
    for idx, bar_index in enumerate(targets, start=1):
        key = str(bar_index)
        if key in cache:
            print(f"  [{idx}/{total}] ponto {bar_index} cache", flush=True)
            continue
        snap = build_snapshot_at_bar(
            bar_index=bar_index,
            m15_closes=m15,
            m5_closes=m5,
            us_symbols=us_syms,
            eu_symbols=eu_syms,
            macro_cfg=macro_cfg,
        )
        orch._bt_snapshot = snap
        stream.set_bar_index(bar_index)
        try:
            await gemini_payload_for_bar(
                orch,
                anchor=anchor,
                runtime=runtime,
                cache=cache,
                bar_index=bar_index,
                cache_file=cache_file,
            )
            calls += 1
            elapsed = time.perf_counter() - t0
            per = elapsed / max(calls, 1)
            remaining = (total - idx) * per
            print(
                f"  [{idx}/{total}] ponto {bar_index} ok | {elapsed:.0f}s | ~{remaining / 60:.0f} min restantes",
                flush=True,
            )
        except Exception as exc:
            failures += 1
            print(f"  [{idx}/{total}] ponto {bar_index} ERRO: {exc}", flush=True)
    return calls, failures
