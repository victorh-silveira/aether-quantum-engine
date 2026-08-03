"""Watchdog de ingestao: detecta inanicao de ticks e reconecta o stream."""

from __future__ import annotations

import asyncio
import contextlib
import logging
from enum import StrEnum
from typing import Any


logger = logging.getLogger("AETH")


class WatchdogState(StrEnum):
    """Estado de saude da ingestao de mercado."""

    HEALTHY = "HEALTHY"
    STALE_DATA = "STALE_DATA"


def _watchdog_cfg(orch: Any) -> dict[str, Any]:
    """Extrai bloco orchestrator da configuracao do motor."""
    chunk = orch.config.get("orchestrator") if isinstance(orch.config, dict) else {}
    return chunk if isinstance(chunk, dict) else {}


def watchdog_enabled(orch: Any) -> bool:
    """Indica se o watchdog de ingestao esta habilitado."""
    return bool(_watchdog_cfg(orch).get("watchdog_enabled", True))


class AetherWatchdog:
    """Monitora TickBuffer e forca reconnect do stream em inanicao de fluxo."""

    def __init__(
        self,
        orch: Any,
        *,
        stale_seconds: float = 30.0,
        poll_interval: float = 5.0,
        reconnect_cooldown_seconds: float = 60.0,
    ) -> None:
        self._orch = orch
        self._stale_seconds = max(5.0, float(stale_seconds))
        self._poll_interval = max(1.0, float(poll_interval))
        self._reconnect_cooldown = max(0.0, float(reconnect_cooldown_seconds))
        self._state = WatchdogState.HEALTHY
        self._task: asyncio.Task[None] | None = None
        self._recovering = asyncio.Lock()
        self._last_reconnect_mono = 0.0

    @property
    def state(self) -> WatchdogState:
        """Estado atual do watchdog."""
        return self._state

    async def start(self) -> None:
        """Inicia loop de monitoramento em background."""
        if self._task is not None and not self._task.done():
            return
        self._task = asyncio.create_task(self._run_loop(), name="aether-watchdog")

    async def stop(self) -> None:
        """Cancela task de monitoramento."""
        task = self._task
        if task is None:
            return
        task.cancel()
        with contextlib.suppress(asyncio.CancelledError):
            await task
        self._task = None

    async def _run_loop(self) -> None:
        """Loop perpetuo de verificacao de atividade de ticks."""
        try:
            while getattr(self._orch, "running", False):
                await asyncio.sleep(self._poll_interval)
                await self._evaluate()
        except asyncio.CancelledError:
            raise

    async def _evaluate(self) -> None:
        """Avalia idade do ultimo tick e dispara recuperacao se necessario."""
        orch = self._orch
        if not getattr(orch, "running", False):
            return
        stream = getattr(orch, "stream", None)
        if stream is None or not stream.is_synchronized or not orch.ws.is_running:
            return
        buffer = stream.tick_buffer
        now = asyncio.get_running_loop().time()
        last_tick = buffer.last_tick_monotonic()
        if last_tick <= 0.0:
            ready_mono = float(getattr(orch, "_stream_ready_mono", 0.0))
            if ready_mono > 0.0 and (now - ready_mono) < self._stale_seconds:
                return
            if ready_mono <= 0.0:
                return
        elif (now - last_tick) <= self._stale_seconds:
            if self._state == WatchdogState.STALE_DATA:
                self._state = WatchdogState.HEALTHY
            return
        if self._recovering.locked():
            return
        if (
            self._reconnect_cooldown > 0.0
            and self._last_reconnect_mono > 0.0
            and (now - self._last_reconnect_mono) < self._reconnect_cooldown
        ):
            self._state = WatchdogState.STALE_DATA
            return
        self._state = WatchdogState.STALE_DATA
        await self._recover_stale_stream(now - max(last_tick, 0.0))

    async def _recover_stale_stream(self, stale_age: float) -> None:
        """Salva risco e reconecta WebSocket sem encerrar o motor."""
        async with self._recovering:
            orch = self._orch
            now = asyncio.get_running_loop().time()
            if (
                self._reconnect_cooldown > 0.0
                and self._last_reconnect_mono > 0.0
                and (now - self._last_reconnect_mono) < self._reconnect_cooldown
            ):
                return
            logger.warning(
                "WATCHDOG: STALE_DATA | ultimo tick ha %.1fs | reconectando stream",
                stale_age,
            )
            try:
                await orch._save_full_state()
            except Exception as exc:
                logger.warning("WATCHDOG: falha ao salvar snapshot antes do reconnect: %s", exc)
            ok = await stream_reconnect(orch)
            self._last_reconnect_mono = asyncio.get_running_loop().time()
            if ok:
                self._state = WatchdogState.HEALTHY
                logger.info("WATCHDOG: ingestao restaurada apos reconnect")


async def stream_reconnect(orch: Any) -> bool:
    """Delega reconnect controlado ao StreamHandler."""
    stream = getattr(orch, "stream", None)
    if stream is None:
        return False
    reconnect = getattr(stream, "reconnect_stream", None)
    if not callable(reconnect):
        return False
    return bool(await reconnect(orch))


def build_watchdog(orch: Any) -> AetherWatchdog | None:
    """Instancia watchdog conforme configuracao do orquestrador."""
    if not watchdog_enabled(orch):
        return None
    cfg = _watchdog_cfg(orch)
    return AetherWatchdog(
        orch,
        stale_seconds=float(cfg.get("watchdog_stale_tick_seconds", 30.0)),
        poll_interval=float(cfg.get("watchdog_poll_interval_seconds", 5.0)),
        reconnect_cooldown_seconds=float(cfg.get("watchdog_reconnect_cooldown_seconds", 60.0)),
    )


async def start_ingestion_watchdog(orch: Any) -> None:
    """Inicia watchdog de ingestao no orquestrador."""
    watchdog = build_watchdog(orch)
    if watchdog is None:
        orch._ingestion_watchdog = None
        return
    orch._ingestion_watchdog = watchdog
    await watchdog.start()


async def stop_ingestion_watchdog(orch: Any) -> None:
    """Encerra watchdog de ingestao se ativo."""
    watchdog = getattr(orch, "_ingestion_watchdog", None)
    if isinstance(watchdog, AetherWatchdog):
        await watchdog.stop()
    orch._ingestion_watchdog = None
