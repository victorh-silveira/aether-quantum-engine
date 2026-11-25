"""Adaptador HTTP/cache local para o calendario economico semanal (FairEconomy/ForexFactory)."""

from __future__ import annotations

import asyncio
import json
import logging
import time
import urllib.request
from datetime import datetime
from pathlib import Path
from typing import Any

from src.domain.fundamental.economic_calendar import EconomicEvent


logger = logging.getLogger("AETH")
DEFAULT_CALENDAR_URL = "https://nfs.faireconomy.media/ff_calendar_thisweek.json"
DEFAULT_CACHE_PATH = "data/economic_calendar_cache.json"
DEFAULT_REFRESH_SECONDS = 21600


def parse_calendar_payload(raw_items: Any) -> list[EconomicEvent]:
    """Converte payload JSON semanal em lista imutavel de EconomicEvent com epoch UTC."""
    if not isinstance(raw_items, list):
        return []
    events: list[EconomicEvent] = []
    for item in raw_items:
        if not isinstance(item, dict):
            continue
        date_raw = str(item.get("date") or "").strip()
        country = str(item.get("country") or "").strip().upper()
        impact = str(item.get("impact") or "").strip()
        title = str(item.get("title") or "").strip()
        if not date_raw or not country or not impact or not title:
            continue
        try:
            dt = datetime.fromisoformat(date_raw)
        except ValueError:
            continue
        if dt.tzinfo is None:
            continue
        events.append(
            EconomicEvent(
                title=title,
                country=country,
                epoch=int(dt.timestamp()),
                impact=impact,
                forecast=str(item.get("forecast") or ""),
                previous=str(item.get("previous") or ""),
            )
        )
    events.sort(key=lambda ev: ev.epoch)
    return events


def _download_json_sync(url: str, timeout: float) -> Any:
    """Executa download sincrono do feed JSON exclusivamente via HTTPS."""
    if not url.startswith("https://"):
        raise ValueError("esquema de URL nao permitido")
    request = urllib.request.Request(url, headers={"User-Agent": "AetherQuantumEngine/1.0"})  # noqa: S310
    with urllib.request.urlopen(request, timeout=timeout) as response:  # noqa: S310
        return json.loads(response.read().decode("utf-8"))


class EconomicCalendarClient:
    """Cliente resiliente com cache em memoria e disco, TTL de 6h e politica fail-open."""

    def __init__(
        self,
        source_url: str = DEFAULT_CALENDAR_URL,
        cache_path: str | Path = DEFAULT_CACHE_PATH,
        refresh_interval_seconds: int = DEFAULT_REFRESH_SECONDS,
        timeout_seconds: float = 8.0,
    ) -> None:
        self.source_url = source_url
        self.cache_path = Path(cache_path)
        self.refresh_interval_seconds = max(300, int(refresh_interval_seconds))
        self.timeout_seconds = max(1.0, float(timeout_seconds))
        self._events: list[EconomicEvent] = []
        self._last_refresh: float = 0.0

    def _load_disk_cache(self, now: float) -> bool:
        """Carrega eventos do cache em disco se ainda estiverem dentro do TTL."""
        if not self.cache_path.is_file():
            return False
        try:
            payload = json.loads(self.cache_path.read_text(encoding="utf-8"))
            fetched_at = float(payload.get("fetched_at") or 0.0)
            events = parse_calendar_payload(payload.get("events"))
            if events:
                self._events = events
                self._last_refresh = fetched_at
                return (now - fetched_at) < self.refresh_interval_seconds
        except (OSError, ValueError, TypeError):
            return False
        return False

    def _save_disk_cache(self, raw_items: Any, fetched_at: float) -> None:
        """Persiste payload bruto no disco sem propagar erro de I/O."""
        try:
            self.cache_path.parent.mkdir(parents=True, exist_ok=True)
            body = json.dumps({"fetched_at": fetched_at, "events": raw_items}, ensure_ascii=False)
            self.cache_path.write_text(body, encoding="utf-8")
        except OSError as exc:
            logger.warning("CALENDARIO_MACRO: falha ao gravar cache local: %s", exc)

    async def get_events(self) -> list[EconomicEvent]:
        """Retorna eventos em cache ou atualiza remotamente quando o TTL expira."""
        now = time.time()
        if self._events and (now - self._last_refresh) < self.refresh_interval_seconds:
            return list(self._events)
        if not self._events and self._load_disk_cache(now):
            return list(self._events)
        try:
            raw_items = await asyncio.to_thread(_download_json_sync, self.source_url, self.timeout_seconds)
            parsed = parse_calendar_payload(raw_items)
            if parsed:
                self._events = parsed
                self._last_refresh = now
                self._save_disk_cache(raw_items, now)
                return list(self._events)
            logger.warning("CALENDARIO_MACRO: feed vazio recebido; mantendo cache anterior (%d)", len(self._events))
        except Exception as exc:
            logger.warning("CALENDARIO_MACRO: falha ao consultar feed (%s); operando em fail-open", exc)
        return list(self._events)
