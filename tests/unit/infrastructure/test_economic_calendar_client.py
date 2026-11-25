"""Testes unitarios (100% cobertura) do cliente HTTP/cache de calendario economico."""

from __future__ import annotations

import json
import time

import pytest

from src.infrastructure.market import economic_calendar_client as mod
from src.infrastructure.market.economic_calendar_client import (
    EconomicCalendarClient,
    parse_calendar_payload,
)


SAMPLE_FEED = [
    {
        "title": "Core PCE Price Index m/m",
        "country": "USD",
        "date": "2026-11-25T08:30:00-05:00",
        "impact": "High",
        "forecast": "0.3%",
        "previous": "0.2%",
    },
    {"title": "Sem data", "country": "USD", "impact": "High"},
    {"title": "Data invalida", "country": "USD", "date": "invalida", "impact": "High"},
    {"title": "Sem tz", "country": "USD", "date": "2026-11-25T08:30:00", "impact": "High"},
    "item_nao_dict",
]


def test_parse_calendar_payload_and_download_helper(monkeypatch):
    assert parse_calendar_payload("not_a_list") == []
    events = parse_calendar_payload(SAMPLE_FEED)
    assert len(events) == 1
    assert events[0].country == "USD"
    assert events[0].impact == "High"
    assert events[0].epoch > 0

    class _FakeResponse:
        def __enter__(self):
            return self

        def __exit__(self, *args):
            return False

        def read(self):
            return json.dumps(SAMPLE_FEED).encode("utf-8")

    monkeypatch.setattr(mod.urllib.request, "urlopen", lambda req, timeout=8.0: _FakeResponse())
    downloaded = mod._download_json_sync("https://example.test/cal.json", 5.0)
    assert isinstance(downloaded, list)
    with pytest.raises(ValueError, match="esquema"):
        mod._download_json_sync("file:///etc/passwd", 5.0)


@pytest.mark.asyncio
async def test_client_memory_cache_disk_cache_and_fail_open(tmp_path, monkeypatch):
    cache_file = tmp_path / "cal_cache.json"
    calls = 0

    def fake_download(url: str, timeout: float):
        nonlocal calls
        calls += 1
        return SAMPLE_FEED

    monkeypatch.setattr(mod, "_download_json_sync", fake_download)
    client = EconomicCalendarClient(cache_path=cache_file, refresh_interval_seconds=3600)

    # 1. Primeira chamada baixa da rede e salva no disco
    ev1 = await client.get_events()
    assert len(ev1) == 1
    assert calls == 1
    assert cache_file.is_file()

    # 2. Segunda chamada usa cache em memoria (sem nova requisicao)
    ev2 = await client.get_events()
    assert len(ev2) == 1
    assert calls == 1

    # 3. Novo cliente carrega direto do cache em disco dentro do TTL
    client2 = EconomicCalendarClient(cache_path=cache_file, refresh_interval_seconds=3600)
    ev3 = await client2.get_events()
    assert len(ev3) == 1
    assert calls == 1

    # 4. Cache corrompido no disco + erro de rede -> fail-open seguro
    bad_file = tmp_path / "bad_cache.json"
    bad_file.write_text("{invalido", encoding="utf-8")

    def fail_download(url: str, timeout: float):
        raise OSError("network down")

    monkeypatch.setattr(mod, "_download_json_sync", fail_download)
    client3 = EconomicCalendarClient(cache_path=bad_file, refresh_interval_seconds=3600)
    assert await client3.get_events() == []

    # 5. Cache valido no disco mas sem eventos + feed remoto vazio -> retorna []
    empty_file = tmp_path / "empty_cache.json"
    empty_file.write_text(json.dumps({"fetched_at": time.time(), "events": []}), encoding="utf-8")
    monkeypatch.setattr(mod, "_download_json_sync", lambda u, t: [])
    client4 = EconomicCalendarClient(cache_path=empty_file, refresh_interval_seconds=3600)
    assert await client4.get_events() == []

    # 6. Falha de I/O ao gravar cache em disco nao propaga excecao
    monkeypatch.setattr(mod, "_download_json_sync", lambda u, t: SAMPLE_FEED)
    client5 = EconomicCalendarClient(cache_path=tmp_path / "ro_dir", refresh_interval_seconds=3600)
    (tmp_path / "ro_dir").mkdir()
    assert len(await client5.get_events()) == 1
