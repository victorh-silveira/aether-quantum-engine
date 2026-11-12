"""Testes de telemetria estendida do servico indicador publico."""

from __future__ import annotations

from indicator.domain import Signal
from indicator.institutional import market_session_status_brt
from indicator.metrics import Metrics


def test_indicator_metrics_render_runtime_telemetry():
    metrics = Metrics()
    metrics.set_symbols({"OTC_SSMI": "Swiss 20", "R_25": "V25"})
    metrics.update(Signal("OTC_SSMI", 900, 1700000700, "SEM SINAL", None, "pullback-1-test", "afastado_da_media", 1.0))
    metrics.record_deriv_rtt("ticks_history", 42.5, ok=True)
    metrics.record_deriv_rtt("active_symbols", 150.0, ok=False)
    metrics.record_tick_duration(0.312)

    rendered = metrics.render()
    assert 'aether_indicator_real_market_open{symbol="R_25"} 1' in rendered
    assert 'aether_indicator_deriv_otc_open{symbol="OTC_SSMI"}' in rendered
    assert 'aether_indicator_reason_total{symbol="OTC_SSMI",period="900",reason="afastado_da_media"} 1' in rendered
    assert 'aether_indicator_deriv_ws_rtt_ms{method="ticks_history"} 42.500' in rendered
    assert 'aether_indicator_deriv_ws_errors_total{method="active_symbols"} 1' in rendered
    assert "aether_indicator_tick_duration_seconds 0.312000" in rendered


def test_market_session_status_brt_weekend_and_custom():
    from datetime import UTC, datetime, timedelta, timezone

    brt = timezone(timedelta(hours=-3))
    sat = datetime(2026, 11, 14, 10, 0, tzinfo=brt).astimezone(UTC).timestamp()
    sun_late = datetime(2026, 11, 15, 21, 0, tzinfo=brt).astimezone(UTC).timestamp()
    wed_afternoon = datetime(2026, 11, 11, 15, 0, tzinfo=brt).astimezone(UTC).timestamp()

    assert market_session_status_brt("R_25", sat) == (True, True)
    assert market_session_status_brt("OTC_SSMI", sat) == (False, False)
    assert market_session_status_brt("OTC_N225", sun_late) == (True, True)
    assert market_session_status_brt("OTC_SSMI", sun_late) == (False, False)
    assert market_session_status_brt("OTC_SSMI", wed_afternoon) == (False, True)
    assert market_session_status_brt("OTC_CUSTOM", wed_afternoon) == (True, True)


async def test_pullback_service_tick_records_duration():
    from indicator.pullback_service import PERIODS, PullbackService

    class FakeSink:
        async def promote_setups(self):
            return None

    now = 1_700_001_000
    metrics = Metrics()
    service = PullbackService(object(), FakeSink(), metrics)
    service.symbols = {"OTC_SSMI": "Swiss 20"}
    service.last_catalog = now
    service.last_epoch = {("OTC_SSMI", p): now // p * p - p for p in PERIODS}

    await service.tick(now)
    assert metrics.last_tick_duration_seconds >= 0.0
    assert "aether_indicator_tick_duration_seconds" in metrics.render()


async def test_run_indicator_attaches_rtt_callback(tmp_path, monkeypatch):
    import asyncio
    from types import SimpleNamespace

    import indicator.service as ind_service

    attached: list[object] = []

    class FakeSource:
        def __init__(self, url: str) -> None:
            _ = url
            self.clients = [SimpleNamespace(_on_rtt=None)]
            attached.append(self.clients[0])

        async def close(self) -> None:
            return None

    class FakeStore:
        def __init__(self, dsn: str) -> None:
            _ = dsn

        async def start(self) -> None:
            return None

        async def latest_setups(self) -> list[object]:
            return []

        async def close(self) -> None:
            return None

    class FakeMetrics(Metrics):
        async def start(self, port: int = 9100) -> None:
            _ = port

        async def close(self) -> None:
            return None

    async def fake_run(self) -> None:
        raise asyncio.CancelledError

    monkeypatch.setattr(ind_service, "PullbackSource", FakeSource)
    monkeypatch.setattr(ind_service, "PullbackStore", FakeStore)
    monkeypatch.setattr(ind_service, "Metrics", FakeMetrics)
    monkeypatch.setattr(ind_service.PullbackService, "run", fake_run)

    import pytest

    with pytest.raises(asyncio.CancelledError):
        await ind_service.run_indicator("dsn", tmp_path, "url", 9102)
    assert callable(attached[0]._on_rtt)
