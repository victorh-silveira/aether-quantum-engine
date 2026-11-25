"""Contratos do coletor publico e inventario de pesquisa de frequencia."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from scripts.analysis import frequency_data as data


def _bar(epoch: int, **changes: float) -> dict:
    """Cria uma vela normalizada com amplitude valida."""
    return {"epoch": epoch, "open": 10, "high": 12, "low": 9, "close": 11, **changes}


def test_normalization_rejects_conflicts_invalid_and_forming() -> None:
    """Duplicatas identicas sao contadas e conflitos eliminam a chave inteira."""
    rows = [
        _bar(0),
        _bar(0),
        _bar(900),
        _bar(900, close=10),
        _bar(2700),
        _bar(3600),
        _bar(4501),
        _bar(5400, close=float("nan")),
        _bar(6300, high=8),
        {},
    ]
    clean, inventory = data.validate_rows(rows, 900, 3600)
    assert [row["epoch"] for row in clean] == [0, 2700]
    assert inventory["duplicates"] == 2
    assert inventory["conflicting_epochs"] == 1
    assert inventory["invalid"] == 4
    assert inventory["forming_removed"] == 1
    assert inventory["gap_count"] == 1
    assert inventory["gap_seconds"] == 1800


def test_empty_inventory_is_explicit() -> None:
    """Ausencia de velas nao vira cobertura positiva."""
    clean, inventory = data.validate_rows([], 900, 3600)
    assert clean == []
    assert inventory["first_epoch"] is None
    assert inventory["last_epoch"] is None


class _Source:
    """Entrega paginas deterministicas sem acesso a rede."""

    def __init__(self, pages: list) -> None:
        self.pages = iter(pages)
        self.calls: list[tuple] = []

    async def candles(self, symbol, count, end, period):
        """Registra cursor e devolve a proxima pagina ou falha controlada."""
        self.calls.append((symbol, count, end, period))
        value = next(self.pages)
        if isinstance(value, Exception):
            raise value
        return value


@pytest.mark.asyncio
async def test_pagination_preserves_real_warmup_count() -> None:
    """Sessoes fechadas nao encurtam aquecimento para uma duracao de calendario."""
    start = 1_800_000
    before = [_bar(start - (index + 1) * 1800) for index in range(510)]
    source = _Source([[_bar(start), _bar(start + 900)], before])
    rows, meta = await data.collect_pair(source, "OTC_FTSE", 900, start, start + 1800, delay=0)
    assert len(rows) == 502
    assert meta["warmup_bars_before_start"] == 500
    assert source.calls[1][2] == start - 1
    assert all(row["period"] == 900 and row["symbol"] == "OTC_FTSE" for row in rows)


@pytest.mark.asyncio
async def test_partial_history_and_errors_are_recorded() -> None:
    """Falha da API preserva o que foi coletado e impede completude silenciosa."""
    source = _Source([[_bar(1800)], RuntimeError("indisponivel")])
    rows, meta = await data.collect_pair(source, "OTC_FTSE", 900, 1800, 3600, delay=0)
    assert len(rows) == 1
    assert "indisponivel" in meta["error"]
    assert meta["warmup_bars_before_start"] == 0
    empty, _ = await data.collect_pair(_Source([[]]), "OTC_FTSE", 900, 1800, 3600, delay=0)
    assert empty == []


@pytest.mark.asyncio
async def test_forward_page_is_not_accepted_as_progress() -> None:
    """API que ignora cursor termina com diagnostico verificavel."""
    _, meta = await data.collect_pair(_Source([[_bar(4500)]]), "OTC_FTSE", 900, 1800, 3600, delay=0)
    assert "posterior" in meta["error"]


@pytest.mark.asyncio
async def test_rate_limit_retry_is_bounded_and_keeps_cursor(monkeypatch: pytest.MonkeyPatch) -> None:
    """Retentativa por rate-limit preserva cursor sem perder pagina anterior."""
    sleeps = []

    async def sleep(seconds):
        sleeps.append(seconds)

    monkeypatch.setattr(data.asyncio, "sleep", sleep)
    source = _Source([RuntimeError("rate limit"), []])
    _, meta = await data.collect_pair(source, "OTC_FTSE", 900, 1800, 3600)
    assert sleeps == [30]
    assert source.calls[0] == source.calls[1]
    assert meta["error"] == ""
    source = _Source([RuntimeError("rate limit")] * 9)
    _, meta = await data.collect_pair(source, "OTC_FTSE", 900, 1800, 3600)
    assert "rate limit" in meta["error"]
    assert len(source.calls) == 9


@pytest.mark.asyncio
async def test_resume_verifies_dataset_hashes(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """Retomada so reutiliza pares congelados integros com aquecimento completo."""
    monkeypatch.chdir(tmp_path)
    monkeypatch.setattr(data, "SYMBOLS", ("OTC_FTSE",))
    monkeypatch.setattr(data, "freeze_inputs", lambda output: {"git_revision": "test"})
    calls = []

    class Source:
        async def close(self):
            return None

    async def pair(source, symbol, period, start, cutoff, *, seed_rows):
        calls.append(period)
        return [{"symbol": symbol, "period": period, **_bar(0)}], {
            "error": "",
            "rows_persisted": 1,
            "warmup_bars_before_start": 500,
        }

    monkeypatch.setattr(data, "PublicDeriv", Source)
    monkeypatch.setattr(data, "collect_pair", pair)
    output = tmp_path / "coleta"
    first = await data.collect(output, 3600, 1800)
    assert first["rows"] == 2
    assert first["calendar"]["coverage_complete"] is False
    second = await data.collect(output, 3600, 1800)
    assert second["sha256"] == first["sha256"]
    assert len(calls) == 2
    (output / "OTC_FTSE_900.parquet").write_bytes(b"alterado")
    with pytest.raises(ValueError, match="Hash divergente"):
        await data.collect(output, 3600, 1800)


def test_calendar_weekly_extremes_do_not_certify_year(tmp_path: Path) -> None:
    """Mesmo extremos anuais nao certificam completude de um cache semanal."""
    path = tmp_path / "calendar.json"
    assert data.calendar_inventory(path, 0, 100)["coverage_complete"] is False
    path.write_text(
        json.dumps(
            {
                "events": [
                    {"date": "2026-01-01T00:00:00+00:00"},
                    {"date": "2026-11-25T00:00:00+00:00"},
                    {"date": "2026-01-01T00:00:00"},
                    {"date": "invalido"},
                    {},
                ]
            }
        ),
        encoding="utf-8",
    )
    report = data.calendar_inventory(path, 0, 100)
    assert report["event_count"] == 2
    assert report["coverage_complete"] is False
    assert len(report["sha256"]) == 64


def test_freeze_is_stable_and_does_not_overwrite_previous_inputs(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Reexecucao preserva configuracao e calendario usados na primeira coleta."""
    monkeypatch.chdir(tmp_path)
    (tmp_path / "config").mkdir()
    (tmp_path / "data").mkdir()
    (tmp_path / ".git").mkdir()
    (tmp_path / ".git" / "HEAD").write_text("a" * 40)
    (tmp_path / "config" / "settings.json").write_text("{}")
    (tmp_path / "data" / "economic_calendar_cache.json").write_text('{"events": []}')
    paths = (
        "src/indicator/strategy_persistent_trend.py",
        "src/indicator/pullback.py",
        "src/domain/fundamental/economic_calendar.py",
        "src/domain/risk/pullback_stake.py",
    )
    for name in paths:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("codigo")
    output = tmp_path / "frozen"
    first = data.freeze_inputs(output)
    (tmp_path / "config" / "settings.json").write_text('{"changed": true}')
    second = data.freeze_inputs(output)
    assert second == first
    assert (output / "settings_baseline.json").read_text() == "{}"
    assert (output / "economic_calendar_snapshot.json").read_text() == '{"events": []}'
